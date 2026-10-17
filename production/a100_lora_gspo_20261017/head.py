#!/usr/bin/env python3
"""Trainer-head supervisor for the Qwen3.5-4B LoRA GSPO run (2026-10-17).

Waits for the rollout backends, starts rlforge.rollout.router and the DDP trainer, then watches
both. Writes RUN/state.json every 30 s and RUN/events.jsonl; on any failure writes RUN/FAILED.json
(needs_thread_wakeup) with the log tails, cleans up only its own process groups, never retries.
Usage: head.py PLAN_JSON
"""
import datetime, json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path, obj):
    tmp = Path(str(path) + '.tmp'); tmp.write_text(json.dumps(obj, indent=2) + '\n'); os.replace(tmp, path)


def get(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status
    except Exception:
        return None


def tail(path, n=60):
    try:
        return Path(path).read_text(errors='replace').splitlines()[-n:]
    except OSError:
        return []


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    run = Path(plan['run_dir']); run.mkdir(parents=True, exist_ok=True)
    out = run / 'trainer_out'
    env = dict(os.environ)
    env.update({k: str(v) for k, v in plan['env'].items()})
    procs, state = {}, {'status': 'starting', 'plan': sys.argv[1], 'started_utc': now(), 'pid': os.getpid(),
                        'needs_thread_wakeup': False}

    def event(kind, **kw):
        with open(run / 'events.jsonl', 'a') as f:
            f.write(json.dumps({'utc': now(), 'event': kind, **kw}) + '\n')

    def save(**kw):
        state.update(kw, checked_utc=now(), procs={k: p.pid for k, p in procs.items()},
                     alive={k: p.poll() is None for k, p in procs.items()},
                     checkpoints=sorted(x.name for x in out.glob('checkpoint-*')) if out.exists() else [])
        atomic(run / 'state.json', state)

    def spawn(name, argv, extra_env=None):
        log = open(run / f'{name}.log', 'ab', buffering=0)
        p = subprocess.Popen(argv, env={**env, **(extra_env or {})}, cwd=str(run), stdout=log,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        procs[name] = p; event('launched', name=name, pid=p.pid, argv=argv); save()
        return p

    def cleanup():
        for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 5)):
            for p in procs.values():
                if p.poll() is None:
                    try:
                        os.killpg(p.pid, sig)
                    except ProcessLookupError:
                        pass
            t = time.time() + wait
            while time.time() < t and any(p.poll() is None for p in procs.values()):
                time.sleep(0.5)

    def fail(reason):
        receipt = {'utc': now(), 'status': 'failed', 'reason': reason, 'needs_thread_wakeup': True,
                   'next_action': 'Inspect logs; keep evidence; bounded repair; resume from latest checkpoint-*',
                   'trainer_tail': tail(run / 'trainer.log'), 'router_tail': tail(run / 'router.log', 20)}
        atomic(run / 'FAILED.json', receipt); event('failed', reason=reason)
        cleanup(); save(status='failed', needs_thread_wakeup=True, error=reason)
        sys.exit(1)

    def on_signal(signum, _frame):
        fail(f'supervisor received signal {signum}')
    signal.signal(signal.SIGTERM, on_signal); signal.signal(signal.SIGINT, on_signal)

    backends = plan['backends']
    deadline = time.time() + plan.get('backend_wait_s', 1800)
    while True:
        healthy = [b for b in backends if get(b['url'] + '/health') == 200]
        save(status='waiting_backends', healthy_backends=len(healthy), total_backends=len(backends))
        if len(healthy) >= plan['min_healthy']:
            break
        if time.time() > deadline:
            fail(f'only {len(healthy)}/{len(backends)} backends healthy before deadline')
        time.sleep(20)
    event('backends_ready', healthy=len(healthy), total=len(backends))

    py = plan['python']
    r_argv = [py, '-m', 'rlforge.rollout.router', '--host', '0.0.0.0', '--port', str(plan['router_port']),
              '--stage-root', str(run / 'router_stage'), '--log', str(run / 'router_events.jsonl'),
              '--min-healthy', str(plan['min_healthy'])]
    for b in backends:
        r_argv += ['--backend', b['url'] + (',agent=' + b['agent'] if b.get('agent') else '')]
    spawn('router', r_argv)
    t = time.time() + 120
    while get(f"http://127.0.0.1:{plan['router_port']}/server_info?config_format=json", timeout=30) != 200:
        if procs['router'].poll() is not None or time.time() > t:
            fail('router did not come up')
        time.sleep(3)
    event('router_ready')

    spawn('trainer', plan['trainer_argv'], plan.get('trainer_env'))
    save(status='running')
    while True:
        time.sleep(30)
        tr, ro = procs['trainer'].poll(), procs['router'].poll()
        if tr is not None:
            event('trainer_exit', rc=tr)
            if tr != 0:
                fail(f'trainer exited rc={tr}')
            cleanup(); save(status='complete', needs_thread_wakeup=True,
                            next_action='Review final checkpoint; decide next segment')
            atomic(run / 'DONE.json', {'utc': now(), 'status': 'complete'})
            return
        if ro is not None:
            fail(f'router exited rc={ro}')
        # a crashed rank can leave accelerate's launcher hanging on the survivors
        for ip in plan.get('worker_argvs', {}):
            wx = run / f'worker_{ip}_exit.json'
            if wx.exists() and (json.loads(wx.read_text()).get('rc') != 0
                                or time.time() - wx.stat().st_mtime > 600):  # rc 0 = stage end; trainer exit follows
                fail(f'trainer worker node {ip} exited: ' + wx.read_text()[:300])
        try:
            last = (run / 'trainer.log').stat().st_mtime
        except OSError:
            last = time.time()
        if time.time() - last > plan.get('stall_s', 3600):
            fail(f"trainer.log silent for more than {plan.get('stall_s', 3600)} s (stall)")
        logs = [run / 'trainer.log'] + [run / f'trainer_node_{ip}.log' for ip in plan.get('worker_argvs', {})]
        if any('ChildFailedError' in l or 'CUDA out of memory' in l for lg in logs for l in tail(lg, 300)):
            fail('trainer rank crashed (ChildFailedError / CUDA OOM in trainer.log)')
        save(status='running')


if __name__ == '__main__':
    main()
