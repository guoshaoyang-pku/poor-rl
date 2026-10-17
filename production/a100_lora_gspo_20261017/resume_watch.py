#!/usr/bin/env python3
"""Bounded auto-resume for the Qwen3.5-4B LoRA GSPO run (2026-10-17). The containers have no ssh to each other, so
the two roles coordinate only through files on the shared NFS.

head (a100_t1, next to head.py) follows RUN. On RUN/FAILED.json with a trainer-side reason it
  1. waits for the failed head, router and trainer to exit, for worker_IP_exit.json from every worker node, for free
     GPUs here, and for a live node watcher with free GPUs on every worker node;
  2. writes the next run (r5 -> r6) with make_plan.py from the newest valid RUN/trainer_out/checkpoint-*:
     INIT_ADAPTER = that checkpoint, DATA_START = RUN's --dataset-start-index + its rollout_state prompt_index,
     PRIOR_UPDATES = RUN's prior_updates + its step. Every other plan field (LR, B, ...) must equal RUN's;
  3. unloads the stale trl-policy-v* adapters from every backend, runs start_head.sh (and start_eval.sh), writes
     NEW/RESUMED.json and a line in STATE/ledger.jsonl, then follows NEW.
node (each worker node, a100_t1_2) follows RUN. When NEW/RESUMED.json (from_run RUN) appears it runs worker.sh on
  NEW/plan.json and follows NEW. A worker it launched that is still running KILL_AFTER s after FAILED.json is stopped.
Limits: at most MAX_RESUMES per WINDOW_H (ledger). Operator stops (the head got a signal), rollout fleet or router
start failures, and a RUN/NO_RESUME file are never resumed. Anything it cannot do safely writes
RUN/RESUME_BLOCKED.json (needs_thread_wakeup) and the watcher exits. It only signals processes it started: everything
it launches carries RESUME_WATCH=ROLE:RUN in its environment, which every descendant inherits, orphaned or not.
Usage: resume_watch.py {head|node|plan} RUN_DIR [options]    (plan: print what resuming RUN would launch; no writes)
"""
import argparse, datetime, json, math, os, re, signal, subprocess, sys, time, traceback, urllib.request
from pathlib import Path

KIT = Path(__file__).resolve().parent
RESUMABLE = ('trainer exited rc=', 'trainer rank crashed', 'trainer worker node', 'silent for more than',
             'router exited rc=')  # head.py fail() reasons; the rest (signal, backends, router start) need a human
KNOBS = ('MODEL', 'TRAINER_NODES', 'CPS_PER_RANK', 'MAX_STEPS', 'SAVE_STEPS', 'SB_TOKENS', 'SB_ACT_GB', 'MIN_HEALTHY',
         'INIT_ADAPTER', 'DATA_START', 'PRIOR_UPDATES', 'NGEN', 'CURRICULUM', 'STAGE', 'STAGE_DONE',
         'DROP_ZERO_VAR', 'ROLLOUT_HOSTS', 'AHEAD', 'DATA')  # the make_plan.py environment
RESUME_FLAGS = ('--init-adapter', '--dataset-start-index')
STAGE_FLAGS = ('--num-generations', '--max-steps')  # may change at a curriculum stage advance / mid-stage resume


class Blocked(Exception):
    pass


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path, obj):
    tmp = Path(str(path) + '.tmp'); tmp.write_text(json.dumps(obj, indent=2) + '\n'); os.replace(tmp, path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def my_ip():
    return subprocess.check_output(['hostname', '-I'], text=True).split()[0]


def next_run(run):
    m = re.fullmatch(r'(.*?)(\d+)', run.name)
    return run.parent / f'{m[1]}{int(m[2]) + 1}'


def launch(script, *args, marker):
    """Run one of the kit's detaching start scripts with RESUME_WATCH=marker; return the pid it prints."""
    out = subprocess.run(['bash', str(script), *map(str, args)], capture_output=True, text=True, timeout=300,
                         env={**os.environ, 'RESUME_WATCH': marker})
    m = re.search(r'\bpid (\d+)', out.stdout)
    if out.returncode or not m:
        raise Blocked(f'{script.name} rc={out.returncode}: {(out.stdout + out.stderr)[-400:]}')
    return int(m[1])


# ---- processes (Linux /proc)
def cmdline(pid):
    try:
        return Path(f'/proc/{int(pid)}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except (OSError, TypeError, ValueError):
        return ''


def alive(pid, needle):  # running (zombies have no cmdline) and the command line contains needle: no pid-reuse hits
    c = cmdline(pid) if pid else ''
    return bool(c) and needle in c


def owned(marker):
    """Live processes started by launch(..., marker=marker) and all their descendants (the environment is inherited)."""
    want, out = f'RESUME_WATCH={marker}'.encode(), []
    for p in Path('/proc').iterdir():
        if p.name.isdigit() and int(p.name) != os.getpid():
            try:
                if want in (p / 'environ').read_bytes().split(b'\0') and cmdline(p.name):
                    out.append(int(p.name))
            except OSError:
                pass
    return out


def stop(pids, term_wait=30):
    """SIGTERM, then SIGKILL, processes this watcher started (or their descendants). Returns the survivors."""
    for sig, wait in ((signal.SIGTERM, term_wait), (signal.SIGKILL, 10)):
        for p in pids:
            try:
                os.kill(p, sig)
            except (ProcessLookupError, PermissionError):
                pass
        t = time.time() + wait
        while time.time() < t and any(cmdline(p) for p in pids):
            time.sleep(1)
    return [p for p in pids if cmdline(p)]


def gpu_busy_mib():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=60).stdout
        return max(int(x) for x in out.split())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


# ---- backends
def http(url, body=None, timeout=10):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def policy_adapters(url):
    return sorted((m['id'] for m in json.loads(http(url + '/v1/models'))['data']
                   if re.fullmatch(r'trl-policy-v\d+', str(m.get('id')))), key=lambda n: int(n[12:]))


def unload_stale(backends):
    """Unload every trl-policy-vK the dead router left on the backends: the new run restarts at v1."""
    report = {}
    for b in backends:
        u = b['url']
        try:
            names = policy_adapters(u)
        except Exception as e:  # noqa: BLE001  a down backend comes back with no adapters
            report[u] = {'unreachable': repr(e)[:200]}
            continue
        errs = {}
        for n in names:
            try:
                http(u + '/v1/unload_lora_adapter', {'lora_name': n}, timeout=120)
            except Exception as e:  # noqa: BLE001
                errs[n] = repr(e)[:200]
        try:
            left = policy_adapters(u)
        except Exception as e:  # noqa: BLE001
            left = [f'unreachable after unload: {e!r}'[:200]]
        report[u] = {'unloaded': [n for n in names if n not in left], 'left': left, **({'errors': errs} if errs else {})}
    return report


# ---- checkpoint and plan
def safetensors_ok(path):
    try:
        with open(path, 'rb') as f:
            n = int.from_bytes(f.read(8), 'little')
            if n > 1 << 27:
                return False
            hdr = json.loads(f.read(n))
        end = max(v['data_offsets'][1] for k, v in hdr.items() if k != '__metadata__')
        return end > 0 and path.stat().st_size == 8 + n + end
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        return False


def pick_checkpoint(run):
    """Newest checkpoint-N with a complete adapter and a rollout_state.json (a save cut by the crash is skipped)."""
    ck = [(int(m[1]), p) for p in (run / 'trainer_out').glob('checkpoint-*')
          if (m := re.fullmatch(r'checkpoint-(\d+)', p.name))]
    skipped = []
    for step, p in sorted(ck, reverse=True):
        rs = read_json(p / 'rollout_state.json')
        if (isinstance(rs, dict) and isinstance(rs.get('prompt_index'), int) and (p / 'adapter_config.json').is_file()
                and safetensors_ok(p / 'adapter_model.safetensors')):
            return step, p, rs, skipped
        skipped.append(p.name)
    return None, None, None, skipped


def argval(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def normalized(plan):
    """The plan without its resume point and with its own run dir abstracted; must not change across resumes."""
    def strip(argv):
        out, it = [], iter(argv)
        for x in it:
            if x in RESUME_FLAGS or x in STAGE_FLAGS:
                next(it, None)
            else:
                out.append(x)
        return out
    p = {k: v for k, v in plan.items() if k not in ('resume', 'curriculum')}
    p['curriculum_schedule'] = (plan.get('curriculum') or {}).get('schedule')
    p['trainer_argv'] = strip(p['trainer_argv'])
    p['worker_argvs'] = {ip: strip(a) for ip, a in p['worker_argvs'].items()}
    return json.loads(re.sub(re.escape(plan['run_dir']) + r'(?=[/"])', '{RUN}', json.dumps(p)))


def derive(run, kit, advance=False):
    """Plan of the run that resumes RUN from its newest valid checkpoint, checked against RUN's plan.
    advance=True: RUN finished its curriculum stage (DONE.json); the new run is the next stage."""
    plan = json.loads((run / 'plan.json').read_text())
    nxt, argv = next_run(run), plan['trainer_argv']
    start0 = int(argval(argv, '--dataset-start-index') or 0)
    prior0 = int((plan.get('resume') or {}).get('prior_updates') or 0)
    step, ck, rs, skipped = pick_checkpoint(run)
    if ck:
        init, start, prior = str(ck), start0 + rs['prompt_index'], prior0 + step
    else:  # failed before its first save: rerun RUN's own starting point
        init, start, prior = argval(argv, '--init-adapter'), start0, prior0
    env = {k: v for k, v in os.environ.items() if k not in KNOBS}
    env.update(TRAINER_NODES=','.join(plan['trainer_nodes']), CPS_PER_RANK=argval(argv, '--completions-per-step'),
               MAX_STEPS=argval(argv, '--max-steps'), SAVE_STEPS=argval(argv, '--save-steps'),
               SB_TOKENS=argval(argv, '--subbatch-tokens'), SB_ACT_GB=plan['env']['RLFORGE_SB_ACT_GB'],
               MIN_HEALTHY=str(plan['min_healthy']), MODEL=argval(argv, '--model'), DATA_START=str(start),
               ROLLOUT_HOSTS=','.join(sorted({b['url'].split('//')[1].split(':')[0] for b in plan['backends']})),
               PRIOR_UPDATES=str(prior), NGEN=argval(argv, '--num-generations'), DATA=argval(argv, '--train'),
               AHEAD=argval(argv, '--rollout-ahead-batches') or '1',
               DROP_ZERO_VAR=plan['env'].get('RLFORGE_DROP_ZERO_VAR', '0'), **({'INIT_ADAPTER': init} if init else {}))
    cur = plan.get('curriculum')
    if cur:
        stage, done = int(cur['stage']), int(cur.get('stage_done') or 0) + (step if ck else 0)
        if advance:
            stage, done = stage + 1, 0
        env.update(CURRICULUM=cur['schedule'], STAGE=str(stage), STAGE_DONE=str(done))
    elif advance:
        raise Blocked('stage advance requested for a run without a curriculum')
    out = subprocess.run([sys.executable, str(kit / 'make_plan.py'), nxt.name], env=env, capture_output=True,
                         text=True, timeout=120)
    if out.returncode:
        raise Blocked(f'make_plan.py rc={out.returncode}: {out.stderr[-400:]}')
    new = json.loads(out.stdout)
    a, b = normalized(plan), normalized(new)
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    want = ['--init-adapter', init, '--dataset-start-index', str(start)] if init else []
    for av in [new['trainer_argv'], *new['worker_argvs'].values()]:
        got = [x for i, x in enumerate(av) if x in RESUME_FLAGS or (i and av[i - 1] in RESUME_FLAGS)]
        if got != want:
            raise Blocked(f'derived argv resume flags {got} != {want}')
    B, lr = new['batch_completions_per_update'], new['lr']
    if (diff or new['run_dir'] != str(nxt) or B != plan['batch_completions_per_update'] or lr != plan['lr']
            or not math.isclose(lr, 2e-5 * math.sqrt(B / 1024), rel_tol=1e-9)):
        raise Blocked(f'derived plan differs from {run.name} outside the resume point: {diff} (lr {lr}, B {B})')
    return new, {'kind': 'stage_advance' if advance else 'resume', 'curriculum': new.get('curriculum'),
                 'from_run': str(run), 'to_run': str(nxt), 'checkpoint': init, 'checkpoint_step': step,
                 'rollout_state': rs, 'skipped_checkpoints': skipped, 'dataset_start_index': start,
                 'prior_updates': prior, 'lr': lr, 'batch_completions_per_update': B,
                 'plan_check': f'equal to {run.name}/plan.json outside run_dir and the resume point'}


def has_next_stage(run):
    cur = (read_json(run / 'plan.json') or {}).get('curriculum')
    return bool(cur) and int(cur['stage']) + 1 < len(cur['schedule'].split(','))


# ---- roles
class Watcher:
    def __init__(self, a, role, hb_name):
        self.a, self.run, self.kit, self.ip = a, Path(a.run), Path(a.kit), my_ip()
        self.sd = Path(a.state_dir); self.sd.mkdir(parents=True, exist_ok=True)
        self.hb = self.sd / hb_name
        old = read_json(self.hb) or {}
        if (old.get('host') == self.ip and old.get('status') in ('watching', 'resuming')
                and alive(old.get('pid'), 'resume_watch.py')):
            sys.exit(f'another watcher (pid {old["pid"]}) owns {self.hb}')
        self.st = {'role': role, 'host': self.ip, 'pid': os.getpid(), 'started_utc': now(), 'seq': 0}
        signal.signal(signal.SIGTERM, lambda *_: (self.beat(status='stopped'), sys.exit(0)))

    def event(self, kind, **kw):
        with open(self.sd / 'events.jsonl', 'a') as f:
            f.write(json.dumps({'utc': now(), 'role': self.st['role'], 'host': self.ip, 'event': kind, **kw}) + '\n')

    def beat(self, **kw):
        self.st.update(kw, utc=now(), seq=self.st['seq'] + 1); atomic(self.hb, self.st)

    def gpu_ok(self, mib):
        return self.a.gpu_mib < 0 or (mib is not None and mib <= self.a.gpu_mib)

    def block(self, run, reason):
        atomic(run / 'RESUME_BLOCKED.json', {
            'utc': now(), 'status': 'blocked', 'reason': reason, 'needs_thread_wakeup': True,
            'next_action': 'Inspect FAILED.json and resume_watch/events.jsonl; repair; resume by hand; restart the watchers',
            'watcher': {'role': self.st['role'], 'host': self.ip, 'pid': os.getpid()}})
        self.event('blocked', run=str(run), reason=reason[:500])
        self.beat(status='blocked', following=str(run), reason=reason[:500])
        return 2


class Head(Watcher):
    def __init__(self, a):
        super().__init__(a, 'head', 'head_watch.json')
        self.st.update(max_resumes=a.max_resumes, window_h=a.window_h)

    def ledger(self):
        try:
            return [json.loads(l) for l in open(self.sd / 'ledger.jsonl') if l.strip()]
        except OSError:
            return []

    def waited(self, cond, timeout):
        t = time.time() + timeout
        while not cond():
            if time.time() > t:
                return False
            self.beat(); time.sleep(self.a.poll_s)
        return True

    def node_ready(self, wip, run):
        f = self.sd / f'node_{wip}.json'
        seq0 = (read_json(f) or {}).get('seq')

        def ok():  # the heartbeat advanced while we waited, so that watcher is alive now
            n = read_json(f) or {}
            return (n.get('seq') != seq0 and n.get('following') == str(run) and n.get('status') == 'watching'
                    and self.gpu_ok(n.get('gpu_busy_mib')))
        return self.waited(ok, self.a.node_wait_s)

    def resume(self, run, failed, advance=False):
        a, reason = self.a, str(failed.get('reason', ''))
        if (run / 'NO_RESUME').exists():
            raise Blocked('NO_RESUME present: not resumed')
        recent = []
        if not advance:
            if not any(r in reason for r in RESUMABLE):
                raise Blocked(f'not an auto-resumable failure: {reason[:300]}')
            t = time.time() - 3600 * a.window_h
            recent = [e for e in self.ledger() if e.get('kind') != 'stage_advance'
                      and datetime.datetime.fromisoformat(e['utc']).timestamp() > t]
            if len(recent) >= a.max_resumes:
                raise Blocked(f'resume budget used: {len(recent)} resumes in the last {a.window_h} h '
                              f'(max {a.max_resumes})')
        plan = json.loads((run / 'plan.json').read_text())
        if plan['trainer_nodes'][0] != self.ip:
            raise Blocked(f'this host {self.ip} is not the trainer head {plan["trainer_nodes"][0]}')
        self.beat(status='resuming', following=str(run), reason=reason[:300])
        hs = read_json(run / 'state.json') or {}
        pids = [hs.get('pid'), *(hs.get('procs') or {}).values()]
        if not self.waited(lambda: not any(alive(p, f'{run}/') for p in pids), a.head_exit_wait_s):
            raise Blocked(f'failed head/router/trainer of {run.name} still running after {a.head_exit_wait_s} s')
        pids = owned(f'head:{run}')  # of a run this watcher launched: ranks orphaned by the head's SIGKILL, its eval
        if pids:
            left = stop(pids)
            self.event('reaped', run=str(run), pids=pids, left=left)
            if left:
                raise Blocked(f'could not stop leftovers {left} of {run.name}')
        for wip in plan['worker_argvs']:
            if not self.waited(lambda: (run / f'worker_{wip}_exit.json').exists(), a.worker_exit_wait_s):
                raise Blocked(f'worker node {wip} of {run.name} did not exit within {a.worker_exit_wait_s} s')
        if not self.waited(lambda: self.gpu_ok(gpu_busy_mib()), a.gpu_wait_s):
            raise Blocked(f'GPUs on {self.ip} still busy after {a.gpu_wait_s} s')
        for wip in plan['worker_argvs']:
            if not self.node_ready(wip, run):
                raise Blocked(f'no live node watcher with free GPUs on {wip} following {run.name}')
        new, info = derive(run, self.kit, advance=advance)
        nxt = Path(new['run_dir'])
        if nxt.exists():
            raise Blocked(f'{nxt} already exists')
        stale = unload_stale(new['backends'])
        left = {u: r['left'] for u, r in stale.items() if r.get('left')}
        if left:
            raise Blocked(f'stale adapters still loaded: {left}')
        nxt.mkdir(parents=True)
        atomic(nxt / 'plan_gen.json', new)
        head_pid = launch(self.kit / 'start_head.sh', nxt / 'plan_gen.json', marker=f'head:{nxt}')
        eval_pid = eval_err = None
        if a.eval != 'off':
            try:
                eval_pid = launch(self.kit / 'start_eval.sh', nxt, *a.eval.split(), marker=f'head:{nxt}')
            except Exception as e:  # noqa: BLE001  the run goes on without its live eval
                eval_err = repr(e)[:300]
        rec = {'utc': now(), 'status': 'stage_advanced' if advance else 'resumed', **info, 'failure_reason': reason[:500],
               'failed_utc': failed.get('utc'), 'stale_adapters': stale, 'head_pid': head_pid, 'eval_pid': eval_pid,
               'eval_error': eval_err, 'resume_in_window': len(recent) + 1, 'max_resumes': a.max_resumes,
               'window_h': a.window_h, 'watcher': {'host': self.ip, 'pid': os.getpid()}, 'needs_thread_wakeup': False,
               'next_action': 'check the first steps of to_run (status.py); root-cause from_run/FAILED.json at review'}
        atomic(nxt / 'RESUMED.json', rec)
        with open(self.sd / 'ledger.jsonl', 'a') as f:
            f.write(json.dumps({'utc': rec['utc'], 'kind': info['kind'], 'from_run': str(run), 'to_run': str(nxt),
                                'head_pid': head_pid,
                                'eval_pid': eval_pid}) + '\n')
        self.event('resumed', **{k: rec[k] for k in ('from_run', 'to_run', 'checkpoint', 'dataset_start_index',
                                                     'prior_updates', 'head_pid', 'eval_pid')})
        return nxt

    def main(self):
        run, missing = self.run, 0
        try:
            while True:
                done = {e['from_run']: e['to_run'] for e in self.ledger()}
                if str(run) in done:  # resumed by an earlier incarnation of this watcher
                    run = Path(done[str(run)]); continue
                if (run / 'DONE.json').exists():
                    if has_next_stage(run):
                        self.event('stage_done', run=str(run))
                        run, missing = self.resume(run, {'reason': 'curriculum stage complete'}, advance=True), 0
                        continue
                    self.event('done', run=str(run)); self.beat(status='done', following=str(run)); return 0
                if (run / 'RESUME_BLOCKED.json').exists():
                    self.beat(status='blocked', following=str(run)); return 2
                failed = read_json(run / 'FAILED.json')
                if failed:
                    self.event('failed_seen', run=str(run), reason=str(failed.get('reason'))[:300])
                    run, missing = self.resume(run, failed), 0
                    continue
                hp = (read_json(run / 'state.json') or {}).get('pid')
                missing = missing + 1 if hp and not alive(hp, f'{run}/') else 0
                if missing >= 3:
                    raise Blocked(f'head pid {hp} of {run.name} exited without FAILED.json or DONE.json')
                self.beat(status='watching', following=str(run), head_pid=hp, reason=None)
                time.sleep(self.a.poll_s)
        except Blocked as e:
            return self.block(run, str(e))
        except Exception:  # noqa: BLE001
            return self.block(run, 'resume watcher error: ' + traceback.format_exc()[-1500:])


class Node(Watcher):
    def __init__(self, a):
        super().__init__(a, 'node', f'node_{my_ip()}.json')

    def main(self):
        run, ended, ip, a = self.run, None, self.ip, self.a
        try:
            while True:
                plan = read_json(run / 'plan.json')
                if plan is not None and ip not in plan.get('worker_argvs', {}):
                    self.event('error', run=str(run), reason=f'{ip} is not a worker node')
                    self.beat(status='error', following=str(run), reason=f'{ip} is not a worker node of {run.name}')
                    return 1
                exitf = run / f'worker_{ip}_exit.json'
                mine = read_json(run / f'worker_{ip}_launched.json') or {}
                own = mine.get('by') == 'resume_watch' and mine.get('host') == ip
                stage_end = (run / 'DONE.json').exists() and has_next_stage(run)
                if (run / 'DONE.json').exists() and not stage_end:
                    self.event('done', run=str(run)); self.beat(status='done', following=str(run)); return 0
                end = stage_end or (run / 'FAILED.json').exists() or (run / 'RESUME_BLOCKED.json').exists()
                ended = (ended or time.time()) if end else None
                if end and own and not exitf.exists() and time.time() - ended > a.kill_after_s:
                    pids = owned(f'node:{run}')
                    left = stop(pids)
                    atomic(exitf, {'utc': now(), 'rc': None, 'stopped_by': 'resume_watch node', 'pids': pids,
                                   'left': left})
                    self.event('worker_stopped', run=str(run), pids=pids, left=left)
                nxt = next_run(run)
                if end and exitf.exists() and (read_json(nxt / 'RESUMED.json') or {}).get('from_run') == str(run):
                    pid = launch(self.kit / 'worker.sh', nxt / 'plan.json', marker=f'node:{nxt}')
                    atomic(nxt / f'worker_{ip}_launched.json', {'utc': now(), 'pid': pid, 'by': 'resume_watch',
                                                                'host': ip, 'watcher_pid': os.getpid()})
                    self.event('worker_launched', run=str(nxt), pid=pid)
                    run, ended = nxt, None
                    continue
                if (run / 'RESUME_BLOCKED.json').exists() and (exitf.exists() or not own):
                    self.event('blocked', run=str(run)); self.beat(status='blocked', following=str(run)); return 2
                self.beat(status='watching', following=str(run), worker_exited=exitf.exists(),
                          gpu_busy_mib=None if a.gpu_mib < 0 else gpu_busy_mib())
                time.sleep(a.poll_s)
        except Exception:  # noqa: BLE001
            self.event('error', run=str(run), reason=traceback.format_exc()[-1500:])
            self.beat(status='error', following=str(run), reason=traceback.format_exc()[-500:])
            return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('role', choices=['head', 'node', 'plan'])
    ap.add_argument('run')
    ap.add_argument('--kit', default=str(KIT), help='dir with make_plan.py, start_head.sh, start_eval.sh, worker.sh')
    ap.add_argument('--state-dir', default=None, help='default: RUN/../../resume_watch')
    ap.add_argument('--poll-s', type=float, default=30)
    ap.add_argument('--max-resumes', type=int, default=2)
    ap.add_argument('--window-h', type=float, default=6)
    ap.add_argument('--head-exit-wait-s', type=float, default=600)
    ap.add_argument('--worker-exit-wait-s', type=float, default=2700, help='longer than the 30 min NCCL timeout')
    ap.add_argument('--kill-after-s', type=float, default=2100, help='node: stop its own worker this long after FAILED')
    ap.add_argument('--gpu-wait-s', type=float, default=900)
    ap.add_argument('--node-wait-s', type=float, default=600)
    ap.add_argument('--gpu-mib', type=int, default=2000, help='max memory.used per GPU that counts as free; -1: skip')
    ap.add_argument('--eval', default='20 2', help='start_eval.sh EVERY N for each resumed run, or off')
    a = ap.parse_args()
    a.run = os.path.abspath(a.run)
    a.state_dir = a.state_dir or str(Path(a.run).parent.parent / 'resume_watch')
    if not re.fullmatch(r'.*?\d+', Path(a.run).name):
        sys.exit(f'run dir name must end in a number: {a.run}')
    if a.role == 'plan':
        try:
            new, info = derive(Path(a.run), Path(a.kit))
        except Blocked as e:
            sys.exit(f'blocked: {e}')
        print(json.dumps({**info, 'plan': new}, indent=2))
        return 0
    return (Head(a) if a.role == 'head' else Node(a)).main()


if __name__ == '__main__':
    sys.exit(main())
