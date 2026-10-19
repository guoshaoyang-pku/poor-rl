#!/usr/bin/env python3
"""Check a finished dry drill against DRY_ROOT/scenario.json['expect']; prints PASS/FAIL lines.
Usage: check_dry.py DRY_ROOT"""
import json, os, sys
from pathlib import Path

root = Path(sys.argv[1]); runs = root / 'runs'; sd = root / 'resume_watch'
exp = json.loads((root / 'scenario.json').read_text())['expect']
ok = True


def check(cond, msg):
    global ok
    ok &= bool(cond)
    print(('PASS ' if cond else 'FAIL ') + msg)


def rj(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


hw, nw = rj(sd / 'head_watch.json') or {}, next((rj(p) for p in sd.glob('node_*.json')), {})
check(hw.get('status') == exp['final'], f"head watcher ends {hw.get('status')!r}, want {exp['final']!r}")
check(nw.get('status') == exp['node_final'], f"node watcher ends {nw.get('status')!r}, want {exp['node_final']!r}")
ledger = [json.loads(l) for l in open(sd / 'ledger.jsonl')] if (sd / 'ledger.jsonl').exists() else []
adv = [e for e in ledger if e.get('kind') == 'stage_advance']
ledger = [e for e in ledger if e.get('kind') != 'stage_advance']
check(len(adv) == exp.get('stage_advances', 0), f'{len(adv)} stage advances, want {exp.get("stage_advances", 0)}')
check(len(ledger) == len(exp['resumes']), f'{len(ledger)} resumes in the ledger, want {len(exp["resumes"])}')
p1 = rj(runs / 'r1/plan.json')
for e in exp['resumes']:
    to = runs / e['to']; r = rj(to / 'RESUMED.json') or {}; p = rj(to / 'plan.json') or {}
    check(r.get('checkpoint') == str(runs / e['ckpt']), f"{e['to']}: resumed from {r.get('checkpoint')}")
    check((r.get('dataset_start_index'), r.get('prior_updates')) == (e['data_start'], e['prior']),
          f"{e['to']}: DATA_START {r.get('dataset_start_index')} PRIOR_UPDATES {r.get('prior_updates')}, "
          f"want {e['data_start']} {e['prior']}")
    check(r.get('skipped_checkpoints', []) == e.get('skipped', []), f"{e['to']}: skipped {r.get('skipped_checkpoints')}")
    check(p.get('lr') == p1['lr'] and p.get('batch_completions_per_update') == p1['batch_completions_per_update'],
          f"{e['to']}: lr {p.get('lr')} B {p.get('batch_completions_per_update')} equal to r1")
    want = f"init_adapter={runs / e['ckpt']} dataset_start_index={e['data_start']} "
    logs = [to / 'trainer.log', *to.glob('trainer_node_*.log')]
    check(all(want in lg.read_text() for lg in logs) and len(logs) == 2,
          f"{e['to']}: rank 0 and the node-1 worker started with {want.strip()}")
    st = r.get('stale_adapters', {})  # the failed run's last versions; the mock's seeded v57..60 before the 1st resume
    gone = {'trl-policy-v25'} | ({'trl-policy-v57'} if e is exp['resumes'][0] else set())
    check(st and all(not v.get('left') and gone <= set(v.get('unloaded', [])) for v in st.values()),
          f"{e['to']}: stale adapters unloaded on all {len(st)} backends ({next(iter(st.values()), {}).get('unloaded')})")
    check(any(json.loads(l).get('by') == 'resume_watch' for l in map(Path.read_text, to.glob('worker_*_launched.json'))),
          f"{e['to']}: node-1 worker launched by the node watcher")
    check(r.get('head_pid') and r.get('eval_pid'), f"{e['to']}: head pid {r.get('head_pid')} eval pid {r.get('eval_pid')}")
if exp.get('blocked'):
    b = rj(runs / exp['blocked']['run'] / 'RESUME_BLOCKED.json') or {}
    check(exp['blocked']['reason'] in b.get('reason', '') and b.get('needs_thread_wakeup'),
          f"{exp['blocked']['run']}: RESUME_BLOCKED {b.get('reason', '')[:120]!r}")
    check(not (runs / f"r{int(exp['blocked']['run'][1:]) + 1}").exists(), 'no run dir after the blocked one')
if exp.get('stopped_worker'):
    x = next((rj(p) for p in (runs / exp['stopped_worker']).glob('worker_*_exit.json')), {}) or {}
    check(x.get('stopped_by') == 'resume_watch node' and x.get('left') == [],
          f"{exp['stopped_worker']}: hung worker stopped by the node watcher (pids {x.get('pids')})")
check(not any('COLLISION' in p.read_text() for p in runs.glob('*/trainer.log')), 'no stale adapter name collision')
def cmdline(p):
    try:
        return (p / 'cmdline').read_bytes().decode(errors='replace').replace('\0', ' ')
    except OSError:
        return ''


left = {int(p.name): c[:120] for p in Path('/proc').iterdir() if p.name.isdigit() and int(p.name) != os.getpid()
        and f'{root}/' in (c := cmdline(p)) and 'mock_vllm' not in c}
check(not left, f'no drill process left running {left}')
print('ALL PASS' if ok else 'SOME FAILED')
sys.exit(0 if ok else 1)
