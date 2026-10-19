#!/usr/bin/env python3
"""Alerts (never actions) for the 4B GSPO line; follows RUN and its resumes/stage advances via the resume_watch ledger.

  step_time  : two consecutive optimizer steps with fwd_bwd + rollout_wait + weight_sync > STEP_S (default 900 s)
  rank_trunc : over the most recent WINDOW scored ranking groups (default 320, about 5 steps at 64 groups/step),
               truncated ranking completions > TRUNC_FRAC (default 1%) of the ranking completions
Each alert is appended to RUN/ALERTS.jsonl and written once to RUN/ALERT_<kind>.json (re-armed when it clears), for
the coordinating session's patrol to pick up. Usage: alert_watch.py RUN_DIR
"""
import ast, datetime, json, os, re, sys, time
from pathlib import Path

RUN = Path(sys.argv[1])
NFS = Path('/home/cfs/guoshaoyang/aiq_rl/q35_4b_gspo_20261017')
STEP_S = float(os.environ.get('ALERT_STEP_S', 900)); TRUNC = float(os.environ.get('ALERT_TRUNC_FRAC', 0.01))
WINDOW = int(os.environ.get('ALERT_WINDOW_GROUPS', 320))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def steps(run):
    out = []
    try:
        for line in open(run / 'trainer.log', errors='replace'):
            if "'grad_norm'" in line:
                d = ast.literal_eval(re.search(r'\{.*\}', line).group(0))
                out.append(sum(float(d.get(k, 0) or 0) for k in ('perf/fwd_bwd_s', 'perf/rollout_wait_s', 'perf/weight_sync_s')))
    except (OSError, ValueError, SyntaxError, AttributeError):
        pass
    return out


def rank_trunc(run):
    try:
        rows = [json.loads(l) for l in open(run / 'task_split.jsonl') if '"ranking"' in l]
    except (OSError, ValueError):
        return None
    rows = [r for r in rows if r.get('task') == 'ranking'][-WINDOW:]
    if len(rows) < WINDOW:
        return None
    n = sum(r['n'] for r in rows)
    return sum(r.get('trunc', 0) for r in rows) / max(1, n)


def next_run(run):
    try:
        for l in open(NFS / 'resume_watch/ledger.jsonl'):
            e = json.loads(l)
            if e.get('from_run') == str(run):
                return Path(e['to_run'])
    except (OSError, ValueError):
        pass
    return None


def alert(run, kind, active, **kw):
    sticky = run / f'ALERT_{kind}.json'
    if active and not sticky.exists():
        rec = {'utc': now(), 'kind': kind, 'run': str(run), **kw,
               'note': 'alert only: no action taken; report to the coordinating session'}
        sticky.write_text(json.dumps(rec, indent=1) + '\n')
        with open(run / 'ALERTS.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
    elif not active and sticky.exists():
        sticky.rename(run / f'ALERT_{kind}.cleared_{int(time.time())}.json')


run = RUN
while True:
    nxt = next_run(run)
    if nxt is not None and nxt.exists():
        run = nxt
        continue
    s = steps(run)
    if len(s) >= 2:
        alert(run, 'step_time', s[-1] > STEP_S and s[-2] > STEP_S, last_two_step_s=[round(s[-2]), round(s[-1])],
              threshold_s=STEP_S, step=len(s))
    t = rank_trunc(run)
    if t is not None:
        alert(run, 'rank_trunc', t > TRUNC, ranking_trunc_frac=round(t, 4), threshold=TRUNC, window_groups=WINDOW)
    time.sleep(120)
