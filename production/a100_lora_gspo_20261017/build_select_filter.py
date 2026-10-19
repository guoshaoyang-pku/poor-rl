#!/usr/bin/env python3
"""Informative-select stream for r13 (approved 2026-10-19): continue the mix45 stream from ABS_START, keep ranking rows
unchanged and in order, and draw select rows only from prompts that are not saturated in the TRAINING-side history.

Saturated = seen with >= MIN_SAMPLES completions across runs r4..r12 (task_split.jsonl, rollout-time scores of training
prompts; heldout290 is never read except to assert disjointness) and either always exact or never exact.
Kept select pool = mixed + unseen; re-shuffled each pass (seed). Blocks of 20 = 11 ranking + 9 select (45% select),
each block shuffled, as in train_think_mix45_s20261018.jsonl.
Usage: build_select_filter.py ABS_START OUT_NAME
"""
import collections, glob, hashlib, json, random, sys

D = '/home/cfs/guoshaoyang/aiq_rl/q35_4b_gspo_20261017'
SRC = f'{D}/data/train_think_mix45_s20261018.jsonl'
ABS_START, OUT = int(sys.argv[1]), sys.argv[2]
MIN_SAMPLES, SEED = 16, 20261019
RUNS = ['r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12']

hist = collections.defaultdict(lambda: [0, 0])
for r in RUNS:
    for f in glob.glob(f'{D}/runs/{r}/task_split.jsonl'):
        for line in open(f):
            x = json.loads(line)
            hist[x['qid']][0] += x['exact']; hist[x['qid']][1] += x['n']

src = open(SRC).read().splitlines()
rest = src[ABS_START:]
task = lambda r: json.loads(r)['task']  # noqa: E731
qid = lambda r: json.loads(r)['question_id']  # noqa: E731
ranking = [r for r in rest if task(r) == 'ranking']
select_all = {qid(r): r for r in src if task(r) != 'ranking'}  # unique select prompts of the release


def status(q):
    e, n = hist.get(q, (0, 0))
    if n < MIN_SAMPLES:
        return 'unseen' if n == 0 else 'thin'
    return 'solved' if e == n else 'never' if e == 0 else 'mixed'


cls = collections.Counter(status(q) for q in select_all)
keep = [r for q, r in select_all.items() if status(q) in ('mixed', 'unseen', 'thin')]
heldout = {json.loads(l)['question_id'] for l in open(f'{D}/data/heldout290_d2fix_canonical.jsonl') if l.strip()}
assert not heldout & set(select_all) and not heldout & {qid(r) for r in ranking}, 'heldout overlap'

rng = random.Random(SEED)
out, si, pool = [], 0, keep[:]
rng.shuffle(pool)
for b in range(len(ranking) // 11):
    block = ranking[b * 11:(b + 1) * 11]
    for _ in range(9):
        if si == len(pool):
            pool = keep[:]; rng.shuffle(pool); si = 0
        block.append(pool[si]); si += 1
    rng.shuffle(block)
    out += block
path = f'{D}/data/{OUT}'
open(path, 'w').write('\n'.join(out) + '\n')
meta = {'source': SRC, 'abs_start': ABS_START, 'seed': SEED, 'min_samples': MIN_SAMPLES, 'history_runs': RUNS,
        'history_source': 'runs/<r>/task_split.jsonl (training rollouts only; heldout290 not used)',
        'select_unique_by_status': dict(cls), 'select_kept_unique': len(keep),
        'select_removed_unique': cls['solved'] + cls['never'], 'rows': len(out),
        'tasks': dict(collections.Counter(task(r) for r in out)), 'ranking_rows_from_source': len(ranking),
        'heldout_overlap': 0, 'sha256': hashlib.sha256(open(path, 'rb').read()).hexdigest()}
json.dump(meta, open(path.replace('.jsonl', '.meta.json'), 'w'), indent=1)
print(json.dumps(meta, indent=1))
