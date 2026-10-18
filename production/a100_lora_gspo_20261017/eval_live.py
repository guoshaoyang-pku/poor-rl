#!/usr/bin/env python3
"""Low-frequency checkpoint eval, co-located on the rollout GPUs through the router (no reserved GPU).

For every complete RUN/trainer_out/checkpoint-N with N % EVERY == 0 (and once for the run's --init-adapter, N=0),
load that adapter through the router under its own name `eval-<run>-ckN` (the trainer's trl-policy-vK adapters are
never used: the trainer unloads them on its own schedule, which raced the old live eval), sample N_PER_Q natural-16k
completions per heldout290 question (T=1, top_p=1, top_k=0, max_tokens 16384, thinking template), grade with
aiq_think_reward_v4.score_one (strict: status == exact), unload the adapter, and append a summary to
RUN/eval_live.jsonl (raw samples: RUN/eval_live/ckN.jsonl.gz). Failed requests are retried up to 3 rounds; a point is
written only if every request was graded (`errors` must be 0), otherwise an `incomplete` record is written instead.
Needs one free LoRA slot per backend (serve_node.sh MAX_LORAS >= trainer's live adapters + 1).
Usage: eval_live.py RUN_DIR [EVERY=20] [N_PER_Q=2]
"""
import ast, collections, datetime, gzip, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RUN = Path(sys.argv[1]); EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 20; N = int(sys.argv[3]) if len(sys.argv) > 3 else 2
NFS = Path('/home/cfs/guoshaoyang/aiq_rl/q35_4b_gspo_20261017')
plan = json.loads((RUN / 'plan.json').read_text())
argv = plan['trainer_argv']
MODEL = argv[argv.index('--model') + 1]
INIT = argv[argv.index('--init-adapter') + 1] if '--init-adapter' in argv else None
PRIOR = int((plan.get('resume') or {}).get('prior_updates') or 0)
ROUTER = f"http://127.0.0.1:{plan['router_port']}"
sys.path.insert(0, str(NFS / 'data'))
from aiq_think_reward_v4 import score_one  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)
rows = [json.loads(l) for l in open(NFS / 'data/heldout290_d2fix_canonical.jsonl') if l.strip()]
prompts = []
for r in rows:
    msgs = r['prompt'] if isinstance(r['prompt'], list) else ast.literal_eval(r['prompt'])
    prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True))
(RUN / 'eval_live').mkdir(exist_ok=True)


def post(path, body, timeout=600):
    req = urllib.request.Request(ROUTER + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def done_steps():
    try:
        recs = [json.loads(l) for l in open(RUN / 'eval_live.jsonl') if '"step"' in l]
        return {r['step'] for r in recs if r.get('status') in ('ok', 'incomplete')}  # errors are retried
    except OSError:
        return set()


def complete(ck):
    return (ck / 'rollout_state.json').exists() and (ck / 'adapter_model.safetensors').exists()


def one(args):
    i, k, name = args
    body = {'model': name, 'prompt': prompts[i], 'max_tokens': 16384, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 0,
            'seed': 20261017 + 1000 * i + k}
    try:
        out = json.loads(post('/v1/completions', body, timeout=3600))
        text, ntok = out['choices'][0]['text'], out['usage']['completion_tokens']
    except Exception as e:  # noqa: BLE001
        return {'i': i, 'k': k, 'error': repr(e)[:300]}
    s = score_one(text, rows[i]['answer'], n_tokens=ntok, cap=16384, num_choices=int(rows[i]['num_choices']))
    return {'i': i, 'k': k, 'qid': rows[i]['question_id'], 'task': rows[i]['task'], 'status': s['status'],
            'reward': s['reward'], 'ntok': ntok, 'text': text}


def evaluate(step, path):
    name = f'eval-{RUN.name}-ck{step}'
    t0 = time.time()
    post('/v1/load_lora_adapter', {'lora_name': name, 'lora_path': str(path)})
    try:
        res, todo = {}, [(i, k, name) for i in range(len(rows)) for k in range(N)]
        for _ in range(3):
            with ThreadPoolExecutor(96) as ex:
                for r in ex.map(one, todo):
                    res[(r['i'], r['k'])] = r
            todo = [(i, k, name) for (i, k), r in res.items() if 'error' in r]
            if not todo:
                break
    finally:
        try:
            post('/v1/unload_lora_adapter', {'lora_name': name})
        except Exception:  # noqa: BLE001
            pass
    vals = list(res.values())
    ok = [r for r in vals if 'error' not in r]
    by = collections.defaultdict(list)
    for r in ok:
        by[r['task']].append(r['status'] == 'exact')
    pct = lambda xs: round(100 * sum(xs) / max(1, len(xs)), 2)  # noqa: E731
    summ = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'step': step, 'updates': PRIOR + step,
            'adapter': str(path), 'n_per_q': N, 'graded': len(ok), 'errors': len(vals) - len(ok),
            'status': 'ok' if len(ok) == len(vals) == len(rows) * N else 'incomplete',
            'acc_strict': pct([r['status'] == 'exact' for r in ok]),
            'by_task': {t: pct(x) for t, x in by.items()},
            'unparsed': pct([r['status'] == 'unparsed' for r in ok]), 'trunc': pct([r['status'] == 'trunc' for r in ok]),
            'mean_tokens': round(sum(r['ntok'] for r in ok) / max(1, len(ok))), 'wall_s': round(time.time() - t0)}
    with gzip.open(RUN / 'eval_live' / f'ck{step}.jsonl.gz', 'wt') as f:
        for r in vals:
            f.write(json.dumps(r) + '\n')
    with open(RUN / 'eval_live.jsonl', 'a') as f:
        f.write(json.dumps(summ) + '\n')


def router_ready():
    try:
        with urllib.request.urlopen(ROUTER + '/server_info?config_format=json', timeout=30) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def ended():
    """FAILED/DONE written by the current head of RUN (a marker older than its start is ignored)."""
    try:
        t0 = datetime.datetime.fromisoformat(json.loads((RUN / 'state.json').read_text())['started_utc']).timestamp()
    except Exception:  # noqa: BLE001
        t0 = 0
    return any((RUN / f).exists() and (RUN / f).stat().st_mtime >= t0 - 5 for f in ('FAILED.json', 'DONE.json'))


while not ended():
    if not router_ready():
        time.sleep(30)
        continue
    seen = done_steps()
    todo = []
    if INIT and 0 not in seen and Path(INIT).exists():
        todo.append((0, Path(INIT)))
    for ck in sorted((RUN / 'trainer_out').glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[1])):
        n = int(ck.name.split('-')[1])
        if n % EVERY == 0 and n not in seen and complete(ck):
            todo.append((n, ck))
    for step, path in todo[-1:]:  # newest pending point only; older ones are skipped if the run moved on
        try:
            evaluate(step, path)
        except Exception as e:  # noqa: BLE001
            with open(RUN / 'eval_live.jsonl', 'a') as f:
                f.write(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'step': step,
                                    'status': 'error', 'error': repr(e)[:300]}) + '\n')
    time.sleep(60)
