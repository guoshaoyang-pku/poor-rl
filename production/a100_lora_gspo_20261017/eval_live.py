#!/usr/bin/env python3
"""Low-frequency live eval, co-located on the rollout GPUs through the router (no reserved GPU).

Whenever the newest loaded policy adapter (router_events.jsonl `load_lora` trl-policy-vK) is >= EVERY versions past
the last eval, sample N natural-16k completions per heldout290 question with that adapter (T=1, top_p=1, top_k=0,
max_tokens 16384, thinking template), grade with aiq_think_reward_v4.score_one (strict: status == exact), and append
a summary line to RUN/eval_live.jsonl (raw samples: RUN/eval_live/vK.jsonl.gz). Version vK = policy after K-1 updates.
Usage: eval_live.py RUN_DIR [EVERY=20] [N=2]
"""
import ast, collections, datetime, gzip, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RUN = Path(sys.argv[1]); EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 20; N = int(sys.argv[3]) if len(sys.argv) > 3 else 2
NFS = Path('/home/cfs/guoshaoyang/aiq_rl/q35_4b_gspo_20261017')
plan = json.loads((RUN / 'plan.json').read_text())
MODEL = plan['trainer_argv'][plan['trainer_argv'].index('--model') + 1]
URL = f"http://127.0.0.1:{plan['router_port']}/v1/completions"
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


def newest_version():
    best = 0
    try:
        for line in open(RUN / 'router_events.jsonl'):
            e = json.loads(line)
            if e.get('event') == 'load_lora' and e.get('ok') and str(e.get('name', '')).startswith('trl-policy-v'):
                best = max(best, int(e['name'].rsplit('v', 1)[1]))
    except OSError:
        pass
    return best


def one(args):
    i, k, name = args
    body = {'model': name, 'prompt': prompts[i], 'max_tokens': 16384, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 0,
            'seed': 20261017 + 1000 * i + k}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            out = json.loads(resp.read())
        text, ntok = out['choices'][0]['text'], out['usage']['completion_tokens']
    except Exception as e:  # noqa: BLE001
        return {'i': i, 'k': k, 'error': repr(e)[:300]}
    s = score_one(text, rows[i]['answer'], n_tokens=ntok, cap=16384, num_choices=int(rows[i]['num_choices']))
    return {'i': i, 'k': k, 'qid': rows[i]['question_id'], 'task': rows[i]['task'], 'status': s['status'],
            'reward': s['reward'], 'ntok': ntok, 'text': text}


last = -10 ** 9
while not (RUN / 'FAILED.json').exists() and not (RUN / 'DONE.json').exists():
    v = newest_version()
    if v and v - last >= EVERY:
        t0 = time.time()
        with ThreadPoolExecutor(96) as ex:
            res = list(ex.map(one, [(i, k, f'trl-policy-v{v}') for i in range(len(rows)) for k in range(N)]))
        ok = [r for r in res if 'error' not in r]
        by = collections.defaultdict(list)
        for r in ok:
            by[r['task']].append(r['status'] == 'exact')
        summ = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'version': v, 'updates': v - 1,
                'n_per_q': N, 'graded': len(ok), 'errors': len(res) - len(ok),
                'acc_strict': round(100 * sum(r['status'] == 'exact' for r in ok) / max(1, len(ok)), 2),
                'by_task': {t: round(100 * sum(x) / len(x), 2) for t, x in by.items()},
                'unparsed': round(100 * sum(r['status'] == 'unparsed' for r in ok) / max(1, len(ok)), 2),
                'trunc': round(100 * sum(r['status'] == 'trunc' for r in ok) / max(1, len(ok)), 2),
                'mean_tokens': round(sum(r['ntok'] for r in ok) / max(1, len(ok))), 'wall_s': round(time.time() - t0)}
        with gzip.open(RUN / 'eval_live' / f'v{v}.jsonl.gz', 'wt') as f:
            for r in res:
                f.write(json.dumps(r) + '\n')
        with open(RUN / 'eval_live.jsonl', 'a') as f:
            f.write(json.dumps(summ) + '\n')
        last = v
    time.sleep(60)
