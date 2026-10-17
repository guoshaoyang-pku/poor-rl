#!/usr/bin/env python3
"""Compact status of a run: supervisor state, per-step trainer metrics, router sync events.
Usage: status.py RUN_DIR [N_STEPS]"""
import ast, json, re, sys
from pathlib import Path

run = Path(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
st = json.loads((run / 'state.json').read_text())
print('state', st['status'], st.get('checked_utc'), 'alive', st.get('alive'), 'ckpts', st.get('checkpoints', [])[-3:])
if (run / 'FAILED.json').exists():
    print('FAILED', json.loads((run / 'FAILED.json').read_text())['reason'])
keys = ('step', 'reward', 'reward_std', 'grad_norm', 'loss', 'learning_rate', 'gspo/abs_log_rho_mean',
        'gspo/log_rho_mean', 'gspo/seq_clip_low_frac', 'gspo/seq_clip_high_frac', 'completions/mean_length',
        'completions/clipped_ratio', 'entropy', 'perf/mfu_fwd_bwd', 'step_time')
rows = []
for line in (run / 'trainer.log').read_text(errors='replace').splitlines():
    m = re.search(r"\{'loss'.*\}|\{'.*'grad_norm'.*\}", line)
    if m:
        try:
            rows.append(ast.literal_eval(m.group(0)))
        except Exception:
            pass
for r in rows[-n:]:
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()
           if k in keys or k.startswith('reward') or 'time' in k or k.startswith('completions/')})
print('logged_steps', len(rows))
tb = [l for l in (run / 'trainer.log').read_text(errors='replace').splitlines() if 'Traceback' in l or 'Error' in l]
print('error_lines', len(tb), tb[-3:])
