# Example: MCQ + ranking RL on ArchitectureIQ-style data

Data format (`train.jsonl`, one JSON per line):

```json
{"prompt": [{"role": "user", "content": "...question with choices A/B/C...\nFinish with <answer>X</answer>."}], "answer": "C", "source": "my-pack", "task": "mcq"}
{"prompt": [{"role": "user", "content": "...rank 5 losses...\nAnswer like <answer>A<B<C<D<E</answer> (lower held-out loss first)."}], "answer": "A<B<E<C<D", "source": "loss_ranked", "task": "ranking"}
```

- `prompt`: chat list (applied through the model's chat template) or a plain string.
- `answer`: a single letter (MCQ) or a 5-letter chain (ranking, best-first).
- `source` (optional): enables per-source reward/accuracy/truncation curves in the
  task log and auto-report.
- `task` (optional): informational; the grader detects ranking by gold shape.

Run (see repo README for the full layout):

```bash
export ROOT=$PWD/demo VENV=/path/to/venv MODEL=/path/to/model
mkdir -p $ROOT/data && cp train.jsonl eval.jsonl $ROOT/data/
GSPO=1 MAX_STEPS=200 bash ../../scripts/run_async_dp.sh full _demo
```

Notes:

- Ranking gold letters must be shuffled per question (never `A<B<C<D<E` for every
  row -- a constant-order key hands out free reward; the ArchitectureIQ pool
  asserts zero such rows after generation).
- Keep the answer contract short (one closing tag). Contracts that demand extra
  sections push completions into the token cap and turn `-2` from a real signal
  into background noise.
- Eval rows must be disjoint from train rows at the question, pair, and
  source-instance level, or the held-out curve is meaningless.
