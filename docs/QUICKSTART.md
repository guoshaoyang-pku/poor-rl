# Quickstart

## Install

```bash
pip install -e ".[panel]"
```

## Quickstart (single node, 8 GPUs)

```bash
export ROOT=/path/to/project   # will hold data/ logs/ runs/ evals/
export VENV=/path/to/venv
export MODEL=/path/to/base/model
export RLFORGE_BASE_MODEL=$MODEL

# train.jsonl rows: {"prompt": [...chat...] or "...", "answer": "C" | "A<B<C<D<E", "source": "..."}
ROOT=$ROOT VENV=$VENV MODEL=$MODEL \
GSPO=1 GSPO_EPS_LOW=0.007 GSPO_EPS_HIGH=0.008 MAX_STEPS=1500 \
REPORT_TO=swanlab \
bash scripts/run_async_dp.sh full _run1 &

# watchdog: in-loop held-out eval + keep-best + auto-stop
python -m rlforge.watchdog --run $ROOT/runs/async_dp_run1 \
    --trainer-log $ROOT/logs/trainer_dp_run1.log --launcher-pid $! \
    --eval-gpu 7 --eval-data data/eval.jsonl &

# panel 1 (general): SwanLab comparison dashboard
ROOT=$ROOT VENV=$VENV bash scripts/panel.sh  # http://127.0.0.1:5092

# panel 2 (RL-specific): auto HTML report every 10 min
python -m rlforge.report --run $ROOT/runs/async_dp_run1 \
    --trainer-log $ROOT/logs/trainer_dp_run1.log \
    --eval-history $ROOT/evals/async_dp_run1/history.jsonl \
    --out $ROOT/runs/async_dp_run1/report
```

### Adaptive clip-fraction caps

With staleness >1 the low-side sequence clip can engage on a large fraction of
sequences. Instead of guessing `eps`, cap the clipped fraction directly and let
`eps` widen up to a ceiling:

```bash
ADAPT_CLIP_LOW_MAX_FRAC=0.1 ADAPT_CLIP_HIGH_MAX_FRAC=0.1 GSPO_EPS_MAX=0.1 \
ROOT=$ROOT VENV=$VENV MODEL=$MODEL GSPO=1 \
bash scripts/run_async_dp.sh full _run1 &
```

Watch `gspo/eps_low`, `gspo/eps_high`, and `gspo/seq_active_clip_*_frac` in the
tracker to see the controller at work.

## Custom rewards

Any `module:function` with the TRL signature works:

```bash
python -m rlforge.trainer --reward my_project.rewards:my_fn ...
```

The launcher also accepts a `REWARD` environment variable
(default `rlforge.rewards.mcq:mcq_reward`).

See [`examples/custom_reward/arith.py`](../examples/custom_reward/arith.py) for a
minimal annotated example (truncation detection, task-log counters, dataset-column
pass-through). The built-in grader (`rlforge.rewards.mcq`) scores MCQ letters and
5-choice ranking chains, and writes per-task / per-source reward/accuracy/
truncation counters to `$RLFORGE_TASK_LOG` — the report turns them into curves.
Eval and training import the same grader code, so numbers are comparable by
construction.
