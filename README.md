# rlforge

Async **GSPO/GRPO** RL post-training for generative models (LLM/VLM), built on
[TRL](https://github.com/huggingface/trl)'s experimental `AsyncGRPOTrainer` + a
vLLM rollout server — plus the guardrails you need to run experiments unattended
on a single GPU node, fully offline.

Developed and validated on an 8xH200 node training a 0.8B model for 1500 steps:
held-out accuracy 0.11 (base) -> 0.81, with every failure mode below hit (and
fixed) in production first.

## Why not just TRL / verl?

| | rlforge | TRL stock | verl |
|---|---|---|---|
| GSPO sequence-level IS | yes, with `seq_mean` normalization | sync only, token-mean normalization | yes (set `loss_agg_mode=seq-mean-token-mean`) |
| Async rollout (staleness) | yes, static GPU split (e.g. 4+4) | yes (experimental) | colocate or separate_async |
| Reward truncation signal | length-aware (`-2` on cap hit) | n/a | reward managers don't pass lengths by default |
| Unattended runs | watchdog (5 stop rules) + in-loop held-out eval + keep-best | no | no |
| Offline auto-report | self-contained HTML + metrics.json every N min | wandb (needs net) | wandb |
| Scale | single node, full-DP, ~0.5-3B | single/multi node | multi-node, FSDP, 8B-70B+ |

Scope: RL post-training of generative models at single-node scale. For classic
control (DQN etc.) use Stable-Baselines3/CleanRL; for >=8B or multi-node use verl.

## Install

```bash
pip install -e .          # or: pip install -r requirements.txt && pip install -e . --no-deps
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
bash scripts/run_async_dp.sh full _run1 &

# watchdog: in-loop held-out eval + keep-best + auto-stop
python -m rlforge.watchdog --run $ROOT/runs/async_dp_run1 \
    --trainer-log $ROOT/logs/trainer_dp_run1.log --launcher-pid $! \
    --eval-gpu 7 --eval-data data/eval.jsonl &

# auto-report (local wandb): refreshes report.html + metrics.json every 10 min
python -m rlforge.report --run $ROOT/runs/async_dp_run1 \
    --trainer-log $ROOT/logs/trainer_dp_run1.log \
    --eval-history $ROOT/evals/async_dp_run1/history.jsonl \
    --out $ROOT/runs/async_dp_run1/report
```

## Custom rewards

Any `module:function` with the TRL signature works:

```bash
python -m rlforge.trainer --reward my_project.rewards:my_fn ...
```

The built-in grader (`rlforge.rewards.mcq`) scores MCQ letters and 5-choice
ranking chains, and writes per-task / per-source reward/accuracy/truncation
counters to `$RLFORGE_TASK_LOG` (the report turns them into curves). Eval and
training import the same grader code, so numbers are comparable by construction.

## The GSPO contract (what we got wrong before you)

- **Normalization matters more than the ratio.** Averaging the loss over the
  global token count weights every sequence by its length; combined with a `-2`
  truncation penalty, the longest (truncated) sequences dominate the gradient and
  the policy collapsed ~12x faster than token-level GRPO at the same lr. Use
  `seq_mean` (uniform per sequence) — it's also the paper's objective.
- **Token-level clip values are a no-op at sequence level.** eps=0.2/0.28 never
  engages on a per-sequence ratio. The paper's 3e-4/4e-4 is right for on-policy;
  with staleness >1, calibrate empirically (we run 0.007/0.008 at staleness 3)
  and watch `gspo/seq_clip_low_frac` — sustained >=0.5 is the collapse signature.
- **Off-policy depth pushes rho below 1 systematically.** With staleness>=2 the
  low-side clip does real work; that's the safe direction, but read it together
  with the held-out curve, not alone.

`tests/test_gspo_core.py` pins all of this: ratio, both normalizations, clip
engagement, gradient direction, and zero-gradient-outside-clip, against a naive
reference implementation.

## Pitfalls this framework guards against

1. **TRL's 120s request timeout** kills slow-but-fine long completions forever ->
   set `--request-timeout` for the worst case (we use 3600).
2. **Reward parsers that test one separator** (`">" in gold`) silently mis-score
   every ranking answer when the key format changes -> shape-based detection.
3. **Truncated completions scored as "unparseable"** -> the reward sees token
   counts, and `-2` actually fires.
4. **Eval numbers without generation conditions** -> the eval report records
   decoding params, truncation rate, per-task breakdown and a constant baseline.
5. **save_total_limit deleting your peak** -> watchdog copies the best
   checkpoint to `keep_best/` the moment a new best eval lands.
6. **Missing tokenizer files in checkpoints** -> the evaluator builds a shim
   from the base model (never weights).

## Repo layout

```
src/rlforge/trainer.py    GSPOAsyncGRPOTrainer + CLI (python -m rlforge.trainer)
src/rlforge/gspo.py       naive reference math for the GSPO loss
src/rlforge/rewards/      reward protocol + built-in MCQ/ranking grader
src/rlforge/eval_mcq.py   held-out evaluator (vLLM), shared scoring path
src/rlforge/watchdog.py   in-loop eval / keep-best / auto-stop
src/rlforge/report.py     auto HTML report (local wandb substitute)
scripts/run_async_dp.sh   single-node launcher (vLLM server + DP trainer)
tests/                    GSPO core math + reward/parser tests
examples/aiq_mcq/         data format + a runnable example config
```

## Citation

GSPO: [arXiv 2507.18071](https://arxiv.org/abs/2507.18071). Built on TRL and vLLM.

## License

Apache-2.0
