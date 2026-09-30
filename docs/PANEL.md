# The RL panel

Two complementary views, both fully offline:

## 1. TensorBoard (generic metrics, pulled from the ecosystem)

The trainer is a HF Trainer, so all of HF's tracker integrations work with zero
custom code:

```bash
REPORT_TO=tensorboard bash scripts/run_async_dp.sh full _run1   # writes <run>/tb
ROOT=/path/to/project bash scripts/panel.sh                     # tensorboard on :6006
```

You get loss, reward, KL, entropy, grad-norm, completion-length, lr, MFU curves
with run comparison, smoothing, and wall-clock/relative/step x-axes. Other
backends need no code either: `REPORT_TO=wandb` (needs net), `mlflow`, `swanlab`
(local-first), comma-separate several.

## 2. Built-in HTML report (RL-specific curves TB cannot compute)

```bash
python -m rlforge.report --run <run> --trainer-log <trainer log> \
    --eval-history <eval history.jsonl> --base-eval <base eval.json> \
    --out <run>/report --interval 600
```

Refreshes every 10 minutes while the run is live:

- held-out eval ladder per checkpoint (with base model + constant baselines),
- **per-task reward / accuracy / truncation** curves reconstructed from the
  reward-call counters (reward includes the -2 truncation penalty; accuracy does
  not -- the pair of curves is what separates "getting worse" from "getting
  truncated"),
- per-source (dataset subset) curves when the dataset carries a `source` column,
- length / entropy / GSPO clip-fraction health panel,
- a self-contained `report.html` (images inlined) + `metrics.json` for scripting.

## What verl does, for reference

verl's "panel" is also just wandb/console tracking of trainer metrics plus its
rollout dumps; the RL-specific reading (per-task decomposition, held-out ladders)
is left to the user there too. Our layer-2 report is the piece that generic
trackers don't give you out of the box, on either stack.
