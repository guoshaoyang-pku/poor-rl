# Precision recipes

RL fine-tuning updates are tiny (lr ~1e-6..1e-5). In bf16, a weight w gets an
update u only if |u| is larger than roughly 2^-8 * |w| (bf16 has 8 mantissa bits);
smaller updates are silently rounded away. At lr=2e-6 that rounding floor is a
real fraction of the intended update, so **pure bf16 training can eat learning
entirely** -- we observed exactly this in our precision arms before settling on
the recipes below.

## The three recipes (launcher env: `DTYPE` x `MIXED_PRECISION`)

| Recipe | env | master weights | compute | when |
|---|---|---|---|---|
| pure fp32 | `DTYPE=none MIXED_PRECISION=no` | fp32 | fp32 | debugging; arm A reference |
| pure bf16 | `DTYPE=bfloat16 MIXED_PRECISION=no` | bf16 | bf16 | fast default at 0.8B with lr >= 2e-6 |
| fp32 master + bf16 compute | `DTYPE=none MIXED_PRECISION=bf16` | fp32 | bf16 (autocast) | **the low-precision recipe**: small lr, long runs, or whenever updates approach the bf16 rounding floor |

The third recipe is the one to build on for bf16-and-lower work: optimizer states
and master weights stay fp32 (accelerate keeps the model in fp32 and autocasts the
forward), so a 2e-6 update always lands, while the matmuls run at bf16 speed.

## Going lower than bf16 (planned)

- fp8 (E4M3) compute via torchao/transformer-engine on Hopper+: only with fp32
  masters, and only after the bf16-master recipe is validated on the target task.
- fp8 KV cache on the rollout server (`KV_DTYPE=fp8`): orthogonal to training
  precision but shifts sampled logprobs slightly; check `gspo/seq_clip_low_frac`
  and the reward curve against a bf16-KV control before adopting.

## How to verify updates are not being eaten

1. Log `grad_norm` (already in the trainer logs) and lr; the expected update RMS
   is ~ lr * grad_norm / sqrt(n_params) per step.
2. Compare against the dtype rounding floor: fp32 ~ 1e-7 relative, bf16 ~ 4e-3
   relative. If expected relative update < floor, switch masters to fp32.
3. Cheap empirical check: two 20-step smokes, pure bf16 vs fp32-master, same seed;
   if the pure-bf16 reward curve lags measurably, updates are being rounded away.
