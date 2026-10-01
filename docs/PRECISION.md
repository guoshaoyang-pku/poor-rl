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

## Going lower than bf16

- FP8 rollout is independent of actor precision. The launcher defaults to `KV_DTYPE=fp8` on the validated H200 setup; set `KV_DTYPE=auto` to disable it or compare against BF16/auto. It can reduce KV-cache memory substantially. On other accelerators, verify vLLM support before use.
- `ROLLOUT_QUANTIZATION=fp8` asks vLLM to quantize rollout weights/compute. This can save additional memory, but has a larger numerical effect than FP8 KV cache and is model/backend dependent. The following short GSPO integration smoke passed, but FP8-weight quality and long-run stability are **not yet validated**; use it as an experimental arm, not the default.
- Keep actor master weights FP32 (`DTYPE=none`) with BF16 autocast (`MIXED_PRECISION=bf16`) for this experiment. Do not raise the learning rate merely to compensate for FP8 inference; first check policy-ratio/clip health against a BF16 rollout control.

### H200 smoke and paired held-out validation (vLLM 0.30, Qwen3.5-0.8B, 2026-10-01)

- With the same 24,576-token max model length, 0.85 GPU-memory utilization, eager mode, and single-GPU serving, `KV_DTYPE=fp8` reported 15,949,824 cached tokens (649.0 max-length sequences) vs 8,764,179 tokens (356.6 sequences) for BF16/auto KV: **1.82x cache capacity**. Both modes loaded and completed generation requests.
- Paired evaluation used the same `async_dp_flip450` checkpoint-500, 50 held-out questions, two samples per question, T=1.0/top-p=1.0/max-tokens=16,384, 16 concurrent requests, and matched request seeds. Across three repeated passes, mean accuracy was 33.7% (auto) vs 35.0% (FP8); parse rate 95.7% vs 95.3%; truncation 0% in both. Median generated-token throughput was 284 vs 309 tokens/s (**+8.6%** FP8), while mean request latency was 1.54 vs 1.46 s. This is a small, single-model/task probe: no statistically reliable accuracy gain is claimed, and other prompt lengths/load levels may show different throughput.
- FP8-weight GSPO was run for 3 steps on 360-1 H200 (vLLM 0.30, Qwen3.5-0.8B), with FP32 actor/master weights, BF16 autocast/compute, FP8 rollout weights, FP8 KV cache, 2e-6 LR, and matched 3-step BF16-weight control. Both completed live weight sync each step and produced changed FP32 checkpoints; nonzero gradients were observed (FP8 norm 9.1–16.1, BF16 norm 6.6–12.4). The brief run proves the training/sync/update path executes, **not** learning quality. Both arms showed very high `gspo/seq_clip_low_frac` (~0.98–1.00), so diagnose ratio/logprob alignment on a representative run before scaling up; FP8 weights are not approved as a production default.
- FP8 KV cache is marked **operationally validated and default for H200 runs**. On unsupported accelerators or when investigating regressions, explicitly set `KV_DTYPE=auto`. Temporary vLLM services were stopped after testing; the only remaining GPU usage was an unrelated user-owned benchmark, left untouched.

## How to verify updates are not being eaten

1. Log `grad_norm` (already in the trainer logs) and lr; the expected update RMS
   is ~ lr * grad_norm / sqrt(n_params) per step.
2. Compare against the dtype rounding floor: fp32 ~ 1e-7 relative, bf16 ~ 4e-3
   relative. If expected relative update < floor, switch masters to fp32.
3. Cheap empirical check: two 20-step smokes, pure bf16 vs fp32-master, same seed;
   if the pure-bf16 reward curve lags measurably, updates are being rounded away.
