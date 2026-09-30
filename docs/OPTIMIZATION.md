# Optimization inventory (with estimates)

Workload profile, measured on the v3full run (8xH200, Qwen3.5-0.8B, GSPO,
staleness=3, CPS=256x4, NGEN=16, completion cap 16k):

- **Long-CoT phase** (first ~200 steps of a reasoning run): completions ~10-16k
  tokens, prompts ~6.5k tokens. Rollout-dominated (40-100 s/step).
- **Short-answer phase** (converged): completions ~16 tokens. 13.8 s/step,
  trainer + pipeline-overhead dominated.

Recon on the deployed stack (TRL 1.14 / vLLM 0.30), so we don't claim wins that
are already on: flash-attn3 attention, chunked lm_head (8192-token logprob
chunks, logits never fully materialized), prefix caching (vLLM default ON),
beta=0 (no ref-model forward), packed padding-free rows.

## Rollout (vLLM server)

| # | knob | estimate (long-CoT) | mechanism | A100 |
|---|---|---|---|---|
| R1 | prefix caching **verify** | +20-30% | TRL sends NGEN=16 separate n=1 requests with the identical 6.5k prompt; 15/16 prefills should be cache hits. Already default-on; we assert the hit rate in logs | ok |
| R2 | `KV_DTYPE=fp8` | +20-50% when KV-bound | halves KV bytes -> ~2x resident sequences | **no** (Hopper+) — keep opt-in |
| R3 | `ASYNC_SCHED=1` | +5-10% | overlaps scheduler CPU with GPU at high request churn (our n=1x16 pattern) | ok |
| R4 | `FI_SAMPLER=1` | +3-8% | batched flashinfer sampling kernel | test |
| R5 | speculative decoding | **rejected** | batch-bound rollout; spec-decode helps low-batch latency, not throughput | — |

## Trainer

| # | knob | estimate | mechanism | A100 |
|---|---|---|---|---|
| T1 | `LIGER=1` (base kernels only) | +8-15% | fused RMSNorm/RoPE/SwiGLU. TRL blocks `use_liger_kernel` in async mode and fused-linear-CE would bypass logprob scoring, so we monkey-patch elementwise kernels only | ok (triton) |
| T2 | `GRAD_CKPT=0` | +15-30% | removes the recompute forward; memory fits at <=1B on 140GB cards | keep ON (80/40GB) |
| T3 | `OPTIM=adamw_torch_fused` | +3-8% | fused optimizer kernel | ok |
| T4 | `DYNAMO=inductor` | +10-20% if stable | torch.compile; risk: dynamic shapes from packed rows | ok |
| T5 | `TF32=1` | 2-3x on fp32 matmuls | only for the fp32-master recipe; bf16 compute unaffected | ok (and recommended there) |
| T6 | FA3 -> FA2 auto-fallback | compat, not speed | TRL hardcodes `kernels-community/flash-attn3` (Hopper-only); auto-swap by compute capability | **required** |

## Pipeline / wall clock

| # | knob | estimate | mechanism |
|---|---|---|---|
| P1 | `INFLIGHT` 512 -> 2048 | +5-15% long-CoT | keeps the server saturated across phase transitions (a step consumes 1024 completions; staleness=3 wants a 2-3k outstanding pool) |
| P2 | watchdog `--eval-mode server` | 30-60 min per 1500-step run | per-checkpoint vLLM boot is 2-4 min + VRAM capture + trainer-GPU contention, x15 evals. Server mode probes the live policy (caveat: a few steps past the checkpoint) |
| P3 | weight sync 1.6 GB/step over NCCL | negligible (~10 ms) | measured, leave alone |

## Combined estimate

- Long-CoT phase: rollout +25-40% x trainer +30-60% (they overlap in the async
  pipeline) -> **~1.6-2.2x end-to-end**.
- Short-answer phase: trainer/overhead-bound -> **~1.3-1.7x**.

## Measured results (360-1, 20-step smokes, identical data/seed)

_Filled in after the benchmark runs — see the report section below._
