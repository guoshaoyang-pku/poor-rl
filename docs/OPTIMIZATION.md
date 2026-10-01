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
| R2 | `KV_DTYPE=fp8` (default on H200; `auto` override) | +4-11% observed at short-answer load; capacity +82% | halves KV bytes -> more resident sequences | H200 verified; A100 not validated |
| R3 | `ASYNC_SCHED=1` | +5-10% | overlaps scheduler CPU with GPU at high request churn (our n=1x16 pattern) | ok |
| R4 | `FI_SAMPLER=1` | +3-8% | batched flashinfer sampling kernel | test |
| R5 | `ROLLOUT_QUANTIZATION=fp8` | test | quantizes rollout weights in vLLM; model/kernel support and quality must be checked on target hardware | Hopper+ |
| R6 | speculative decoding | **rejected** | batch-bound rollout; spec-decode helps low-batch latency, not throughput | — |

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

## Measured results (fixed-length inference, 360-1 GPU 2, 2026-10-01)

Model: Qwen3.5-0.8B-ms; vLLM 0.30; BF16 compute; FP8 KV in both arms; FP8 arm additionally uses FP8 weight quantization. Prefix caching disabled. Each request uses exactly 1,024 input and 256 generated tokens. Concurrency is the client in-flight limit; vLLM caps were 512 sequences / 32,768 batched tokens / 512 CUDA-graph size. All requests succeeded.

| Concurrency | BF16 tok/s | FP8 tok/s | FP8 speedup | BF16 p95 (s) | FP8 p95 (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 486 | 478 | 0.98x | 0.54 | 0.56 |
| 8 | 3,279 | 3,355 | 1.02x | 0.64 | 0.62 |
| 32 | 8,948 | 9,497 | 1.06x | 0.94 | 0.86 |
| 64 | 11,529 | 14,010 | 1.22x | 1.61 | 1.21 |
| 128 | 10,632 | 18,282 | 1.72x | 3.46 | 1.87 |
| 256 | 7,764 | 21,900 | 2.82x | 8.79 | 3.28 |
| 512 | 10,025 | 24,227 | 2.42x | 15.09 | 6.16 |

FP8 weight quantization gave little benefit at low concurrency, but reached 2.4–2.8x throughput at 256–512 requests in this short-output test. The highest measured aggregate throughput was at 512; 256 had lower p95 latency. Treat these as short-output saturation results, not a guarantee for 8k–16k completions; long-context KV pressure needs a separate sweep. Raw results and GPU samples are in `/data/home/guoshaoyang/aiq_rl/runs/rlforge_mfu_20261001/bench/inference_fixed_20261001/` (`sweep_bf16.json`, `sweep_fp8.json`, `gpu_metrics_*.csv`).

## Training batch and variable-length findings (2026-10-01)

TRL's installed `AsyncGRPOTrainer` supports both a fixed-count planner and a token-budget planner. With token budgeting, each per-rank row is capped at `token_budget` tokens; short completions pack more samples, long completions fewer. Rows are packed without per-sample padding, position ids reset at each sequence, and only small inter-rank alignment padding is added then stripped before the model forward. The planner balances rows by sum of squared sequence lengths (attention cost), not only token count. `CPS/NGEN` fixes gradient-accumulation microbatches per optimizer step, but with dynamic token budgeting the realized samples per step are not a fixed CPS count; monitor `batch/samples_per_step` and `batch/microbatches_per_step`. Its other packing metrics include `batch/row_tokens_max`, `batch/row_fill_frac`, `batch/row_imbalance`, and `batch/pad_frac`. A variable-length synthetic planner probe packed 20 samples into row batches of 1–6 samples under the 24,576-token cap, with no drops.

With `token_budget=None`, TRL queries vLLM's max model length (24,576 here), so the per-row cap can be too aggressive for a stable training-memory target. RLForge's current trainer/launcher does not expose an override yet; expose a `TOKEN_BUDGET`/CLI option before a controlled CPS/token-budget sweep. In an isolated one-GPU synthetic forward/backward probe, one packed row at 8,192 / 16,384 / 24,576 tokens peaked at about 31/33, 64/78, and 91/124 GiB allocated/reserved; the 24,576 case emitted a failed-allocation warning before succeeding. This is a capacity probe, not an end-to-end RL training result, and 24,576 should not be treated as a safe setting.

An end-to-end optimizer-step batch/microbatch ceiling was **not established**: the only otherwise-free 360-1 card could serve rollout, but candidate trainer cards 3 and 7 returned `CUDA-capable device(s) is/are busy or unavailable`; GPU 4 was occupied. At the same time, 360-2 was running the user's Suika DQN jobs and ophis had active GPU workloads. Four 360-1 launcher attempts stopped before a training step: the first lacked the local attention-kernel override, the next combined rollout and trainer on one GPU and failed NCCL weight transfer, and the separate-card trainer probes on GPUs 3 and 7 failed with `CUDA-capable device(s) is/are busy or unavailable`. No existing jobs were stopped or changed. Use a separate healthy trainer GPU pair for the true CPS/NGEN sweep. Start with a conservative per-row budget (e.g. 16k, not the 24,576 default), raise CPS only while tracking peak allocated/reserved memory, step samples, time, and loss/ratio health. The 1-GPU probe artifact is `bench/train_token_budget_probe.json` under the run directory above.
