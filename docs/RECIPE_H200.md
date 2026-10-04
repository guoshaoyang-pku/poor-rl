# Recipe: H200 (sm90) — the production track

The H200 recipe is the battle-tested line: developed and run on an 8xH200 node
(0.8B async GSPO, 1500 steps, held-out 0.11 -> 0.81), with optional cross-node
rollout extension to a second node. All code lives in `src/rlforge/`; the exact
as-run tree is additionally snapshotted in `production/v3_1/` (md5-pinned).

## Stack

| Layer | Choice | Where |
|---|---|---|
| Algorithm | Async GSPO, `seq_mean` IS, staleness cap 3 | `src/rlforge/trainer.py`, `src/rlforge/gspo.py` |
| Rollout | vLLM TP=1 data-parallel replicas, group-affinity routing | `src/rlforge/dp_route.py`, `scripts/serve_dp.sh` |
| Prompt sharing | prefix_share: prompt forwarded once per group, G branches; token-budget sub-batches (default 131k), memory-adaptive activation ckpt | `src/rlforge/prefix_share.py` |
| Logprob | fast_logprob fused bf16/tf32 path (replaces TRL chunked lm_head: 5.4x fwd / 5.0x bwd) | `src/rlforge/fast_logprob.py` |
| Load balance | BalancedGroupRowBatcher (heaviest/mean rank load 1.36 -> 1.04) | `src/rlforge/prefix_share.py` |
| Scoring | non-blocking judge scorer; stale-drop audit | `src/rlforge/score_loop.py`, `src/rlforge/drop_audit.py` |
| KL | explicit frozen anchor (`RLFORGE_REF_MODEL`), k3 KL | `src/rlforge/trainer.py` |
| Precision | fp32 master weights + bf16 training compute + bf16 rollout; optional native FP8 full FT is experimental | [`PRECISION.md`](PRECISION.md), [`RECIPE_FP8.md`](RECIPE_FP8.md) |
| Fused ops | conv/RMSNorm/SwiGLU kernels, **off by default** (p99 log-ratio 2.5e-3) | `src/rlforge/fused_ops.py` |

## Launch

```bash
bash scripts/aiq/run_g32_v3_2.sh   # v3.2 trainer-throughput path (defaults on)
V32=0 bash scripts/aiq/run_g32_v3_2.sh   # falls back to the v3.1 trainer path
```

Key env knobs: `FAST_LOGPROB=1`, `SUBBATCH_TOKENS=131072`, `SB_CKPT=auto`,
`BALANCE_ROWS=on`, `INFLIGHT`/`QUEUE_MAXSIZE`, judge knobs `SCORE_CONC`,
`JUDGED_STALE`, `AIQ_HALLUC_*`. Full SOP (layout, in-flight/queue sizing,
preflight, smoke checks, restart traps): [`SOP_0.8B.md`](SOP_0.8B.md).

Cross-node rollout (one vLLM DP server spanning two hosts, head + `--headless`,
IB weight sync 0.145 s/step @0.8B): [`../tools/cross_node/INTEGRATION.md`](../tools/cross_node/INTEGRATION.md).

## Measured

- v2 -> v3.1: **16x samples/s, 23x trained tok/s** (rollout_wait 91.5 s -> 0.41 s;
  trainer busy 7.3% -> 94.8%).
- v3.1 -> v3.2: **2.03x fwd_bwd**, end-to-end 65 -> 37 s/step
  ([`V3_2_THROUGHPUT_2026-10-03.md`](V3_2_THROUGHPUT_2026-10-03.md)).
- Numerics gate: per-seq dlog-ratio |mean| <= 2.5e-8, p99 <= 1.1e-7,
  grad cos >= 0.99994; prefix_share bit-exact vs FA3 per-seq (2.10x/group).
  Gate harness: `tools/prefix_gate/`, `scripts/v32_gate.py`.

## Low-precision boundaries on H200

- **FP4 (MXFP4/NVFP4)**: measured net-negative on sm90 — Marlin MXFP4 is a
  weight-only A16 kernel (unpacks 4-bit to bf16 in registers, same MMA FLOPs as
  bf16 plus unpack overhead): 27B bf16 1,545 vs MXFP4 941 tok/s @256 concurrency.
  The +53% KV capacity never converted to throughput. FP4 is a Blackwell (sm100)
  primitive; on H200 it buys memory, not speed. Details: [`PRECISION.md`](PRECISION.md).
- **Native FP8 training + rollout (experimental)**: FP32 masters/gradients/Adam,
  shared E4M3 CUTLASS forward and E5M2 gradient GEMMs with FP32 output. A matched
  32-completion forward/backward batch measured 1,119.34 ms BF16 vs 1,117.89 ms
  FP8 (1.001x) with alignment off and opt-in trainer graphs; this excludes
  optimizer, rollout and judge. The aligned eager configuration measured
  1,599.55 ms vs 840.59 ms fused BF16; optional exact conv compilation reached
  about 1,111 ms. Correct FP32 serving passed four prefill completions from one
  prompt (max absolute mean log-ratio 0.001178), while one unique decode case
  failed the 0.004 gate (+0.006154). Long-run quality remains open.
  The old 0.53x mixed LoRA/full FT, and the 6.7x kernel claim lacks a recovered
  source-bound trace. See [`RECIPE_FP8.md`](RECIPE_FP8.md) and the
  [`historical correction`](reports/FP8_HISTORY_2026-10-05.md).

Bottleneck-chain analysis and every production pitfall (symptom -> root cause ->
fix -> evidence): [`INFRA_HANDOFF.md`](INFRA_HANDOFF.md).
