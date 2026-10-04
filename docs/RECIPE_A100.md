# Recipe: A100 (sm80) — the wave-2 e2e track

The A100 recipe targets **Qwen3.8-27B LoRA RL on A100-40GB nodes** (4 GPUs per
node, TCP networking): FSDP2/HSDP training over a frozen bf16 base with an fp32
LoRA master, chunked prefix sharing, and multi-backend async rollout with
adapter-only policy sync. One launcher integrates three independently developed
tracks (prefix / rollout / hsdp).

> **Status: CPU e2e verified, GPU smoke pending.** `tools/e2e/cpu_e2e_test.sh`
> passes end-to-end on CPU with fake vLLM backends. GPU smoke on the Tione
> A100 hosts has not run yet (GPU use was refused by the permission system at
> integration time). Treat this track as validated logic awaiting hardware
> numbers, unlike the H200 track which carries production measurements.

## Stack

| Layer | Choice | Where |
|---|---|---|
| Trainer | FSDP2 (1-D) or HSDP (`--replicate R`: shard within host, replicate across hosts — only LoRA grads cross hosts), frozen bf16 base, LoRA fp32 master | `tools/e2e/e2e_gspo.py` |
| Prompt sharing | `prefix_share_sm80`: port of the H200 prefix_share to sm80 + FSDP2 — FA2 (`flash_attn_func`) / SDPA `causal_lower_right` / mask fallback, chunked branch backward with fp32 accumulators, per-layer FSDP2 hooks preserved, optional pinned-CPU activation offload | `src/rlforge/prefix_share_sm80.py` |
| Rollout | Multi-backend vLLM router (any TP/precision mix): group-affine prompt pinning (prefix-cache affinity), LRU re-placement, fan-out pause/resume/reset, all-or-nothing adapter load | `src/rlforge/rollout/router.py` |
| Adapter sync | Adapter-only LoRA push: router reads the PEFT dir on the trainer host, optionally casts bf16 (halves bytes, lossless for serving), HTTP-pushes to a `lora_agent` per host, loads as `policy-v{k}` into every backend | `src/rlforge/rollout/lora_agent.py` |
| Async depth | batch b submitted once step b-1-max_stale finished, newest adapter; step 1 runs lr=0 so its log-ratio is the pure train/inference mismatch per rollout precision | `tools/e2e/e2e_gspo.py` |

## Launch

```bash
# CPU end-to-end correctness test (tiny model + fake vLLM backends):
bash tools/e2e/cpu_e2e_test.sh

# GPU (one host, 4 trainer GPUs, real vLLM backends behind the router):
bash tools/e2e/serve_backend.sh   # per-backend vLLM launcher
torchrun --nproc_per_node 4 tools/e2e/e2e_gspo.py \
    --router http://127.0.0.1:8410 --out RUN_DIR \
    --backend-precision 0=bf16_tp2,1=w4a16_tp1

# Multi-host HSDP comms bench (run on every host):
ssh a100_t1   'bash tools/hsdp/run_node.sh hsdp_2x4 0 2 1,2,3,4 <HEAD_IP> 29761 -- --comp-lens 2048,8192 --micro 4'
ssh a100_t1_2 'bash tools/hsdp/run_node.sh hsdp_2x4 1 2 0,1,2,3 <HEAD_IP> 29761 -- --comp-lens 2048,8192 --micro 4'
```

## Precision notes for sm80

- A100 has **no fp8 tensor cores** (sm90 feature) and no fp4 path (sm100): the
  trainer-side compute recipe is bf16 base + fp32 LoRA master, and rollout
  precision is whatever the backend mix declares (`--backend-precision`) — the
  e2e driver reports per-precision abs log-rho so quantization drift is measured,
  not assumed.
- HSDP keeps cross-host traffic to LoRA grads only (reduce-scatter intra-host +
  all-reduce of grad shards across hosts, one sync per optimizer step via
  `set_requires_gradient_sync(False)` on non-final micro-steps).

## Tests

- `tests/test_prefix_sm80_cpu.py` — sm80 prefix-share CPU correctness
- `tests/test_rollout_router.py` — router dispatch / failover / adapter sync
- `tools/hsdp/cpu_tests.sh` — gloo CPU correctness for the HSDP bench
