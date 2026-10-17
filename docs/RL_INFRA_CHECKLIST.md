# Async GRPO/GSPO infra checklist

These are lessons from the production lines: 0.8B on H200/H100 and 4B LoRA on 5×A100, October 2026. Each entry gives the symptom, how to tell, the fix, and where the fix lives. Go through the list before you trust a step-time number.

## Measure before concluding

| Pitfall | How to tell | Rule |
|---|---|---|
| A one-off `nvidia-smi` snapshot | The rollout read 0% once, but its 20-minute average was 25% (H100) and 27–39% (A100) | Sample utilization every 2–3 s for at least 3 minutes. Report the mean and the fraction of samples where all GPUs are idle. |
| "100% util" on trainer GPUs while they wait in NCCL | Utilization was 100% but power was 85–105 W on an A100 (compute draws 300–400 W). py-spy showed every rank in `broadcast_object_list` | Check `power.draw` next to utilization. py-spy dump every rank before calling the trainer the bottleneck. |
| Who waits for whom | The step logs `perf/rollout_wait_s` against `perf/fwd_bwd_s`, and the router `/metrics` exposes per-backend `num_requests_running` | Use both. The router sum across backends going to 0 means rollout is not being fed. |
| 99% util while rollout concurrency drains | A100 4B r7: rollout GPUs read 99% util (304–311 W) for 3 minutes, but the router's `num_requests_running` sum fell from 1603 to 443 with 0 waiting | Utilization says a kernel is running, not how full the batch is. Track the running and waiting sums over time. |

## Rollout child (one Python process, GIL-bound)

| Pitfall | How to tell | Fix |
|---|---|---|
| The rollout child pins one core | `top` shows 100% on the `multiprocessing.spawn` child of rank 0. py-spy shows the event loop as `active+gil` | `rlforge/fast_rollout.py` (fastroll), wired in `score_loop.py`. On the A100 4B run, fastroll together with queue 2B raised rollout-node util from 27–39% (all GPUs idle in 20–60% of samples) to 99%, with all 2048 of 2048 slots running. |
| Prompt re-tokenized for every completion | 33% of the child's CPU went to `apply_chat_template` (3.4–3.6k-token prompts) | Memoized chat template. It returns identical ids; T1 in `tests/test_fastroll.py`. |
| Look-ahead gate polling | About 30% went to `sleep(0.01)` loops across thousands of tasks | One pump coroutine with the same admission rule; T3. |
| `asyncio.wait(all inflight, FIRST_COMPLETED)` | About 13% at 8k in-flight tasks | O(done) wait proxy; T2. |
| `--queue-maxsize` smaller than the batch | The log says `all 32 scoring slots busy … rollout-buffer backpressure`, then backends drain to 0 | Use queue ≥ 2B. The A100 4B run had 512 against B=2048. |
| One batch of look-ahead with B equal to the slot count | A100 4B r7: B=2048 on 16×128 slots. Each version bump admits one whole batch, which starts together and drains through its tail; `perf/rollout_wait_s` alternates 2–28 s and 225–298 s | `--rollout-ahead-batches K` (`AHEAD=K` in the A100 kit) admits K future batches; K must be below `--max-staleness`. Watch the staleness histogram and the long-length drop fractions in `drop_audit`: long samples dropped as stale are a length bias. |

## Rollout servers

| Pitfall | How to tell | Fix |
|---|---|---|
| KV preemption | `vllm:num_preemptions_total` reached 6% of requests at 186 concurrent 6–9k-token sequences per A100-40G. At `--max-num-seqs 128` and 6.2k-token mean completions (r7), a 183 s window had 1009 preemptions against 1565 finished requests | Cap `--max-num-seqs` (128 on A100-40G for a 4B model with 16k completions). As completions grow, 128 still preempts; the throughput cost of a lower cap is not measured yet. |
| Group-affine pinning plus a long tail | One engine holds the last long group while the others sit at 0 | Keep enough groups in flight, which needs queue and look-ahead room. The router re-places pins when a backend gets busier. |
| Too many threads per process | `fork rejected by pids controller`, ThreadPoolBuildError EAGAIN (H100 container, pids.max 9830) | Set `RAYON_NUM_THREADS=4` and `OMP_NUM_THREADS=4` for vLLM. |
| Prefix caching | Check `vllm:prefix_cache_hits_total / queries_total`. A100 4B reached 88% | Keep `--enable-prefix-caching`. Route a group to one backend (router or dp-route). |

## Trainer

| Pitfall | How to tell | Fix |
|---|---|---|
| No flash_attn on pre-Hopper GPUs | TRL hardcodes Hopper FA3. Without `flash_attn` the trainer falls back to sdpa with masks: slower and more memory | Build an sm80 wheel (`FLASH_ATTN_CUDA_ARCHS=80`, about 6 min). The trainer picks `flash_attention_2` automatically. |
| Zero-variance groups | 38% of groups (77% of easy MCQ) had identical rewards: zero gradient at full cost | `RLFORGE_DROP_ZERO_VAR=1`. Replacements are admitted, so B stays B informative samples. |
| OOM as completions grow | A worker rank OOMs while rank 0 waits out the 30-minute NCCL timeout | Size `--subbatch-tokens` and `RLFORGE_SB_ACT_GB` for the length tail. head.py scans worker logs for OOM and fails fast. |
| Sub-batches too small | `v3_2/subbatches_per_row` of 18 and pad of 20% | Raise sub-batch tokens once memory allows. Smaller G (curriculum) shortens rows. |
| Load imbalance across ranks | `batch/row_imbalance` 1.5 with G32 (4 groups per rank) | Use smaller G or more groups per rank, with `--balance-rows on`. |
| Weights-only LoRA checkpoints | There is no optimizer.pt | Continue with `--init-adapter` + `--dataset-start-index` (the rollout_state `prompt_index`). Adam restarts with warmup. |

## Reference step times: A100 4B LoRA GSPO, B=2048, 16 trainer GPUs (2 nodes, TCP), 16 rollout GPUs

Measured from `trainer.log`. r5 is the mean of its last 10 steps (41–50); r7 is the mean of steps 2–6 (step 1 includes startup).

| | r5 | r7 |
|---|---|---|
| Setup | G32, sdpa, sub-batch 16384, queue 512 | G8, flash_attention_2, sub-batch 24576, fastroll, queue 4096, zero-variance drop |
| `perf/step_s` | 622 s | 543 s (436, 608, 387, 706, 576) |
| `perf/fwd_bwd_s` | 503 s | 393 s (373–427) |
| `perf/rollout_wait_s` | 108 s | 138 s (2–298, alternating) |
| `perf/weight_sync_s` | 2.2–2.8 s | 2.2–2.4 s |
| Trained tokens per step | 12.7M | 13.3M |
| Trained tokens/s (wall clock) | 20.4k | 24.6k |
| fwd_bwd time per trained token | 39.6 µs | 29.5 µs |

In r7 the trainer is 25% faster per token, but the step is tail-bound: `perf/rollout_wait_s` alternates between short and long (2–28 s, then 225–298 s). The look-ahead gate admits only one future batch, and at B=2048 that batch fills all 2048 slots at once, then drains through its long tail.

## LR rule (user, 2026-10-17)

Full-parameter LR is 2e-6 at B=1024 and scales with √B. LoRA LR is 10× that.

`lr_lora = 2e-5 · sqrt(B/1024) = 1e-5 · sqrt(B/256)`

That gives 2.83e-5 at B=2048. B is completions per optimizer update across all ranks. Change kernel and micro-batch sizes freely; change LR only through B.
