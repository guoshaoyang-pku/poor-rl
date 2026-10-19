# Qwen3.5-4B LoRA GSPO on all A100 nodes (2026-10-17)

Overnight RL from the 4B LoRA SFT ep4 start (merged), on all five A100 containers.

## Layout (run r4)

| Role | Node | GPUs |
|---|---|---|
| Trainer rank 0, router, supervisor `head.py`, live eval | a100_t1 (172.16.0.116) | 8 |
| Trainer node 1 (`worker.sh`) | a100_t1_2 (172.16.0.105) | 8 |
| Rollout: one TP1 `vllm serve` per GPU, plus a `lora_agent` (`serve_node.sh`) | a100_t1_4 (.48) ×8, a100_t1_3 (.15) ×6, a100_perm (.123) ×8 | 22 |

- Trainer: DDP across 16 ranks, using NCCL over TCP on eth0 with IB disabled. The frozen base is bf16 and the fp32 LoRA adapter has r32/α64 on all 12 projection kinds (attention, GatedDeltaNet, MLP).
- Rollout: `rlforge.rollout.router` routes each prompt group to one backend. Each policy version is cast to bf16 (130 MB) and pushed to every host's lora_agent, then loaded as `trl-policy-vK` on all 22 backends.
- Persistent storage: everything lives on the shared NFS at `/home/cfs/guoshaoyang/aiq_rl/q35_4b_gspo_20261017/`, which holds `code/`, `data/`, `models/`, `env/` and `runs/rN/`. Checkpoints are written every 10 steps to `runs/r4/trainer_out/checkpoint-*`.

## Settings

| Item | Value |
|---|---|
| Start | `q35_4b_sft_ep4_merged` (heldout290 16.59% strict, mean 2.9k tokens) |
| Data / reward | `aiq_rl_v1.2_d2fix/train_think.jsonl` (sha 57225919…) / `aiq_think_reward_v4.think_reward` (sha 7c1d4238…) |
| Algorithm | GSPO seq_mean, eps 4e-3 / 4e-3, KL 0, no LLM judge (AIQ_HALLUC=0) |
| Group / cap | G=32, max_completion 16384 |
| Async | max_staleness 3, rollout look-ahead 1 batch |
| Batch B | 2048 completions per update (128 per rank × 16 ranks; 64 groups) |
| LR | 2.83e-5 = 2e-5 × sqrt(2048/1024), the user's rule (LoRA LR = 10× full-param, √batch scaling from 2e-6 at 1024). TRL default 5-step warmup |
| Prefix sharing | on, using an sdpa fallback because the A100 venv has no flash_attn; subbatch 32768 tokens |

## Code changes over poor-rl main (`src/rlforge`)

- `trainer.py`:
  - LoRA flags and `peft_config`, ported from poor-rl f9e019a; the LoRA path had been lost in the v3.2 merge.
  - Pre-Hopper attention falls back to sdpa when `flash_attn` is absent.
- `trainer.py` (r5): `--init-adapter` and `--dataset-start-index` continue from a weights-only LoRA checkpoint.
- `prefix_share.py`: `_backbone` strips the PEFT wrapper. Without this, the VLM `*ForConditionalGeneration` has no `embed_tokens` (r2 crash).
- `head.py`: probes the router with `/server_info?config_format=json` (the r1 failure). It also detects a crashed rank, a stalled trainer, and a worker-node exit.

## Runs

| Run | Outcome |
|---|---|
| r1 | Head readiness probe bug, so the trainer never started (FAILED.json kept). Rollout node logs for all later runs are under `runs/r1/nodes/` |
| r2 | PEFT + VLM `embed_tokens` crash at the first forward |
| r3 | Single-node trainer (8 GPUs); 2 steps verified, then stopped by the operator to move to 2 trainer nodes. fwd_bwd took 407–439 s/step |
| r4 | 2 trainer nodes, 59 updates (03:41–08:30). Rank 14 on t1_2 hit a CUDA OOM as completions grew to about 3.6k tokens; rank 0 then hung until the 30 min NCCL timeout. Checkpoints 10–50 are on NFS. heldout290 strict was 18.5% at v4, 26.0% at v24 and 29.3% at v44 |
| r5 | From 09:00. Continues r4 checkpoint-50 via `--init-adapter` (weights only; Adam restarts with a 5-step warmup) and `--dataset-start-index 3182`. Memory: subbatch 16384, SB_ACT_GB 10; LR and B unchanged. Total updates = 50 + r5 step. Its `eval_live` `updates` field is relative to r5. Auto-resume watchers have been armed on r5 since 12:16 (see below) |

## Operate

- `status.py RUN [N]`: supervisor state plus the last N steps' metrics.
- `eval_live.py RUN 20 2`: heldout290 pass@1 (n=2) with the live adapter every 20 versions, co-located on the rollout GPUs. Results go to `RUN/eval_live.jsonl`.
- Stop a rollout host: `stop_node.sh runs/r1`.
- Failure receipt: `RUN/FAILED.json` (needs_thread_wakeup) together with `state.json` and `events.jsonl`. With the resume watchers running, it is followed by `NEXT/RESUMED.json` or `RUN/RESUME_BLOCKED.json`.

## Auto-resume (`resume_watch.py`)

Two watchers resume a failed run from its latest checkpoint without a human. The containers have no ssh to each other, so the watchers communicate only through files on the NFS.

| Role | Node | What it does |
|---|---|---|
| `head` | a100_t1 | Follows `runs/rN`. When `FAILED.json` appears, it creates `runs/rN+1` (newest checkpoint, `make_plan.py`, stale adapters unloaded, `start_head.sh`, `start_eval.sh 20 2`, `RESUMED.json`) and then follows the new run |
| `node` | a100_t1_2 | Follows the same run. When `runs/rN+1/RESUMED.json` appears, it runs `worker.sh` on the new plan and then follows the new run |

A resume runs these steps in order. If any step fails, the watcher writes `RUN/RESUME_BLOCKED.json` (needs_thread_wakeup) and exits.

1. **Gate.**
   - Only trainer-side failures are resumed: trainer rc≠0, a rank OOM/ChildFailedError, a worker-node exit, a stall, or a router exit.
   - These are never resumed: a signal to head.py (operator stop), backend-wait or router-start failures, and runs that contain a `NO_RESUME` file.
   - Budget: at most 2 resumes per 6 h, counted in `resume_watch/ledger.jsonl`.
2. **Wait.**
   - The failed head, router and trainer have exited (≤10 min).
   - Every `worker_IP_exit.json` exists (≤45 min, longer than the 30 min NCCL timeout).
   - The t1 GPUs are at ≤2000 MiB (≤15 min).
   - The node watcher's heartbeat is advancing, it follows this run, and its GPUs are free (≤10 min).
3. **Plan.**
   - Uses the newest `checkpoint-N` that has a complete `adapter_model.safetensors` (header vs file size) and a `rollout_state.json`. A save cut short by the crash is skipped.
   - `DATA_START` = the run's `--dataset-start-index` + `prompt_index`. `PRIOR_UPDATES` = the run's prior updates + N.
   - If the run has no checkpoint, its own starting point is rerun.
   - The new plan must equal the old one except for run_dir and the resume flags, so LR, B and memory settings cannot change.
4. **Unload.** Every `trl-policy-vK` on every backend is unloaded (`/v1/models`, then `/v1/unload_lora_adapter`). The new run restarts at v1, so a leftover name would collide. The resume is refused if any adapter remains.
5. **Launch.**
   - Starts the head and the live eval.
   - Writes `RESUMED.json` (checkpoint, DATA_START, PRIOR_UPDATES, LR, B, per-backend unload report, pids, budget used) and a ledger line.

The watchers only signal processes they started. Everything a watcher launches carries `RESUME_WATCH=head:RUN` or `RESUME_WATCH=node:RUN` in its environment. Every descendant inherits it, including trainer ranks orphaned when head.py SIGKILLs the launcher, so `/proc/*/environ` identifies a watcher's own processes exactly.

- The node watcher stops a worker it launched only if that worker is still alive 35 min after `FAILED.json`. It then writes `worker_IP_exit.json` itself.
- The head watcher stops any leftovers of a run it launched (ranks, the eval) before it checks the GPUs.
- For r5, which was started by hand, nothing carries the marker, so the watchers only wait.

Operate:

- Start (deployed alongside r5):
  - `ssh a100_t1 /home/cfs/.../code/start_watch.sh head /home/cfs/.../runs/r5`
  - `ssh a100_t1_2 /home/cfs/.../code/start_watch.sh node /home/cfs/.../runs/r5`
  - After a block, repair the run, then start both watchers again on the new run.
- State: `/home/cfs/.../resume_watch/` holds `head_watch.json`, `node_172.16.0.105.json` (heartbeats), `events.jsonl`, `ledger.jsonl`, and the logs `head_172.16.0.116.log` and `node_172.16.0.105.log`.
- Preview: `resume_watch.py plan RUN` prints the plan that a resume would launch now, without writing anything. For r5 at checkpoint-20 it gives r6 with DATA_START 4462 and PRIOR_UPDATES 70, and its only differences from r5 are the resume flags.
- Stop a run with no resume: `touch RUN/NO_RESUME`, then stop head.py (SIGTERM to head.py is not resumed in any case). `kill` on a watcher pid sets its status to `stopped`.
- Counting: total updates = `PRIOR_UPDATES` + the run's step. `eval_live` `updates` stays relative to its own run. Each resume restarts Adam with a 5-step warmup, because checkpoints are weights-only.
- Dry drill (no GPU, on a100_t1): `dryrun/run_dry.sh A|B|C PORT`.
  - The drill uses the real head.py, router, worker.sh and make_plan.py. The backends are mocks (`mock_vllm.py`), and the trainer, worker and eval are fakes (`fake_trainer.py`).
  - `check_dry.py` checks the receipts against `scenarios.json`.
  - 2026-10-17 results: all checks passed for A, B and C. Evidence is in `a100_t1:/home/tione/guoshaoyang/aiq_rl/q35_resume_dry/{A,B,C}-1017_121341/`.

| Drill | Failure path | Result |
|---|---|---|
| A | r1: rank-0 OOM. r2: trainer rc=1, and the node watcher stops the hung node-1 worker. r3: node-1 rank crash | r1→r2 (ckpt-20, DATA_START 1280, PRIOR 20), r2→r3 (ckpt-20, 2560, 40), then r3 blocked on the budget (2 per 6 h) |
| B | r1: OOM right after a save that was cut short (`checkpoint-30` truncated) | r1→r2 from checkpoint-20 (checkpoint-30 skipped); r2 completes and both watchers end `done` |
| C | head.py receives SIGTERM (operator stop) | Not resumed; `RESUME_BLOCKED.json` |

All three drills also checked that stale adapters were unloaded on every mock backend, that the resumed runs had no adapter-name collision, and that no drill process was left running.
