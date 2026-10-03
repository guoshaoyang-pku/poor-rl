# Production snapshot: v3.1 infra (2026-10-03)

This directory holds the exact code that runs the 0.8B async GSPO line on the reference cluster (8×H200, plus a second host for rollout). Every file is byte-identical to the cluster tree `rlforge_v3_1`; see `MD5SUMS`. It is a **snapshot, not a fork to edit**. New work goes into `src/rlforge/` once the merge described below is done.

Background, measured results and pitfalls: [`docs/INFRA_HANDOFF.md`](../../docs/INFRA_HANDOFF.md). Runbook: [`docs/SOP_0.8B.md`](../../docs/SOP_0.8B.md).

## What is in it

| file | md5 (8) | what |
|---|---|---|
| `src/rlforge/trainer.py` | `0229d8f8` | GSPO trainer on TRL 1.14 `AsyncGRPOTrainer`. Adds a per-sequence forward for hybrid models and these flags: `--prefix-share`, `--token-budget`, `--dp-route`, `--queue-maxsize`, `--drop-audit`, `--score-concurrency`, `--judged-max-staleness`, `--reward-early-hooks`, `--score-task-max-s` |
| `src/rlforge/prefix_share.py` | `ede53a03` | Each GRPO group's prompt is forwarded once, and the G completions branch from its GatedDeltaNet state and attention K/V. Passed the gate: bit-exact against the FA3 per-sequence path, and 2.10× faster per group |
| `src/rlforge/dp_route.py` | `815fd1ed` | Pins every request of a group to one vLLM DP replica (`X-data-parallel-rank`), for prefix-cache affinity |
| `src/rlforge/drop_audit.py` | `9900deed` | Observation only: stale-drop rates bucketed by length, generated vs. trained length distributions, and rewards of dropped vs. trained samples |
| `src/rlforge/score_loop.py` | `de8b32c0` | Non-blocking scorer: N groups are scored concurrently, so a slow LLM-judge group delays only itself. Also judged-sample staleness allowance and a scoring timeout |
| `scripts/run_g32_v3_1.sh` | `da30395d` | Production launcher, single host or cross-host. Includes preflight, watchdog, teardown, the run.json manifest, a groups/step divisibility guard and `DRY=1` |
| `scripts/drop_audit_report.py` | `32032b08` | Summarizes `drop_audit.jsonl` over a range of steps |
| `tests/test_drop_audit.py`, `tests/test_v3_1_compose.py` | | CPU tests for the audit and for audit and scorer composed together |

**Judged-sample staleness, in detail.** `--judged-max-staleness S` (the launcher's `JUDGED_STALE`) is an **absolute** cap. A judged sample is kept while its staleness is at most `STALE + min(judge_versions, S − STALE)`, where `judge_versions` is the number of weight syncs that happened while its group was being scored. `auto` means `S = STALE + 2`.

## Why it is separate from `src/rlforge/`

The trainer diverged into several copies:

| copy | has |
|---|---|
| `src/rlforge/` in this repo (a542944) | LoRA, adaptive GSPO clip caps, SwanLab, boundary-aware hybrid packing |
| this snapshot | per-sequence hybrid forward, prefix sharing, DP routing, drop audit, non-blocking scorer, and the v3 knobs |

The next step is to merge them into one `src/rlforge/trainer.py`. The requirement is that, with the production flags, it behaves exactly like `0229d8f8`; after the merge it needs a GPU smoke and a step-1 numerics probe. Until then, use this snapshot for production-equivalent runs.

## How to use it

```bash
# Build a code tree: repo package + snapshot modules on top. Do not edit the repo copy.
cp -r src/rlforge /path/to/rlforge_v3_1/src/rlforge
cp production/v3_1/src/rlforge/*.py /path/to/rlforge_v3_1/src/rlforge/
# The launcher puts $RLFORGE_V3/src first on PYTHONPATH and checks prefix_share's md5.
RLFORGE_V3=/path/to/rlforge_v3_1 DRY=1 RUN_NAME=<run> INIT_MODEL=<ckpt> bash production/v3_1/scripts/run_g32_v3_1.sh full
```

The launcher's defaults are those of the reference deployment: host IPs, `/data/...` paths, and a sourced site env file that sets `ROOT` and `VENV`. Override them with environment variables (`HEAD_IP`, `REMOTE_IP`, `REMOTE_SSH`, `SERVER_GPUS`, `REMOTE_GPUS`, `TRAINER_GPUS`, `INIT_MODEL`, `DATA`, `PORT`, `RPC_PORT`, …), or copy the script and edit the site block. To run the cross-host launcher on one host, use `tools/cross_node/selfssh/ssh`. The task reward module, its judge prompt and the benchmark data are not part of this repo.

Verified for this snapshot (2026-10-03, on 360-2, CPU only):
- the overlay imports cleanly;
- `--help` lists all nine flags;
- `tests/test_drop_audit.py` passes (rc 0, 77 s);
- `tests/test_v3_1_compose.py` passes (rc 0, 12 s).

On GPU it runs the `async_g32_v3_1*` runs (from 19:2x CST), and v3 (`async_g32_v3_infra_st3_kl005_20261003`, steps 1–189) ran the same files except the two v3.1 additions.
