# Cross-node DP rollout — integration package

360-2 = head (HTTP API, load balancer, local DP ranks, trainer). 360-1 = headless DP ranks.
TRL keeps **one** `--server-url` and **one** NCCL weight-transfer group; it needs no code change.
Background, measurements and design choice: `aiq_rl/docs/reports/CROSS_NODE_ROLLOUT_2026-10-03.md`.

Verified on 2026-10-03 (vLLM 0.30.0, TRL 1.14.0, Qwen3.5-0.8B, 360-2 GPU2 head DP0 + 360-1 GPU4 headless DP1 + trainer 360-2 GPU3):
- `/get_world_size` = 2. NCCL picked `NET/IB` (mlx5_0/mlx5_4) across hosts, with OOB over bond4.
- Weight-transfer init: 8.9 s, once per run.
- Sync of 1.706 GB / 473 tensors: send median **0.120 s** (14.2 GB/s); pause+send+resume **0.145 s**.
- After a perturbed sync, greedy output changed on both replicas. After restoring, greedy tokens were bit-exact with the baseline on both replicas.

---

## 0. Pre-flight (every launch)

```bash
# 360-1: which GPUs are actually free? (GPU 0/1/3/7 bad or foreign, 5/6 = another user)
ssh 360-1 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader'
#   GPU 2 may still hold someone's 27B server (port 8106): do NOT kill it. Wait, or use only GPU 4.
# Same checkpoint bytes on both hosts, same path (/data/shared is LOCAL disk on each host, not shared):
CKPT=/data/shared/guoshaoyang/aiq_rl_store/models/sft_rc_ckpt57_20261003
ssh 360-2 "rsync -a $CKPT/ 10.234.161.2:$CKPT/"          # 1.5 GB, seconds over bond4
ssh 360-2 "cd $CKPT && sha256sum * | sort" > /tmp/a; ssh 360-1 "cd $CKPT && sha256sum * | sort" > /tmp/b; diff /tmp/a /tmp/b && echo SAME
# Ports: head API $PORT and DP RPC 13345 must be free on 360-2.
```

## 1. Environment (both hosts, and the trainer)

```bash
export VLLM_HOST_IP=$(ip -4 -o addr show bond4 | awk '{print $4}' | cut -d/ -f1)   # 10.234.161.3 on 360-2, .2 on 360-1
export NCCL_SOCKET_IFNAME=bond4 GLOO_SOCKET_IFNAME=bond4   # docker0 exists on both hosts and is not routable between them
export NCCL_IB_HCA=^mlx5_bond_0                            # the bonded mlx5 is Ethernet on both hosts; use the IB HCAs only
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0   # no nvcc on either host
```

**TRL's master address.** TRL's `WeightTransferClient` uses `vllm.utils.network_utils.get_ip()` as the TCPStore and NCCL master address. `get_ip()` returns `VLLM_HOST_IP` if it is set, and otherwise the interface on the default route. On 360-2 the default route is already bond4 (10.234.161.3). Still, **export `VLLM_HOST_IP=10.234.161.3` in the trainer's environment** so it never resolves to docker0/lo. The 360-1 engines connect to that address on a random port, and connectivity was verified for this.

## 2. Head on 360-2 (existing launcher, no code edit)

`run_async_dp.sh` and `run_g32_think.sh` already run
`CUDA_VISIBLE_DEVICES=$SERVER_GPUS vllm serve ... --tensor-parallel-size $TP --weight-transfer-config '{"backend":"nccl"}' $VLLM_EXTRA`.
Make it a DP head by passing TP=1 and extra flags, for example 3 local ranks on GPUs 0,1,2 plus 2 remote ranks on 360-1:

```bash
# on 360-2, with the env from §1 exported
DP_TOTAL=5; DP_LOCAL=3
SERVER_GPUS=0,1,2 TP=1 TRAINER_GPUS=3,4,5,6,7 NUM_TRAINER=5 \
VLLM_EXTRA="--data-parallel-size $DP_TOTAL --data-parallel-size-local $DP_LOCAL \
  --data-parallel-address 10.234.161.3 --data-parallel-rpc-port 13345 --api-server-count 1 \
  --kv-cache-dtype fp8 --max-num-seqs 1024 --max-num-batched-tokens 32768 \
  --max-cudagraph-capture-size 1024 --async-scheduling" \
  bash run_g32_think.sh full _<suffix>
```

Notes:
- `--max-num-seqs` applies **per replica**. The value must not exceed the Mamba-state blocks that the memory budget allows. When `max_num_seqs > available Mamba cache blocks` it fails with a ValueError (D3 saw it at 4096 vs 2070 blocks). At UTIL 0.90 on an otherwise free H200, 4096 fits. On shared GPUs, use 1024.
- The launcher's 15-minute `/health` wait covers the remote ranks. The head reports healthy only after every DP rank has handshaken, so start §3 within that window.
- `--api-server-count 1`: one API process is enough at this request rate, and it keeps the dev endpoints and `/metrics` in one process.
- Client in-flight: the trainer's `--max-inflight-tasks` (INFLIGHT) must cover all replicas. Scale it about linearly with DP_TOTAL relative to what saturated one server.

## 3. Headless ranks on 360-1 (GPUs 2,4)

```bash
# on 360-1, AFTER the head process has started; same model path, same vLLM flags
D=/data/home/guoshaoyang/crossnode_20261003
cd $D && ROLE=headless GPUS=2,4 DP_SIZE=5 DP_LOCAL=2 START_RANK=3 HEAD_IP=10.234.161.3 RPC_PORT=13345 \
  UTIL=0.90 MODEL=$CKPT SERVED_NAME=$CKPT MAX_LEN=24576 NAME=dp_headless \
  VLLM_EXTRA="--kv-cache-dtype fp8 --max-num-seqs 1024 --max-num-batched-tokens 32768 --max-cudagraph-capture-size 1024 --async-scheduling" \
  bash serve_dp.sh            # writes $D/dp_headless.pid and $D/dp_headless.log
```

`serve_dp.sh` (this directory) exports the env from §1. It also sets `VLLM_SERVER_DEV_MODE=1` and `--weight-transfer-config nccl`, and passes `--dtype bfloat16 --max-model-len ${MAX_LEN:-18432}`.
- **Match the head's `--max-model-len` (the launcher uses 24576): set `MAX_LEN=24576`.**
- `--gpu-memory-utilization` should match the head unless the 360-1 GPU is shared. If another job is on the card, lower UTIL and set max-num-seqs ≤ the Mamba-state blocks it allows.
- If only one 360-1 GPU is free: `GPUS=4 DP_LOCAL=1`, and `DP_SIZE` on both sides = DP_LOCAL(head) + 1.

Check:
```bash
ssh 360-2 'curl -s localhost:$PORT/get_world_size'        # == DP_TOTAL (TP=1)
ssh 360-2 'curl -s localhost:$PORT/metrics | grep -c "^vllm:generation_tokens_total{"'   # one series per engine
```

## 4. Watchdog (required)

Observed live on 2026-10-03, after the 360-1 rank was killed by an external SIGTERM:
- The head kept `/health` = **200**.
- Requests sent to the dead rank **hung**.
- The next `/init_weight_transfer_engine` hung **300 s** (TCPStore `wait timeout ... /broadcast_from/0/0`) and then returned **500**.
- During a run the trainer would stall until `request_timeout` (3600–5400 s) or the 300 s control timeout, and then crash.

Run this watchdog on 360-2 next to the launcher:
```bash
REMOTE=10.234.161.2 REMOTE_PIDFILE=/data/home/guoshaoyang/crossnode_20261003/dp_headless.pid \
KILL_PIDS="<run_g32_think.sh pid> <head vllm pid>" nohup bash watch_ranks.sh > logs/watch_ranks.log 2>&1 &
```
It SIGTERMs the launcher, so the launcher's TERM trap writes `stop_reason` and stops vLLM. Then tear down the remote ranks (§6).

Other failure modes:

| Event | Effect | Handling |
|---|---|---|
| Headless rank dies | Silent stall, see above | `watch_ranks.sh` |
| Head dies | TRL HTTP errors, trainer exits; headless ranks may linger and hold GPU memory | Teardown §6 |
| An external cleanup sweep kills 360-1 vLLM processes (this is what killed the test rank) | Same as headless rank dying | Agree that one owner holds 360-1 GPUs 2 and 4 for the whole run |
| Model bytes differ between hosts | Replicas silently sample from different policies until the first weight sync overwrites them (TRL syncs every step) | sha256 check in §0 |
| Weight-sync hang | `weight_sync_timeout` (TRL default 1800 s) | Leave at default; the watchdog catches the real cause |

Timeouts: keep the launcher's `REQUEST_TIMEOUT` (long completions need it). Leave `VLLM_ENGINE_READY_TIMEOUT_S` (600 s) at its default; it covers headless startup.

## 5. Optional: group-affine routing (`dp_route.py`)

vLLM's DP load balancer ignores prefixes, so the 32 samples of a group are spread across replicas. The header `X-data-parallel-rank: <int>` bypasses the balancer (`vllm/entrypoints/generate/base/serving.py:238`). `dp_route.GroupAffineRolloutLoop` pins all in-flight requests that share a prompt to one replica, chosen as the least-loaded one by in-flight count.

Measured at DP=2, G=32, 5.5k prompt / 1k out:
- Computed prefill dropped 15.0k → 9.2k tok/s (**−38%**), and the prefix hit rate rose from 0.855 to 0.913.
- **gen tok/s did not change** (18.4k vs 18.4k).

**Default off.** Turn it on only if the trainer starts to see prefill pressure: longer prompts, shorter outputs, or larger DP.

To enable it, the module must be importable in the spawned rollout child:
```bash
cp tools/cross_node/dp_route.py /data/home/guoshaoyang/rlforge/src/rlforge/dp_route.py   # on 360-2 (and wherever the trainer runs)
```
and in `rlforge/trainer.py` `main()`, before the trainer is constructed:
```python
import os
if os.environ.get("RLFORGE_DP_ROUTE") == "1":
    from rlforge import dp_route
    dp_route.install()     # AsyncRolloutWorker._loop_cls -> GroupAffineRolloutLoop (pickled by reference into the child)
```
Then launch with `RLFORGE_DP_ROUTE=1`. With DP=1 it detects `dp == 1` and sends requests exactly as before.

## 6. Teardown

```bash
# 1. trainer / launcher on 360-2 (its trap stops the head vLLM)
ssh 360-2 'kill -TERM <run_g32_think.sh pid>'
# 2. headless ranks on 360-1 (only our pidfile; never pattern-kill)
ssh 360-1 'D=/data/home/guoshaoyang/crossnode_20261003; kill $(cat $D/dp_headless.pid); for i in $(seq 30); do kill -0 $(cat $D/dp_headless.pid) 2>/dev/null || break; sleep 2; done'
# 3. verify both sides
ssh 360-1 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n "3p;5p"'
ssh 360-2 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; ss -ltn | grep -E ":13345|:$PORT"'
```
Do not use `pkill -f "vllm serve"`: it matches the ssh command line itself, and on 360-1 it would also kill other users' and agents' servers.

## Not verified

- Cross-node gen tok/s per replica. The 360-1 rank was killed before the throughput run. Same-host DP replicas get 9.2k each while sharing one GPU, and 18.7k on a full GPU.
- Head-dies behaviour of the headless processes (do they exit on their own?). Teardown kills them explicitly.
- Long-run stability with DP > 2, and the `/pause` path against a dead engine (inferred from the init hang and the code, not observed).
- Under 0.3σ noise, the two replicas' greedy outputs differed from each other. Both changed and both restored bit-exact. The likely cause is near-tie argmax on a destroyed model, but that is not proven. A logprob-gap check (`sync_test.py` now prints it) was cut off by the external kill.
