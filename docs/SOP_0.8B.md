# 0.8B 异步 GSPO 运行 SOP（v3.1，2026-10-03）

本 SOP 适用于以下场景：Qwen3.5-0.8B 这类混合模型（GatedDeltaNet 与 attention 按 3:1 混合）、单机 8×H200（可另加一台做 rollout），prompt 约 5–6k token、completion 约 2k token、上限 16k，GRPO 组大小 G=32。

背景、证据和踩过的坑见 [`INFRA_HANDOFF.md`](INFRA_HANDOFF.md)。生产代码是 [`production/v3_1/`](../production/v3_1/)，launcher 是 `production/v3_1/scripts/run_g32_v3_1.sh`，下文的环境变量都指这个 launcher。

主机名、IP、路径都是参考部署（360-1 / 360-2）的值，换环境时用环境变量覆盖。

---

## 1. 一句话版本

rollout 用 TP=1 的 vLLM DP 副本，每卡一个；KV cache 用 bf16。API server 数等于 DP 数，keep-alive 设 3600 s。trainer 用 `--prefix-share on --token-budget 0`，每行放一整组，trainer rank 数必须整除每步组数。在途请求数加队列长度约等于 1 步的样本量。drop audit 打开。每次改 infra，都在 step 1（lr=0）对照上一个 run 的数值。

## 2. GPU 布局

| 规则 | 原因 |
|---|---|
| rollout 每卡一个 TP=1 副本，组成一个 DP server（`--data-parallel-size`） | 0.8B 用 TP2 时每卡 11.5k tok/s，TP1 单卡 18.7k [实测] |
| 跨机时一个 DP server 横跨两台机：本机起 head，另一台起 `--headless` 引擎 | TRL 只看到一个 URL 和一个 NCCL 组，不用改代码 |
| trainer rank 数 N 必须整除每步组数（launcher 有守卫） | 每行一整组时，每个 micro-batch 是 N 组 |
| eval watcher 单独占 1 张卡 | 每个 checkpoint 评测约 520–530 s，赶得上每 25 步存一次的节奏 [实测] |
| 预检确认要用的卡确实空闲（显存 > `GPU_BUSY_MIB`=1000 MiB 视为占用） | 容器里 `--query-compute-apps` 返回空 |

参考布局：

| 布局 | rollout | trainer | eval | 实测 / 预估 |
|---|---|---|---|---|
| v3（跨机） | 360-2 GPU 0,1,3 + 360-1 GPU 2,4（DP5） | 360-2 GPU 4–7（4 rank） | 360-2 GPU 2 | 每步约 60 s，trainer busy 95%，rollout 富余约一半 [实测] |
| v3.1 单机 | 360-2 GPU 0,1 + 3（DP3，借助 `tools/cross_node/selfssh/ssh`） | 360-2 GPU 4–7 | 360-2 GPU 2 | 每步 53–83 s，仍受 trainer 限制；staleness 升到 2.5–3 [实测] |

## 3. 代码和环境

1. 代码树：把 `production/v3_1/src/rlforge/*.py` 覆盖到仓库的 `src/rlforge/` 上（另建一个目录，不要改仓库）。做法见 [`production/v3_1/README.md`](../production/v3_1/README.md)。launcher 用 `RLFORGE_V3=<这个树>` 指向它。
2. venv：vLLM 0.30.0、torch 2.13、TRL 1.14.0（experimental async_grpo）、transformers 5.17，FA3 用 `kernels-community/flash-attn3` 的本地 kernel。
3. 机器上没有 nvcc 时，必须设：`VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0`（launcher 已经设好）。
4. 仓库自带的 `scripts/run_async_dp.sh` 默认值还是旧的（`TP=4`、`KV_DTYPE=fp8`）。0.8B 别直接用这些默认值。

## 4. 参数

### 4.1 算法（v3 / v3.1 的值；改动属于实验决策）

| 变量 | 值 | 备注 |
|---|---|---|
| `NGEN` | 32 | 组大小 |
| 每步组数 | 32（= 1024 条） | 由 `CPS = 组数 × NGEN / N` 决定（4 rank → `CPS=256`，gas 8）。也可以设 `GROUPS_PER_STEP=32`，launcher 会自动推出 CPS |
| `STALE` | 3 | |
| `GSPO_EPS_LOW/HIGH` | 3e-3 / 3e-3 | seq_mean；旧消融显示 ≥5e-3 明显变差 |
| `KL_BETA` | 0.05 | ref = 启动时的初始模型（见 §11） |
| `LR` | 2e-6 | 5 步 warmup |
| `MAX_COMPLETION` | 16384 | |
| `MAX_STEPS` / `SAVE` | 600 / 25 | `save_total_limit=5`，旧 checkpoint 会被轮转删除 |

### 4.2 rollout（vLLM）

| 设置 | 值 | 依据 |
|---|---|---|
| `KV_DTYPE` | `auto`（bf16） | 0.8B 的 KV 使用率只有 7–51%，fp8 KV 不提速，还增加数值误差 [实测] |
| `--max-num-seqs` / `--max-num-batched-tokens` / `--max-cudagraph-capture-size` | 2048 / 32768 / 2048 | stock 配置在 2048 并发时只有 1023 条在跑；调优后单卡 17.2–18.8k tok/s [实测] |
| `--async-scheduling` | 开 | |
| `GPU_MEM_UTIL` | 0.85 | |
| `MAX_MODEL_LEN` | 24576 | prompt 加 completion 必须放得下 |
| `API_SERVER_COUNT` | = DP（launcher 默认就是） | 只开 1 个时：/pause 44–73 s、424 次断连、每副本约 6k tok/s [实测] |
| `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` | 3600 | 默认 5 s 会导致 439 次 ServerDisconnected [实测] |
| `TMPDIR` | 短路径 | ZMQ ipc 路径有 107 字符上限 |
| `DP_ROUTE` | on | 同一组钉在同一个副本上，prefill 少 38%，gen tok/s 不变 [实测] |
| 不要开 | MTP 投机解码、一卡两实例 | 吞吐分别 +17% / +14%，但都没在权重同步下验证过 |

### 4.3 在途请求和队列（最容易出事的两个参数）

先判断当前瓶颈在哪一侧：
- `perf/rollout_wait_s / perf/step_s` < 0.1：trainer 是瓶颈；
- 该比值 > 0.25：rollout 是瓶颈。

**trainer 是瓶颈时**：`INFLIGHT + QUEUE_MAXSIZE` 约等于 1.1 步的样本量。v3 用的是 640 + 512。
- 缓冲越深，样本越旧。长组发出最早、完成最晚，会最先超过 STALE。
- v3 在 INFLIGHT 1280、队列 1024 时，step 11 出现过：8–16k 的组丢了 2/2，截断组丢了 6/8，<2k 的组 0/10。改成 640 + 512 后，累计丢弃降到 0.07% [实测]。

**rollout 是瓶颈时**：每个副本保持 ≥ 256 条在途。同时用下面的估算检查最长组能否在 STALE 步内完成：
- 每条序列的速度 ≈ 每副本 tok/s ÷ 每副本在途数；
- 最长成员的 token 数 ÷ 每条序列的速度，应小于 STALE × step_s。
- 例：INFLIGHT 4096、5 个副本时，每副本约 820 条在途，每条约 23 tok/s，6k 的组要约 4.4 min，超过 3 步 [估算]。

**副本数越少，单个副本的在途数越多，组完成得越晚，staleness 越高。** v3.1 单机 DP3 时，staleness 在 2.5–3；v3 的 DP5 约 2.0 [实测]。

**绝对不要**把队列扩到 (stale+1)·S。

## 5. judge（LLM 评审，可选）

1. judge 开着时必须同时设 `SCORE_CONC=32` 和 `REWARD=<线程安全的 reward>`，否则串行打分会卡住生成。launcher 会拒绝不满足这一条的组合。
2. 供给：中转按账号、按模型限流，每个模型每滚动 60 s 放行 10 次。同一账号再加 key 没用，只有换模型桶（先做一致性校准）或换账号才能加供给。参考部署的 relay 支持 `--pool` 模式，可以同时用两个模型桶（每个 9 次/分钟）。
3. 需求定在供给的 75%：`FRAC = 0.75 × supply_per_min × step_s / 60 / (K × 每组存活 rollout 数)`。
   - 只用一个模型桶时：K=2，FRAC 约 0.12，RPM 9。
   - 用两个模型桶时：FRAC 约 0.24，RPM 18。
4. `AIQ_HALLUC_BURST ≥ K × NGEN × FRAC`，K=2 时取 16。BURST 8 时，第二个 judge 组会遇到空桶，有效 FRAC 只剩约 0.06 [实测]。
5. `EARLY_HOOKS=0`。提前送审会让判决偏向短 rollout [实测 CPU]。judge 调用顺序用 shuffle（默认开）。
6. judge 组的 staleness：auto 补偿等于该组打分期间实际发生的同步次数，在 GPU 上不够用。实测 judged_keep_frac 只有 0.5。改用 judge 组单独的固定上限，验收标准是 `sample/judged_keep_frac ≥ 0.95`。

## 6. 预检清单（launcher 会自动做大部分）

- [ ] `RUN_NAME` 是新名字。launcher 没有默认值，`runs/<run>/run.json` 已存在时拒绝启动。
- [ ] 所有要用的卡空闲，两台机都要查。
- [ ] 两台机上 `model.safetensors` 的 sha256 一致。`/data/shared` 是各机自己的本地盘，不是共享挂载。
- [ ] `prefix_share.py` 的 md5 等于 `PREFIX_SHARE_MD5`（ede53a03），align=64，没有开 SDPA 回退，FA3 本地 kernel 目录存在。
- [ ] 端口（8011、13411）空闲。
- [ ] 每步组数能被 trainer rank 数整除（`GPS_GUARD=1`）。
- [ ] reward 文件的 md5 已经记进 run.json。
- [ ] 正在被某个 run 使用的 launcher **不要改**。bash 是边执行边读脚本的，新版本请另存一个文件名。

## 7. 启动

```bash
cd $ROOT
# 1) 只打印计划和命令，不启动任何东西
DRY=1 RUN_NAME=<run> INIT_MODEL=<ckpt> [其余变量] bash run_g32_v3_1.sh full
# 2) 正式启动（setsid：launcher pid == pgid）
RUN_NAME=<run> INIT_MODEL=<ckpt> [其余变量] setsid nohup bash run_g32_v3_1.sh full \
  > logs/<run>.driver.log 2>&1 < /dev/null &
```

单机跑跨机 launcher（把"远端"rank 起在本机）时，加上：

```bash
PATH=<repo>/tools/cross_node/selfssh:$PATH SELFSSH_HOST=self-local REMOTE_SSH=self-local \
REMOTE_IP=<本机 bond IP> SERVER_GPUS=0,1 REMOTE_GPUS=3
```

## 8. 冒烟：前 1–5 步看什么

| 指标 | 期望 | 不对时 |
|---|---|---|
| `batch/samples_per_step` | 正好等于 组数 × 32 | 检查 `--token-budget 0` 和 CPS |
| mb audit（前 `RLFORGE_MB_AUDIT_STEPS` 步） | 每个 rank 的 microbatch 数相同 | 不同就说明 DDP 迟早会死锁，停掉 |
| `perf/pause_s` | < 1 s | 查 API server 数 |
| trainer 日志里的 `ServerDisconnected` | 0 | 查 keep-alive |
| 每副本 gen tok/s（vLLM /metrics） | 约 10–19k | 查 API server 的 CPU 占用 |
| step 1（lr=0）的 `gspo/abs_log_rho_p50/p90` | 与上一个 run 的 step 1 一致（v3：6.5e-4 / 1.5e-3） | 说明数值路径变了，停掉排查 |
| step 1 的 `kl_ref` | 约 5.5e-4（ref 等于初始模型时） | 同上 |
| `sample/staleness_mean` | trainer 是瓶颈时 ≤ 约 2.5 | 调小 INFLIGHT 或队列 |
| `drop_audit.jsonl` 的分桶 | 长桶的丢弃率不能明显高于短桶 | 同上 |
| judge 开着时：`sample/judged_keep_frac`、halluc_step 里的 `skipped_rate` / `timeouts` | ≥ 0.95；skipped_rate 接近 0 | 调 BURST、judge 组 staleness |

## 9. 监控阈值（运行中）

- 立即处理：trainer 日志出现 `Traceback`、`out of memory`、`NCCL error`、`DistStoreError`；driver 日志出现 `stop_reason`；trainer 进程或 eval watcher 退出；单步超过 150–200 s。
- 趋势要盯的：`gspo/seq_clip_low_frac` 和 `gspo/seq_clip_high_frac`（实验侧目标是各 10–25%），entropy 斜率（上一条主线塌缩前 entropy 从 0.66 涨到 1.4–1.8），`kl_ref`，held-out exact 和长度中位数。
- 比较学习效果时，用 rollout 侧所有被打分组的 reward 加 held-out eval。trainer 日志里的 reward 只来自被训练的样本，会随丢弃率移动。

## 10. 停止

```bash
kill -TERM <launcher pid>      # trap 会拆掉 trainer、head、远端 headless 进程组，并写 stop_reason
```

- **绝不用 `pkill -f` / `killall`。** 按名字杀会误伤别人的 server，也会杀掉 ssh 命令本身。v3 就是在 step 189 被别人的批量清理误杀的。
- 远端紧急情况：`ssh <remote> 'kill -TERM -- -<pgid>'`，pgid 在 driver 日志和 `runs/<run>_remote/vllm_headless.pid` 里。
- 停完之后确认两台机的 GPU 显存都已归零。

## 11. 从 checkpoint 重启

checkpoint 只保存模型，没有 resume 功能，所以每次重启都是一个**新 run**（新的 `RUN_NAME`，`INIT_MODEL=<checkpoint>`）。要注意以下几点：

| 陷阱 | 结果 | 做法 |
|---|---|---|
| KL ref = 启动时初始模型的 deepcopy | KL 锚点悄悄移到重启点 | 想保留原锚点，要先实现 `--ref-model`（ROADMAP）；否则就明确记录"ref reset" |
| Adam 状态丢失，warmup 重新跑 | 刚过 warmup 那几步更新更大，abs_log_rho 和 clip 会跳升（v3.1：kl_ref 14 步内从 5e-4 升到 6.6e-3） | 预期内；横轴同时看 updates 和 samples |
| 数据顺序还是同一个 seed | 从头按原顺序再走一遍 | 记录下来 |
| `save_total_limit=5` | 想要的 checkpoint 会被轮转删掉 | 用作重启起点的 checkpoint 先拷到固定路径，并算 sha256 |
| 跨机 | 远端也要有同一份 checkpoint | rsync 过去后比较 sha256（launcher 预检会再查一遍） |

## 12. eval watcher

- 新 run 起来后，在 eval 卡上启动 watcher，用 `--skip-init --every 25`。重启 run 的第 0 步就是旧 run 那个 checkpoint 的评测，不必重复。
- 旧 run 的 watcher 等它评完最后一个 checkpoint 再停。

## 13. 出问题时

先查 [`INFRA_HANDOFF.md` §5](INFRA_HANDOFF.md#5-踩过的坑与经验)，按"症状 → 根因 → 修复"索引。
