# rlforge infra 交接文档（v3 / v3.1 infra 线，2026-10-03）

本文写给接手 rlforge RL infra 的人：现在的 infra 长什么样、为什么长这样、踩过哪些坑、下一步做什么。
操作步骤见 [`SOP_0.8B.md`](SOP_0.8B.md)。生产代码快照见 [`production/v3_1/README.md`](../production/v3_1/README.md)，跨机 rollout 的接入手册见 [`tools/cross_node/INTEGRATION.md`](../tools/cross_node/INTEGRATION.md)。

**标注约定**（沿用源报告）：
- **[实测]**：在机器上跑出来的数字，附来源（日志字段或测试名）。
- **[估算]**：推算或外推，附假设。
- **[代码]**：读 vLLM 0.30.0 / TRL 1.14.0 / transformers 5.17 源码确认。
- **[未测]**：明确没跑过。

**关于主机名和路径**：文中的 `360-1` / `360-2`（两台 8×H200）、端口 8011 / 13411、`$HOME/...` 等都是**参考部署的示例**，不是代码默认值。代码里所有站点相关的值都是环境变量或必填参数。

---

## 0. 一页结论

| 问题 | 结论 |
|---|---|
| 现在的 infra 比 v2 快多少 | trained samples/s **16×**（1.04 → 16.8），trained tok/s **23×**（1,346 → 31,587）[实测，trainer step dict，同一解析脚本，稳态段] |
| 23× 从哪来 | trainer 利用率 6.9% → 95.0%（**13.8×**，来自管线修复 + DP rollout）× trainer 每 token 速度 19.6k → 33.3k tok/s（**1.7×**，来自 prefix sharing）[实测] |
| 口径差异 | v2 开着 LLM judge（K=2），v3 关着。所以有一部分提速来自关 judge |
| 现在的瓶颈 | **trainer**：busy 95%，rollout 产能约有一半富余 [实测 busy；富余为估算] |
| 0.8B 的 SOP 要点 | rollout 用 TP=1 的 DP 副本；bf16 KV；API server 数 = DP；keep-alive 3600 s；INFLIGHT+队列约 1 步样本；trainer rank 数整除每步组数；prefix-share 门禁 md5；drop audit 打开 |
| 27B 能信的数字 | 只有显存相关的数字可信（MXFP4 51.1 → 18.3 GiB，KV +53%）。并发曲线**可能**是用有缺陷的压测客户端测的，**未核实**；FP8 W8A8 **从没测过** |
| 现状 | v3 跑到 step 189 时，被外部的 vLLM 批量清理误杀，watchdog 按设计拆除了整个 run。19:1x 起从 ckpt-175 用 v3.1（judge 开，两个 luna 桶）在 360-2 单机 DP3 重启；下一步换成用上两台机全部好卡的布局（§3.1） |
| 代码统一 | 生产代码在 [`production/v3_1/`](../production/v3_1/)，与集群上正在跑的 `rlforge_v3_1` **逐字节一致**（MD5SUMS）。仓库的 `src/rlforge/` 还是合并前的版本（有 LoRA、自适应 clip、SwanLab、hybrid_packing，但没有 per-seq / prefix share / v3 开关）。把两者合成一份是下一步（§7 #1），合并后必须先 GPU 冒烟并做 step-1 数值探针（§5.6） |

---

## 1. 当前 infra 是什么

### 1.1 架构

```
  ┌──────────────────────────── 主机 A（head，8×H200）─────────────────────────────────────────┐
  │                                                                                              │
  │  production/v3_1/scripts/run_g32_v3_1.sh = setsid 进程组的 owner                            │
  │    ├─ 预检：GPU 显存、两台机 ckpt sha256、端口、prefix_share md5、每步组数整除                   │
  │    ├─ 写 runs/<run>/run.json（布局、全部参数、各文件 md5、stop_reason）                        │
  │    ├─ watchdog（每 WD_INTERVAL s）：head pid、远端 headless pid（ssh kill -0）、/health          │
  │    │     任何一项失败 → SIGTERM launcher → trap 拆除 trainer + head + 远端进程组                  │
  │    │                                                                                         │
  │    ├─ vLLM DP head：API server ×DP（--api-server-count = DP）+ 本机引擎 rank 0..L-1，TP=1       │
  │    │                                                                                         │
  │    └─ accelerate DDP trainer ×N rank（rlforge.trainer, GSPO）                                 │
  │          rank 0 的子进程 = TRL AsyncRolloutWorker：                                          │
  │            生成循环：每组 G=32 个 n=1 请求 ──HTTP──► API servers                             │
  │                 dp_route：X-data-parallel-rank 头把整组钉在一个副本上（prefix cache 亲和）      │
  │            打分循环：score_loop 同时给 N 组打分（judge 组只拖慢自己）◄── reward（+ 可选 judge） │
  │                 │                                                                            │
  │                 ▼ RolloutQueue（QUEUE_MAXSIZE）                                                │
  │          RolloutQueueDataset：staleness 过滤（> STALE 丢弃）◄── drop_audit（只观测，rank 0）    │
  │                 ▼                                                                            │
  │          GroupRowBatcher（一行 = 一整组）→ prefix_share forward（policy + KL ref）              │
  │          → GSPO seq_mean loss → 优化器 step → NCCL 权重广播到全部 DP 引擎（world = DP+1）        │
  │                                                                                              │
  │  eval watcher（1 张卡；参考部署用任务目录里的 eval 脚本，通用版是 rlforge.watchdog）              │
  └──────────────────────────────────────────────────────────────────────────────────────────────┘
          ║ ZMQ 请求/输出（KB 级）走两机共享的 TCP bond；NCCL 权重广播走 IB（排除 Ethernet 模式的 HCA）
  ┌──────── 主机 B（可选）────────┐
  │ vllm serve --headless         │  引擎 rank L..DP-1，一个 setsid 会话（pid == pgid），停 = 杀整组
  └───────────────────────────────┘
```

### 1.2 组件清单

| 组件 | 仓库位置 | 作用 | 默认 | 状态 |
|---|---|---|---|---|
| GSPO 异步 trainer | `production/v3_1/src/rlforge/trainer.py`（生产）；`src/rlforge/trainer.py`（仓库旧版，有 LoRA / 自适应 clip / SwanLab） | TRL `AsyncGRPOTrainer` 子类；GSPO seq_mean、KL ref、per-seq forward 和全部 v3/v3.1 开关 | — | 生产版在 v3.1 run 上跑；两份待合并 |
| prefix sharing | `production/v3_1/src/rlforge/prefix_share.py` | 每组 prompt 只前向一次，G 条 completion 从 prompt 状态分支 | `--prefix-share off` | 与 FA3 per-seq **逐位一致**，2.10× [实测] |
| per-seq forward | 生产 `trainer.py: per_seq_logprobs` | 混合模型每条序列单独前向（避免 GatedDeltaNet 状态跨样本泄漏） | `--per-seq-forward auto`（混合模型自动开） | 与干净前向逐位一致 [实测，v2 集群] |
| 边界感知打包 | `src/rlforge/hybrid_packing.py`（只在仓库旧版 trainer 里） | 打包行里给 DeltaNet/conv 传 `cu_seqlens` | 打包路径自动 | 见 [`PACKING.md`](PACKING.md) |
| 组亲和路由 | `production/v3_1/src/rlforge/dp_route.py` | 同组请求钉到一个 DP 副本 | `--dp-route off` | prefill −38%，gen tok/s 不变 [实测] |
| drop audit | `production/v3_1/src/rlforge/drop_audit.py` + `production/v3_1/scripts/drop_audit_report.py` | 按长度分桶的 stale 丢弃率、生成/训练长度分布、丢弃组与训练组的 reward | `--drop-audit off`（launcher 默认 on） | v3 生产用的是旧版 3048dc5b；v3.1 修复版 9900deed 只在 CPU 上测过 |
| 非阻塞打分 | `production/v3_1/src/rlforge/score_loop.py` | 并发打分 + judge 组的 staleness 补偿 + 打分超时 | `--score-concurrency 0`（= 原版 TRL） | CPU 测试全过；19:1x 起在 v3.1 run 上跑（GPU 冒烟发现见 §5.5 #3、#6） |
| 单机 launcher（仓库旧版） | `scripts/run_async_dp.sh` | 单机：vLLM server + DDP trainer | **仍是旧默认 TP=4、KV_DTYPE=fp8**，未按 v3 修正 | 0.8B 不要直接用默认值，见 SOP §3 |
| 生产 launcher（单机 / 跨机） | `production/v3_1/scripts/run_g32_v3_1.sh` | head + 远端 headless（ssh）、watchdog、预检、run.json、组数整除守卫 | 默认值是参考部署的路径 / IP，全部可用环境变量覆盖 | v3.1 run 在用 |
| 跨机工具 | `tools/cross_node/` | `serve_dp.sh`、`watch_ranks.sh`、`sync_test.py`、`bench_groups.py`、`selfssh/ssh`（单机跑跨机 launcher 用） | — | 见 [`INTEGRATION.md`](../tools/cross_node/INTEGRATION.md) |
| 压测工具 | `tools/bench/` | `bench_sat.py`（闭环饱和压测）、`serve_arm.sh`、`offload_parity.py`（**写好了没跑过**） | — | — |
| prefix 门禁 | `tools/prefix_gate/gate_v3_prefix.py`（+ 门禁结果 json、a679 → ede53a03 补丁） | 6 项数值门禁 + DDP + 速度 | — | 换模型、换库版本时必须重跑 |
| eval watcher | `src/rlforge/watchdog.py` | 在线 held-out eval、keep-best、5 条自动停止规则 | — | 已有 |

### 1.3 必须记住的不变量

| 不变量 | 内容 | 依据 |
|---|---|---|
| 每步样本数 | `--prefix-share on --token-budget 0` 时：samples/step = 32·N·gas **精确成立**，gas = CPS/NGEN | [代码] + [实测] v3 每步恰好 1024 |
| 整除 | 一行一整组时，每步组数必须是 trainer rank 数的倍数。4 rank → 32 组；5 rank → 30 或 35 组；6 rank → 30 或 36 组 | [代码] |
| token-budget 打包会忽略 CPS | 不设 `--token-budget 0` 时，TRL 的 TokenBudgetBatcher 决定每步样本数，CPS 只决定 gas | [实测] v2 每步只训了 125–189 条，而不是 1024 |
| NCCL 组大小 | `/get_world_size` 返回 TP·DP（含远端），权重传输组 = DP+1 | [实测] DP=5 时 world=5 |
| staleness 打戳 | 一组的版本号在**第一个成员发出时**打上，组在**最慢成员完成时**才推进队列 | [代码] TRL `_generate_loop` |
| KL 参考 | trainer 在启动时 deepcopy **初始模型**当 KL ref（`trainer.py` 约 L105） | [代码] |
| checkpoint | 只存模型，不存优化器；`save_total_limit=5` 会轮转删除 | [代码] |

---

## 2. 怎么走到这里：v2 → v3 的瓶颈链

### 2.1 v2 的病理 [实测，v2 trainer/vLLM 日志]

| 现象 | 数字 | 根因 |
|---|---|---|
| trainer 大部分时间空等 | steps 2–7：Σfwd_bwd / Σstep = 61.8 / 898.6 s = **6.9%** | 管线 bound，不是算力 bound |
| 生成的样本大多被丢掉 | 训练比例 f = 1060 / (1060 + 3821) = **0.22**；sftrc57 整个 run 丢 110,294 条、训 26,680 条（**78–80%** 被丢） | STALE=1，加上下面的 batcher |
| 每步只训 125–189 条 | launcher 注释承诺 1024 条 | 没设 token_budget → 每行按 24,576 token 装 3–5 条 → CPS 不再决定每步样本数 |
| 打分串行阻塞 | 每个 judge 组最多卡 90 s；score_block_s 11–30 s | TRL `_score_loop` 一次只打一组；16 槽的打分队列满了以后生成循环也停 |
| judge 本身拿不到结果 | 延迟均值 32–37 s、p90 50–60 s；69–78% 超时；约 40% 的结果在组放弃等待之后才到（花了钱又白扔） | 客户端 8 线程，中转 `UPSTREAM_SEM=4`，上游 9.9 次/分钟 |
| judge 的作用 | sftrc57 里只改了 127 条 reward（13.8 万条 rollout 的 **0.09%**） | 覆盖率太低，当探测器有用，当奖励项基本无效 |
| rollout 用 TP2 | 每卡 11.5k tok/s，单卡 TP1 是 18.7k | 0.8B 做张量并行，每层 all-reduce 的开销不值 |

### 2.2 瓶颈链（按发现顺序，每修一处，瓶颈就移到下一处）

1. **judge 串行阻塞打分** → v3 先关 judge；`score_loop.py` 做好后再开（v3.1）。
2. **stale 丢弃 80%** → `STALE` 1 → 3（eps 1e-3 → 3e-3，两侧对称），每步 32 组 = 1024 条（`--token-budget 0`，一行一整组）。
3. **rollout TP2** → 5 个 TP=1 的 DP 副本（360-2 三张 + 360-1 两张，跨机 DP）。
4. **trainer 前向的 token 里 68–90% 是重复的 prompt** [实测，v2] → prefix sharing（逐位一致，单组 2.10×）。
5. **冒烟时发现单个 API server 撑不住 5 个副本** → `--api-server-count = DP`。
6. **uvicorn keep-alive 5 s** → `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=3600`。
7. **trainer 成了瓶颈 + 缓冲太深 → 长组被系统性丢弃** → INFLIGHT 1280 → 640，QUEUE_MAXSIZE 1024 → 512。
8. **现在：trainer-bound（busy 95%）** → 下一批杠杆见 §7。

### 2.3 实测提速 [实测，v3 取 steps 4–85，旧 run 取稳态；同一解析脚本，数据来自 trainer step dict]

| run | samples/s | trained tok/s（墙钟） | trainer busy | fwd_bwd 期间 tok/s | samples/step | 中位 step_s | 中位 completion 长度 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1_kl005 | 0.77 | 958 | 5.0% | 19.3k | 159 | 187 | 1865 |
| v2_kl005 | 1.04 | 1,346 | 6.9% | 19.6k | 156 | 141 | 1846 |
| sftrc57 | 1.61 | 1,109 | 7.2% | 15.4k | 189 | 103 | 925 |
| **v3** | **16.8** | **31,587** | **95.0%** | **33.3k** | 1024 | 60 | 1898 |

**归因（v3 对 v2_kl005，两者 completion 长度接近）**：
- 23× trained tok/s = **13.8×**（利用率 6.9% → 95%：关 judge + STALE 3 + 每步 32 组 + DP rollout + 多 API server）× **1.7×**（每 token 速度 19.6k → 33.3k：prefix sharing）。
- 按样本算 16×，低于修订后预估的 19–28 条/s [估算]。主要原因是 v3 不再丢长组，平均每条样本更长、更贵。
- **口径不对等**：v2 开 judge（K=2），v3 关 judge。

**预估 vs 实测**（用来校准以后的估算）：
- 最初 A/B/C 三档预估约 10× / 20× / 25× [估算]；
- prefix sharing 单卡实测后下调到 11–16×；
- 门禁补丁（2.10×）后上调到 13–19×；
- 最终实测 16×（样本）/ 23×（token）。
- prefix sharing 的 token 比是 3.4–3.6×，墙钟只有 1.7–2.1×：剩下的大头是 32×2k 的 completion 本身。

### 2.4 时间线（2026-10-03，CST）

| 时间 | 事件 |
|---|---|
| 约 13:00 | v2 停掉，开始端到端诊断（日志拆解、TRL 代码阅读、360-1 rollout 扫描） |
| 13:2x | 跨机权重同步实测通过；prefix sharing 首版 1.71×（SDPA 版） |
| 14:00 | prefix 门禁：SDPA 版不过，FA3 + 64 对齐补丁后逐位一致，2.10× |
| 14:04 | smoke1：远端 ZMQ ipc 路径超过 107 字符 → watchdog 拆除生效 |
| 14:21 | smoke2：1 个 API server → /pause 44–73 s、424 次断连、每副本约 6k tok/s |
| 14:41 | smoke3：5 个 API server → /pause 0.03–0.4 s、每步 52–61 s；keep-alive 导致 439 次断连 |
| 14:51 | smoke4：keep-alive 修复 + drop audit → 0 断连，audit 异常 0 |
| 15:02 | v3 正式启动（INFLIGHT 1280） |
| 15:23 | step 14 停掉、重启：step 11 出现长组成批丢弃 → INFLIGHT 640 / QUEUE 512 |
| 17:2x | 测出 judge 中转的限流规则（按账号、按模型） |
| 18:39 | v3.1 包构建完成（CPU 测试全过，未经审查，未上 GPU） |
| 18:40:19 | v3 在 step 189 被外部 vLLM 批量清理 SIGTERM 远端 rank → watchdog 拆除整个 run |
| 19:1x | judge relay 加 pool 模式（两个 luna 模型各一个 60 s 窗口），24 并发全 200 |
| 19:2x | v3.1 从 ckpt-175 启动（judge 开，360-2 单机 DP3）；step 1–14 冒烟发现见 §3.1 |

---

## 3. 现状与代码位置

### 3.1 run 状态

- v3 run（`async_g32_v3_infra_st3_kl005_*`，600 步计划）跑到 step 189 被外部误杀（§5.7 第 5 条）。
  - 最后一个 checkpoint 是 ckpt-175：held-out exact 约 0.217–0.219，reward 0.313 [实测]。
  - 被杀前的稳态 [实测]：每步约 60 s（49–67 s），staleness 约 2.0，累计 stale 丢弃 0.07%，entropy 0.60–0.64；|log ρ| p90 从 step 1 的 1.6e-3 升到 step 86 的 2.7e-3，低侧 clip 从 0.2% 升到 6.7%（eps 3e-3）。
- **v3.1 重启（`async_g32_v3_1_ck175_judge_*`，19:1x 起）**：从 ckpt-175 起（KL ref 也随之换成 ckpt-175，实验侧接受），360-2 单机：rollout GPU 0,1,3（DP3，用 `tools/cross_node/selfssh/ssh` 让"远端" rank 在本机起），trainer GPU 4–7，eval GPU 2；judge 开（K=2，FRAC 0.24，RPM 18，BURST 8），两个 luna 模型各一个限流桶。
  - step 1–14 [实测]：仍是 trainer-bound（fwd_bwd 50–80 s，rollout_wait 0.4–4 s，step 53–83 s，均值约 66 s）；staleness 2.5–3（v3 约 2.0）；abs_log_rho p90 3.4–4.5e-3，seq_clip_low 15–37%（v3 同期 4–8%）；kl_ref 14 步内 5e-4 → 6.6e-3（Adam 重置后头几步更新更大）；judge 每个 pseudo-step 只发出约 4 次调用（§5.5 #3），judge 样本只保留一半（§5.5 #6）。
  - 下一步：换成两台机全部好卡的布局，BURST 16，judge 组单独的 staleness 上限，eps ±4e-3（实验侧决定）。
  - 重启的语义陷阱（KL anchor、Adam、warmup、数据顺序）见 [`SOP_0.8B.md` §11](SOP_0.8B.md#11-从-checkpoint-重启)。

### 3.2 代码谱系（为什么要统一）

| 树 | trainer md5 | 有什么 | 缺什么 |
|---|---|---|---|
| 本仓库开源前（a542944） | `226cf989`（581 行） | LoRA（`LORA=1`）、自适应 GSPO clip cap、SwanLab、`hybrid_packing.py`（611ceff） | per-seq、prefix share、v3 开关 |
| 集群 v2 树 | `661ff7e4`（594 行） | `per_seq_logprobs`（混合模型一次前向一条序列，FA3 来自 `kernels-community/flash-attn3`，与干净前向逐位一致） | LoRA、自适应 clip、v3 开关 |
| v3 生产（冻结） | `5ec9b09f` | prefix_share `ede53a03`、dp_route `815fd1ed`、drop_audit `3048dc5b`、`--queue-maxsize`、mb audit、poslog | v3.1 修复 |
| v3.1 包（最新，CPU 测过） | `0229d8f8` | v3 生产 + drop-audit 修复（`9900deed`）+ 非阻塞打分（`score_loop.py` `de8b32c0`） | GPU 冒烟、审查 |
| **本仓库 `production/v3_1/`** | `0229d8f8` | 与 v3.1 包逐字节一致 | 仓库旧版的 LoRA / 自适应 clip / SwanLab / hybrid_packing |
| 本仓库 `src/rlforge/`（a542944） | `226cf989` | 见第一行 | 待与 `production/v3_1` 合并（§7 #1） |

规则：
- 以后改 infra 先改本仓库，再部署到集群。
- 集群上的冻结树只读，每个 run 的 `run.json` 记录所用文件的 md5。
- `prefix_share.py` 必须与门禁通过的 `ede53a03` 逐字节一致。如果确实要改，就重跑门禁并更新 launcher 里的 `PREFIX_SHARE_MD5`。

### 3.3 仓库文件

```
production/v3_1/                  生产快照（与集群 rlforge_v3_1 逐字节一致，见 MD5SUMS 和 README）
  src/rlforge/trainer.py          0229d8f8：--prefix-share --token-budget --dp-route --queue-maxsize --drop-audit
                                  --score-concurrency --judged-max-staleness --reward-early-hooks --score-task-max-s；
                                  RLFORGE_MB_AUDIT_STEPS / RLFORGE_POSLOG_STEPS
  src/rlforge/prefix_share.py     ede53a03（门禁通过版）
  src/rlforge/dp_route.py         815fd1ed 组亲和路由
  src/rlforge/drop_audit.py       9900deed（v3.1 修复版）
  src/rlforge/score_loop.py       de8b32c0 非阻塞打分
  scripts/run_g32_v3_1.sh         生产 launcher（单机 / 跨机；DRY=1 只打印不启动）
  scripts/drop_audit_report.py    汇总 drop_audit.jsonl 的一段 step
  tests/test_drop_audit.py  tests/test_v3_1_compose.py
src/rlforge/                      仓库旧版（LoRA、自适应 clip、SwanLab、hybrid_packing），待合并
tools/cross_node/                 serve_dp.sh watch_ranks.sh sync_test.py bench_groups.py INTEGRATION.md selfssh/
tools/bench/                      bench_sat.py serve_arm.sh offload_parity.py
tools/prefix_gate/                gate_v3_prefix.py、门禁结果 json、prefix_share a679 → ede53a03 补丁
```

**不在仓库里的**：任务相关的 reward 模块及其 judge prompt、benchmark 数据、任务专用 launcher。参考部署里这些在私有的任务目录下。

### 3.4 集群上的产物（示例布局）

以参考部署为例，`$ROOT` 是任务目录：

| 产物 | 路径 |
|---|---|
| driver / trainer / vLLM head 日志 | `$ROOT/logs/<run>.driver.log`、`logs/trainer_dp_<run>.log`、`logs/vllm_dp_<run>.log` |
| run 目录 | `$ROOT/runs/<run>/{run.json, drop_audit.jsonl, poslog/, task_split.jsonl, rollout_samples.jsonl, checkpoint-*}` |
| 远端 headless | 远端机 `$ROOT/runs/<run>_remote/{vllm_headless.log, vllm_headless.pid}` |
| eval | `$ROOT/evals/<run>_heldout/`、`logs/<run>.eval_watch.log` |
| 冻结代码树 | `$HOME/rlforge_v3_prod/`（v3）、`$HOME/rlforge_v3_1/`（v3.1，含 `merge/*.diff` 来源记录） |

---

## 4. 0.8B SOP 摘要

完整清单见 [`SOP_0.8B.md`](SOP_0.8B.md)。要点：

1. **布局**：rollout 用 TP=1 的 DP 副本，每卡一个；trainer rank 数必须整除每步组数。8 卡参考：rollout 3（+ 远端 2）+ trainer 4 + eval 1。
2. **vLLM**：
   - `KV_DTYPE=auto`（bf16 KV）；
   - `--max-num-seqs 2048 --max-num-batched-tokens 32768 --max-cudagraph-capture-size 2048 --async-scheduling`；
   - `--gpu-memory-utilization 0.85`；
   - `API_SERVER_COUNT=DP`、`VLLM_HTTP_TIMEOUT_KEEP_ALIVE=3600`；
   - `VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0`（没有 nvcc 的机器）。
3. **异步参数**：
   - NGEN 32，每步 32 组（1024 条），STALE 3；
   - trainer-bound 时 INFLIGHT + QUEUE_MAXSIZE ≈ 1.1 步样本（v3：640 + 512）；
   - rollout-bound 时每副本保持 ≥ 256 在途，同时检查最长组能否在 STALE 步内完成。
4. **trainer**：`--prefix-share on --token-budget 0 --dp-route on --drop-audit on`；launcher 校验 prefix_share md5。
5. **冒烟**：`MAX_STEPS=5`。看 step 1–3：
   - 样本数、各 rank 的 mb 计数；
   - /pause、断连、每副本 tok/s；
   - abs_log_rho p50/p90 对比上一个 run；kl_ref；staleness；drop 分桶。
6. **停止**：只 `kill -TERM <launcher pid>`；绝不用 `pkill -f`。

---

## 5. 踩过的坑与经验

格式：**症状 → 根因 → 修复/规则 → 证据**。

### 5.1 rollout / vLLM

| # | 症状 | 根因 | 修复 / 规则 | 证据 |
|---|---|---|---|---|
| 1 | 0.8B rollout 每卡吞吐低 | 旧 launcher 默认 `TP=4`（生产用 TP2）。0.8B 每层 all-reduce 的开销不值 | **0.8B 用 TP=1 的 DP 副本** | TP2 两卡合计 23.1k（每卡 11.5k）vs TP1 单卡 18.7k [实测，360-1] |
| 2 | 加并发、开 fp8 KV、开 CPU KV offload 都不涨吞吐 | 0.8B 的 KV 从来不是瓶颈：并发 256–2048 时 KV 使用率只有 7–51%。EngineCore 把一个 CPU 核跑满 100%，GPU 有 25–30% 时间空转 | **SOP 用 bf16 KV（`KV_DTYPE=auto`）**。fp8 KV 不更快，还额外增加数值误差 | stock 并发 256/512/1024/2048 = 16.8k/17.9k/16.1k/14.9k；tuned 17.2k/18.8k/18.7k/18.1k，prefix 命中 0.88–0.90 [实测，360-1 单卡，~5.5k prompt，2048 max tokens，ignore_eos，G=16，闭环，fp8 KV] |
| 3 | stock 配置在 2048 并发时只有 1023 条在跑 | `--max-num-seqs` 默认 1024 | `--max-num-seqs` 调大，但不能超过显存能放下的 Mamba 状态块数（共享卡上 4096 对 2070 块直接 ValueError） | [实测] |
| 4 | "MFU 拉到 40%" 的目标 | 0.8B decode 是内存/状态流量 bound | 别追 0.8B 的 MFU：实测约 4%，上限约 10% | [实测 + 估算] |
| 5 | 想用投机解码 | — | MTP 2 token：20.7k/21.9k/21.7k（**+17%**），但 KV 使用率升到 39–100%，prefix 命中降到约 0.68；MTP 1 token 18.8k（无收益）。**没在权重同步下测过**：drafter 的 `mtp.*` 权重是否会被同步、返回的 logprob 是否来自 target 模型，都未验证 | [实测，无权重更新] |
| 6 | 想用 bf16 SSM 状态省带宽 | — | mamba SSM cache 改 bf16：18.5k，无收益 | [实测] |
| 7 | 单卡想再挤一点 | EngineCore 是 CPU bound | 一卡两实例：10.7k + 10.8k = 21.4k（+14%）。可作为 rollout-bound 时的备选，没在同步下测过 | [实测] |
| 8 | 5 个副本时 /pause 要 44–73 s，每副本只有约 6k tok/s，424 次 ServerDisconnected | launcher 写死 `--api-server-count 1`，一个 Python 进程处理全部 1280 个在途请求，CPU 100% | **`API_SERVER_COUNT=DP`**（这本来就是 vLLM 默认值）。pause/update 广播到所有引擎，与 API server 数无关 | 修复后 /pause 0.03–0.4 s，每步 124–135 s → 52–61 s，gen 48–65k tok/s [实测 smoke2 → smoke3] |
| 9 | step 边界附近出现 439 次 ServerDisconnected（重试一次就好） | uvicorn 默认 keep-alive 5 s；aiohttp 复用了已被服务端关闭的连接 | **`VLLM_HTTP_TIMEOUT_KEEP_ALIVE=3600`** | 修复后 0 次 [实测 smoke4 及 v3 全程] |
| 10 | 远端 headless 启动即死：`ipc path ... is longer than 107 characters` | vLLM 在 `$TMPDIR/<uuid4>` 绑 ZMQ ipc socket，Unix socket 路径上限 107 字符 | **TMPDIR 用短路径**（不要放在 run 目录下） | smoke1 [实测]；watchdog 正确拆除了两台机 |
| 11 | 没有 nvcc 的机器上 TP2 崩溃 `Could not find nvcc`；27B 在 EngineCore init 时死 | flashinfer 的 allreduce 和 sampler 都要 JIT | **`VLLM_ALLREDUCE_USE_FLASHINFER=0`、`VLLM_USE_FLASHINFER_SAMPLER=0`** | [实测，360-1 / 360-2 都没有 nvcc] |
| 12 | 组亲和路由后 prefix 命中只有 0.913，到不了理论上限 0.969 | 混合模型下 vLLM 把 attention block 设为 **1072 token**，prefix 只能按整块命中，每个请求的尾块都要重算 | 理解即可；预热 1 个请求也没用（0.912） | [实测，DP=2，G=32] |
| 13 | 旧压测的结论自相矛盾 | `/tmp/bench_serve.py`：(a) 每次发同一个 prompt，prefill 全部命中缓存；(b) httpx 默认 `max_connections=100`，"256/768 并发"实际在途 ≤ 100 | **压测一律用 `tools/bench/bench_sat.py`**（闭环保持 C 个在途、组内共享 prompt、ignore_eos、读 /metrics 稳态窗口）。基于旧脚本的结论（如 27B "42% MFU、接近算力上限"）**已撤回** | [实测，代码审查] |
| 14 | KV offload 的长序列无损性 | — | `tools/bench/offload_parity.py` 写好了，**没跑过**。这件事不在关键路径上 | [未测] |

### 5.2 TRL 异步管线

| # | 症状 | 根因 | 修复 / 规则 | 证据 |
|---|---|---|---|---|
| 1 | step_s ≈ 93–150 s，而 fwd_bwd 只有 7–10 s | TRL `_score_loop` 串行打分；judge 组每组最多卡 90 s；`_groups_to_score`（maxsize 16）满了以后生成循环在 `put_nowait` 上空转 | 开 judge 时**必须**配 `--score-concurrency 32`（非阻塞打分） | v2 score_block_s 均值 23 s、最大 45 s [实测]；CPU 模拟：非 judge 组产出 33 → 139（4.2×），生成最长停顿 46.1 → 0.29 s [实测，test a] |
| 2 | 每步只训 125–189 条，78–80% 被 stale 丢弃 | 没设 token_budget → TokenBudgetBatcher 每行装 3–5 条，每步样本数由 token 决定、不由 CPS 决定；策略版本更新太快 | **`--token-budget 0`**（一行一整组）；STALE 3 | v3 每步恰好 1024 条 [实测] |
| 3 | trainer-bound 时 staleness 贴着上限（2.75–2.97，cap 3），长组被成批丢弃 | 缓冲太深：INFLIGHT 1280 + TRL 队列 1024 ≈ 2.3 步样本在排队；长组最早发出、最晚完成，最先超龄 | **trainer-bound 时 INFLIGHT + QUEUE_MAXSIZE ≈ 1.1 步样本**（v3：640 + 512） | step 11 丢弃（组）：<2k 0/10、2–4k 4/20、4–8k 4/8、8–16k 2/2、截断 6/8 → 修复后累计丢弃 0.07%，staleness 约 2.0 [实测 drop audit] |
| 4 | 想用更大的 queue 让打分永不反压 | FIFO 越深，trainer 拿到的样本越旧 | **不要**把 queue 扩到 (stale+1)·S。只有 trainer-bound 时才调，而且是往小调（约 S/2） | 同上 |
| 5 | 长组在 rollout-bound 时也被丢 | token 窗口 ≈ (stale..stale+1)·S·C/(f·INFLIGHT)，与引擎速度无关；INFLIGHT 越大，单序列 decode 越慢 | 先算"最长组能否在 STALE 步内完成"再定 INFLIGHT（SOP §4.3） | INFLIGHT 4096 → 每副本约 820 条、约 23 tok/s/序列，6k 的组要约 4.4 min > 3 步 [估算] |
| 6 | 截断组（16k）几乎从不进训练，−2 截断惩罚等于没训练 | 同 3/5，外加组版本按第一个成员打戳 | 记在 ROADMAP：截断/超时组不要整组丢弃，交给 GSPO clip 处理。**v3 没做** | v2 截断组占 14.4% 的组、23% 的生成 token [实测] |
| 7 | trainer 日志里的 reward 曲线随丢弃率移动 | trainer reward 只来自被训练的样本，带长度选择 | 比较学习效果用 rollout 侧所有被打分组的 reward + held-out eval | [代码] |
| 8 | 打分 hang 住时 run 静默挂起 | 原版心跳与进度无关 | `--score-task-max-s`（默认 max(600, 3×judge 超时)）把 hang 变成 RuntimeError → `failed_event` | test g1/g2 [实测 CPU] |

### 5.3 trainer

| # | 症状 | 根因 | 修复 / 规则 | 证据 |
|---|---|---|---|---|
| 1 | prefix sharing 首版门禁不过：序列 log-ratio p99 1.75e-3，单桶最大 6.5e-4，逐 token 0.0108，grad cos 0.99918，泄漏测试 17/18 | prefix attention 走的是 SDPA，生产走的是 FA3；共享本身没有引入偏差 | **prefix 用模型自带的 FA3；共享前缀按 FLA 的 64 token chunk 对齐，不足 64 的尾巴在每个分支里重算** | 补丁后：序列 log-ratio 0.0、逐 token 0.0、grad cos 0.99996（FA3 反向本身的噪声底）、泄漏 18/18、2-rank DDP 通过 [实测，8 个真实组 × 32 条，含 16k 截断] |
| 2 | 合并、部署时 prefix_share 被换回了旧版 | 多个 agent 并行，构建 agent 交回的是未打补丁的 `a679ebcd` | **launcher 校验 `PREFIX_SHARE_MD5`**；还要校验 align=64、没开 SDPA 回退、FA3 本地 kernel 目录存在 | 负向测试：把 align 改成 1，launcher 拒绝启动 [实测] |
| 3 | 想用 5 或 6 张 trainer 卡 | 一行一整组时，每个 micro-batch = N 组 | 每步组数必须是 N 的倍数（5 卡：30/35 组）。launcher 设了 `GROUPS_PER_STEP` 守卫 | [代码] + launcher 守卫用例 [实测] |
| 4 | 显存 | 一行 ≈ 14 GB + 每千 token 0.23 GB | 0.8B 下单卡峰值约 57 GB/rank（含优化器）。"截断 > 50% 就 OOM"的估计**偏保守**：重启后 step 2 有 92% 截断，没有 OOM | smoke2、v3 step 2 [实测] |
| 5 | 混合模型打包前向的 logprob 漂移 | GatedDeltaNet 的 conv/递归状态不会在 `position_ids` 边界重置 | 三选一：prefix share（生产）、`--per-seq-forward auto`（混合模型一次前向一条）、`hybrid_packing` 边界修复。见 [`PACKING.md`](PACKING.md) | [实测] |
| 6 | 从 checkpoint "续跑"后曲线不连续 | checkpoint 只存模型；KL ref 是**启动时**模型的 deepcopy；没有 resume；warmup 重跑；数据从 seed 0 重放 | 每次重启都是**新 run**。要保留原 KL anchor，需要 `--ref-model`（ROADMAP） | [代码] |
| 7 | 想要的 checkpoint 被删了 | `save_total_limit=5` 轮转 | 要当重启起点的 checkpoint，在轮转前拷出去 | ckpt-25 在 ckpt-150 时已被删 [实测] |
| 8 | prefix sharing 速度比 token 比低 | 剩下的是 completion 本身；`causal_lower_right` mask 优化在 autocast 下失败（q/k fp32 vs v bf16）已回退 | 后续优化项（§7） | 单组 forwarded token 降到 0.31×，墙钟 2.10× [实测] |

### 5.4 跨机 DP rollout

| # | 症状 | 根因 | 修复 / 规则 | 证据 |
|---|---|---|---|---|
| 1 | 远端 rank 死了，head 的 `/health` 仍是 200，路由到它的请求挂起，下一次 weight-transfer init 挂 300 s 后 500 | head 不感知 headless 的死亡 | **watchdog 必须有**：远端 pid（ssh `kill -0`）+ head pid + `/health`，任何一项失败就 SIGTERM launcher，拆掉整个 run | [实测，外部 SIGTERM 掉 360-1 rank 时]；v3 在 step 189 被误杀时 watchdog 按设计拆除 |
| 2 | NCCL 两端可能选到不一致的设备 | 一个 bond HCA 是 Ethernet link layer；docker0 两边都有但跨机不通 | `NCCL_IB_HCA=^<Ethernet 模式的 bond HCA>`；`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` / `VLLM_HOST_IP` 钉在共享 TCP bond 上；trainer 进程也要 export | [实测] |
| 3 | 副本悄悄从不同的策略采样 | 两台机"同路径"是各自的本地盘，内容可能不同 | 预检比较两台机 `model.safetensors` 的 sha256 | launcher 预检 [实测] |
| 4 | 担心跨机权重同步慢 | — | 不慢：1.706 GB / 473 张量，send 中位 0.120 s（14.2 GB/s），pause+send+resume 0.145 s；本机同步 0.17 s/step | [实测]；IB 366 Gb/s（ib_write_bw）|
| 5 | 想当然地认为 "greedy 输出变了 = 同步成功" | 噪声扰动后两个副本的 greedy 彼此不同（各自都变了，恢复都逐位精确），原因未证 | 同步验证用 `sync_test.py`：扰动后每个副本都要变、恢复后每个副本都要与基线逐位一致；不能只看一个副本的 greedy | [实测 + 未证] |
| 6 | 一台机没加载 `nvidia_peermem` | 那一侧 NCCL 走 IB 但没有 GPUDirect RDMA（经 host 内存中转） | 0.8B 无所谓；27B 全量同步要考虑（§6） | [实测] |
| 7 | headless 先于 head 启动会连不上 | headless 要连 head 的 `--data-parallel-rpc-port` | 先起 head，再起 headless；head 在所有 rank 握手前 /health 不会是 200 | [代码 + 实测] |
| 8 | 组亲和路由值不值 | 0.8B 下 decode 是瓶颈 | 可选。prefix 命中 0.855 → 0.913，实际 prefill −38%，gen tok/s 不变（18,407 vs 18,378）。DP 越大、prompt 越长、输出越短越有用。v3 开着 | [实测] |

### 5.5 judge / LLM reward

| # | 症状 | 根因 | 修复 / 规则 | 证据 |
|---|---|---|---|---|
| 1 | 加了第二把 key，judge 吞吐不变 | 中转的限流是**按账号、按模型**，每个模型每滚动 60 s 放行 10 次；被拒的请求很快返回 503 `model:rate_limited`，不计数 | 同一账号加 key 没用。要么换第二个模型桶（需先做一致性校准），要么换账号 | 两把 key 同时发 12+6 条，合计只放行 10 条；两个 luna 模型各 10 条；一个 sol 模型 30 并发不限流 [实测] |
| 2 | judge 超时率 70% | 需求约为供给的 2×（v2）/ 7×（v3 全量 K=2） | 按供给的 75% 定需求：**FRAC = 0.75 × supply/min × step_s/60 / (K × 每组存活 rollout 数)**。v3 K=2 → FRAC 约 0.11–0.12，RPM 9 | [实测供给 + 估算] |
| 3 | RPM 桶把 judge 调用卡得远低于供给 | 桶只拒不等；K=2 时两个 judge 组常常前后脚到达，第二组遇到的是空桶 | **burst ≥ K × NGEN × FRAC**（不是 NGEN × FRAC）；v3.1 改成 16 | CPU：burst 4 → 每组 4.0 次，burst 16 → 10.1 次 [实测 test d_burst]；GPU（v3.1，FRAC 0.24，BURST 8，RPM 18）：一个 pseudo-step 想发 15 次、只发出 4 次、桶拒 11 次，有效 FRAC 约 0.06，relay 每分钟只有 3–14 次 OK（供给 18）[实测] |
| 4 | 长 CoT 很少被 judge 到 | 原版按完成顺序（短的先完成）送审 | 打乱顺序（shuffle），并监控 `verdict_by_len_q / live_by_len_q` 是否平坦 | [代码 + 实测 CPU] |
| 5 | "每条 rollout 一完成就提前送审"看起来能省延迟 | 饱和时短 rollout 抢先占满桶 | **关闭**（`--reward-early-hooks` 仅实验用） | 判决率按长度四分位：提前送审 0.656/0.344/0.234/0.016，关闭 0.125/0.078/0.141/0.156 [实测，test h] |
| 6 | judge 组比同组其他样本更容易被 stale 丢 | judge 等待要 1.5–2 个版本；trainer-bound 时样本本来就在缓冲里排到 staleness 2.5–3 | auto 补偿 = 该组打分期间实际经过的同步次数（上限 STALE+2）。**GPU 上不够**：打分只要约 20 s，补偿常为 0，而样本到达时已贴着上限。改成 judge 组单独的固定上限（实验侧定为 STALE+4），验收 `judged_keep_frac ≥ 0.95` | CPU：test e1/g5 [实测]；GPU（v3.1 step 14）：judged_keep_frac 0.5、judged_allowance_mean 0、judged_staleness_mean 3，未评判组 0.969 [实测] |
| 7 | 两段式 reward（先给暂定值、judge 回来再纠正） | GSPO clip 会把大部分迟到的纠正清零，而暂定的正 advantage 已经强化了幻觉 | **分析过，没做** | NONBLOCKING_SCORING §6 |
| 8 | drop audit 和 judge 同时开时出现 `decision_mismatch` | audit 按固定 max_staleness 预测丢弃，而 judge 组有补偿 | 预期行为：mismatch 数 = 被补偿保留的 judge 样本数 | compose 测试 39 == 39 [实测 CPU] |

### 5.6 数值门禁（任何 infra 改动都要过）

**规则**：任何 infra 改动都在 step 1（lr=0）做 mismatch 探针，对照上一个 run 的 step 1。

| 指标 | v2（TP2，bf16 KV） | v3（TP1，bf16 KV，prefix share） | 容差 |
|---|---|---|---|
| abs_log_rho p50 / p90 | 6.4e-4 / 1.17e-3 | 6.5e-4 / 1.5e-3（smoke2）；6.8e-4 / 1.57e-3（launch）| p90 升了约 20%，归因于 TP2 → TP1，仍远小于 eps 3e-3 |
| signed log_rho_mean | −3.0e-4（系统性为负） | — | ±1e-4 |
| seq_clip_low / high | 0.18 / 0.078（eps 1e-3） | 0.29–0.4% / 0（eps 3e-3） | ±0.05（step 1 约 1000 条时可收紧到 ±0.03） |
| kl_ref | 5.6e-4 | 5.5–5.8e-4 | — |

来源 [实测]：v2 trainer log；v3 smoke2 / launch step 1。

- log_rho 系统性为负 → 低侧 clip 占主导 → 只清零 A<0 的序列 → 数值误差本身就把训练偏向正样本。所以**改数值的 infra 等于改目标函数**。
- fp8 KV 的对照：fp8 KV 的旧主 run 在 step 1 是 seq_clip_low 0.62、abs_log_rho 1.8e-3，bf16 KV 的 v2 是 0.18、7.2e-4 [实测]。两个 run 的其他设置不完全相同，但方向与 logprob-gap probe 一致（[`PRECISION.md`](PRECISION.md)）。

### 5.7 运维与多 agent 协作

| # | 症状 | 根因 | 规则 | 证据 |
|---|---|---|---|---|
| 1 | 改了 launcher，正在跑的 run 行为异常 | bash 边执行边读脚本文件 | **run 用着的 launcher 绝不改**；新版本另存一个文件名（v3 → v3_1） | [实测，事故规避] |
| 2 | GPU 上有没人认领的 vLLM server，挤得测试 OOM | 某个 agent 重试后留下的 server | 一台机一个 owner，放 lockfile；起任何 server 都记 pid 文件 | [实测，两次] |
| 3 | 给 workflow 内部 agent 发消息，结果多出一个副本，两份互相覆盖文件 | 消息没被转发，而是另起了一个新 agent | 不要给 workflow 内部的 agent 发消息；要改方向就停掉 workflow、重新起 | [实测] |
| 4 | 构建 agent 清理时误杀了两个别人的 `sleep 300` 进程 | 按模式匹配杀进程 | **永远不要 `pkill -f` / `killall`**；只杀自己记录的 PID/PGID | [实测] |
| 5 | v3 在 step 189 被拆除 | 另一个工具的 agent 在 360-1 上做了大范围 vLLM 清理，SIGTERM 了我们的 headless rank；watchdog 按设计拆除整个 run | 同上；跨机 run 期间远端 GPU 必须约定独占 | 18:40:19 [实测] |
| 6 | 预检看不出谁在占卡 | 容器里 `nvidia-smi --query-compute-apps` 返回空 | 预检用 `nvidia-smi --query-gpu=index,memory.used`，按显存阈值判断（launcher 默认 > 1000 MiB 算忙） | [实测] |
| 7 | `pkill -f "vllm serve"` 会杀掉 ssh 命令本身和别人的 server | 模式匹配 | 拆除只用 launcher 的 trap，远端只杀记录下来的 pgid | [代码] |
| 8 | 子 agent 中途全部 403 退出 | 中转余额 / 额度不足 | 长 workflow 前先查余额；审查 agent 没跑完的结论不能当"已审查" | 18:1x v3.1 的审查 agent 全部 403；19:29 开源合并 workflow 的 6 个 agent 全部 403，合并没做成，只推了生产快照 [实测] |
| 9 | 同一份 checkpoint 两台机都要有 | 各自的本地盘 | rsync 后比较 sha256；launcher 预检会再查一遍 | [实测] |
| 10 | eval 来不来得及 | 每个 checkpoint 的 held-out eval 约 520–530 s | 每 25 步（约 25 min）存一次的节奏下跑得完 | [实测] |

---

## 6. 27B

### 6.1 实测过的（360-2，Qwen3.8-27B，单卡/配置，max-len 8192，util 0.80，fp8 KV，1024 in / 256 out）

| 项 | bf16 | MXFP4（Marlin，weight-only A16） | 结论 |
|---|---:|---:|---|
| 模型显存 | 51.1 GiB | 18.29 GiB | −32.8 GiB [实测] |
| KV token | 835,584 | 1,280,000 | +53% [实测] |
| 256 并发吞吐 | 1,545 tok/s | 941 tok/s | **0.61×** [实测，但见 6.2] |

另外几项：
- **FP4 在 Hopper 上只换显存、不换速度**：没有 FP4 tensor core，Marlin 是 A16 kernel。
- **LoRA 让 rollout 吞吐降约 15%** [实测，0.8B]。
- **旧 FP8 trainer 探针不足以否定原生 FP8 full FT**：旧测试是冻结 BF16 base 的 FP8 LoRA；6.7× kernel 结论未找到绑定源码的原始 trace。2026-10-05 的 FP32 主参数＋原生 FP8 前反向已通过执行验证；关闭对齐、开启 CUDA graphs 的固定 G32 探针与对应 BF16 对照持平。对齐路径卷积融合提速 1.435×，仍慢于 fused BF16；decode 一致性门槛未通过，完整 RL 收益与训练质量尚未验证。见 [`RECIPE_FP8.md`](RECIPE_FP8.md) 和[历史核对](reports/FP8_HISTORY_2026-10-05.md)。
- 多模态 checkpoint 做 MXFP4 要排除 vision tower（`"ignore": ["*visual*"]`）；没有 nvcc 时要设 `VLLM_USE_FLASHINFER_SAMPLER=0` [实测]。

### 6.2 错的 / 未核实的

- **"27B 已经用到 42% MFU、接近算力上限"已撤回**：它基于 `/tmp/bench_serve.py`，该脚本所有请求同一个 prompt（prefill 全部命中缓存）。
- **[`PRECISION.md`](PRECISION.md) 里 27B bf16 vs MXFP4 的并发扫描（64–768）是否用了同一个客户端，未核实。** 如果用了，"256/768 并发"实际在途 ≤ 100，那条并发曲线（以及"bf16 在 256 并发达峰"）不能用。显存、KV 容量这些数字不受影响。
- 0.8B 的 "fp8 持平" **不能**推到 27B：两者的瓶颈性质不同。

### 6.3 先做什么（按顺序）

1. **用 `tools/bench/bench_sat.py` 重测 27B bf16 的并发曲线**：闭环、组内共享真实 prompt、ignore_eos、读 /metrics 稳态窗口，并发 64–1024。其他决策都依赖这条曲线。
2. **FP8 W8A8**（`--quantization fp8`，激活也走 FP8 tensor core）vs bf16，同一套压测；再做一次 logprob-gap probe（PRECISION.md 的方法）。这是 27B 上最可能有效、却**从没测过**的精度杠杆。Hopper FP8 峰值是 bf16 的 2 倍，预估 1.3–1.6× [估算]。
3. **DP vs TP**：bf16 27B 单卡放得下（51.1 GiB + 58.6 GiB KV），所以 TP1×DP 可行；在相同卡数下对比 TP1×DP 和 TP2。0.8B 的结论（DP 每卡效率高约 1.6×）不能直接搬。
4. **权重同步成本**：TRL 在 rollout DP>1 时，即使是 LoRA 也会回退到 merged **全量**同步（[代码]，`async_grpo_trainer.py`）。
   - 27B 每次同步约 54 GB bf16，14 GB/s 下约 4 s [估算]；
   - 360-2 没有 `nvidia_peermem`（无 GPUDirect RDMA），实际可能更慢；
   - 可选方向：DP=1 + TP（可以只同步 adapter，233 MB 是解析值），或者给 TRL 打补丁让 adapter 同步到所有 DP 副本。
5. **prefix sharing @27B**：未测。代码支持 Qwen3.5/3.8 混合文本层；上线前必须用 `tools/prefix_gate/` 重跑 6 项门禁和显存测试。
6. **trainer 侧**保持 fp32 master + bf16 compute。

**预期**：相对当前 bf16 基线，按每 GPU 秒算现实是 **3–5×** [估算]，最乐观约 6×。10× 只在换指标时成立，比如"有效样本/s"或"达到目标 reward 的时间"：解决截断、减少无效生成能贡献很大的倍数。

---

## 7. 未完成事项 / 下一批杠杆

| # | 项 | 预期 | 前提 / 风险 | 状态 |
|---|---|---|---|---|
| 1 | 把 `production/v3_1` 合进 `src/rlforge`（保留仓库的 LoRA / 自适应 clip / SwanLab / hybrid_packing），然后 GPU 冒烟 + step-1 探针 | 只剩一份代码 | 用生产 flag 时行为必须与 0229d8f8 一致；§5.6 容差内；mb 计数一致 | **下一步** |
| 2 | trainer 加到 5 rank（每步 30/35 组） | +20–25% [估算] | 每步样本数变化属于算法参数，需要实验侧同意；rollout 少一卡 | 实验侧选择保持 4 卡 / 32 组，以便前后曲线可比 |
| 3 | trainer kernel：`LIGER=1`（只用基础 kernel）、`torch.compile`、选择性 grad-ckpt（全量 grad-ckpt 现在开着）、重新实现被回退的 `causal_lower_right` mask（显式 bf16 cast）、`causal_conv1d` | +10–30% [估算] | 每项单独 A/B + step-1 探针；没有 nvcc，wheel 需要离线装 | 未开始 |
| 4 | 非阻塞 judge 上 GPU（v3.1） | 把 judge 开回来，吞吐影响预计接近 0 [估算] | 看 `judged_keep_frac`、`score_saturated_s_total`、`verdict_by_len_q` 是否平坦 | 在跑；已发现 BURST 和 judge 组 staleness 两个问题（§5.5 #3、#6） |
| 5 | resume + `--ref-model` | 崩溃不再等于整个 run 作废；KL anchor 可固定 | `save_only_model=False`、数据游标和 RNG 状态、`resume_from_checkpoint` | 未开始 |
| 6 | stale 丢弃的长度偏差 | 截断/超时组不整组丢弃，交给 GSPO clip | 先用 drop audit 量化 | ROADMAP |
| 7 | MTP 投机解码 | rollout +17% [实测，无同步] | perturb/restore 同步测试；`mtp.*` 权重是否被同步；acceptance 会随策略偏离下降 | 只在 rollout-bound 时有意义 |
| 8 | worker 里提前中止注定超窗的组 | rollout +5–10% [估算] | 不改 RL 语义（这些组本来就会被丢） | 未开始 |
| 9 | 每次同步后逐副本 probe logprob 校验 | 及早发现某个副本漏收权重 | 只记录 | 未开始 |
| 10 | 27B 线 | §6.3 | — | 未开始 |
| 11 | `offload_parity.py` | 长序列 KV offload 是否无损 | 不在关键路径上 | 未跑 |

---

## 8. 来源

内部报告（不在仓库里，数字已经抄进本文）：E2E_THROUGHPUT_PLAN、CROSS_NODE_ROLLOUT、PREFIX_SHARE_TRAINER、NONBLOCKING_SCORING、V3_LAUNCH、V3_1_INFRA_BUNDLE、CCTQ_RATE_LIMIT、ROADMAP_NOTES（均为 2026-10-03），以及当天 infra 会话的逐条汇报日志。

仓库内：[`SOP_0.8B.md`](SOP_0.8B.md) · [`production/v3_1/README.md`](../production/v3_1/README.md) · [`tools/cross_node/INTEGRATION.md`](../tools/cross_node/INTEGRATION.md) · [`PITFALLS.md`](PITFALLS.md) · [`OPTIMIZATION.md`](OPTIMIZATION.md) · [`PRECISION.md`](PRECISION.md) · [`PACKING.md`](PACKING.md) · [`LORA.md`](LORA.md) · [`ROADMAP.md`](ROADMAP.md)
