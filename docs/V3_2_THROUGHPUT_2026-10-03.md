# V3.2 trainer 吞吐优化（rlforge_v3_2，2026-10-03）

**一句话**：trainer 每步 fwd_bwd 从 **63.4 s 降到 31.2 s（2.03×）**，所有改动都数值等价（逐 token 差 ≤ 4e-6，per-seq log-ratio 差约 1e-7，grad cos ≥ 0.99994，与 FA3 反向本身的噪声底持平）。测法：真实长度分布的 8 个 micro-batch（= 1 步），单张 H200 上逐行跑完，再按 4 rank 取 max 合成一步。v3_1 在同一测法下是 63.4 s，主实验实测中位数是 61.2 s，口径对得上。显存峰值与长尾组解耦：32×16k 的长尾组从 120 GiB 降到 87 GiB，同时快 1.9×。端到端预计约 **1.5–1.7×**：trainer 提速后，在 v3.1e 的 INFLIGHT=640 下 rollout 会成为新瓶颈（§6）。

标注沿用 INFRA_HANDOFF：**[实测]**、**[估算]**、**[代码]**、**[未测]**。

---

## 0. 交付清单

| 项 | 位置 |
|---|---|
| 代码 | `360-2:/data/home/guoshaoyang/rlforge_v3_2`：由 rlforge_v3_1 复制后修改；v3_1 目录未改动，`find -newer` 为空 |
| 新/改文件 | `src/rlforge/prefix_share.py`（md5 `9686c54b`）、`src/rlforge/trainer.py`（`1a84deee`）、新增 `src/rlforge/fast_logprob.py`（`7840fc84`）、新增 `src/rlforge/fused_ops.py`（实测后**不启用**）、`scripts/v32_gate.py`（门禁/测速）、`scripts/v32_diag.py`、`scripts/v32_profile.py`、`scripts/v32_lp_unit.py`、`scripts/v32_ops_unit.py`、`tests/test_v3_2_subbatch.py`（CPU） |
| launcher | `360-2:/data/home/guoshaoyang/aiq_rl/run_g32_v3_2.sh`：由 `run_g32_v3_1.sh`（md5 da30395d）复制，v3_1 文件未改 |
| 门禁/测速原始结果 | `ophis-gpu:/data4/guoshaoyang/aiq_v32/out/*.json|*.log`（numerics_b、step_c/d/e、ddp、memcal、speed*） |
| 测试环境 | ophis-gpu GPU1（空闲 H200；GPU6 中途出现他人进程后即停用，其上的绝对时间作废）。venv 版本与 360-2 一致：torch 2.13.0+cu130、transformers 5.17.0、trl 1.14.0、fla 0.5.2、FA3 本地 kernel（从 360-2 拷贝）。权重：v3_1b_ckpt50（转 bf16 存储，加载时上转 fp32 master，与生产一致） |

## 1. 做了什么

| # | 改动 | 开关（launcher 默认） | 数值 |
|---|---|---|---|
| A | **快速 log-prob**（`fast_logprob.py`）：替换 TRL 的 `_ChunkedLogProbFunction`。logits 仍是 bf16 GEMM（与 TRL 相同），lm_head 每次调用只 cast 一次；在线 logsumexp/entropy 以及反向的 `g·(onehot−p)` 各编译成一个融合 kernel；反向两个大 GEMM 改用 TF32 tensor core（fp32 累加）。TRL 原实现是纯 fp32 SIMT sgemm | `FAST_LOGPROB=1`，`RLFORGE_LOGPROB_BWD=tf32` | lp/entropy 差 ≤ 2e-6，grad cos 1.0000000，范数比 0.99998 [实测] |
| B | **按 token 预算切子批**（`plan_subbatches` / `subbatch_logprobs` / `_compute_loss_subbatched`）：每组 completion 按长度降序，用 DP 做最优分桶（最小化 padding + 每桶开销 λ=1024 token），再按预算（默认 131072 padded token）贪心切成子批；每个子批共享同一 prompt 前缀，前缀在每个子批重新前向。每个子批都单独 forward+backward，所以显存峰值只取决于预算，与长尾无关 | `SUBBATCH_TOKENS=131072` | 逐 token **完全一致**（diag：0.0）[实测] |
| B' | **按显存预算自适应 ckpt**：每个子批按 `RLFORGE_SB_ACT_GB`（默认 80 GiB）和实测的每 token 激活成本（不 ckpt 一层 128 KB/token，ckpt 一层 4.7 KB/token，base 16 KB/token）算出最多能有几层不重算，后 n 层保留激活，其余层重算 | `SB_CKPT=auto` | ckpt 不改数值 |
| C | **rank 负载均衡**（`BalancedGroupRowBatcher`）：每个 micro-batch 内先用 LPT 放整组；之后最多做 6 次块迁移，每次把最重行中某组的一块 completion 移到最轻行（计入额外 prompt 和每块固定开销 6144 token 当量）。每个 micro-batch 的样本数（128）和每步样本数（1024）不变，哪些样本进哪一步也不变 | `BALANCE_ROWS=on` | loss 按 micro-batch 总样本数归一（见 §2.2） |
| D | 融合 conv1d / RMSNorm / SwiGLU（`fused_ops.py`，Inductor） | **不启用**（未设 `RLFORGE_FUSED_OPS`） | per-seq p99 2–2.5e-3，**超出门槛**，否决（§4） |

**DDP 安全性**（B 的关键）[代码 + 实测]：一行有多个子批时，除最后一个子批外都走**未包装的 module**（accelerate 的 autocast 包装在 module 上，DDP reducer 不会被触发），并在 compute_loss 内部完成 backward，梯度累加到 `.grad`。最后一个子批走 DDP 包装的 `model`，loss 交还给 trainer，由 `accelerator.backward` 做唯一一次同步 backward。因此不管各 rank 切出几个子批，每个 rank 每个 micro-batch 都**恰好 1 次 DDP forward、1 次同步 backward**，micro-batch 数仍固定为 gas=8，不需要用空批补齐。

## 2. 正确性

### 2.1 数值门禁（`v32_gate.py numerics`，真实 trainer 代码路径对比）

方法：两条路径都调用**真实的** `GSPOAsyncGRPOTrainer.compute_loss`。旧路径是 v3_1 分支，用 TRL 原版 logprob；新路径是 `_compute_loss_subbatched`，打开 fast logprob。模型构造与 TRL 完全相同：FA3、fp32 master、bf16 autocast、非重入 grad ckpt。KL ref 是 bf16 eval 副本，加了 2% 相对噪声，使 kl_ref 落在 0.054–0.060，与生产的 1e-2 同一量级（ckpt175 bf16 当时还在传）。old_log_probs = 旧路径 lp + 每条序列 N(0, 3e-3) 偏移 + 每个 token N(0, 0.05)，使 seq clip 真实触发（clip_low 3–12%）。

数据：真实 prompt（v3.1e task_split 的 qid → 训练池）+ 真实 rollout 文本窗口；长度分四种 profile：

- **longtail**：24 条 16384 截断 + 8 条 8–15k，共 58 万 token，即 v3_1 关 ckpt 时 OOM 的那类组；
- **typical**：v3.1e 中位组，含 1 条截断；
- **short**；
- **mixed2**：一行 1.5 组，其中半组是 partial group。

门槛：|mean| < 2e-4，p99 < 1e-3，grad cos > 0.99，范数比在 ±2% 以内。

| 组 | 配置 | per-seq Δlog-ratio mean / p99 / max | 逐 token max | loss 相对差 | grad cos | 范数比 | 子批数 | 时间（s） | 峰值（GiB） | 结果 |
|---|---|---|---|---|---|---|---|---|---|---|
| typical | v3_1 | — | — | — | — | — | 1 | 7.45 | 25.2 | 基线 |
| | 131072:auto | −2.5e-8 / 6.5e-8 / 6.7e-8 | 3.8e-6 | 0 | 0.999951 | 1.00020 | 1 | 5.00 | 76.8 | PASS |
| | 65536:auto | 同上 | 3.8e-6 | 6e-8 | 0.999951 | 1.00007 | 2 | 4.09 | 86.5 | PASS |
| | 32768:none | 同上 | 3.8e-6 | 6e-8 | 0.999951 | 1.00006 | 4 | 3.97 | 100.6 | PASS |
| short | 131072:auto | −1.6e-8 / 9.2e-8 / 1.1e-7 | 3.8e-6 | 0 | 0.999964 | 0.99992 | 1 | 1.64（v3_1 2.80） | 90.7 | PASS |
| mixed2 | 131072:auto | −2.1e-8 / 5.7e-8 / 6.0e-8 | 3.8e-6 | 4e-7 | 0.999941 | 0.99992 | 2 | 4.79（v3_1 8.65） | 85.6 | PASS |
| longtail | v3_1 | — | — | — | — | — | 1 | 46.0 | **120.1** | 基线 |
| | 131072:auto | −2.3e-8 / 3.3e-8 / 3.3e-8 | 3.8e-6 | −4e-7 | 0.999979 | 1.00018 | 5 | **24.0** | **86.6** | PASS |
| | 65536:auto | 同上 | 3.8e-6 | −2e-7 | 0.999978 | 1.00012 | 10 | 23.5 | 94.7 | PASS |
| | 32768:none | 同上 | 3.8e-6 | −7e-7 | 0.999979 | 0.99988 | 28 | 24.7 | 105.8 | PASS |

全部 16 个组合都 PASS，完整表在 `numerics_b.json`。上表峰值不含 AdamW 状态（numerics 模式不建 optimizer）；step 模式含 AdamW，峰值 92.8 GiB。

grad cos 卡在 0.99994–0.99998，不是新路径引入的：FA3 反向用 atomic add，本身不确定。v3 门禁测到的噪声底是 0.99996，同一路径两次 backward 也在这个量级。

**逐项归因**（`v32_diag.py`，同一 typical 组，无梯度，相对 v3_1 路径）[实测]：

| 改动 | per-seq Δ p99 | 逐 token Δ 均值 |
|---|---|---|
| 同一路径重跑（确定性） | 0 | 0 |
| fast_logprob | 6.5e-8 | 1.7e-7 |
| 子批（1 / 2 / 4 个子批，DP 分桶） | **0（逐位一致）** | 0 |
| 融合 conv1d | 2.5e-3 | 1.6e-2 |
| 融合 RMSNorm | 2.1e-3 | 1.6e-2 |

### 2.2 GSPO seq_mean 归一与梯度累积

- v3_1：每行 `.mean()` 覆盖 32 条，再 /gas，DDP 再 /R，所以每条序列权重 = 1/(32·8·4) = 1/1024。
- v3_2：每个子批 loss = Σ_seq(seq_mean_loss + β·seq_mean_k3) × R/n_mb / gas，其中 n_mb = micro-batch 的总样本数，由 collator 的 `global_n_forward_tokens / mean_seq_len` 精确反推。DDP 平均后每条序列权重 = 1/(n_mb·gas) = 1/1024，与 v3_1 相同。
- 各行样本数不同时（balance 拆组、partial group）v3_2 仍然精确；v3_1 的行内 `.mean()` 在这种情况下会给不同序列不同权重。
- 门禁里 mixed2 行有 48 条序列，loss 与梯度和 v3_1 一致：此时 n_mb = 行内条数，两种写法等价。

### 2.3 DDP

见 §7：真实 torch DDP，2 rank，gas=2，第一个 micro-batch 在 no_sync 下。两个 rank 每个 micro-batch 的子批数不同（3–6 个）；另做了一组强制不均的布局（rank0 20 条、rank1 44 条，跨组拆分）。三组测试都**无死锁**，每个 rank 每 2 个 micro-batch 恰好 2 次 DDP forward。DDP 平均后的梯度与"各 rank 用 v3_1 路径求梯度、再手工 all-reduce"比较：cos 0.999994，范数比 1.00003–1.00008。

## 3. 测速（真实长度分布的一步，单 H200 = GPU1，[实测]）

方法（`v32_gate.py step`）：

- 取 v3.1e task_split 中第 160–191 组，按顺序组成 8 个 micro-batch × 4 组 = 1 步。prompt 用真实 prompt。completion 长度围绕该组记录的 `mean_tokens` 做 lognormal 抽样（CV 0.5，按 prefix pad_frac 0.08 标定），截断数取该组记录值，内容是真实文本。
- 布局用对应的 batcher 生成每个 rank 的行，每行单独 fwd+bwd 计时（含 KL ref 前向、AdamW 状态已分配）。
- micro-batch 时间 = 4 行的 max，因为 rank 在每个 micro-batch 的 collective 处互等。一步 = 8 个 micro-batch 之和。
- 校验：v3_1 在该测法下 63.4 s/步，主实验 69 步的实测 fwd_bwd 中位数 61.2 s（每步 545 万 forwarded token），口径吻合。

| 配置（逐项叠加） | fwd_bwd/步（s） | 完全均衡下界（s） | 峰值显存（GiB） | forwarded tok/s（4 rank） | 相对 v3_1 |
|---|---|---|---|---|---|
| v3_1（生产） | **63.4** | 48.7 | 39.9 | 8.8 万 | 1.00× |
| + A fast_logprob | 40.8 | 31.6 | 39.9 | 13.7 万 | 1.55× |
| + B 子批 131072，全 ckpt | 39.5 | 30.3 | 38.6 | 14.2 万 | 1.61× |
| + B' auto ckpt（80 GiB） | 38.3 | 28.6 | 91.4 | 14.6 万 | 1.66× |
| **+ C rank 均衡（= v3_2 默认）** | **31.2–31.8** | 28.7 | 92.8–93.9 | **17.6–18.0 万** | **2.03×** |
| 同上，子批 262144 | 31.0 | 28.6 | 94.7 | 18.1 万 | 2.05× |
| 对照：v3_1 去掉 KL ref | 52.0 | 40.2 | 38.8 | 10.8 万 | — |
| 对照：v3_2 默认去掉 KL ref | 25.0 | 22.9 | 91.0 | 22.4 万 | — |

（forwarded token = 每步 560 万 unshared token，与 TRL 的 `batch/forwarded_tokens_per_step` 同口径。）

单组测速（GPU1，fwd+bwd+ref，`numerics_b` 中的 t）：

| 组 | v3_1 | v3_2 131072:auto | 加速 |
|---|---|---|---|
| typical | 7.45 s | 5.00 s（65536:auto 4.09 s） | 1.5–1.8× |
| short | 2.80 s | 1.64 s | 1.7× |
| mixed2 | 8.65 s | 4.79 s | 1.8× |
| longtail | 46.0 s / 120 GiB | 24.0 s / 87 GiB | 1.9× |

### 3.1 逐项结论

1. **最大头是 log-prob（A）**。profile 显示 v3_1 一行的 CUDA 时间约 47% 花在 TRL 的 chunked logprob 上：反向的 `grad_logits @ W` / `grad_logits.T @ h` 是纯 fp32 SIMT/xmma sgemm，没有用 tensor core，而且每个 [2048×8192] 的 tile 要经过约 8 次 fp32 逐元素 pass。替换后单测 fwd 5.4×、bwd 5.0×（N=40k token）。这一项就让整步快了 1.55×。patch_chunked_lm_head 确认在用：prefix 路径直接调用 `_ChunkedLogProbFunction`，现在由 `RLFORGE_FAST_LOGPROB` 切换。
2. **子批 + 自适应 ckpt（B/B'）**。在典型组上只省 3–6%：重算只是 forward 的一份，而且 0.8B 每层不 ckpt 要 128 KB/token，80 GiB 只够约 25k token 全层不重算。真正的价值在于**显存与长尾解耦**：longtail 组 120 → 87 GiB，并且更快（24 s 对 46 s）。预算本身对速度不敏感：65k / 131k / 262k 差别 < 3%；32k 以下前缀重复和 GDN 小 batch 的开销开始显现。
3. **rank 均衡（C）收益 +20%**。v3_1 每行一整组，4 个 rank 每个 micro-batch 等最重的一组：真实 profile 下 max/mean 约 1.3–1.5，所以 63.4 s 中有 15 s 是等待（完全均衡下界 48.7 s）。第一版逐条迁移 completion，组被切得过碎，额外的 prompt 前向和小调用开销抵消了收益。改成块迁移并计入每块固定开销后，每个 micro-batch 平均只多 1.8 块，max/mean 从 1.36 降到 1.04（CPU 模拟），实测 38.3 → 31.2 s，距下界 28.7 s 只差 9%。
4. **padding（任务第 2 项）**：
   - 生产日志里 14–22% 的 `batch/pad_frac` 是 collator 为凑矩形加在 rank 之间的 padding。`compute_loss` 在前向前就剥掉了，**不耗 FLOPs**，只多占 dispatcher 的广播字节。均衡后各行长度接近，这一项也随之下降。
   - 真正耗算力的是 prefix 分桶的 padding（`prefix_share/pad_frac` ≈ 8%）。DP 最优分桶把它降到 4.0%（typical）和 0.5%（longtail）。λ 取 1024 token 时 DP 分桶本身比 v3_1 的 0.8 比例分桶略慢：全 ckpt 下 +5% 的调用开销，抵掉了 padding 的收益。所以 padding 已不是瓶颈，没有再往下压。
5. **KL ref（任务第 3 项）**：ref 与 policy 权重不同，任何一层都无法共享前向，包括 prompt 部分；合并成一次调用也省不掉算力。v3_2 中 ref 与 policy 在**同一子批内**各前向一次。ref 走 no-grad + fast logprob（logprob 那部分便宜了 5×），不 ckpt、不存激活。去掉 ref 的对照见 §7，用来量化剩余成本。
6. **其他低风险项**：
   - TF32 全局开关：主体是 bf16 autocast，无效；只在 fast_logprob 的反向 GEMM 内局部开启（A 已包含）。
   - torch.compile 局部融合（D：conv / RMSNorm / SwiGLU）：典型组再快 21%（4.04 → 3.18 s）、显存降 20%，但 RMSNorm 的 reduction 顺序和 conv 的 1-ulp bf16 差异经 24 层放大，per-seq p99 到 2–2.5e-3，**超出 1e-3 门槛，不上**。代码保留在 `fused_ops.py`，默认关闭。
   - 更大 micro-batch：每个 micro-batch 的组数是算法参数（4 rank × gas 8 = 32 组/步），不改；单次调用内的 token 规模由子批预算控制，131k 与 262k 差别小于 3%。

## 4. 被否决 / 没做的

| 项 | 原因 |
|---|---|
| 融合 conv/norm/SwiGLU（`RLFORGE_FUSED_OPS`） | +21% 速度，但 per-seq log-ratio p99 2.5e-3 > 1e-3 [实测]。若以后放宽门槛（例如 rollout 侧 bf16 噪声本身已是 1e-3 量级），可以重新评估 |
| 缓存 prompt 末状态、各子批从同一状态分支（不重复 prompt 前向） | 预算 ≥ 64k 时，典型组只有 1–2 个子批，重复的 prompt 前向不到 4%；长尾组 5 个子批，约 3%。收益小于 DDP 下"prompt 图跨子批 backward"带来的复杂度和风险，没做 |
| 关掉全部 ckpt / 固定隔层不 ckpt | 仍按显存预算由 auto 决定；0.8B 每层不 ckpt 要 128 KB/token，全关只放得下约 25k token/子批 |

## 5. 显存

- 每 token 激活（memcal，GPU1）：全 ckpt 113 KB/token（主要是 24 层 × fp32 残差 4 KB 作为 ckpt 输入）；不 ckpt 时每层 +123 KB/token。
- auto 按 `RLFORGE_SB_ACT_GB=80` 选层数。step 实测峰值 92.8–94.7 GiB，含 AdamW 状态和 bf16 ref。生产 rank 另有 DDP bucket（约 3 GiB）和权重同步缓冲，预计峰值 ≤ 105 GiB，H200 141 GiB 上留有 35 GiB 以上余量 [估算]。
- 若 smoke 时峰值偏高：调低 `RLFORGE_SB_ACT_GB`（60 → 每个子批少几层不 ckpt，速度约降 2%），或 `SUBBATCH_TOKENS=65536`。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 由 launcher 默认打开。

## 6. 端到端预期与风险

- **trainer**：fwd_bwd 61 → 约 30 s/步 [估算 = 实测比例 2.03× × 生产 61.2 s]，step 约 33 s（另加约 3 s 权重同步、rollout 等待和 python 开销）。
- **rollout 会成为瓶颈** [估算]：
  - 现在 rollout 每分钟出 31.5 组（任务日志实测，2742 组 / 87 min）。这正好等于 trainer 的消费速度：INFLIGHT=640 + 队列 512 把 rollout 节流在 trainer 的速度上，所以 rollout 的真实上限没有直接测量。
  - INFRA_HANDOFF 估算 rollout 有约 50% 富余，对应约 40–45 s/步。
  - 所以 v3.1e 的 INFLIGHT=640 下，端到端预计 64 s → 约 40 s/步，即 **1.5–1.7×**。若要拿满 trainer 的 2×，需要另行决定是否提高 INFLIGHT（会改变 staleness 分布，属实验侧决策；本次按要求保持 640）。
- **staleness**：trainer 变快后队列里积压的样本变少，staleness 应该下降，丢弃也会减少（drop_audit 可直接观察）。
- 风险：
  1. **未在 4 卡生产环境冒烟**：360-2 / 360-1 没有空闲的 4 卡 trainer + rollout 资源，只做了单卡门禁和 2-rank DDP 测试（§7）。上线前务必先跑 smoke（§8）。
  2. `torch.compile` 在每个 rank 首次调用时编译 fast_logprob 的两个 tile kernel，step 1 多约 20–60 s。动态形状下若重编译超过 8 次，会回退 eager：结果仍正确，只是变慢。
  3. balance 打开后，各 rank 每个 micro-batch 的序列数不同：`[rlforge][mb_audit]` 的 `local_seqs` 会不一样，这是预期行为；`microbatches` 仍必须是 8。
  4. 第一步数值探针：按 SOP，对比新 run step 1 的 abs_log_rho p50/p90、seq_clip_low、kl_ref 与 v3.1e step 1。门禁显示 per-seq 差约 1e-7，预期在噪声内。
  5. 从 checkpoint 重启的语义（Adam 重置、warmup 重跑、数据顺序从 seed 0 开始）与 v3.1e 从 ck50 重启时相同（SOP §11）。

## 7. DDP 测试与 KL 成本 [实测]

**DDP**（`torchrun --nproc_per_node 2 v32_gate.py ddp`，ophis GPU1 上 2 个进程用 gloo；DDP 本身用的是 torch 的 reducer）：

| 布局 | 各 rank 每个 mb 的样本数 | 各 rank 每个 mb 的子批数 | DDP forward 次数/rank | 死锁 | grad cos | 范数比 |
|---|---|---|---|---|---|---|
| balanced（默认参数） | [32, 32] / [32, 32] | [4, 5] / [5, 5] | 2 / 2 | 无 | 0.999994 | 1.00003 |
| balanced（min_gain=0） | 同上 | [4, 5] / [5, 5] | 2 / 2 | 无 | 0.999994 | 1.00008 |
| **强制不均**（一组拆到两个 rank） | [20, 32] / [44, 32] | [3, 5] / [6, 5] | 2 / 2 | 无 | 0.999994 | 1.00008 |

参照梯度的算法：每个 rank 用 v3_1 路径（行内 `.mean()`），按 `len(row)·R/n_mb` 缩放成每条序列 1/n_mb 的权重，然后 all-reduce 求平均。这同时验证了不均布局下的 per-seq 归一。

**KL ref 成本**：v3_2 去掉 ref 是 25.0 s/步，带 ref 是 31.2 s/步，ref 占 6.2 s（20%）；v3_1 中 ref 占 11.4 s（52.0 → 63.4）。fast_logprob 让 ref 的 logprob 部分变便宜，但它的 24 层前向是省不掉的：权重不同，prompt 部分也无法与 policy 共享。仍可考虑的方向：
- ref 改成每 K 步才算一次的近似（**改目标函数，没做**）；
- ref 用更小的 dtype（同样会改数值，没做）。

## 8. 上线（由主会话执行；本次开发期间 v3.1e 未受任何影响）

**冒烟没跑**：360-2 GPU 0–7 和 360-1 GPU 2/4 都被 v3.1e 占用，没有空闲的 4 卡 trainer + rollout 资源。在 360-2 上只做了 DRY 预检（launcher 语法、代码 md5、门禁 md5、FA3 目录、reward 文件都通过；端口和 GPU 被占用属预期）以及 CPU 导入/单测。下面是完整命令，由主会话切换。

```bash
# ---- 0) on 360-2: freeze the latest v3.1e checkpoint as the init (both hosts need the same file at the same path)
cd /data/home/guoshaoyang/aiq_rl
R=async_g32_v3_1e_ck50_ref175_20261003
CK=$(ls -d runs/$R/checkpoint-* | sed 's/.*-//' | sort -n | tail -1); echo "latest ckpt $CK"
DST=/data/shared/guoshaoyang/aiq_rl_store/models/v3_1e_ckpt${CK}_20261003
mkdir -p $DST && cp runs/$R/checkpoint-$CK/{model.safetensors,config.json,generation_config.json,tokenizer.json,tokenizer_config.json,chat_template.jinja} $DST/
ssh 10.234.161.2 "mkdir -p $DST" && scp -q $DST/* 10.234.161.2:$DST/      # launcher preflight compares sha256 on both hosts

# ---- 1) stop v3.1e (its trap tears down trainer + head + 360-1 ranks); stop its eval watcher separately if needed
kill -TERM $(python3 -c "import json;print(json.load(open('runs/$R/run.json'))['launcher_pid'])")

# ---- 2) smoke: 5 steps, same algorithm, new run dir
COMMON="INIT_MODEL=$DST RLFORGE_REF_MODEL=/data/shared/guoshaoyang/aiq_rl_store/models/v3_ckpt175_20261003 \
  CPS=256 NGEN=32 STALE=3 INFLIGHT=640 QUEUE_MAXSIZE=512 LR=2e-6 KL_BETA=0.05 GSPO_EPS_LOW=4e-3 GSPO_EPS_HIGH=4e-3 \
  PREFIX_SHARE=on TOKEN_BUDGET=0 DP_ROUTE=on DROP_AUDIT=on GRAD_CKPT=1 \
  REWARD=aiq_think_reward_v3:think_reward AIQ_HALLUC=1 AIQ_HALLUC_BASE_URL=http://127.0.0.1:3130/v1 AIQ_HALLUC_K=2 \
  AIQ_HALLUC_LUNA_REWARD=1 AIQ_HALLUC_JUDGE_FRAC=0.24 AIQ_HALLUC_RPM=18 AIQ_HALLUC_BURST=16 AIQ_HALLUC_THREADS=16 \
  SCORE_CONC=32 JUDGED_STALE=7 RLFORGE_JUDGED_STALE_FIXED=1 EARLY_HOOKS=0 \
  SERVER_GPUS=0,1,3 REMOTE_GPUS=2,4 REMOTE_IP=10.234.161.2 TRAINER_GPUS=4,5,6,7"
# v3_2 knobs are launcher defaults: V32=1 FAST_LOGPROB=1 SUBBATCH_TOKENS=131072 SB_CKPT=auto BALANCE_ROWS=on RLFORGE_SB_ACT_GB=80
# (do NOT pass PREFIX_SHARE_MD5 -- the v3.1e env carries the v3_1 value 897dbfc9; v3_2's default is 9686c54b)
env $COMMON RUN_NAME=async_g32_v3_2_smoke_ck${CK}_20261003 MAX_STEPS=5 SAVE=1000 \
  setsid nohup bash run_g32_v3_2.sh full > logs/async_g32_v3_2_smoke_ck${CK}_20261003.driver.log 2>&1 < /dev/null &
#   smoke checks (logs/trainer_dp_<run>.log):
#     [rlforge] v3_2 subbatch_tokens=131072 sb_ckpt=auto ... ; [rlforge][mb_audit] microbatches=8 on every rank
#     (local_seqs may differ per rank -- expected with BALANCE_ROWS=on); no OOM; nvidia-smi GPU4-7 peak <= ~110 GiB
#     perf/fwd_bwd_s ~28-33 (step 1 +20-60 s torch.compile); v3_2/subbatches_per_row ~1-5
#     step-1 probe vs v3.1e step 1: gspo/abs_log_rho_p50/p90, seq_clip_low_frac, kl_ref within SOP 5.6 tolerances
#   if peak memory is high: RLFORGE_SB_ACT_GB=60 (or SUBBATCH_TOKENS=65536); fallback to the v3_1 trainer path: V32=0

# ---- 3) full run (after the smoke passes; its launcher exits by itself after 5 steps)
env $COMMON RUN_NAME=async_g32_v3_2_ck${CK}_ref175_20261003 MAX_STEPS=$((550 - CK)) SAVE=25 \
  setsid nohup bash run_g32_v3_2.sh full > logs/async_g32_v3_2_ck${CK}_ref175_20261003.driver.log 2>&1 < /dev/null &
# MAX_STEPS = 550 - CK keeps v3.1e's total step budget (v3.1e: MAX_STEPS=550 from ck50); use 550 to match the literal value.
```

**预期**：
- fwd_bwd 约 61 → 30 s/步（2.0× [实测比例]）；
- trainer 侧 trained tok/s 约 3.2 万 → 6.5 万；
- 端到端 step_s 64 → 约 35–42 s（1.5–1.8× [估算]，上限取决于 rollout，在 INFLIGHT=640 下 rollout 会成为新瓶颈：看 `perf/rollout_wait_s` 是否从 0.4 s 升到 5–10 s）；
- staleness_mean 下降，stale 丢弃减少。

**回退**：`V32=0` 等价于 v3_1 trainer 路径（同一代码树：`--subbatch-tokens` 为 0 时走 v3_1 的 compute_loss，logprob 也回到 TRL 原版）。也可以直接用未改动的 `run_g32_v3_1.sh`。
