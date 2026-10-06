# H200 RL infra 迭代

Qwen3.5-0.8B；主参数和优化器为 FP32，训练计算精度见表（v3.2的logprob反向另用TF32）。Rollout 默认BF16；FP8行的trainer／rollout均用原生FP8 GEMM，敏感算子保留BF16／FP32。总吞吐只统计实际参加训练的 completion token。

| 阶段 | 训练计算 | 训练＋rollout H200 | 计时点 n | 中位秒/步 | 总 token/s | 相对 v1 | 等 rollout | MFU估算¹ | 首步训推差² |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0 | FP32 | 4＋4 | 9 | 120.17 | 1,607 | 0.81× | 54.7% | 12.1% | 2.04 |
| v1（近似 TRL baseline） | BF16 | 4＋4 | 17 | 113.80 | 1,972 | 1.00× | 84.9% | 1.1% | 1.20 |
| v2 | BF16 | 4＋2 | 4 | 136.38 | 1,551 | 0.79× | 90.9% | 0.9% | 1.17 |
| v3 | BF16 | 4＋5 | 186 | 61.06 | 31,392 | 15.92× | 1.9% | 15.6% | 1.64 |
| v3.1 | BF16 | 4＋3 | 25 | 63.87 | 30,755 | 15.60× | 2.0% | 15.0% | 1.52 |
| v3.1b | BF16 | 4＋5 | 61 | 65.11 | 30,516 | 15.47× | 2.4% | 14.8% | 1.48 |
| v3.1e | BF16 | 4＋5 | 110 | 65.27 | 30,957 | 15.70× | 1.9% | 15.0% | 1.54 |
| v3.2 | BF16 | 4＋5 | 447 | 36.18 | 53,221 | 26.99× | 7.4% | 25.9% | 1.54 |
| v3.2b | BF16 | 4＋5 | 447 | 36.15 | 52,825 | 26.79× | 7.8% | 25.7% | 1.46 |
| v3.2（4＋4对照） | BF16 | 4＋4 | 17 | 35.57 | 53,691 | 27.23× | 6.4% | 26.7% | 1.49 |
| v3.2（本轮对照） | BF16 | 4＋4 | 17 | 37.24 | 52,058 | 26.40× | 6.5% | 24.8% | 1.55 |
| v4（量化／缓存修复） | FP8 | 4＋4 | 17 | 38.33 | 52,992 | 26.87× | 8.2% | 4.3% | 14.42 |
| v4（compact backward＋128 GiB） | FP8 | 4＋4 | 17 | 32.34 | 58,363 | 29.60× | 5.8% | 4.7% | 14.42 |
| v4（view-base修复）³ | FP8 | 4＋4 | 17 | 33.82 | 57,103 | 28.96× | 5.2% | 4.6% | 14.33 |
| v4（stride统一） | FP8 | 4＋4 | 17 | 32.23 | 61,268 | 31.07× | 4.1% | 5.0% | 14.00 |
| v4（融合／缓存／MB1024）⁴ | FP8 | 4＋4＋2 reference | 3 | 71.67 | 135,439 | 68.68× | 8.6% | 10.5% | 14.28 |
| **v4（同源码复现，step 4–6）⁴** | FP8 | **4＋4＋2 reference** | **3** | **64.76** | **136,825** | 69.38× | **9.0%** | 10.6% | 14.50 |
| v4（同次复现，step 7–10）⁴ | FP8 | 4＋4＋2 reference | 4 | 69.18 | 108,421 | 54.98× | 23.8% | 8.3% | 14.50 |
| **v4（judge 链路修复，step 4–6）⁴** | FP8 | **4＋4＋2 reference** | **3** | **67.86** | **137,792** | 69.87× | **8.6%** | 10.8% | 14.46 |
| v4（β=0，4＋4静态） | FP8 | 4＋4 | 3 | 63.33 | 116,249 | 58.95× | 32.8% | 9.0% | 14.34 |
| v4（β=0，4＋4自适应） | FP8 | 4＋4 | 3 | 63.53 | 141,979 | 72.00× | 16.5% | 11.2% | 14.33 |
| **v4（β=0，4＋6静态）** | FP8 | 4＋6 | 3 | 58.01 | 151,401 | 76.78× | 11.3% | 11.8% | 14.20 |

### v4 setting

| 项目 | 设置 |
|---|---|
| baseline | 保留 GSPO 与原 reward／Luna judge；本轮去掉 KL reference，β=0；旧 v4 为 β=0.05 |
| 卡数 | 4＋4 共 8 张；4＋6 共 10 张，额外 2 张为跨机 rollout 副本 |
| batch | 每卡真实 microbatch 1024；GAS=1；global batch=4096；独立 G32；subbatch token=65536 |
| 精度 | FP32 主参数、梯度与 Adam；trainer／rollout 均用 FP8 GEMM；attention KV 为 BF16、GDN state 为 FP32 |
| 输入 | v3.1e ckpt100；rl_pool_b_v0_think；aiq_think_reward_v3；LR=2e-6，warmup=5，seed=0 |
| clip | 实际训练 low/high=0.004/0.004；同 batch 诊断 0.004/0.008、0.008/0.004、0.016/0.004 |
| 等待调节 | 按 optimizer step 的等待、队列、staleness/drop 调整入场数，完成已入场的完整 G32；4＋6 本次使用静态设置 |

4＋4＋2 reference 中的 2 张卡运行冻结 BF16 reference，为 β=0.05 的 KL 项提供 logprob；不负责 reward 或 policy 更新。β=0 后，这两张卡可改为 rollout，构成 4＋6。

总吞吐＝Σ 实际参加训练的 completion token／Σ step 秒，包含等待和权重同步。旧 20 步对照统计 step 4–20，历史长 run 统计 step≥4；新配方主窗口预先固定 step 4–6，10 步复现另列 step 7–10。本轮三个 β=0 run 均完成 6 步、exit=0，输入哈希相同；实际异步采样 batch 不同，每配置只跑了一次，未验证长期训练收益。

v0、v1 与旧 4＋4 BF16 对照使用同 checkpoint、KL reference、数据和公共超参数；历史主线各阶段的 checkpoint、batch 与 judge 设置不同，4＋5 主线另有 1 张评测卡。表中倍数表示各阶段的总吞吐。

¹ MFU 只统计 4 张训练卡并计入等待。v0–v3.2 使用 TRL 通用 FLOPs 公式，FP32／BF16 峰值取 67／989.5 TFLOP/s。v4 使用捕获的投影／head、attention／GDN 有用 FLOPs，按共享前缀 token 代理缩放；公式为 Σ(FLOPs精度／峰值精度)÷(4×总秒)，FP8／BF16 峰值取 1,979／989 TFLOP/s（[H200 规格](https://www.nvidia.com/en-us/data-center/h200/)）。不计 rollout／reference、padding 和重算，attention 长度与 head 占比也只是近似；**两种估算法不能用来判断跨精度 MFU 退化**。本轮 4＋4 静态／自适应／4＋6 的纯前反向阶段估算为 **14.4%／14.4%／14.3%**，整模型 MFU 尚非实测。

² 首步训推差＝LR 为 0 时，序列平均 log-ratio 绝对值的 p90，表内单位为 0.001；各组实际采样 batch 不同。与下面的逐 token 一致性测试口径不同。

³ view-base 轮的 judge 上游失败占 54.5%，仅作为系统速度观测。⁴ 新配方每卡 MB1024、GAS1、global4096；旧 4＋4 对照 global1024。旧 10 步复现及时 judge 为 3/228；修复链路后为 151/154。本轮三个 β=0 run 为 160/163、137/140、120/120，均保留原 reward 与 judge；这些计数覆盖整个 run 的已评分记录，也包括后来丢弃的样本。

本轮 4＋4 静态／自适应／4＋6 吞吐为最强 BF16 的 **2.17×／2.64×／2.82×**，包含精度、真实 microbatch、调度与 β 的改变。自适应 4＋4 的 inventory 在 step 4、5 后从 352→308→269，等待由 32.8% 降至 16.5%。4＋6 等待进一步降至 11.3%，总吞吐比自适应 4＋4 高 6.6%；按全部物理卡归一，自适应 4＋4 为 17,747 token/s/卡，4＋6 为 15,140 token/s/卡。

### 训推差异

同 checkpoint、同 token、零参数更新，两端均为原生 FP8；1 个实际样本、128 个 completion token：

| logprob 差 | 有符号均值 | 逐 token 绝对差 P90 |
|---|---:|---:|
| trainer−decode | −0.00434 | 0.1214 |
| trainer−prefill | −0.00219 | 0.1254 |
| prefill−decode | −0.00215 | 0.1431 |

**P90 门槛 0.004，三项均未通过。** 无 optimizer update、主权重与 serving 权重未变；诊断 serving 的 batch／prefix cache 配置与生产不同，尚不能代表全 batch 一致性。性能提升已测到，训推一致仍需修复。

### GSPO clip

4＋6 的 step 4–6，共 12,288 条 sequence；同一批实际 ratio 与 advantage 的阈值诊断，实际训练仍为第一行设置：

| ε low／high | 下侧越界 | 上侧越界 | 实际下侧 clip（负 advantage） | 实际上侧 clip（正 advantage） |
|---|---:|---:|---:|---:|
| **0.004／0.004** | 95.57% | 0.0814% | **43.84%** | 0.0081% |
| 0.004／0.008 | 95.57% | 0% | 43.84% | 0% |
| 0.008／0.004 | 75.62% | 0.0814% | 35.61% | 0.0081% |
| 0.016／0.004 | 5.84% | 0.0814% | 2.56% | 0.0081% |

分母均为全部 sequence；实际 clip 指 surrogate 对 ratio 的平坦分支，不是梯度范数裁剪。放宽 high 几乎没有作用，放宽 low 可减少平坦分支；这不能证明训练质量改善，也不能修复当前训推数值偏差与 policy age。日志的 token clip 约 17.1%／9.3%，与上表的 GSPO sequence clip 不同。

### MFU／稀疏化排查

四卡真实训练捕获一暖机后的 step；另做同 checkpoint 的单 H200 算子与固定 G32 对照。Linear、MLP 和 LM head 的 **dgrad／wgrad 已使用原生 FP8**：前向权重／激活 E4M3，反向梯度 E5M2，GEMM 高精度累加并写回 FP32 参数梯度；主参数和 Adam 为 FP32。attention、GDN、norm／conv 和归约仍有 BF16／FP32 运算。

| 排查项 | 数字 | 结论／范围 |
|---|---:|---|
| FP32 Adam | 6.43 ms／G32 | 独立暖机 GPU 计时，非四卡整步 |
| 权重同步 | 0.263 秒／步 | 未 profile 的 4＋6，占总墙钟 0.46% |
| 梯度同步 | 61 buckets／update，3.010 GB FP32 | 仅末尾一次 DDP backward；没有逐 G32 全量通信 |
| 最后 DDP backward | 1.05–1.10 秒；NCCL 与计算不重叠 0.05–0.36 秒 | 四卡 trace，含本地反向；不是纯 wire 时间 |
| batch 准备 | collator 2.07 秒／步；每 rank H2D 1.007 GB | 完整 batch 先广播再 slice；理想分片约 0.252 GB／rank |
| 指标 gather | 72 bytes／rank；rank0 3.089 秒 | rank 到达差 3.087 秒；不是传输百字节耗时三秒 |
| 计算提交 | 每 rank 约 50 个 subbatch／步 | MB1024、GAS1；trace step 每 rank 108–114 万 kernel |
| GPU 时间占比 | trace 训练区间 69–77% 有 GPU 活动 | 包含 NCCL 等待；profile 放大耗时，不能当生产 SM 利用率或可省预算 |
| 原生 2:4 sparse FP8 | 大 down GEMM 最好 1.38×，head 1.03× | 改为 tensorwise scale 的单算子诊断；原 rowwise 大 GEMM 多数更慢 |
| 零梯度 head | 固定 G32 compute 1.0018×，wall 0.9989× | 无整模型净收益；真实 token／权重，合成 advantage／old logprob，未使用生产 clip mask |
| 固定 head tile Graph | eager 0.481 ms → graph 0.563 ms（0.855×） | 含全部输入刷新、输出 clone；数值／新权重检查通过，graph＋静态输入占 378 MiB |

主吞吐仍为 **151,401 token/s＝76.78×v1**。本轮 100×目标按 v1 的 1,972 token/s 计算：**197,200 token/s**；同工作量和其他等待不变，compute 需 **47.82→34.39 秒／步**。同方法 MFU 14.31%→20% 对应 **34.22 秒／步**，均未达到。trace 的约 40 秒 profiler stop 开销与被扰动的 step 不计入吞吐。

优先减少 conv／norm／GDN、量化与反向写回的成本，提前生成 host 的 G32／索引计划，再处理 collator／整份 batch 复制及 4＋6 admission。生产 43.85% 是 GSPO 平坦分支的**序列比例**；先统计真实 inactive token 和完整零梯度 subbatch，再决定跳过反向。2:4 权重稀疏不会自动加速 wgrad，行 mask 也未必满足 dgrad 的转置约束；本轮稀疏和 Graph 候选均未启用。

训推共同前向仍需对齐权重／scale、量化粒度、prefill／decode 与 GDN state；上面的整模型 gate 仍未通过。原始数值、源码哈希及测量边界：[本轮排查证据](reports/V4_MFU_SPARSE_2026-10-06.json)、[探针与 trace 解析源码](reports/V4_MFU_SPARSE_2026-10-06_sources.tar.gz)。

### 改动简述

- **v0→v1**：主参数和优化器保持 FP32，训练计算改为 BF16 autocast；v0 使用 SDPA＋PyTorch fallback。
- **v2→v3**：调整 ranking reward；TP rollout 改为 TP1 多副本 DP，共享 prompt 前缀，修复 API 连接超时，调整 stale 和队列。
- **v3.1 系列**：恢复并发 judge，增加 judged staleness 补偿、丢弃审计和固定 KL reference；**v3.2** 加入 fast logprob、token 子批、自适应 activation checkpoint、rank 负载均衡。
- **v4**：前反向 GEMM 使用 FP8、更新与累加保持 FP32；修复量化重复编译、Adam 更新后量化缓存不刷新、vLLM 缓存 dtype 和 backward stride。融合 conv＋SiLU，减少 Norm 保存的中间量和 scheduler／输出处理，增大真实 microbatch。
- **本轮 β=0**：去掉 KL reference，保留原 reward／judge；测试 4＋6 与逐步 admission 控制，补齐 MFU 估算、零更新训推差异、GSPO 两侧 clip 诊断。
- **保留的限制**：FP8 KV、MB2048、合并独立 G32 的候选未带来预期收益；共同算子与融合梯度候选的整模型 gate 尚未通过。主入口只集成了部分可选优化，本表高吞吐来自冻结实验配方。

数值、窗口、输入／源码哈希：[本轮 β=0 证据](reports/V4_BASELINE_2026-10-06.json)、[历史 FP8 证据](reports/FP8_SCALE_2026-10-05.json)。复核源码：[本轮脚本与数值](reports/V4_BASELINE_2026-10-06_sources.tar.gz)、[冻结实验基座](reports/FP8_SPRINT_2026-10-06_sources.tar.gz)；需要原 checkpoint、数据、reward 与记录的运行环境。
