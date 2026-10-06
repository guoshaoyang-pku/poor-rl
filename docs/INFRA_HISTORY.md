# H200 RL infra 迭代

Qwen3.5-0.8B；主参数和优化器为 FP32，训练计算精度见表（v3.2的logprob反向另用TF32）。Rollout 默认BF16；FP8行的trainer／rollout均用原生FP8 GEMM，敏感算子保留BF16／FP32。总吞吐只统计实际参加训练的 completion token。

| 阶段 | 训练计算 | 训练＋rollout H200 | 计时点 n | 中位秒/步 | 总 token/s | 等 rollout | MFU估算¹ | 首步训推差² |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v0 | FP32 | 4＋4 | 9 | 120.17 | 1,607 | 54.7% | 12.1% | 2.04 |
| v1（近似 TRL baseline） | BF16 | 4＋4 | 17 | 113.80 | 1,972 | 84.9% | 1.1% | 1.20 |
| v2 | BF16 | 4＋2 | 4 | 136.38 | 1,551 | 90.9% | 0.9% | 1.17 |
| v3 | BF16 | 4＋5 | 186 | 61.06 | 31,392 | 1.9% | 15.6% | 1.64 |
| v3.1 | BF16 | 4＋3 | 25 | 63.87 | 30,755 | 2.0% | 15.0% | 1.52 |
| v3.1b | BF16 | 4＋5 | 61 | 65.11 | 30,516 | 2.4% | 14.8% | 1.48 |
| v3.1e | BF16 | 4＋5 | 110 | 65.27 | 30,957 | 1.9% | 15.0% | 1.54 |
| v3.2 | BF16 | 4＋5 | 447 | 36.18 | 53,221 | 7.4% | 25.9% | 1.54 |
| v3.2b | BF16 | 4＋5 | 447 | 36.15 | 52,825 | 7.8% | 25.7% | 1.46 |
| v3.2（4＋4对照） | BF16 | 4＋4 | 17 | 35.57 | 53,691 | 6.4% | 26.7% | 1.49 |
| v3.2（本轮对照） | BF16 | 4＋4 | 17 | 37.24 | 52,058 | 6.5% | 24.8% | 1.55 |
| Native FP8（量化／缓存修复） | FP8 | 4＋4 | 17 | 38.33 | 52,992 | 8.2% | — | 14.42 |
| Native FP8（compact backward＋128 GiB） | FP8 | 4＋4 | 17 | 32.34 | 58,363 | 5.8% | — | 14.42 |
| Native FP8（view-base修复）³ | FP8 | 4＋4 | 17 | 33.82 | 57,103 | 5.2% | — | 14.33 |
| Native FP8（stride统一） | FP8 | 4＋4 | 17 | 32.23 | 61,268 | 4.1% | — | 14.00 |
| Native FP8（融合／缓存／MB1024）⁴ | FP8 | 4＋4＋2 reference | 3 | 71.67 | 135,439 | 8.6% | — | 待验证 |
| **Native FP8（同源码复现，step 4–6）⁴** | FP8 | **4＋4＋2 reference** | **3** | **64.76** | **136,825** | **9.0%** | — | 待验证 |
| Native FP8（同次复现，step 7–10）⁴ | FP8 | 4＋4＋2 reference | 4 | 69.18 | 108,421 | 23.8% | — | 待验证 |
| **Native FP8（judge 链路修复，step 4–6）⁴** | FP8 | **4＋4＋2 reference** | **3** | **67.86** | **137,792** | **8.6%** | — | 待验证 |

本轮FP8与4＋4对照统计步骤4–20；历史长run统计step≥4。总吞吐＝Σ completion token / Σ step秒，等待和MFU按计时加权。v0、v1和4＋4对照使用同一checkpoint、KL reference、数据及公共超参数；v0是FP32训练基线，使用SDPA＋PyTorch fallback。历史各行的checkpoint、batch和judge配置不同；4＋5主线另有1张评测卡。

¹ MFU仅包含4张训练卡，计入等待；BF16按989.5 TFLOP/s/卡、FP32按67 TFLOP/s/卡计算（[H200规格](https://www.nvidia.com/en-us/data-center/h200/)）。使用TRL通用FLOPs公式，未精确计入混合attention、共享前缀、reference与重计算，属于估算。不同精度的MFU百分比不能表示速度倍数。

² 首步训推差＝LR为0时，序列平均log-ratio绝对值的p90，单位为0.001；各组实际采样batch不同。
FP8完整混合精度MFU暂无，不能套用BF16峰值。本轮数据与源码哈希见[FP8证据](reports/FP8_SCALE_2026-10-05.json)；单卡固定G32仅投影＋head的GEMM估算：BF16／TF32 11.60%，修复后的FP8 5.08%，compact128 5.83%；计时包含BF16 reference，范围不同，不进入此表。
最新固定G32中，原生FP8投影＋head实测约1,282 TFLOP/s，为H200稠密FP8峰值的64.8%；这是GEMM内核利用率，包含checkpoint重算，不是整模型MFU。

³ 该轮judge上游失败占54.5%，源码与设置对齐，但实际奖励服务状态不同；速度仅作为系统观测。

⁴ 新配方真实microbatch为每rank 1024、GAS=1、global batch=4096，保留独立G32；旧4＋4对照global batch=1024。另用2张独立H200计算完整固定BF16 reference，共10张卡。三次run的主窗口预先固定step 4–6，10步复现另固定step 7–10；训推逐token一致性未通过；原10步复现及时judge verdict为3/228（1.3%），修复连接回退与取消后为151/154（98.1%），模型、prompt、RPM与90秒预算保持原设置。短窗口吞吐不能代表相同有效reward或学习质量。实验代码见[冻结源码包](reports/FP8_SPRINT_2026-10-06_sources.tar.gz)，运行基座为公开提交`2d9295e`；需要原checkpoint、数据、reward和对应运行环境，仓库主入口尚未集成全部实验配方。

改动简述：

- **v0→v1**：主参数和优化器保持FP32，训练计算改为BF16 autocast。
- **v2**：ranking reward从线性分数改为逆序数奖励，trainer主体相同。
- **v3**：TP rollout改为TP1多副本DP；每步固定1024样本，stale从1改为3，共享prompt前缀，暂关judge，并修复API server与连接超时。
- **v3.1系列**：恢复并发judge，增加judged staleness补偿与丢弃审计；调整队列和跨机卡数，v3.1e固定KL reference。
- **v3.2**：fast logprob、按token切子批、自适应activation checkpoint、rank负载均衡；v3.2b沿用同一infra继续训练。
- **Native FP8**：前反向GEMM使用FP8，更新与累加保持FP32；修复变长量化重复编译、fused AdamW缓存不刷新、vLLM AOT缓存dtype错配。
- **FP8早期优化**：Norm反向保存输入并重算中间值，减少activation checkpoint；统一backward布局，总吞吐提高14.1%。
- **FP8新配方**：融合conv＋SiLU、仅打包小维度Norm反向；减少native scheduler／KV bookkeeping和输出处理；缓存sampled-token文本并使用FlatLogprobs；提前在独立卡计算完整BF16 reference；真实microbatch增大到1024，使用整数token计数和丢弃审计。保持FP32主参数、梯度和Adam，以及BF16 attention KV。
- **Judge链路**：修复首个地址连接卡住时的取消与回退，每地址TCP／TLS预算3秒，保持整体89秒deadline；复用健康连接，及时判分恢复至98.1%。
- **未采用的修改**：FP8 attention KV完整cohort慢11.9%；两个独立G32合并投影＋MLP虽logprob逐值一致，却慢9.7%、显存113.8 GiB；MB2048也因rollout等待未达到2×。GDN与其他训推对齐候选的整模型gate仍未通过，本轮保留实验源码与证据。

可选原生logprob缓存已集成为`RLFORGE_SERVING_LOGPROBS_CACHE=1`（默认关闭，安装包后由vLLM general plugin加载）；58,262位置的原生字段、UTF-8及累计logprob逐值一致，main／spawn均通过。开启时需使用记录的vLLM源码版本；stream／top-K／echo等请求回到原生路径。

主入口支持`--num-generations 32 --completions-per-step 1024 --microbatch-per-rank 1024 --exact-token-counts`，即每卡真实microbatch 1024、GAS=1。指标通信由8–9次合成1次；四卡NCCL的64组对照最大差1.2×10⁻¹⁰，G32分组和样本字段保持一致。整数计数仅在显式开启时应用，要求记录的TRL源码版本；其余FP8实验配方仍见源码包。

同八卡对齐实测，v3.2总吞吐是v1的**27.2倍**；同4＋5卡历史记录，v3.1e→v3.2提高**1.72倍**。

可选decode调度与输出处理已集成为`RLFORGE_SERVING_DECODE=1`（默认关闭），保留原生边界fallback与依赖源码检查。11,776步原生状态对照、13种混合场景和main／spawn均通过；整理后的单卡完整生成比冻结实验链慢1.9%，尚未证明完整RL无回归，不替换上表实验配方。

本轮收束新增可选`RLFORGE_FP8_FORWARD=torch`（两端进程都需设置，默认`native`）：trainer与自身serving投影／head共用Torch原生FP8 GEMM，修复16-byte scale指针对齐，FP8输入和BF16输出保持不变；H200测试32 passed。同一单卡fixture、15层checkpoint、完整本地BF16 reference下，BF16／原生FP8／共享Torch FP8为2,908／1,760／1,794 ms，峰值显存86.59／53.23／53.23 GiB；共享前向额外耗时1.89%。fixture重放历史completion到共同prompt，原始配对不可得，仅代表计算对照。FP32 Adam另一次CUDA实测6.41 ms，约占该固定G32 policy compute的0.53%。

已用实际输入定位residual舍入、SwiGLU、主／gated RMSNorm和FA3 split规则差异；算子gate通过，整模型训推仍未通过。固定split候选的128 token logprob差p90：trainer–decode 0.1154、trainer–prefill 0.0863、prefill–decode 0.0835，门槛0.004；它尚未与最新pointwise候选完整组合。融合候选修复QK布局后，同一训练计算路径的整模型前向logprob逐值一致，计算快3.65%、显存75.76→61.27 GiB；梯度误差与原实现重复运行接近，但严格梯度gate失败，未集成。近似混合精度useful peak-equivalent为17.27%（单卡含本地reference），整模型／完整RL MFU仍未实测；不替换表中的MFU。详细数值、gate和源码哈希见[FP8证据](reports/FP8_SCALE_2026-10-05.json)。
