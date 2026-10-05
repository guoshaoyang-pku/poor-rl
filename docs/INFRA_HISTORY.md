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
| **Native FP8（stride统一，最终）** | FP8 | **4＋4** | **17** | **32.23** | **61,268** | **4.1%** | — | 14.00 |

本轮FP8与4＋4对照统计步骤4–20；历史长run统计step≥4。总吞吐＝Σ completion token / Σ step秒，等待和MFU按计时加权。v0、v1和4＋4对照使用同一checkpoint、KL reference、数据及公共超参数；v0是FP32训练基线，使用SDPA＋PyTorch fallback。历史各行的checkpoint、batch和judge配置不同；4＋5主线另有1张评测卡。

¹ MFU仅包含4张训练卡，计入等待；BF16按989.5 TFLOP/s/卡、FP32按67 TFLOP/s/卡计算（[H200规格](https://www.nvidia.com/en-us/data-center/h200/)）。使用TRL通用FLOPs公式，未精确计入混合attention、共享前缀、reference与重计算，属于估算。不同精度的MFU百分比不能表示速度倍数。

² 首步训推差＝LR为0时，序列平均log-ratio绝对值的p90，单位为0.001；各组实际采样batch不同。
FP8完整混合精度MFU暂无，不能套用BF16峰值。本轮数据与源码哈希见[FP8证据](reports/FP8_SCALE_2026-10-05.json)；单卡固定G32仅投影＋head的GEMM估算：BF16／TF32 11.60%，修复后的FP8 5.08%，compact128 5.83%；计时包含BF16 reference，范围不同，不进入此表。

³ 该轮judge上游失败占54.5%，源码与设置对齐，但实际奖励服务状态不同；速度仅作为系统观测。

改动简述：

- **v0→v1**：主参数和优化器保持FP32，训练计算改为BF16 autocast。
- **v2**：ranking reward从线性分数改为逆序数奖励，trainer主体相同。
- **v3**：TP rollout改为TP1多副本DP；每步固定1024样本，stale从1改为3，共享prompt前缀，暂关judge，并修复API server与连接超时。
- **v3.1系列**：恢复并发judge，增加judged staleness补偿与丢弃审计；调整队列和跨机卡数，v3.1e固定KL reference。
- **v3.2**：fast logprob、按token切子批、自适应activation checkpoint、rank负载均衡；v3.2b沿用同一infra继续训练。
- **Native FP8**：前反向GEMM使用FP8，更新与累加保持FP32；修复变长量化重复编译、fused AdamW缓存不刷新、vLLM AOT缓存dtype错配。
- **FP8优化**：Norm反向保存输入并重算中间值，减少activation checkpoint；统一backward布局，整轮无编译回退，总吞吐比历史最强提高14.1%。G32→G64计算时间约翻倍，batch翻倍尚无收益；训推差未过0.004 gate，配置仍为实验性。

同八卡对齐实测，v3.2总吞吐是v1的**27.2倍**；同4＋5卡历史记录，v3.1e→v3.2提高**1.72倍**。
