# suika_dqn Experiment Log

Engine: suika/part2 + rules.two_watermelon=merge_disappear_score (git f46d14e).
Code fingerprint (training .py set): sha256:abc14e8b1e1d, committed as root-repo f46d14e.

## wave1_20260928 — 16-arm recipe sweep (QR-DQN Ape-X)

- 2026-09-28 00:22 launch. Nodes: node_d (arms on gpu0-7), node_c (gpu0-7).
- Layout: 1 learner + 20 actors + 1 evaluator per arm; eval seeds 0:16 greedy q5min.
- Anchor: K=128, obs333 (boundary feats), MLP[1024,1024] dueling QR64, double DQN,
  n-step3, gamma0.995, PER(0.6), batch8192, lr3e-4 warmup2k cosine, ema0.003,
  mirror_aug, eps 0.02-0.5 geometric, max_reuse16, grad_budget120k.
- node_d arms: anchor_g0, bigbatch_g1(16k,lr4e-4), lrhi_g2(1e-3), bigmodel_g3(2048^2),
  gammahi_g4(0.999), nstep5_g5(n5,g0.997), noper_g6(alpha0), plaindqn_g7(nq1).
- node_c arms: anchor_s1_g0(seed1), k64_g1, k256_g2, epslow_g3(0.005-0.15),
  lrlo_g4(1e-4), nstep1_g5, small_g6(512^2,b4k), reuse32_g7.
- Prechecks: nvidia-smi 8xA100 idle both nodes; torch 2.6.0+cu124 cuda=True;
  env smoke obs_dim=333 both nodes.
- Launch health (t+3min): all 16 arms training, no tracebacks;
  env_sps ~1.1-1.3k/arm, combined ~19k env steps/s; reuse32 grad_norm ~260 (watch).
- Metrics root (local aggregate): dqn_runs/<node>/<arm>/{metrics,eval,episodes}.jsonl

## wave1 FINAL RESULTS (killed 2026-09-28 09:30 after ~9h, 30-51M env steps/arm)

MILESTONE: plaindqn_g7 and gammahi_g4 both REACHED WATERMELON (maxfruit=10).
plaindqn_g7 eval max 2354 > historical engine record 1844 (beam teacher, old rule).

Final ranking (eval mean @ last, 16-seed greedy):
  1. plaindqn_g7  1315  (max 2354, p2000=0.062, fruit=10)  <- plain dueling DQN, n_quant=1
  2. gammahi_g4   1230  (max 2251, p2000=0.062, fruit=10)  <- gamma 0.999
  3. noper_g6     1152  (PER off, surprisingly strong)
  4. small_g6     1124  (512^2 b4k, hit 120k grad budget)
  5-16. others 695-1130; lrhi_g2 (lr 1e-3) collapsed late 858->742 (avoid high LR).

Lessons: plain Dueling DQN > QR-DQN64 (winner, also faster wall-clock);
gamma 0.999 > 0.995 (matches ~500-1000 drop episode length); K 64-256 no big diff;
bigbatch fine after transient; PER non-decisive; small model competitive per-grad.

## wave2_20260928 — SCALE phase (launched 09:44)

Recipe: winner = plain dueling DQN nq1, gamma 0.999, K128, 1024^2, b8192, lr3e-4,
n3, PER0.6, eps 0.02-0.5, mirror_aug, replay 4M, budget 600k grads.
Scale: 2 arms/node x 100 actors = ~5.1-5.8k env steps/s/arm (4.75x wave1),
~23k env/s combined, reuse-capped at 16 (~10-11 grad/s/arm).
Arms:
  node_d gpu0 w2_warm_g999     warm-start from plaindqn_g7 step98k/env50.2M ckpt
  node_d gpu1 w2_fresh_g999    fresh control
  node_c   gpu0 w2_fresh_g999_b16k  batch16384 lr4e-4
  node_c   gpu1 w2_fresh_g9995   gamma 0.9995
Launch health (t+4min): all 4 arms training, no tracebacks, warm arm q_mean=1092
(inherited), fresh arms q_mean~80-150.

wave2 snapshot @ t+5.8h (~140M env steps/arm, ~7k env/s/arm measured):
  w2_warm_g999       eval mean 1499 / p25 1278 / max 2369
  w2_fresh_g999      eval mean 1468 / p25 1276 / max 2204
  w2_fresh_g999_b16k eval mean 1388 / p25 1200 / max 1966
  w2_fresh_g9995     eval mean 1377 / p25 1166 / max 2048
All 4 arms > wave1 champion (1315). Curves: dqn_runs/wave2_curves.png
(suika_dqn/plot_wave2.py; local monitor loop re-pulls + re-plots every 30min).
Reading: all arms ramp to ~1200-1300 within ~10M steps; warm arm starts at
champion level and keeps a small edge early; after ~60M the four interleave,
eval mean plateau-ish 1400-1500 with ±100-150 eval-to-eval noise (32 seeds);
train rolling score still creeping 1000->1200. NOTE: all wave2 runs use the
buggy n-step successor (D10); absolute numbers pending revalidation.

wave2 TERMINATED ~15:55 (t+6.5h, ~150M env steps/arm) to free the fleet for
wave3 (user call: start the fixed-stack experiments now).

## wave3_20260928 — ARCH V2 phase (launched ~16:10)

Stack: n-step successor FIX everywhere + gamma=1 (D4) + arch v2 (D7/D11),
fresh start, eval seeds 0:16 greedy q5min, 4 arms x 100 actors:
  node_d gpu0 w3_mlp_deep   MLP[2048,2048,2048]+head1024 (13.42M), settle
  node_d gpu1 w3_tf_base    set-transformer v2 5.05M (SAB2->PMA32x256->SAB4)
  node_c   gpu0 w3_tf_deep    tf_base but latent SAB x8 (8.21M)
  node_c   gpu1 w3_tf_tempo   tf_base + tempo env (120fr + 3s full-cross)
Infra notes:
- TF arms: token obs (T=160 x 5 feats = 800, incl velocities; engine get_state
  now exports vx,vy) + per-arm GPU inference server (zmq ROUTER, batch cap 64,
  policy hot-reload) colocated on the learner GPU; evaluator moved to GPU.
- replay 2M for TF arms (token obs 2.4x larger), 4M for MLP arm.
- Launch incident: python3 resolved to system python (no numpy/torch/zmq);
  relaunched with suika-venv on PATH. Also noted: pkill -f matches the ssh
  command's own shell when the literal run path appears in the same command
  (self-kill, exit 255) — kill and relaunch must be separate ssh calls with
  an obfuscated pattern.
- pyzmq installed into both remote venvs (inference server transport).
- OOM incident: TF arms at batch 8192 blew 40G cards (tf_deep SAB8 used ~37GB
  in Stage-1 activations on 160 tokens); relaunched all TF arms at batch 2048
  (lr unchanged 3e-4; reuse ~5-8x at ~10 grad/s) + expandable_segments.
  MLP arm kept batch 8192 and ran through uninterrupted.
- w3_tf_tempo CANCELLED by user (~16:35, before producing results); t1 gpu1
  left free. Tempo-env training question (D2) deferred again.
Fleet after cleanup (3 arms):
  node_d gpu0 w3_mlp_deep  ~5.4k env/s, replay full
  node_d gpu1 w3_tf_base   ~2.6k env/s, infer ~2.4k req/s, gpu 11GB/100%
  node_c   gpu0 w3_tf_deep   ~3.5k env/s, infer ~3.0k req/s, gpu 13GB/87%
  node_c   gpu1 FREE

~17:05 two infra upgrades landed on all 3 arms (full restart, fresh dirs;
old dirs kept as *_slowjunk/*_1gpu_junk):
1. TF arms got a 2nd inference GPU (run_arm infers per cfg infer_gpus;
   actors shard round-robin): tf_base -> perm gpu1+gpu2, tf_deep -> t1 gpu0+gpu1.
2. Bit-exact fast _settle (suika a502524, 290->620 drops/s/core) synced.
Post-restart throughput (~4 min in):
  w3_mlp_deep 5.7k env/s (+5% — was inference/encode-gated, not physics)
  w3_tf_base  2.8k env/s (flat — infer round-trip gated on shared gpu1)
  w3_tf_deep  5.3k env/s (+30% — 2nd infer GPU + fast physics both helped)
Conclusion: physics stopped being the binding constraint; per-step inference
latency is now the limiter on all arms. Next lever = the async rollout +
dynamic-batch design in ASYNC_BATCH_HANDOFF_20260928.md.

wave3 1-hour snapshot (~17:55, arms restarted 17:05 with fast physics):
  w3_tf_deep   20.5M steps (~6.8k/s avg!) eval mean 1404 / p25 1075 / max 2057
  w3_mlp_deep  23.3M steps                 eval mean 1105 / p25  907 / max 1752
  w3_tf_base   11.9M steps (~3.9k/s)       eval mean  616 / p25  456 / max 1118
Reading: tf_deep at 20.5M already beats wave1 champion's final score (1315 @
51M, buggy n-step) and is ahead of wave2's @20M trajectory (~1200-1300).
tf_deep > mlp_deep at comparable steps — early support for set-transformer +
deeper Stage 3. tf_base took off late (29 -> 616 in one eval interval) and
has half the steps (inference-gated until the 2nd GPU). Caveat: 1h is early,
eval noise ±100-150; separation verdict needs several more hours.

## wave3b_20260928 — C physics + resume + batch 64k (launched ~19:20, 3rd attempt)

User directives (18:02): kill w3_tf_base; load C-settle engine (suika
ee14bf1/152129d, 552 drops/s/core, bit-exact, built on both nodes); resume
from ckpt instead of fresh start; ckpt every ~15min (900s, keep 4); scale
actors 100->200/arm, one arm per node; more inference GPUs; batch 65536
with lr raised 3e-4 -> 6e-4 (user: fewer off-policy steps).
Arms:
  node_d w3b_mlp_deep  learner gpu0, infer gpu1+2, resume w3_mlp_deep@24k
  node_c   w3b_tf_deep   learner gpu0, infer gpu0-3, resume w3_tf_deep@10k,
                          grad_accum 32 x 2048 (64k won't fit SAB8 on 40G)
Two resume bugs found and fixed (learner.py):
  1. replay is not persisted; resumed grad_steps with inserts=0 tripped the
     reuse gate -> learner slept forever at 0% CPU. Fix: replay.inserts =
     env_steps on resume.
  2. gate was step-based (grad_steps < inserts*reuse/batch); batch grew 8x
     at resume so old steps blocked 98M/41M fresh inserts. Fix: reuse
     accounting in sample-DRAWS (grad_draws += batch; ckpt saves
     grad_draws+batch; old ckpts restore draws = steps*8192).
Also found 100 stale w3_tf_base actors + infer servers running on perm since
16:51 (survived earlier kills), sharing gpu1/2 with w3b; killed.
Post-fix metrics (t+2.5min): mlp 27k env/s, +574 grad steps, loss 1.43,
q 458, lr 6e-4; tf_deep 22k env/s, first steps loss 17.6 (fresh-replay PER
turbulence, watch). C physics + 200 actors ~2.4x previous throughput.
Disk: perm 6% (418G/7T), t1 4% (7G/200G); ckpt keep=4 bounds growth.

wave3b 1-hour snapshot (~20:20): tf_deep eval mean 1957 @ 66M steps (peak
2020 @ ~48M), p25 1734, max 3165 — crossed the 1844 beam-teacher line on
multiple evals and clearly leads mlp_deep (1489 @ 110M). Note: the 19:20
stuck-launch window produced ~25min of RANDOM-WEIGHT rollouts (no policy.pt
in the fresh run dir, infer servers served the init net); plot_wave2.py now
dedupes relaunched episode streams by (actor,episode) to keep the training
curve honest. Judge by eval, not by that dip.

## w4_20260929 — Qwen3.5-0.8B 直接 RL 文本臂（D13，深夜启动）

用户指令（00:19）：继续 w4；避开跑 tf_deep 的 node_c（只用 node_d）。
设计冻结见 living doc D13 + 格式规范 v1；代码 = 昨晚写好的 qwen 全家桶
（model_qwen / learner_qwen / inference_server_qwen / evaluator_qwen /
run_arm_qwen / qwen_text）。

Placement（perm 实况）：w3b_mlp_deep 占 gpu0-2，egodex 训练占 gpu4-5，
w4 = 2-rank DDP learner (gpu3,6) + 1 infer GPU (gpu7, 与 evaluator 共卡)。
batch 1024（÷2 rank），LoRA r64 全线性层 + V/A/π 头共训（trainable 27.3M /
779.7M），lr 3e-4 头 / 1e-4 LoRA 双组，fp32 权重 + bf16 autocast，梯度检查点。

- **attempt 1（00:33）**：server 未预热直接上线 → 首批真实请求触发逐形状
  Triton autotune（每个新 seq 形状 5-50s 停顿）→ 演员端 10s zmq 超时全灭
  （40/40 RuntimeError，22 局后停摆）。infer 本身健康（预热后 batch64×249tok
  = 394ms ≈ 160 obs/s）。修复：① prompt 长度按 128 桶化（模型只见 7 种形状）；
  ② server/evaluator 绑定前跑全形状 warmup（model_qwen.warmup）；③
  run_arm_qwen 加演员看门狗（死亡可见、全灭即关臂）。
- **attempt 2（01:13）**：推理侧健康（97 req/s 稳定，40 演员，~160 env/s），
  但 learner 第一个 train step OOM：micro 128 × 896 桶 → 激活 ~36G + fp32
  online/target 6.4G 超 40G。修复：micro_bs 128→32（预计峰值 ~17G）。
  causal_conv1d 装轮子失败实录：v1.5 只到 torch2.1；v1.6.2/v1.7 的
  "torch2.6" 轮子实际按旧 c10_cuda_check_implementation 签名编译，与本机
  torch 2.6.0+cu124（新签名带 CUDAErrorLogCapture*）不匹配，torch2.7 轮子
  同样旧签名——放弃，用 fla 的 reference conv（~1ms 级，非瓶颈）。
- **attempt 3（01:38）**：micro 32 依然 OOM（37.06G allocated，同样在
  online pass 第一 chunk）。单进程复现（debug_qwen_mem.py）：target pass
  无梯度峰值仅 8.3G 且逐 chunk 无泄漏；带梯度第一个 chunk 即涨到 ~33G。
  debug_qwen_ckpt.py 三配置探针（ckpt-before-peft / peft-first / ssm bf16）
  全部 23.7G 峰值 @micro16×512——**HF 梯度检查点对 Qwen3.5 混合 GDN 架构
  无降内存效果**（fla chunked GDN 的逐 chunk 状态不被 checkpoint 包住），
  且 ckpt-before-peft 白慢 13 倍（6.6s vs 0.5s/iter）。实测激活
  ~2.55MB/(sample·token)。修复：micro_bs 8（896 桶最坏 24.5G，典型 384 桶
  ~14G）。架构要点：24 层 = 18 线性注意力（GDN，16 头×128×128，config
  mamba_ssm_dtype=fp32）+ 6 全注意力，hidden 1024，vocab 248k，嵌入占
  254M/780M。
- **attempt 4（02:2x）**：micro 8 训练稳定通过：首批 metrics loss 9.37 /
  q_mean 0.30 / grad_norm 48→22 / step_s ~38s（首个 60s 含 JIT）≈
  **0.026 grad/s ≈ 2.7M views/天**；采集 86 env/s；ckpt keep-N 正常。

### w4 中期读数 + BC 暖启转向（09-29 上午）

- **9.5h 读数**：direct-RL 确认在学（eval 724→895→784，median 826，
  max 1521；q_mean 0.3→50；publish→infer 热重载 3 次平滑）。等 env 步
  对齐 wave2-fresh 比较：w4 @3.1M 步用 0.72M 梯度视图 ≈ fresh-MLP
  @2M 步用 235M 视图的分数水平——**早期每视图效率 ~300×**（需注明：
  早期坡道效应 + MLP 65k 大 batch 的视图口径贡献了大半）。
- **w3b_mlp_deep 已到 eval mean 2033 / p2000=56%**（1.41B 步，
  26.7k env/s，预算 24.5%）；tf_xl（t1）1925。用户判定 w4 提升不够，
  要求对齐最强 MLP → **BC 暖启转向**（direct-RL 臂保留作对照）。
- **recipe 对齐审计（w4 vs w3b_mlp_deep）**：对齐项 = 同 env/K/γ/n-step(3
  修复版)/double/PER(α.6,β.4)/mirror/eps 阶梯/评测协议/ema_tau。不对齐
  项 = ① batch 1024 vs 65536；② **有效复用率 0.23× vs 6.8×**（w4 每条
  经验平均只被训练 0.23 次——大部分采集数据从未进梯度！mlp_deep 6.8 次）；
  ③ 暖启动链（w3 系继承 wave2，w4 冷启）。②是重要发现：w4 的
  replay_capacity 2M @114 env/s = 5 小时窗口 vs mlp 4M @26.7k/s = 2.5 分钟
  窗口。RL 精修阶段应 replay ~300k + max_reuse 16 对齐。
- **数据存档现实**：learner 消费 inbox shard 后即删（learner.py:318）——
  昨晚的转移数据**没有存档**（只有 episode 摘要）；且 mlp 系 shard 是
  flat-333 格式，w4 文本接口用不了。补救 = 定向采集：`bc_collector.py`
  （教师 = mlp_deep policy.pt 冻结副本 @2033，DualEnv 双格式观测，
  tokens-800 入库 + flat-333 喂教师，逐状态记教师完整 Q128（f16）+
  argmax 标志 + n-step 语义与 actor.py 完全一致）。48 actors × 600k =
  28.8M 目标，~2k transitions/s，09:50 起跑，冒烟验证教师局分
  1935/2564、argmax 占比 90.7%、Q 范围 [-86, 2060] 正常。
- **bc_learner_qwen.py**（已冒烟通过：l_pi 起始 4.83=ln128 精确吻合、
  val/publish/ckpt 全通）：L = CE(π头, a_T)【仅 expert 样本】+
  smooth_l1(Q(s,·), q_T(s,·))【128 维全向量蒸馏】+ 0.5×smooth_l1 TD
  【double + EMA target】；DDP find_unused=False + no_sync 逐 chunk 累积；
  model_qwen.forward 加 return_pi 合并前向（否则 π 头梯度绕过 DDP 同步，
  各 rank 权重会发散——本次自查抓到的关键 bug）。
- **今晚计划（t1，tf_xl 预计 ~01:50 释放 8 卡）**：rsync ~12M 转移
  （~80GB，t1 根盘 193G 限制）→ torchrun 8 ranks，batch 1024/micro 8，
  budget 12k grads（估 5-8s/步 ≈ 17-27h），val 每 200 步监控 agree_pi。
  预注册 BC 门：agree_pi ≥ 85% 且 BC 后 eval ≥ 1500（teacher 2033 的
  ~75%）；达标后 w4-RL 从 BC ckpt 暖启（treatment）vs 现役 direct-RL
  （control），RL 精修配方按上面②对齐。

吞吐画像（实测）：文本序列化 50ms/64 态；bf16 fwd batch64×249tok=394ms；
单推理卡 ~120-160 obs/s；40 演员 ≈ 100-160 env/s ≈ 10M+ env 步/天。
训练侧预估 ~0.05-0.1 grad/s（batch 1024）≈ 4.5-9M views/天——比 MLP 臂低
~3 个数量级，这就是文本接口的价钱；w4 的成败判据按"每视图学习效率"
（预注册 go/no-go：2e7 views eval≥850；1e8 views ≥1100；3e8 ≥1300）。

### w4 BC 诊断与 v2 软目标修正（09-29 15:2x–16:1x）

对 t1_2 `bc_data/inbox` 抽 200 shards（817k 转移 / 744k 专家状态）离线统计，
脚本 `diag_teacher_marginal.py` / `diag_teacher_gap.py` / `diag_teacher_soft.py`：

- **π 头在 run1-3 里只学到了常数边际分布**：专家样本（91.1%）动作边际熵
  H=3.787 nats（均匀 4.852；best-constant acc 0.0746；effective classes 44.1）。
  实测 val l_pi 3.734(step200)/3.797(step400) == 边际熵，agree_pi 0.092/0.098
  ≈ best-constant —— π 输出与状态几乎无关，不是"接近最优的锐化"，是没学到结构。
- **根因：教师判据比任何可得的回归精度都细**。top1−top2 gap 中位 1.0 / p75 3.0
  / p90 7.0 Q 单位，而 Q~1096、逐状态 std 中位仅 12；qteach 存 f16（该量级
  ULP=1.0），gap<1 的 37% 状态在存储层就是 near-tie。val_qd ~97 → 硬 argmax
  标签不可达（agree_q 0.001→0.011 同证）。**这类教师必须用软目标蒸馏，硬 CE
  在结构上就停在地板。**
- 排除错位：同一前向的 Q 蒸馏在学（l_qd 401→73，val_qd 437→97，l_td 209→134）
  —— token 观测↔qteach 对齐正确；若 obs/act 错位，Q 也会同样学不动。
- **τ 扫描**（选型依据）：目标分布 softmax((q_T−mean_a q_T)/τ) 的熵 —— τ=0.5→0.93
  / 1.0→1.59（top5 质量 0.80）/ 2.0→2.34 / 4.0→3.12 / 8.0→3.83（≈边际，无信号）。
  取 **τ=1.0**。
- **regret 标尺**：regret(random action)=34.3、regret(恒选全局最优列)=30.4，未训练
  网络 val q_regret=29.96 —— 三者同量级，"常数策略"的 regret ≈ 30，故门定
  **q_regret < 10**。

修正（`bc_learner_qwen.py`，config 门控、向后兼容）：`bc_pi_tau>0` 时
L_pi = CE(softmax((q_T−mean_a q_T)/τ), log_softmax(π))，**覆盖全部样本**（教师 Q
每个转移都有，不再限于 91% 的 argmax 样本）；train 日志加 l_pi_hard 对照；val 新增
soft_ce 与其 floor（目标熵）；新增 `--grad-budget` 覆盖（冒烟用）。

运行谱系（同一 run dir，日志按次归档）：
- run1 12:12 起（step200 rank7 SIGABRT → `bc_crash_step200.log`）；run2 15:25 从
  step200 恢复（5.7s/step，step400 val 见上）；run3 15:54 起（同 v1 recipe，ckpt
  瘦身 110MB trainable-only + NCCL 60min 看门狗）。
- **16:0x 主动停 run3，切 v2**（归档 `bc_run2.log` / `bc_run3_pre_v2.log`）：冒烟
  （8 rank × 2 步 × 300k 子集）确认 `pi_tau=1.0` 生效、soft_ce=4.614 / floor=1.625
  （与离线 1.588 吻合）、q_regret=29.96 ≈ 常数基线。
- v2 正式：t1_2 8 卡 `runs/w4bc_20260929/bc_qwen_v2`，`--resume-from step400.pt`
  温启（换 loss 不动参数），launcher `run_bc_v2.sh`，config `w4_bc_qwen_v2.yaml`，
  ~5.7s/step → 11600 步 ≈ 18h。
- **门（修正后）**：q_regret < 10（常数策略 30.4）且 BC 后 eval ≥ 1500；agree_pi /
  agree_q 仅作参考 —— 其上限由教师 margin 与 f16 ULP 决定，near-tie 本质不可分。

**v2.0 首段读数（step 400→600，软 π KD 版）**：val_qd 97.1→53.4（Q 的**水平**在
快速收敛）、l_td 134→69，但 **q_regret 29.96→28.76 ≈ 常数策略基线 30.4 没动**，
soft_ce 4.08（floor 1.61）也没动；**eval（16 种子，evaluator_qwen，argmax-Q）
mean 523.9 / median 493 / max 952 —— 落在随机策略带（物理消融随机策略 475-653）**。
推断：raw-Q 的 smooth_l1 梯度 ~80% 花在"状态水平"（每状态 contrast 只有 ~12，
而 qd 53），**决定 argmax 的动作排序没有专门梯度**（π 头同理被水平项挤占）。

**v2.1（step 600 起）**：新增 Q 向量软排序蒸馏
`bc_qrank_tau: 1.0`，L_qrank = CE(softmax((q_T−mean)/τ), log_softmax(z/τ)) ——
softmax 对状态水平不变，故只监督 contrast；qd/td 继续锚定数值 scale（RL 暖启需要）。
实现：`l_pi_hard/l_qrank` 进 train 日志与 metrics；val 增 `val_qrank`（排序保真度）。
冒烟（2 rank × 2 步）×2 次均通过；正式 run 从 step600.pt（trainable-only，110MB，
strict=False 加载路径验证）温启，run dir 不变，日志归档
`bc_v2.0_steps400-600.log`。评测器（evaluator_qwen，gpu0 与训练共卡）常驻，
每 200 步 publish 后自动补一档 16 种子分数。

## w4 BC 深挖：目标函数的"水平陷阱"与 v3 配方（09-29 17:0x–18:1x）

v2.1（step 600→1050）读数停滞——val_qd 53→43（只是**水平**在降）而 q_regret
28.8→28.7 ≈ 常数策略基线 30.4，eval 546.8 仍在随机带；新增的 qrank 项也没拉动
排名。连续三个子问题用三个探针脚本一次性问清（全部离线、CPU 可跑，不占训练卡）：

- **`probe_bc_features.py`（模型自身特征的最优线性读出）**：取 step800 policy.pt
  的 trunk 隐层（heads 的输入），闭式 ridge 读出——train 32.8k / val 8.2k 状态。
  结果：**heads 29.05 / 隐层最优线性读出 23.45 / 原始 800 维盘面输入 ridge 22.53
  / 2048 维随机 ReLU 特征 22.34**（regret 参照：常数策略 31.1、门 <10）。结论：
  ① 模型 heads 不如一个线性探针；② **隐层不比原始输入多携带任何排序信息**——
  主干学到的是"状态标量"（Q 水平），不是动作排序（每状态 contrast 仅 ~12，
  top1−top2 间隔中位 1.0）。
- **`audit_mirror_headroom.py`（镜像增强审计）**：mirror(unsorted)→orig 迁移
  regret 23.39（参照 orig→orig 22.53），且在**自己镜像过的 val** 上 22.25 自洽
  ——**当前 `_mirror` 没有 bug**（按 (y,x) 重排序的 mirror 反而更差 26.05：现有
  约定是自洽的，别动）。
- **决定性测量（`bc_audit_split.npz` 直接算）**：居中目标的**全局 RMS = 37.1**
  （每状态 std 仅 6.4/12.4/21.8——异方差），因此任何**固定除数**下的点式回归都是
  "量级主导"：零预测器 smooth_l1 = 0.962，完美形状 = 0.937，**排序只值 2.6% 的
  损失**。这解释了为什么 LM（adv 0.94）与一个普通 3 层 MLP（adv 0.937）**停在
  同一个平台**——点式项的梯度只是在缩小 contrast，反而与 listwise 项打架。
- **listwise 项余量巨大**（这才是排序的正确载体）：π 头 KD 的 CE 下界 = 目标熵
  1.59，均匀学生 ≈ 4.85 → **3.3 nats 余量**；qrank（Q 头 log_softmax）同理。
- **v3.1 配方**（arch/learner 全部与 v2 一致，只改损失，`configs/w4_bc_qwen_v3.yaml`）：
  λ_pi=1.0（KD，τ=1）+ λ_qrank=0.5（listwise，τ=1，走 Q 头）+ λ_v=1.0（水平
  /1000）；**λ_qd=λ_td=λ_adv=0**（点式项关闭；代码里 L_adv 也已改为"按状态
  标准化"以免复用踩坑）。副产物：跳过无用的 TD target 前向 → **5.8→3.3 s/step
  （1.8×），12000 步 ≈ 11h**。
- **v3 已见效**（step 1200 vs 800 的 val 对比）：soft_ce 4.08→3.77（floor 1.59）、
  **agree_pi 0.09→0.16**、val_qrank 6.37→4.75、q_regret 28.7→27.8；train 侧
  pi 4.06→3.58、pi_hard 3.79→3.31（硬 argmax CE 也在降）；eval（16 seeds）
  gs600 523.9 → gs800 546.8 → **gs1200 572.1**（随机带 500-650，方向正确）。
- **MLP 参照（本机 MPS，`probe_mlp_ceiling.py`）**：同样的 v3 目标、原始 800 维
  输入、3 层 MLP（1.5M 参数），30 epoch（0.5M 样本）→ **q_regret 17.9 / agree
  0.151**（常数 31.1、RF 22.3）。→ 排序确实可从盘面学到，但**远不是几个 epoch
  能到 "regret<10" 的难度**；门的定义需要按这个量级重新校准（或以 eval 分数
  为主判据）。长跑版（400 epoch、λ_adv=0）在本机继续，作 LM 的上界参照。
- 运行谱系（v3 目录）：v3.0（step1000→1250，λ_adv=1 固定除数）归档
  `bc_v3.0_steps1000-1250.log`；**v3.1 = 同目录，从 step1000.pt 温启**（丢弃
  v3.0 段，避免带进坏目标的动量），launcher `run_bc_v3.sh`（已修 resume 优先
  本目录 → 次选 v2 链）。evaluator_qwen 常驻 gpu0 随 publish 自动补分。

### w4 BC v3.1 收官 + 22:17 外部清场事件（09-29 22:3x 核对）

**v3.1 结果（step 1000 → 5200，t1_2 8 卡，3.3 s/step）**：

| 指标 | v2 末段(1000) | v3.1 @5200 | 预注册门 |
|---|---|---|---|
| q_regret | 28.7（≈常数基线 30.4） | **7.75** | <10 ✅ |
| agree_pi | 0.09 | **0.363** | 参考 |
| agree_q / wellsep | 0.011 / 0.062 | 0.239 / 0.297 | 参考 |
| val_soft_ce（floor 1.64） | 4.08 | 2.87 | — |
| val_qrank | 6.37 | 3.97 | — |
| eval 16 种子 | 523.9（随机带） | **1257–1698**（8 档 ≥1500，p2000 0→0.375） | ≥1500 ✅ |

eval 轨迹（gs1200→5000）：572 / 848 / 1379 / 1374 / 1472 / 1322 / 1432 / 1577 /
1426 / 1686 / 1257 / 1698 / 1433 / 1403 / 1682 / 1566——中位数稳在 1500+。
训练侧 loss 4.95 / pi 2.88 / pi_hard 2.54 / qrank 4.13 / v 0.0048（λ_adv=0）。
→ 上面 336 行"门需要按 MLP 量级重校准"的当时判断**被推翻**：LM+文本接口确实
破了 <10 门（MLP 只到 18.45）。

**结论**：① "水平陷阱"是根因——λ 重配（纯 listwise）后同一模型从随机带直接进入
1500-1700；② MLP 参照（本机 MPS，同数据同目标、原始 800 维输入、1.5M 参数、
400 epoch/3.2M 样本）只到 regret 18.45 / agree 0.127 → **LM+文本接口把同一份
数据用得更好**；③ 距教师（w3b_mlp_deep 末读数 eval 2078.6 @1.5B 步）还差
400-700 分，BC 的定位是"给 RL 一个 1500 起步的 warm start"。

**22:17 事件（外部清场，非本会话）**：t1_2 的 BC v3.1 与 perm 上**全部**训练
（w3b_mlp_deep / w4 direct-RL 等）在 22:17-22:25 窗口被清理；两台容器 uptime
26 天未重启、无进程残留、无替换痕迹。关联证据：perm 22:23 出现 `sft_qwen.py`
（docstring："User spec 2026-09-29, warm start = BC ckpt, budget ~1 h, 后面完全
用 PPO"）、22:26 `run_sft.sh` 与 `runs/w4sft_smoke/`，脚本目标 = **t1_2 8 卡**
→ 判断为 SFT 部署方的资源重整。

本会话处置：22:36 曾误从 step1000 resume（launcher resume 优先级 bug，已修：
优先本目录最新 ckpt）；22:42 发现 SFT 就位后**主动停机让路**。现状：t1_2 已腾空、
`runs/w4bc_20260929/bc_qwen_v3/checkpoints/step5200.pt` 完好（109MB
trainable-only，SFT warm start 即用此文件）。BC 续跑命令：
`RESUME=<step5200.pt> bash run_bc_v3.sh`（12000 步预算剩 6800 步 ≈ 6h）。

### 两条出招路径分岔：Q 头 1540 vs token 生成 500（09-29 23:2x）

回答"1500 分到底学到了什么"：用**同一评测器**（`eval_qwen_policy.py`）、**同一
ckpt**（`bc_qwen_v3/policy.pt` @step5200，全量 3.1GB）跑两条出招路径：

| decode | mean | median | p25 | max | p2000 | moves_mean |
|---|---|---|---|---|---|---|
| `qhead`（Q 头 argmax） | **1540.2** | 1653.5 | 909.5 | 2846 | 0.125 | 160.0 |
| `tokens`（LM 自回归 3 步） | **499.7** | 507.0 | 393.8 | 922 | 0.0 | 79.8 |

token 路径 499.7 = **随机带**（物理消融随机策略 475-653）。且与训练期
`evaluator_qwen` 的 Q 头读数（gs5000: 1566.3）一致 → 评测器无偏差。

**根因（重要）**：BC 的损失只作用在**三个外挂头**上（π 头 128 维 KD、Q 头
listwise、V 头水平/1000），**从未监督 trunk 的 LM 输出 logits**。π 头 ≠ LM
token 分布：前者是外挂 softmax、一次前向出 128 维；后者要 3 步自回归
（d1→d2→d3）才能得到一个动作串。因此 LM head 仍是预训练原样，没学过输出
`"0.461"` 这类答案——**这正是 SFT 阶段 1 存在的理由（D18）**：它是第一份把
教师 Q 的归一化分布压到 digit token 上的监督，把 policy 从"外挂 π 头"换成
"LM 自己的 token 概率"。

**结论**：v3.1 的 1500 分是**纯离线蒸馏**得来的，且只存在于 Q 头 argmax 这条
出招路径；RL 一步都没跑（λ_td=λ_qd=λ_adv=0，无自举、无环境交互）。两条腿的
下一步：Q 头路径可直接进 DQN 式 RL（暖启 step5200）；token 路径必须先 SFT 再
PPO。

**两条工具备注**（踩坑）：① `--decode both` 的 jsonl **只记 token 路径**统计
（`_stats` 只吃 `scores`；`q_scores` 不外传），要 Q 头分数得单独 `--decode qhead`；
② 评测器用 `strict=True` 加载，**吃不了 trainable-only ckpt**（109MB，只有
LoRA+heads）→ 必须用 publish 出的全量 `policy.pt`（3.1GB）。

**被清臂的续跑代价**：perm 全部臂的 replay/EMA 未持久化，续跑 = 从 ckpt 重启 +
replay 空窗抖动（D14 的 resume 记账修复已覆盖该场景）。

## phy_ablation_20260928 — 物理结算消融（15:50–17:00）

问题：settle 60fps/240帧预算下，能否在"末态一致+数值稳定"前提下加速？
工具：`suika_dqn/phy_ablation.py`（固定种子+固定动作脚本跨变体重放；对比吞吐、
分数/步数分布、逐投放偏差、穿透/穿墙/NaN；结果 `phy_ablation_results.json`）。

先修测量再谈消融——profile 发现每帧 ~25µs 中 **C 求解器只占 7.2µs（29%）**，
其余 16.7µs 全是 Python 检查（`_check_game_over`/`_live_particles` 每帧两趟
列表重建 + `p.pos` 每果每帧一次 numpy 数组分配）。

结果（30 种子，本机单核，随机策略；base=290 drops/s 环境口径）：

| 变体 | drops/s | 分数 | 关键观察 |
|---|---|---|---|
| fast（单趟扫描+ffi直读） | 620 | 580±207 | **与 base 逐位一致**（分数路径/盘面全等） |
| fast_it8/6/4 | 631/643/653 | 494/555/597 | 仅 +2~5%；it6/it4 出现 2694 px/s 速度尖峰与 1-2 次穿墙 → 弃 |
| fast_dt45（180帧预算） | 733 | 566±223 | 0 穿墙；与 base 分布不可区分（30种子）→ 备选 |
| fast_dt30（120帧预算） | 845 | 653±258 | **8/30 局穿墙**（水果整体出界）→ 弃 |
| fast_dt30it15 | 771 | 584±185 | 7/30 穿墙 → 弃 |
| fast_check3（每3帧检查） | 771 | 475±169 | 分数 -18%：每步多沉 2 帧 → 堆更实、死更早，**改变了游戏** → 弃 |
| sleep 0.5s | — | — | cpSpaceStep 死循环：合并回调里 kill/加body 违反 step 内约束 → 弃 |

结论与落地：
- **已合入 bit-exact fast 路径**（`suika/part2/suika_env.py::_settle` 单趟扫描，
  `pymunk._chipmunk` 直读坐标/速度，带属性回退）。DQNEnv 端到端 232→452 drops/s
  （1.95x），规则测试通过，与原实现逐位一致。**已跑的 wave3 臂需重启 actor 才生效。**
- "迭代数减半近 2 倍"不成立：Python 开销移除后迭代数只占帧成本 ~8-10%，收益 +2~5%
  且有数值风险。
- 1/30 不可用；1/45 分布级可用但属于"另一个游戏"，若采用需训练+评测环境同步切换，
  先用 CRN 套件放大种子数确认。
- sleeping 需先把合并逻辑改成 post-step callback 才可能启用（大重构，暂不做）。
- tempo_env 若复活，`_any_fully_over_line` 每帧同样有 np.array 分配问题，需同款修复。

## csettle_20260928 — C 扩展下沉逐帧扫描（17:10–18:40）

目标：把 `_settle` 每帧扫描下沉到 C（自包含 .so，经函数指针注入 Chipmunk 入口，不链接）。

**第一个设计（注册表镜像）失败，已回滚。** 用 C 侧哈希表镜像
`Particle.has_collided`（collision 回调写、kill/reset 清），C 里直接判定 game over。
深测 19 万帧发现 6 帧误判、30 集不变量检查 7760 处发散：
- `collide()` 同型分支会把 `has_collided` 置回 False 但不总经过 kill
  （distance 不满足/对方已死），注册表留下脏项；
- `cs_clear` 清表不重哈希，探针链断裂产生重复表项，forget 只删一个；
- 地址复用后新果继承"已碰撞"状态，曾把果子冻在投放口改变整局走向
  （seed 4002：C 路径 595 分 vs Python 877 分）。教训：**镜像可变 Python
  状态进 C 的哈希表，生命周期等价性极难保证，别做。**

**第二个设计（无状态只读扫描）通过全部验证。** C 只算 `(vmax_sq, min_y)`；
`min_y < killy` 是 game-over 的必要条件，仅在此时回调 Python 原有
`_check_game_over()`——谓词与原实现逐位等价，常见帧零 Python 工作。

验证（全部通过）：
- 逐帧双扫 fuzz：19.27 万帧 0 偏差（vmax_sq 与 min_y 同时比对）；
- 全局等价：30 种子 C 路径 vs Python 回退，分数路径逐位一致；
- 30 种子 bit-exact 消融回归：与原始实现全等；
- 规则测试通过。

性能（本机单核，DQNEnv 端到端随机策略）：
- 原始实现 232 drops/s → 单趟 Python 扫描 467 → **C 扫描 552（2.38x）**
- per-drop：settle 124 帧 × (pymunk step ~5.5µs + C 扫描 ~1.5µs) + 非物理 ~0.75ms

已知边界：整个 settle 循环再下沉到 C 不可行——pymunk `Space.step` 在
`cpSpaceStep` 之后还要落地回调内排队的延迟 add/remove（合并的 kill 走这条路），
绕过 pymunk 的 step 会改变行为。当前即 bit-exact 路线的地板。

部署：远端每个跑 actor/评测的 venv 执行 `python part2/build_csettle.py`
（需 cc/gcc；无编译器时自动回退 Python 快扫，行为不变、速度回到 467 级）。
文件：`part2/csettle.c`（97 行）、`part2/csettle_loader.py`、`part2/build_csettle.py`。

## wave3c_20260929 — tf_xl 56M 放大 + 已知 bug 全修（00:34 启动，预算 24h）

背景（用户 00:21 指令）：tf_deep 看似饱和，立即切到更大更深的 tf，修掉之前
所有已知 bug，训练预算一天；8 卡全用、保证 GPU 充分利用。

tf_deep 收官（w3b_tf_deep 停止于 00:30，ckpt 冻结于
runs/wave3b_20260928/w3b_tf_deep_FROZEN/）：eval peak **2008 @ 239M 步**
（p25 1694, max 2826, p2000=0.375），末次 1883 @ 251M。

新臂 w3c_tf_xl（node_c，全新从零训练）：
- 模型：settf d_tok128 / d_lat512 / n_lat64 / stage1=2 / stage3=16 / heads8
  → **55.88M 参数**（tf_deep 为 8.21M，6.8x）。
- 布局：learner 独占 gpu0，evaluator gpu1，推理服务 gpu2-7（6 卡），
  200 actors（C 物理引擎）。replay 8M（~51GB，窗口 ~7min）。
- batch 65536 = 16 × 4096 梯度累积 + **梯度检查点**（model_v2 新增
  grad_ckpt，覆盖 stage1/cross/stage3；no_grad 下自动关闭）。
- 预算：grad_budget 40k（cosine → 0.1x）+ per_beta 同步 40k，按估
  ~0.45 opt/s ≈ 24h；1h 实测后校准。

修复的已知问题（本次代码变更）：
1. **publish 改发 EMA**（原发 online）：online 网把每步梯度噪声直接注入
   200 actor 的出招，是训练 rollout 全局性振荡的主因（实测 2min 桶均值
   std tf=210 vs mlp=139；dip 时 200 actor 同时 p50 腰斩）。EMA 发布即
   DQN 系标准 behavior=target 做法，eval 也更稳。
2. **冷启动随机权重事故（wave3b 19:20）**：learner 进入主循环前先
   publish 一次播种 policy.pt；run_arm 新增闸门——policy.pt 出现前不起
   infer server/actor，learner 早死则全臂退出。
3. **learner 慢（tf 每 opt 步 18.7s 之谜）**：train_step 每个 micro-batch
   3 次 .cpu() 同步（64k=32 累积时近百次/步）。改为全程 GPU 累积、
   末尾一次同步；EMA 更新改 _foreach。预计提速数倍。
4. **evaluator 挤 gpu0**：run_arm 新增 --eval-gpu，evaluator 独占 gpu1。
5. grad_ckpt 首版闭包 bug：lambda 晚绑定捕获循环变量 blk，backward
   重计算错跑 stage3 块（LN 形状不匹配崩溃）→ 默认参数绑定修复。

### wave3c 吞吐调优实录（01:00–01:30）

- 64k batch（16×4096 累积+ckpt+compile）实测 **29s/opt 步**；远程隔离基准
  （gpu1, bench_train_step.py）纯 GPU 26.4s/step、峰值 19.9GB → 瓶颈是 GPU
  计算本身（每步 ≈6 次前向当量：fwd+2bwd+ckpt重算+double/target 两次
  no_grad 前向），环路开销仅 ~3s（PER/inbox/H2D）。compile 只省 warmup。
- **关键推论：learner 是吞吐瓶颈时，每日总抽取数被 GPU 固定**
  （65536/26.4s ≈ 2.5k samples/s ≈ 214M/天），与 batch 大小无关；
  batch 减半只改变 opt 步数（3270→6500 步/天）。24h 预算下选
  **batch 32768（8×4096）+ lr 4.5e-4**（sqrt 缩放）+ grad_budget/
  per_beta 6500。64k 的最初动机（降 off-policy 度）在 reuse≈0.3 的
  learner-bound 状态下已无意义。
- 事故记录：pkill 自匹配 2.0——kill 与 relaunch 合并在一条 ssh 命令里时，
  relaunch 段的明文 "w3c_tf_xl" 出现在 shell 自身 argv，pkill -9 杀死
  自身 shell，后续命令未执行。教训：kill 与 relaunch 必须分两条命令。
- grad_budget 初值 40k 是按错误 sps 估的（以为 0.45 opt/s，实际 0.034），
  已按 24h 实测校准为 6500。

## wave4_killy 分叉消融 — 受控交叉评估（09-29 18:1x–18:35）

**待解问题**：killy 170→200 后，`w4_k200_cont` 的 live 曲线从 fork 点的
~2135 跌到 ~1500 并压平（−30%），而 `w3c_tf_xl` 一直在 2100–2350。
但 killy 同时进入**终止谓词**与（潜在）**观测**，曲线无法区分
"环境变难"与"策略退化"。

**方法**（`xeval_killy.py` + `run_xeval_killy.sh`，commit fb0a767/e6367e1）：
- `SUIKA_KILLY` 在 `part2/config.py` import 之前设置（模块级单例），
  脚本断言 config.pad.killy 已生效，否则拒绝开始。
- **观测不变性证明**：固定动作 probe rollout（4 种子 × 3 步）的 obs+reward
  sha256 在两套死亡线下完全一致（digest `7e016fa908c8caba`）→ killy 只动
  终止，不动输入。
- **权重同源性**：fork ckpt `step3801_env482771285.pt` 在两个 run dir 下
  md5 逐位相同（`78d5c1d24106703d32c7bf60f8309bc7`），故同一份权重可跨
  死亡线直接比较。
- 权重来源已冻结：fork ckpt、w3c 与 w4cont 各自的 `policy.pt` 复制到
  `runs/xeval_killy_perseed/src/` + MD5SUMS（policy.pt 每 ~2s 重发、step
  ckpt 只保留 ckpt_keep=4，不冻结会被覆盖）。
- 64 个种子（0:63），**共同随机数**：同种子同果序列，故可做逐种子配对
  检验。`evaluator.evaluate_once(..., return_per_seed=True)` 为此新增
  （默认关闭，不影响生产 evaluator）。

**结果**（greedy，均值 [bootstrap 95% CI]）：

| 权重 | killy=170 | killy=200 |
|---|---|---|
| fork step3801（同权重，纯环境效应） | **2173** [2066,2281] | **1553** [1458,1648] |
| w3c（k170 续训，@502M 步） | 2154 [2038,2272] | 1595 [1493,1699] |
| w4cont（k200 续训，@498M 步） | 2329 [2206,2454] | 1703 [1584,1826] |

**配对检验**（逐种子分差，64 局）：

| 对照 | 分差 | 95% CI | 赢局 |
|---|---|---|---|
| 环境：同权重 k170→k200 | **+620** | [+503,+743] | 59/64 |
| 策略 @killy=200：w4cont vs w3c | +109 | [−34,+253] | 36/64 |
| 策略 @killy=170：w4cont vs w3c | +175 | [+11,+342] | 40/64 |

**结论**：
1. **曲线上的"崩溃"100% 是环境效应**：同一份权重，死亡线上移 20px
   就掉 620 分（−28.5%），且 59/64 个种子一致。旁证：`p2000` 0.56→0.12、
   `moves_mean` 217→159——局更早结束，不是策略变差。
2. **没有遗忘**：k200 训练过的策略在**两种**死亡线下都不低于 k170 策略
   （+109 / +175），方向上与新环境的适应一致；但 @k200 的 CI 跨 0，
   64 局仍不足以宣称显著收益，只能说"不亏"。
3. 交叉评估复现了生产 evaluator（seeds 0:16 子集：w3c@200 1593 vs live
   1601，w4cont@200 1618 vs live 1601，w3c@170 2249 vs live 2254）——
   工具链自洽，不是另一套口径。

图：`dqn_runs/xeval_killy.png`（左：分组柱+CI；右：配对分差+CI+赢局）。
教训登记为 **D14：当一次改动同时动了环境与策略，必须用"冻结权重 × 交叉
环境"分离效应，否则曲线读数是两个效应的混合**。

## w4bc_20260929 — BC 暖启（Qwen 臂）迁到闲置 node_b（14:47 启动）

动机：预注册的 BC gate 原排在 t1 的 w3c_tf_xl 结束之后（~01:50），但
t1_2/t1_3 两台新节点全时空闲。把 BC 前移 11h，且不动 t1 上的 RL 臂。

- 节点：node_b（8×A100-40G，232 核实为全量节点，nproc=1 是 shim 假象），
  全部 8 卡 DDP（torchrun --nproc_per_node=8），run dir
  `runs/w4bc_20260929/bc_qwen`。
- 数据：perm 的 `runs/bc_data_20260929/inbox` 全量 7056 shards / 180G，
  按 actor 前缀 `a0–a23`（24/48 actor 子集，3528 shards / 90G）经
  **节点间 rsync** 拷到 t1_2 `bc_data/inbox`（≈28MB/s，~55min 完成）。
  BCData 实测 train=12,000,000 val=284,655，与 config 里
  "12000×1024 = 12.3M samples vs 14.4M transitions in the 24-actor subset"
  的预算完全一致。
- **节点间通路**（一次性）：perm 入站仅开 :22，且 t1_2 是
  `PasswordAuthentication no`（仅密钥）、perm 接受 root 口令。解法：在
  t1_2 生成 node-local 密钥对（私钥不出 t1_2），仅把公钥追加进 perm 的
  `~/.ssh/authorized_keys`，此后 t1_2→perm 的 ssh/rsync 直连可用。
  未在远端写入任何个人凭据。
- 环境补齐：t1_2 venv 缺 `transformers` → 装 5.17.0（与 perm 的
  suika-venv 完全同版本），peft 0.21.0 / fla 0.5.2 已一致；
  `causal_conv1d` 两边都缺 → 在 t1_2 补装 1.7.0（源码编译，
  nvcc 12.4；GitHub release 轮子不可达）。8 卡全量启动前跑通
  build_model + GPU 前向 smoke（779.7M 参数）。
- **修掉一个 DDP-only bug**：`bc_learner_qwen.py` 原用
  `device = torch.cuda.current_device()`（返回 int），而 BCLearner 里
  按 `device.index` 使用 → world>1 时 8 个 rank 全部
  `AttributeError: 'int' object has no attribute 'index'`。单卡 smoke
  测不出（world=1 不包 DDP）。已按 `learner_qwen.py` 的写法改成
  `torch.device(f"cuda:{local_rank}")`。
- 吞吐：causal_conv1d 前 **6.15s/opt 步**；装上优化核后 **5.7s/步**
  （≈7%），step50 的 loss 1161.67 vs 源码路径的 1161.37 → 数值等价的
  旁证。12000 步 ≈ **19h**，预计 09-30 上午完成。瓶颈在 GDN 主干算力
  本身（8×A100 合计 ~28% MFU），不是 conv1d，故不再追 kernel。
- gate 不变：`agree_pi ≥ 0.85`（每 200 步测一次，metrics.jsonl），
  过 gate 后拿 `policy.pt` 暖启 Qwen RL 臂。

## tf_xl 吞吐消融（node_d GPU4，09-29 20:2x–21:4x）

**动机**：上一轮把 w3c_tf_xl 的 14.7 s/grad-step 判成"软件病理"，点名三个
嫌疑人（`grad_ckpt` 全量重算、fp32 attention over T=160、未走 SDPA/
FlashAttention），并预测修完能到 20–50k 样本/s。本轮用受控消融检验该判断。

**工具**（新增，均可复跑）：
- `bench_tf_throughput.py` — 复刻 learner 的 step（bf16 autocast +
  micro-batch 累积 + double-DQN 的 3 次前向 + quantile-huber），遍历
  {grad_ckpt}×{eager/sdpa}×{bf16/fp32}×{micro}，CUDA event 计时，记录峰值显存，OOM 容错。
- `model_v2.py` 新增 `SDPABlock` + `attn_impl`（默认 `eager`，生产行为不变）。
  等价性检查：`to_sdpa_state_dict` 把 Block 权重映到 SDPA 键，CPU 上
  max|Δq| = 0（逐位相等）、CUDA 上 rel 3e-7；且 padding 语义被验证有效
  （同一批把 pad 行填成真实果子，Q 变化 2.84）。
- `probe_compile_modes.py` — 每个配置一个**全新进程**，避免 compile 缓存串味。

**消融 1（batch 32768，T=160，3 个计时步）**

| 配置 | ms/步 | 样本/s | 峰值 GB |
|---|---|---|---|
| ck1-eager-bf16-m4096-cp1（**生产**） | 12631.6 | 2594 | 18.2 |
| ck1-**sdpa**-bf16-m4096-cp1 | 11861.5 | 2762 | 18.1 |
| ck1-eager-bf16-m4096-**cp0** | 19647.0 | 1667 | 19.6 |
| ck1-sdpa-**fp32**-m4096-cp1 | 84526.4 | 387 | 29.9 |
| ck0-\*-bf16-m4096 | OOM | | |
| ck1-sdpa-bf16-m8192 / ck0-m8192 | OOM | | |

**消融 2（把 recompute 的钱用更小的 micro 付）**

| 配置 | ms/步 | 样本/s | 峰值 GB |
|---|---|---|---|
| **ck0-sdpa-bf16-m1024-cp1** | **10872.5** | **3013** | 23.8 |
| ck1-sdpa-bf16-m2048-cp1 | 11995.9 | 2731 | 9.7 |
| ck1-sdpa-bf16-m1024-cp1 | 17544.7 | 1867 | 6.3 |
| ck0-\*-bf16-m2048 / m8192 | OOM | | |

**结论 1：三个嫌疑人全部落空，我上一轮的判断是错的。**
1. SDPA 只快 **6%**（12632→11862）。grad_ckpt 开着时每层都要重算，省下的
   显存红利无处分。
2. 去掉 `grad_ckpt` 在 40 GB A100 上 **micro≥2048 直接 OOM**；只有 micro
   1024 装得下（23.8 GB）。重算在这张卡上是**载荷承重**，不是浪费。
3. "fp32 attention"不存在——autocast bf16 一直在生效，而它值 **7.1x**；
   **torch.compile** 值 **1.56x**。两个大杠杆**早就在生产里开着**。
4. 可达最优 = **ck0 + sdpa + micro1024 + compile = 10872 ms/步 = 3013 样本/s
   （比生产快 16%）**。**"20–50k 样本/s"被实测否证，差距 7–17 倍。**

**结论 2：顺手排掉一个我怀疑的真 bug（负结果）。**
`profile_tf_stage.py` 的日志里出现过
`torch._dynamo hit config.recompile_limit (8)`，reason
`GLOBAL_STATE changed: grad_mode autocast(cuda)_enabled` —— 一度以为生产
learner 也在静默回退 eager。干净复测：真实调用模式下只编译 **3 张图**
（grad+autocast / no_grad+autocast / target），把 `recompile_limit` 提到 64
**毫无变化**（12662 vs 12658 ms），compile_online（只编译 online、target 留
eager）反而更差（13720 ms）。**生产没有回退，compile 是好的。** 那次
recompile_limit 是 profile 脚本自己多喂了第三种全局状态（fp32 前向）造成的
——教训：**别按临时脚本的混合调用模式推断生产行为**。

**结论 3：成本结构（每配置一个全新进程，线性拟合 R²=1.0）**

| 变体 | ms/步 | 样本/s | vs 生产 |
|---|---|---|---|
| prod16（T160, s3=16） | 12621.5 | 2596 | 1.00x |
| T=64 | 11504.2 | 2848 | 1.10x |
| s3=8 | 7501.3 | 4368 | 1.68x |
| s3=4 | 4941.7 | 6630 | 2.55x |

step = 2381 + 640×stage3 → **stage3（64 个 latent × 16 层）占 81%，且逐层
线性**；其余全部（stage1 + cross + embedding + 读出 + 优化器）合计 2.38 s。

**结论 4：T=160 里 78% 是 padding，但砍它没用。**
从 t1 上 w3c_tf_xl 的真实 inbox（18 个 shard / 81922 条 transition）统计棋盘
果子数：**mean 15.0、p50 15、p95 25、p99 28、历史最大 36**；>32 个只占
0.078%，从未超过 48。即 160 个 token 槽位里 122 个永远是 padding。
但实测 T 160→64 只省 **8.9%**，与 FLOP 占比（token 级 ~9%）一致——因为算力
91% 在 latent 塔上。**要提吞吐只能砍 stage3 的深度/宽度，那是砍容量。**

**对主线的含义**：A100 上 tf_xl 的 14.7 s/步**不是可修的软件病**，它的成本
就是架构自身的算力；两个能修的杠杆早已开着，剩下 16% 的空间。tf 的价值仍在
"每样本赚"（1/78 样本量追到 MLP 上限 2382→2399），把它换掉或指望 10–20x
提速都不成立。真要再去一层，只有换 141 GB/H200 这类卡把 recompute 和
micro 一起放开（ograd_ckpt 在 H200 上可关、micro 可到 4096+），预估仍有
~1.5–2x；ophis-gpu 现有 5 张空闲 H200 但尚无 suika 部署。

**注意（若要把最优配置落到生产）**：`attn_impl: sdpa` 会改变
`state_dict` 的参数名（`attn.in_proj_weight` → `q/k/v_proj`），
`learner.resume` 直接 `load_state_dict` 会失败，续训前必须先过
`model_v2.to_sdpa_state_dict` 做一次权重映射。

**决策（本轮，已拍板）**：
1. **不迁移在跑的 wave**。node_c 的 tf_xl（2 张卡满载）保持
   eager+grad_ckpt 继续跑——16% 不值得一次权重映射 + 重启中断，而且
   种子/曲线会断。`attn_impl` 默认值仍为 `eager`，生产行为零变化。
2. **下一批 tf 启动时默认 `attn_impl: sdpa` + `micro=1024` +
   `grad_ckpt: false`**（40 GB A100 上 23.8 GB 可容纳，3013 样本/s）。
   续训得先把旧 ckpt 过一遍 `to_sdpa_state_dict`。
3. **真正的杠杆是换卡**：H200/141 GB 上可同时放开 `grad_ckpt: false` 与
   `micro 4096`，预估 1.5–2x。A100 上 tf 的吞吐天花板到此为止。
4. **但不要指望吞吐换分数**：tf 与 MLP 都在 ~2.4k 平台，瓶颈是环境与
   探索而非模型容量（见 wave3c 结论）。本轮的价值是把"tf 慢"归因从
   软件改成架构/硬件，从而否掉"修完就能 10–20x"的路线假设。

## wave5_20260929 — 几何可变（killy/宽度）+ 动作序列存档（23:50 启动）

**目标（用户 09-29 夜）**：板面从"固定 killy"推到"killy 与宽度可变"，看 tf_xl
在更难/更长的局上能否继续涨；同时把**每局完整动作序列**存下来（复盘、可视化）。
只用 node_b 与 node_a（node_d 归 BC/SFT 工作线；node_c 与另一台 8 卡 A100 已划给 LLM 版合成大西瓜训练，不排 RL 任务）。

**killy 语义（易混，务必记住）**：killy 是 y 向下的坐标，**越大 = 死亡线越靠下 =
可用高度越小 = 越难**。stock=170（可用高 505px），220 → 455px。wave4_killy 实测
同权重 170→200 掉 28.5%。所以 wave5 的 killy 220 / 170-250 都是**比 stock 更难**
的方向；若目标是"更多空间、分数近乎无尽"，应往 killy<170 走。

**两臂**（tf_xl 55.9M；`attn_impl=sdpa`、`grad_ckpt=false`、batch 32768=32×1024
累积、200 actors、固定 AdamW 重新预热 lr 2e-4/warmup 300/floor 0.25；均 warm start
自 w4_k200_cont，seed ckpt md5 `f269c60bd48c8f7cb15d61fd33227964`，三节点一致）：

| 臂 | 节点 | 板面 | geo_dim | 评测 |
|---|---|---|---|---|
| w5_k220_fixed | node_a（6 卡：learner gpu0 / eval gpu1 / infer gpu1-5） | killy=220，宽度 stock | 0（结构同 k200_cont） | 单板 12 种子 |
| w5_var | node_b（learner gpu0 / eval gpu1 / infer gpu1-6，**gpu7 留给 LLM 任务**） | 宽 [352,544] × killy [170,250]，按 episode 种子确定性抽样 | 3 | 3×3 网格（宽 352/448/544 × killy 170/210/250）×4 种子 |

`eval_max_moves=1500`：策略若"近乎无尽"不会卡死评测，超限局标 censored，分数按下界。

**代码变更**（commit 见下）：
- `encoding.geom()` 读活的 `config.pad`；`geo_vector()` = `[(W-448)/100, (killy-170)/40,
  W/(bot-killy)-448/505]`，stock 板面恒为 0；token 逐个多 3 维（左右余量、到死亡线的
  深度，**按果心**——与 `suika_env` 的 `p.pos[1] < killy` 判死完全一致）。
- `env.set_geometry / sample_geometry`（种子确定 → 评测共同随机数）；宽度关于
  stock 中心对称（镜像增广仍然合法）；`DQNEnv.reset(geom=...)` 可钉板。
- `model_v2`：`geo_tok/geo_lat` 及 `in_proj` 新增列**零初始化** → stock 板面上与旧模型
  函数逐位等价（max|dq|≈1e-6），可安全 warm start；`adapt_state_dict` 负责
  eager→sdpa 键名映射 + 新列补零，任何其它不匹配直接 `KeyError`（不静默丢权重）。
  优化器状态不迁移（结构已变），用新 AdamW + 预热。
- `learner`：`init_from` 与 `ema` 均走 `adapt_state_dict`；`_mirror` 对 geo 段原样透传。
- `evaluator.evaluate_grid`（钉板网格 + `max_moves` + `action_sink`）。
- **动作序列存档**：actor 每局写 `actions_aN.jsonl`（seed,w,killy,eps,score,moves,
  hex 动作串）；评测写 `eval_actions.jsonl`；`replay_actions.py` 复盘校验/导出逐帧。

**验证**：`test_geometry.py` 在 node_a（强制 CPU）12/12 PASS：stock 板与无几何位逐位
一致、几何种子确定且在界内、warm start 函数保持、镜像对合、复盘一致。上线后从
w5_var 实际数据抽 6 局（3 训练 + 3 评测）复盘，分数与步数全部一致（如 seed=22100101
534x172 3579/335 match）。

**启动与早期状态**（t+8min，仅热身，**不是趋势**）：两臂 0 traceback，env_sps
7.7k–8.2k，grad_sps≈0.06（每步 32 次 micro-batch，设计如此），LR 按 300 步预热
爬升；首次评测 k220_fixed 1344（max 1871）、var 1387（max 3564，网格 min 748）。

**存档与磁盘**：动作串约 1.75MB/min/臂 → ~2.5GB/天/臂（远端 200G 盘，够用；
checkpoint 仍按既定 ~15min 频率，不额外占盘）。不整包拉回：
`pull_wave5.sh` 先在远端 `digest_actions.py` 汇总成 top-K（默认 100）可复盘动作序列
+ 10min 分箱曲线（含按 killy 分组），再只拉 digest/metrics/eval/`eval_actions.jsonl`
（每臂 ~100KB）→ `dqn_runs/<node>/wave5_20260929/<arm>/`。原始 `actions_a*.jsonl`
留在节点。

**待观察**：(1) k220 固定臂能否追回到 k200_cont 的 ~2200 水位（k200_cont 用了
~600M 步才从 ~1500 恢复到 ~2200）；(2) var 臂在各 killy 档的曲线是否同向上升，
还是被容易的档拖着走（看 `by_killy`）；(3) eval 里 censored 是否出现——出现即说明
策略开始"活过窗口"。

## 实验线总览（截至 09-30 00:00）

| 线 | 时间 | 内容 | 结论 |
|---|---|---|---|
| wave1 | 09-28 00:22–09:30 | 16 臂配方扫描（QR-DQN Ape-X） | 朴素 dueling DQN(nq1) 最好，γ=0.999>0.995；首次通关（maxfruit=10），峰值 eval 均值 1315 |
| wave2 | 09-28 09:44–15:55 | 放量 4 臂，~150M 步/臂 | 均值 1377–1499 平台；n-step 后继 bug（D10）待复核 |
| wave3/3b | 09-28 16:10– | 架构 v2：set-transformer vs MLP；C 物理 + resume + 64k batch | tf_base 崩（568）；mlp_deep 2.56B 步/27h 单次评测最高 2495（末 2079）；tf_deep 241M 步最高 2273（末 2008） |
| phy/csettle | 09-28 15:50–18:40 | 物理结算消融 + C 逐帧扫描 | 单核 232→552 drops/s（2.38×），逐位一致；镜像 Python 状态入 C 被否决（D12） |
| wave3c | 09-29 00:34–21h | tf_xl 55.9M，publish 改 EMA、修 learner 同步、evaluator 独占卡 | 606M 步单次最高 2399（末 2136）；与 MLP 同平台 ~2.4k：瓶颈是环境/探索而非容量 |
| wave4_killy | 09-29 18:1x | 冻结权重 × 交叉死亡线 | 同权重 170→200 **−620 分（−28.5%，59/64 种子）**；曲线"崩"是环境效应，非遗忘（D14） |
| w4_k200_cont / scratch | 09-29 | k200 续训 vs 从零 | 续训 ~4.4h 回到 ~2200；从零 96M 步仅 1059 |
| tf_xl 吞吐消融 | 09-29 20:2x | sdpa / grad_ckpt / T | 14.7s/步是架构算力（T 160→64 仅省 8.9%）；新 tf 默认 sdpa+micro1024+ckpt 关；真杠杆是 H200 |
| w4 Qwen/BC/SFT | 09-29 | LLM 直接 RL + BC 暖启 | BC v3.1 q_regret 7.75、eval 1257–1698；decode 路径 qhead 1540 vs tokens 500（随机带）→ SFT stage1；node_d 被占用，**不动** |
| **wave5** | 09-29 23:50 起 | 几何可变 tf_xl ×2 + 动作序列存档 | 见上 |

### wave5 day1 结果（09-30 20:45 检查，t+21h）：峰值创新高后两臂 Q 值发散崩溃

**轨迹**（greedy eval 均值 / 梯度步）：
- w5_k220_fixed：峰值 **2135 @ grad494 / 55M env**（killy=220 更难板面上追平 w3c@k170 水位），
  grad~800 起缓降，**grad~2400（t+8h）崩溃**，之后 13h 停在 ~400-500（≈随机）。
- w5_var：峰值 **1916 @ grad752 / 81M env**（混合网格），同样在 grad~2300 崩溃，
  q_mean 1013 后部分回落到 ~700 但策略没回来。

**崩溃特征**：grad_norm（裁剪前，clip=10 一直开着）从个位数跳到数百~29 万；
q_mean 从 ~475 一路冲到 2816（k220）且 eval 反而下降 → loss 很小而 Q 巨大，
是 **γ=1 自举的典型发散**：非终止态的 Bellman 备份对 Q 加常数仍是解，暖启的
Q（在 killy=200 训练、~350）在新板面上系统性高估，TD 正反馈 + PER 放大高
TD 样本 + EMA target 跟随 → 失控。与数据损坏无关（test_geometry 12/12、
两臂前 7-8h 健康且创新高）。

**亮点**：历史最高分episode诞生——var 臂 **6733 分 / 601 步 @ 510×183**
（eps=0.027），top-5 全部 ≥4352，且都在 killy 171-183、宽 510-538 的"偏大棋盘"
上 → 再次印证"棋盘越大上限越高"。k220 臂 top1=3225（更难板面）。所有 top-100
动作序列已归档本地（`digest_top_actions.jsonl`），可复盘可视化。

**事故教训（D15）**：`ckpt_keep=4 × 15min` 只留最近 1 小时，峰值 ckpt（grad~500）
早已轮转删除——**21 小时里唯一有价值的好策略丢了**，重启只能从
seed_w4_k200_cont 再来。修复方向：(1) evaluator 保存 best-eval ckpt；
(2) 自举 target clamp（q_max≈8000，远高于真实回报 6733）；(3) learner 发散
看门狗（q_mean 超阈值即冻结/回滚）；(4) ema_tau 放缓。

## wave6_20260930 — 地板下移几何（h 500–720）+ 反发散三件套（21:35 起）

**几何（用户 09-30 21:15 拍板）**：killy=170 与顶部间距（killy−top=85）钉死不动；
**地板移动**：`bot = killy + h`，h∈[500,720]（stock=505）；宽度 [400,550] 仍关于
stock 中心对称。球大小、重力、drop 线全部不变。模型几何特征按用户指定 =
`[(W−448)/75, ((bot−killy)−505)/110, ((killy−top)−85)/85]`——即"到地板的距离"与
"顶部到死亡线的距离"直接进模型；token 级的"到死亡线深度"由活几何实时推导
（地板动时 y 反归一化用活的容器高）。stock 处仍全零，暖启等价性保持。

**反发散三件套**（针对 wave5 day1 的 γ=1 自举失控）：
1. **target clamp**：自举 target 截断 [0, `q_target_max`=8000]（真实纪录 6733，
   正常学习永不触顶，只断失控链）。
2. **policy_best.pt**（D15 修复）：eval 均值创新高立即快照，永不轮转删除。
3. **看门狗**：`q_ema > max(3×最近 eval 均值, 1000)` 连续 3 分钟 → 强制存 ckpt
   并以 exit 3 停机。另 ema_tau 0.003→0.001 放缓 target 跟随。

**ckpt 改按梯度步间隔**（不再按墙钟）：生产 100 步（≈30min）keep 6；
debug 20 步（≈5min）keep 8。`ckpt_interval_steps` 覆盖旧 `ckpt_interval_s`。

**臂**（均 tf_xl 55.9M / sdpa / batch 32768=32×1024 / 200 actors / warm start 自
w4_k200_cont / lr 2e-4 warmup 300 / γ=1 不变）：

| 臂 | 节点 | 板面 | 状态 |
|---|---|---|---|
| w6_debug | node_a | w[400,550]×h[500,720] | 22min 烟测后退役：96 梯度步，q_mean 钉在 ~300，gn 0.5–3.5，best-eval 3687 已快照 |
| w6_h720_fixed | node_a（6 卡） | 448×720 固定（"近乎无尽"目标板） | 21:56 起 |
| w6_var | node_b（gpu7 留 LLM） | w[400,550]×h[500,720] 抽样 | 21:35 起 |

评测：h720 臂单板 12 种子；var 臂 3×3 网格（w 400/475/550 × h 500/610/720）×4
种子；`eval_max_moves` 1500→3000（h=720 的局长）。记录三元组改为
(seed, w, h)，wave5 旧格式（w, killy，固定 675 地板）回放向后兼容。

**早期信号（强）**：暖启策略在更大棋盘上**立即**迁移——w6_var 前两次 3×3 网格
评测 mean 3230→3244（max 单局 **11561**，p2000=0.83，0 censored）；debug 臂
双板评测 best 3687。q_mean 全程钉在 ~300 无漂移。注意：wave5 发散发生在
~2400 梯度步（~10h），20 分钟烟测只验证管线与机械装置，长期稳定性靠
clamp + 看门狗兜底。

**验证**：test_geometry 15/15 PASS（t1_3 CPU）：新增地板真实下移
（bot=killy+h=890、果子静止于 y=832 > stock 地板 675）、wave5 格式回放兼容。

### wave6 +2h 诊断：几何输入已接线但模型"几何盲"，geo_lr_mult=10 热修（10-01 00:15）

**症状**：eval/online 涨得慢。online 分板数据：h720 臂 2588→2712（+4.8%/2h）、
var 臂 h700 桶 2838→3014，h500 桶 1757→1769 完全平——增长大致按棋盘面积分配，
是"几何无关通用改进"的特征，不是按板适配。

**根因（称量权重 + 函数测试双重实锤）**：几何通道**输入接线完全正确**
（obs[800:803] = [宽度, killy→地板距离, 顶部→killy 释放高度]，归一化，15/15
测试过），但暖启零初始化 656 步之后：geo_tok/geo_lat absmax 仅 0.003–0.004、
in_proj 几何列 0.0013（token 列 0.175，差 ~135 倍）；函数测试同一棋盘状态喂
两个极端几何（550×720 vs 400×500）ΔQ=0.165/257.8 = **0.06%** —— 模型事实
上分不清棋盘。零初始化是暖启等价性的刻意设计，但 AdamW 下权重增速 ~lr，
照此需数千步才能让几何"可见"。

**热修**：learner 加 `geo_lr_mult`（geo_tok/geo_lat 独立参数组，LR×10）；
续训时参数组结构变化会跳过旧 opt state（打印日志，fresh AdamW，LR 日程按步
续走）。00:19 两臂 kill+relaunch，从 step700/step600 ckpt 无缝续跑
（replay 不持久化，重建中，loss/gn 短暂上扬属预期）。
**验证点**：~1h 后 geo 权重应明显超过 0.004，ΔQ(几何) 应进入可测范围。

**D16（演员长局复盘漂移）**：训练演员进程（`actions_a*.jsonl`）里的千步级长局
**不可逐位复盘**——12168 分局重放到 10910 提前死亡（~90% 处分叉）；短局
（<400 步）全部吻合。原因：演员单进程跨 7 万局不换 `SuikaEnv` 实例，物理空间
跨局存在不可见的残留差异（接触缓存/body 池复用），长跑局对初始微扰敏感。
评测器每局新建 env，6/6 全部 bit-exact。**结论：展示/蒸馏只信评测器动作序列；
演员序列仍可用于统计与中等长度复盘。** 若需训练局长局也可复盘，actor 每局
重建 env（成本 ~每局一次墙体构造，可接受）或把 episode 序号混入 reset 行为
审计——暂不做，记此备查。

**附记（D16，09-30 23:2x 开源导出时发现）**：训练 actor 写的长局动作序列**不保证
逐位可复盘**。w6_var 训练 top1（12168 分 / 1082 步）在新进程里重放到 ~10910 分
提前死亡；而同节点的评测局（evaluator 每次评测是新进程）6/6 全部 bit-exact，
短训练局（~330 步）也一致。机理推断：actor 进程长期存活、Space 反复重建，
刚体/碰撞回调的插入顺序与全新进程不同，碰撞求解顺序出现亚像素级差异——
长局（>1000 步、30+ 果子）把它放大成分数级偏差。**规则：存档/展示级纪录
一律用评测局，或在归档前用 replay_actions 验证**；训练局的 actions 仍可用于
行为分析与教师蒸馏（分数不可信时以重放分数为准）。

**事件（10-01 00:19，两臂同时重启）**：两臂 run_arm 在 00:19:38/44（相差 6 秒）
被**本会话之外的操作者**同步 `learner.py`（内容与本地位逐位相同，mtime 被刷新）
并 kill+重启。learner 从 ckpt step600/700 恢复（各损失 ~50 梯度步），evaluator
以 `best_mean=-inf` 重启 → policy_best 快照被更差的策略覆盖（4249/3682 快照丢失）。
run_arm 无重启循环（learner 死才退出，旧进程无 [exit] 记录），排除自动重启。
处置：(1) evaluator 修补 D15b——启动时从 `policy_best.json` 恢复 best_mean，
重启不再降级快照，已同步两节点（不重启生效于下次重启）；(2) 同步脚本今后一律
`COPYFILE_DISABLE=1`（macOS tar 会撒 `._*` AppleDouble 文件并干扰代码指纹）；
(3) 向用户确认操作者身份。**教训：多会话/多操作者共管同一批节点时，任何重启
都要先查 run.log 的 [code] 与进程树。**

### w5 RL：BC 暖启 nothink LLM RL 起跑（10-01 01:10，node_d）

- **臂**：`w5_rl_qwen_perm`，runs/w5rl_20261001/。learner gpu0-3（DDP4）+ infer gpu4-6 + eval gpu7，160 actors。
- **暖启**：BC v3 续训 chain 最新 ckpt `w4bc_bc/v3_perm/checkpoints/step7800.pt`（q_regret 6.44 / agree_pi 0.40），learner_qwen 新增 `init_from`（strict=False，204 trainable keys 全载）+ t=0 立即 publish。
- **BC 基线（RL eval 路径实测，gs=0）**：mean **1626.6** / median 1796 / max 2726 / p2000=0.375（16 种子）——暖启完美。
- **配方**：lr heads 1.5e-4 / lora 5e-5（scratch 的一半），warmup 100，ema_tau 0.001（放缓），q_target_max=8000 clamp（wave5 防发散保险），batch 1024 micro 8，min_replay 30k，publish 100 步，ckpt 900s keep 8。
- **BC 续训让路**：perm v3_perm 在 step7800 停（已存 ckpt），q_regret 7.75→6.44 收益递减，算力转 RL。
- **目标**：nothink Q 头 RL 在 1626 基线上看到持续提升趋势。

### w5 RL v1 政策退化 → v2 教师锚点热修（10-01 02:30）

- **v1 曲线**：eval 1627(gs0) → 1463(gs100) → **924(gs200, p2000=0)**，三连降判定政策退化。q_mean 稳定 1068、loss 降——Q 拟合良好但**排序被洗**（水平陷阱在 RL 重现：TD 点式回归在 eps≤0.5 混合数据上破坏 BC 的精细排序）。
- **v2 修复**（用户既定"教师预热"方案的实现）：learner 加**冻结 BC 教师锚点**——listwise soft-KD（τ=1，前向 KL，PER 加权）与 TD 损失相加；λ 1.0→0.3 线性衰减 4000 步（教师带跑前期，逐步放手）。eps_max 0.5→0.2 减毒数据。第三份模型拷贝常驻 learner GPU（40G 够）。
- v2 02:30 起跑（run dir `w5_rl_qwen_perm_v2`），基线 eval 1627 复现（同 ckpt 同种子确定性一致）。anchor_kl/lam 入 metrics。

### wave6 夜间（10-01 01:00-03:00）：geo 权重手术 + 第三臂 w6_var_xl 上 node_e

**w6_var geo 手术（01:20）**：geo_lr_mult=10 热修 1 小时后复核——权重确实在长
（geo_tok absmax 0.004→0.0118，~3.25e-5/步），但函数测试 dQ 仍只有 0.03%
（诊断基线 0.06%），照此速度 ~5000 步（>18h）几何才可测。决定直接做手术：
最新 ckpt step902 的 state_dict+ema 里 geo_tok/geo_lat 权重重置为 N(0, 0.02)、
bias 归零（备份 .bak_geo）。函数测试 dQ 0.03%→0.39%/0.91%（mean/best），
进入可学习区间。重启后 eval 3055→3383→3536→3670 无异常（D15b 补丁首次实战，
evaluator 从 policy_best.json 恢复 best_mean=3408.7 成功）。

**w6_var_xl（node_e，stage3 16→24，81.1M 参数）**：用户指示"GPU 冗余→模型可以
更大"。种子构造：step902 全部权重拷入 24 层模型，新增 stage3.16-23 做**恒等
初始化**（out_proj + ff.2 权重/bias 归零 → block(x)=x），geo 通道全部唤醒
（geo_tok/geo_lat std 0.02 + **in_proj 末 3 列几何特征列 std 0.02**——这三列
在暖启时置零且不在 geo_lr_mult 组里，w6_var 上没有此项）。验证：与 16 层
max|ΔQ|=0.43（q~372，0.1%，仅几何列噪声）、argmax 一致、dQ(geo) 0.59%/1.09%。
配置 seed=106、lr 日程从头、fresh AdamW、ckpt 间隔 100 步。首个 eval 3063.5。

**node_e 环境适配记录**（无外网 DNS；GPU 3、7 当前无法创建 CUDA context）：
- python 走 shim：suika-venv/bin/python → mlsbench-diffusers-main conda
  （torch 2.13.0+cu130，py3.11）+ PYTHONPATH=pylibs311（pymunk/pyzmq/pygame/
  cffi 离线 wheel）。弃用 mlsbench-cleanrl（torch 2.4.1 下 inductor 崩 + 两次
  learner "unrecognized error code"）。
- run_arm.py 节点补丁：删 PYTORCH_CUDA_ALLOC_CONF=expandable_segments——
  驱动 580.159.03 下 learner 报 "Invalid access of peer GPU memory over nvlink"，
  删除后稳定运行 >300s。t1（驱动 535）不受影响。（此补丁使 node_e 代码指纹与
  t1 不同，属基础设施差异，已记录于此。）
- 规避 GPU 3/7：learner gpu0、infer [1,2,4,5,6]、evaluator 与 learner 共享 gpu0。
- 教训：pkill -f 的模式会匹配 ssh 远端 shell 自身命令行——用 "run_arm[.]py"
  形式写模式，否则自杀（exit 255）。

**00:19/00:54 重启事件结案**：系本会话工作流的 geo_lr_mult 热修重启（fbed8f1
提交记录"both arms resumed from latest ckpt"），EXPERIMENT_LOG 00:15 条目
计划、00:19/00:54 执行，非外部入侵。evaluator best 覆盖问题已由 D15b 根治。

### 10-01 下午：w6_var 发散退役 + w6_max_fixed 接棒 + XL 迁移 node_f

**w6_var 发散（已退役，best 已保存）**：q_mean 300→417 加速、grad_norm 持续
300-1000（健康 1-10）、eval 从 ~3600 单调跌至 1164——wave5 同款机制
（γ=1 无收缩 + 高估自举；clamp@8000 在发散到达前不起作用）。反发散三件套
延缓但未阻止。可疑加重因子：geo 手术 + geo_lr_mult=10 使几何通路追板级 Q
调整快于价值收敛（h720_fixed 固定几何、geo 输出恒定，同超参完全健康）。
policy_best 3775.75@gs917 已保存为 seed_w6_var_best.pt。
另发现 00:19 代遗留孤儿 evaluator（pid 54236）与新一代并存 ~13h，已清理。

**w6_max_fixed（node_b，新臂）**：550×720 固定板（var 网格中涨最快的格子，
4 种子均 8212），暖启 w6_var best，geo_lr_mult 1、stage3=16、seed 107、
eval-gpu 0（gpu7 让 LLM）。首个 eval 4255.2（同权重在 var 网格上 3775）。

**w6_var_xl 迁移 node_f**：node_e 确认不健康——GPU 3/7 无法建 CUDA context，
learner/evaluator 反复死于 "Invalid access of peer GPU memory over nvlink"
（NaN 看门狗抓到过 1 次 inf 梯度并正确跳过，但最终 nvlink 崩溃杀 learner）。
node_f 八卡全健康（系统 python3.10 + torch 2.13.0+cu130），离线装 pymunk/
pyzmq/pygame/cffi 到 pylibs，shim 同 node_e 方案。注：node_f /tmp 是 974MB
小分区且 100% 满——pip 需 TMPDIR=/data/...，已写入 shim。加 supervisor
自动重启环（run_arm 死→60s 后从最近 ckpt 复活，上限 30 次）。首个 eval
3272.5（同一 seed 在 node_e 病态硬件上只有 1048），env 吞吐 ~10k/s
（t1 臂 ~6.2k/s）。当前三臂：h720_fixed（t1_3）、max_fixed（t1_2）、
var_xl（node_f）。

### w5 RL 收益确认 + 专家迭代（EI）循环上线（10-01 14:30）

**RL 收益确认（64 种子配对评测，gpu7 离线）**：
- BC 基线（step7800 全量 policy）：mean **1582.0** / median 1561 / p2000 0.19 / min 142
- RL gs=1700（v3 run 实时 policy）：mean **1724.3** / median 1728.5 / p2000 0.28 / min 736
- **+142.3（+9.0%）**，mean/median/min/p2000/moves 全统计量一致向上 → 教师锚点配方下的 nothink LLM RL 有真实收益。
- 周期 eval（16 种子）轨迹也走出水平震荡：gs1500=1652 → gs1600=1758 → gs1700=1764 → **gs1800=1905**（med 1914）。
- 教训：16 种子 eval 的 σ≈±150 足以掩盖 +9% 级别的收益；关键节点必须 64 种子配对。

**EI 循环**（expert iteration，利用 v3 的 actions_a*.jsonl 完整对局记录）：
- `ei_dump.py`：回放 top10% 对局（cutoff 2129，选 4000 局 2415..3474 / 86480 局），perm 本节点位精确（48 worker 78s），漂移仅 42/4000（1%）。
- 生成 BC 格式 shard 122 个 / 1.01M transitions，qteach = one_hot(act)×6（τ=1 时 p_act=0.76 软标签）。
- `bc_learner_qwen.py` 新增 `--init-from`（只载权重）；`w6ei_qwen_t1.yaml`（λ_v=0：合成 one-hot 的 level 无意义；lr 1e-4/lora 3e-5 微调 400 步）。
- t1 环境就绪：Qwen3.5-0.8B 模型（12 文件 1.7G）+ 代码 + RL ckpt step1730（md5 核对）。
- `learner_qwen.py` 新增 anchor 热替换（每 25 步轮询 mtime，原子替换即升级教师）——为 EI 循环免重启升级锚点。
- 14:35 t1 八卡起跑 c1 蒸馏（init-from RL step1730）。

**torch 2.13 SDPA 根因确认（15:30）**：w6_var_xl 在 node_f 上 56% 的梯度步
非有限（NaN 保险丝 269 次拦截）。随机数据压测 400 步干净、真实数据 56%
中招——逐后端排查：配置 sdp_backends=[mem_efficient, math]（learner.py 新增，
torch 2.13 的 flash_sdp_enabled() 已变纯 getter，setter 是 enable_flash_sdp）
后 25 分钟**零新增 NaN**，gn 回 1.6-21.6 正常区间。根因：torch 2.13+cu130
在 H200 上 flash/cudnn SDPA bf16 后向对真实观测分布产出 inf 梯度——这同时
解释了 node_e 上 XL step69 的死亡（当时无保险丝），node_e 的病只有 GPU 3/7
context 失败是硬件，learner 之死是软件。t1（torch 2.6 + A100）不受影响，
不加此配置。教训：跨 torch 大版本迁移时，首个小时必须盯 NaN 拦截率。

### π 头部署发现 + v4（AlphaZero 结构）+ EI c2 软目标（10-01 16:30）

**π 头是更好的部署头（64 种子配对）**：
- RL gs=2000：**π-argmax mean 1941.9 / max 3065** vs Q-argmax 1724.3(gs1700) → 免费 +200。
- EI c1 蒸馏 ckpt：Q-argmax 921（one-hot 蒸馏毁 Q）但 **π-argmax 1719.7 ≈ 源策略** → c1 的 π 其实学成了。
- BC 时代征兆一直都在：agree_pi(0.36-0.52) 一贯 ≫ agree_q(0.21-0.38)，只是一直没人直接 eval π。
- 这正是用户 09-30 提的"独立 onehot 头，类似 AlphaZero"结构：π=actor，Q=critic。

**EI c1 失败诊断（硬 one-hot 标签）**：boost=6（p_act=0.76）+ 400 步 lr 1e-4 → qrank CE 梯度 dL/dz_act = q-p ≈ -0.75 每状态猛推 margin；对比度 ~13 的状态与边界之间被 A 头的跨状态泛化抹平（和 BC 时代的水平陷阱同构，方向相反）→ margin 通胀（train qd 1900→8800、qrank 6→12.5 vs 教师 1.59）→ Q-argmax 排序全毁。**π 头存活**因为 CE 对软 one-hot 的梯度有界、不涉及 dueling 分解。
- 修复 = `ei_dump_soft.py`：qteach = 策略自身居中 Q + β·(onehot−1/K)，β=2（保 dark knowledge，margin 增量有界）。

**v4 重启（16:30）**：init_from RL step2102（192 trainable keys，EI c1 弃用）；anchor_teacher.pt 种子=BC step7800（热替换文件就绪）；**actor_decode=pi + eval_decode=pi**（actor 按 π-argmax 行动、eval 量 π 策略）；eps 0.02-0.2。
**EI c2**：v3 全部 160 actor 的对局日志（86k 局）+ RL step2102 软打分在 t1 进行（replay 48 workers → gpu0 打分 ~1.1M 状态 ~60min → 8 卡蒸馏 ~46min → π/Q 双头 64 种子评测）。

### v4 假启动事故（16:45 修复）
- v4 首 eval gs=0 只有 732 → 发现 trainable 提取过滤器的 `.v_head.` 前导点不匹配顶层键 `v_head.0.weight` → 只载了 192 LoRA keys，**三个头全随机**。
- 重新提取 204 keys（含 12 头张量）→ v4 重启；t1 的 c2 软打分同样用了无头 ckpt（targets=噪声+onehot），杀掉重跑。
- ei_cycle.sh 同款 bug 已修。

### Retry 状态复核与 c2 安全续训（10-01 20:12）

- node_c SSH 短暂 connection closed 后恢复；确认 `/path/to` 挂载于持久 XFS 卷。c2 数据目录保留且通过逐 shard 检查：119 个连续编号 `.npz`，988,297 transitions，`obs=(N,800)`、`qteach=(N,128)`，全部数值有限；因此不重跑昂贵的 replay/scoring。
- 旧 dump 日志包含目录消失时写 `eis_00005.npz` 的 `FileNotFoundError`，但后续 119 shards 已完整落盘；当时无 dump writer。保留旧数据，未执行 `rm -rf`。
- c1 已完成 64-seed 对照：Q-head 920.7，π-head 1719.7（源策略约 1720）。两个常驻轮询评测进程已停止，避免继续占 GPU0；结果文件已存在。
- c2 在 node_c 八卡从 `ei_rl_ckpt/rl_latest_trainable.pt`（step 2102、204 trainable keys）重新启动，独立目录 `runs/w6ei_20261001/c2`；CUDA 8/8 可用、源代码和配置 sha256 与本机一致。仓库已有 `test_geometry.py` 与 `test_watermelon_rule.py` 均通过。step 50 loss=4.310/pi=2.567/qrank=3.486、step 100 loss=3.816/pi=2.344/qrank=2.944，训练正常推进；400 步及双头评测待完成。
- node_d v4 仍运行：截至 20:12，grad 544、env 4.73M，最近 eval gs500 mean 1750.6；清理了已完成但仍轮询的旧基线评测，GPU7 留给 v4 evaluator。下一次评测/训练进展待复核。
