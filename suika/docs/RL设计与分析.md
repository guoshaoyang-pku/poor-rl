# 合成大西瓜 RL 设计与分析文档

> **维护约定**：本文档是唯一权威设计文档。每个重大决策、每波实验结论、每次方向讨论后更新对应章节。
> 实验流水账在 `suika_dqn/EXPERIMENT_LOG.md`，本文档只保留结论与依据。
> 最近更新：2026-09-28（wave3 发射：修复版 n-step + γ=1 + arch v2 set transformer + 每臂双卡推理服务；bit-exact 物理 fast 路径已上线；wave2 已终止；非对称 privilege 落实；D11 模型放大）

---

## 0. 项目目标与当前位置

- **目标**：在新规则（双西瓜合并消失 +66）的自有物理引擎上，训出远超所有已知 baseline 的策略，并将引擎运作与策略决策全程可视化。
- **当前最好成绩（内部交叉评测）**（2026-09-28）：eval mean **1654** / 单局 max **2849**（w2warm checkpoint × tempo 环境，32 种子）。这是 settle 训练出的 policy 在另一环境配置上的结果，不代表 tempo 已用于训练。
- **已知参照**（环境与适配状态见 §6）：
  - 随机策略约 ~700（旧记录）；mattjacobs30 旧适配结果 593–1036，已标记待重评；历史 beam 搜索教师 1844（旧规则）。

---

## 1. 游戏规则与计分

### 1.1 水果链与合并得分

11 级：樱桃(0) → 草莓(1) → 葡萄(2) → 橘子(3) → 柠檬(4) → 猕猴桃(5) → 番茄(6) → 桃子(7) → 菠萝(8) → 椰子(9) → 西瓜(10)。

| 合并 | 得分 | 累计（从樱桃建 1 个该果） |
|---|---|---|
| 樱桃+樱桃 | 1 | 1（草莓） |
| 草莓+草莓 | 3 | 5 |
| 葡萄+葡萄 | 6 | 16 |
| 橘子+橘子 | 10 | 42 |
| 柠檬+柠檬 | 15 | 99 |
| 猕猴桃+猕猴桃 | 21 | 219 |
| 番茄+番茄 | 28 | 466 |
| 桃子+桃子 | 36 | 968 |
| 菠萝+菠萝 | 45 | 1981 |
| 椰子+椰子 → 西瓜 | 55 | **4017** |
| 西瓜+西瓜 → 消失（poof） | **+66** | — |

- 掉落果子：代码中的 `PreParticle` 使用 iid Uniform{0..4}（只掉前 5 级）。
- 一个西瓜的理论成本：全樱桃建 = **4017 分**；全柠檬建 = **1329 分**。真实掉落下的期望成本取决于残果、合并顺序和策略，当前没有足够实验支持固定写成 ~1900 或 2200+。
- **双西瓜规则**（本引擎当前规则，`rules.two_watermelon: merge_disappear_score`）：两西瓜接触即共同消失 +66 分，等价官方设定；旧规则（不消失）保留为配置开关。

### 1.2 终止条件的两种模式

| | settle（我们原生） | tempo（参考 mattjacobs30 的节拍/终止设置，当前仅用于交叉评测） |
|---|---|---|
| 节拍 | 最多模拟 240 帧，或速度低于阈值 2 即提前给下一子 | 固定 120 帧（2s）/步，球滚着也能落子 |
| 死亡 | 任一果子**质心**越过 killy 线即死 | 整果**完全越过**线**持续 3 秒**才死 |
| 交叉评测影响 | — | 同一 settle-trained policy 在该整套配置下均值增加约 314（§6.2） |

棋盘几何、重力、摩擦、弹性目前按配置文件对照；tempo 还改变了每步推进和终止判定，因此不能把两种环境当成同一任务。

### 1.3 玩家可见信息

当前果 + 下一果（1 步前瞻）。下一果之后是 iid 随机，**对玩家不可知**——这一点决定了 §2.3 的合法边界。

---

## 2. 问题一：状态表示

### 2.1 现状（wave1/wave2 在用的方案）

333 维扁平向量 → MLP[1024,1024]（LayerNorm+SiLU）→ Dueling 双头：

- **13 维全局**：9 个基础量（当前 type、下一 type、当前半径、果子数、归一化分数、堆顶、平均高度、最大半径、填充率）及 4 个边界量（左/右间隙、相对 killy 的顶部余量、底部间隙与合法跨度的混合量）。
- **320 维盘面**：80 个果槽 × (x, y, radius, type)，按 (y,x) 排序、零填充。

**已知弱点**：按 `(y,x)` 规范排序对输入枚举顺序本身是置换不变的，但水果插入、删除或接近交换时，槽位对应关系会跳变，MLP 需要学习这种不稳定性。这是后续升级集合编码的动机（§2.4）。

### 2.2 当前 + 下一果的建模

已包含在全局特征里。当前实现显式存储 current type、next type、current radius；next radius 可由 next type 和配置表恢复。它覆盖了随机掉落对玩家可见的前瞻，但不等于 tempo 物理状态完整（速度、碰撞标志和越线计时器目前未编码）。

### 2.3 Q-net 要不要包含更长的未来果子序列？

**部署策略：不能看。** next 之后是 iid Uniform{0..4}，真实对局根本拿不到，看了就是作弊且无法部署。

但有两个**合法且高价值**的落地形态（即计划中的 Phase B）：

1. **特权 critic（训练期合法）**：模拟器握着 RNG，未来序列已知。训练特权 Q_priv(s, a, future_seq)，future_seq 过 transformer encoder。用途：
   - (a) 若 critic 的后续策略固定为同一个非特权策略，可对未来随机性求条件期望，作为方差降低的训练信号；
   - (b) 用严格定义的 privileged-vs-non-privileged 对照估计信息价值。若 privileged critic 的后续动作也能看见未来，直接平均会产生先知优势和 strategy fusion，不能当作可部署 Q 或严格上界。
2. **部署期边缘化（待验证）**：未来不可知时，只有在环境模型和后续策略的条件定义明确后，采样未来序列才可解释为 expectimax；简单平均特权 Q 不能自动得到可部署策略。

**特权序列的规格（用户 2026-09-28 拍板）**：未来 **50** 个果子，type embedding + **可学位置编码**（序列有序，与盘面的集合本质不同）→ 2 层小 transformer → 汇聚为 future vector，作为额外 token 注入架构 v2 的 Stage 2 cross-attn。实现用 `env.peek_future(50)`（fork 掉落 RNG 状态预生成再恢复，不污染真实序列）。假设待验：只有残局时最近几个未来果有用——位置编码赋予模型区分近/远的能力，训练后按局面密度可视化未来序列注意力即可检验。

**非对称 privilege 的落实（2026-09-28 与用户确认：critic 可全程特权，actor 全程非特权）**：硬约束是行动来自 argmax Q，所以部署 Q 网本身不能消费未来 token，否则部署时算不出动作——特权 critic 必须是**第二套独立参数的网络**，其输出永不参与行动选择。合法用途按优先级：(a) Phase B 测 oracle gap；(b) **动作蒸馏教师**：Q_priv 在训练环境用未来信息做更好的决策，以交叉熵蒸到非特权 π 头（DAgger 式，π 头即为此预留，D9）——动作蒸馏是监督学习，无 bootstrap 偏置；(c) 若未来引入 policy gradient，V_priv 当 baseline（不改变梯度期望，无偏）。**避免**：用特权网的 bootstrap 目标直接训非特权 Q（clairvoyant targets）——目标含 E[max(·|future)] ≥ 非特权可达价值，系统性乐观且按状态信息价值不等量膨胀，扭曲动作排序（与本节上文 strategy fusion 警告同源）。

### 2.4 架构 v2：Perceiver 式 set transformer（2026-09-28 定稿，wave4 实现）

memory bottleneck 由用户提出，对应文献中的 Perceiver latent array / Set Transformer 的 PMA（pooling by multihead attention）。四段结构：

```
果子 tokens（N 不限，实际 ~100-150；transformer 长度无关，取消 80 上限）
  每个 = type_emb(16d, 可学) + radius + Fourier(x,y) + [vx,vy] → Linear → d_tok=128
  特殊 token ×2：current 果 / next 果（type_emb + radius + 可学角色 emb）
        │  Stage 1：果子级 self-attn ×2（d=128, 8 头, pre-LN）
        ▼    ——邻接/消除通路是果子两两之间的成对几何（距离 vs r_i+r_j），
        │       必须先让果子互相"说话"再压缩（Set Transformer 的 SAB→PMA 结构）；
        │       cross-attn 只是"收集"，果子间不直接通信，堆 cross-attn 层数不能替代本段
        │  Stage 2：cross-attn ×1~2 轮（latent 侧可配 self-attn 迭代精炼）
        ▼    ——L=32 个可学 latent slot 当 query，果子 token 当 K/V；
memory latents = 32 × 256   注意力加权"收集"盘面信息，完成压缩
        │  Stage 3：latent self-attn ×4（d=256, 8 头）
        ▼    ——重计算只发生在 32 个 slot 上（32²=1k vs 150²=22.5k，省 ~20 倍）
读出（三头共享 backbone）：
  V 头：pool(latents) → MLP → 标量
  A 头：128 个可学列 query cross-attn → latents（+可选直连果子 tokens 的 skip path，
        防止 bottleneck 损失列级瞄准精度）→ 每列 advantage
  π 头：per-column 策略 logits（见下"策略头"）
```

- 所有 embedding（type / 角色 / 列 query / latent slot / 未来序列位置编码）全部可学习。
- cross-attn 的机制与成本：输出**只有 L 个 latent 向量**（果子向量是被消费的 K/V，不是输出）；映射 = 每个 slot 对全部果子的 softmax 注意力加权求和。QK^T 每头仅 32×N≈4.8k 项，参数 ~0.5M/层，前向成本可忽略。
- **tempo 模式必须带 (vx, vy)**：落子时球在滚动，盘面是动态的；不给速度 = 看静止照片猜视频。（与 §2.2 修订指出的"速度、碰撞标志和越线计时器未编码"一致，架构 v2 补齐。）
- bottleneck 可靠性：Perceiver 已在 5 万 token 图像输入上验证（压到 512 latents）；本任务 150→32 压缩比温和得多。已知风险 = 列级瞄准精细几何，由 A 头 skip path 兜底（做消融）。
- 总参数 ~8M，前向成本不高于现 MLP 方案。

**规模方向（用户 2026-09-28 拍板）**：cross-attn 压缩 + latent self-attn 加深的结构获认可；Stage 3 层数可增加（4 层起步，向 6-8 层消融），整体模型允许显著大于 ~8M 基线——GPU learner 侧余量 >5×，不构成约束。**真正约束是 actor 端 CPU 单样本推理**（每 env 步一次前向）：模型放大的前提是先做 §5.3 的 actor 批量推理，且每次放大消融需同步监控单核物理+推理吞吐。

**策略头 π（定案：保留）**：DQN 的最小结构只需 V/A 双头（策略=argmax Q），但保留独立 π 头的理由：① 路线图明确有蒸馏（特权 critic / 搜索教师 → 非特权策略），蒸馏的天然载体是 π 上的交叉熵（AlphaZero 式），远稳于回归 128 个 Q 值；② 软策略/熵正则扩展位；③ π 与 A 的分歧本身是训练信号质量的诊断探针；④ 成本≈0。一致性规则：行动一律 argmax Q（A 头）；π 头只承担辅助损失（蒸馏 + 可选向 argmax-Q 自对齐，防走样）。

**实施顺序（修订保留 + 更新）**：当前 `suika_dqn` 训练仍固定使用 settle `DQNEnv`，tempo 只接在交叉评测脚本，random_start 也未接入训练；且 n-step 后继 bug 修复前，旧 checkpoint 的结论需复验。顺序：先修 n-step 并重启验证 → 环境接线（tempo/random_start/Markov 状态补全）用 MLP 做单变量对照 → 架构 v2 → 特权 critic。未来序列 token 保留时间顺序（位置编码），盘面 token 无序（集合）。

---

## 3. 问题二：奖励设计

### 3.1 基础配方

每一步 reward = 该步分数增益 Δscore，用 γ 回传。γ<1 时优化的是折扣后的未来得分；γ=1 且从初始零分状态出发时，累计 Δscore 才严格等于终局总分。

### 3.2 死亡惩罚的定量分析（γ=0.999 下不需要显式惩罚）

用户论证，本文档确认并补充数字：

- γ=0.999 → 有效视野 1/(1−γ) = 1000 步；实际 episode 长度 **~150-200 步**（当前训练是 settle 模式）≪ 1000。
- 折扣衰减：γ^200 ≈ **0.82**——终局结果以 ≥82% 的强度回传到整局每一步。
- 死亡会截断当前状态之后的未来得分；其影响取决于该状态的剩余机会成本，不能用已经累计的整局 2000 分直接代替。
- 显式 −100 是否有益，需要在修正 n-step 实现后做配对 A/B；它可能提供额外终止信号，也可能改变对高堆状态的风险偏好。

**终稿（用户 2026-09-28 拍板）：纯 Δscore + γ=1，零 shaping**，取代此前 γ=0.999 暂定配方。γ=1 时死亡的机会成本（失去的全部未来得分）以权重 1.0 无衰减回传到每个历史状态——死亡惩罚自动内嵌，显式项纯属双重计费；平均步得分 ~9 分的量纲下 −100 只值 ~11 步，却会在濒死状态制造 10-100 倍于典型步奖励的 TD 尖峰，抢占 PER 优先级。濒死/残局信号由激进 random_start 在数据侧供给（D3）。γ=1 下 Q 值量纲不变（本来就是"剩余局分"估计，~1000-3000），lr 与网络无需调整。原 γ=0.999 配方的复验（n-step 修复后）仍按 §8 执行。

### 3.3 存活奖励与 shaping 的边界

- 对方 repo 的 +0.25/步存活在有限 200 步内，γ=0.99 时约为 21.65，γ=0.999 时约为 45.34；250 是无限时域上限。是否改变策略要和剩余得分及死亡概率一起实测，不能只凭无限时域数值判断。
- 若将来要做"死亡恐惧"项，可用**势函数 shaping** F = γΦ(s')−Φ(s)，Φ = −λ·danger(堆顶到 killy 线距离)。要保留 episodic 任务的策略不变性，还需处理终止状态的 potential（通常设 terminal Φ=0）；否则终点残项可能改变策略偏好。
- **我们的立场**：算力大，shaping 意义小——濒死局面的数据靠激进 random_start 直接供给（数据侧解决），比奖励补丁治根。纯 Δscore 为主线；势函数 shaping 最多留一个 A/B 臂验证，不抱预期。

---

## 4. 问题三：RL 算法选取与 baseline 设置

### 4.1 为什么选 DQN 家族

- **V(s) vs Q(s,a)**：V 只评状态，行动还需环境模型做一步推演；Q 把动作选择内化，argmax_a Q(s,a) 直接出策略（model-free）。128 列落点 → 一次前向 128 个分数取最大。
- **Dueling**：Q = V(s) + A(s,a) − mean_a A。盘面整体好坏（V）与各列优劣（A）解耦——大量状态与动作无关（如濒死），V 头独立学好，样本效率高。
- **Double DQN**：缓解 max 操作符的过估计偏差（用 online 选动作、target 估值），不会完全消除偏差。
- **n-step(3)**：在正确拼接 n 步后继的前提下，权衡较长回报与 bootstrap；当前旧 checkpoint 使用过错误后继，相关比较需复验。
- **PER**：按 TD error 优先级采样，死亡/西瓜等高信息量转移多练。
- **Distributional(QR-DQN)**：理论上对随机掉落建模回报分布——但当前 wave1 实测 **plain dueling > QR64**（1315 vs ~1100 且更快），已弃用；修正 n-step 后需复验。
- **暂不采用 PPO**：在当前离散动作、off-policy 数据管线和预算下先以 DQN 为主；PPO 的样本效率差异尚未做受控比较。
- **暂不采用 MCTS/AlphaZero**：当前真物理 rollout 成本较高，已有弱训练结果；这只支持当前实现/预算下的优先级判断。

### 4.2 随机策略/随机环境下如何设 baseline（重要）

**问题本质**：环境随机（掉落序列 iid）+ 训练期策略随机（ε-greedy）→ 单局分数是随机变量 X(θ, ω)。任何基于单局/单种子/单 max 的比较都是噪声。

**方法论（本项目采用的四层）**：

1. **固定种子套件 + CRN（共同随机数）**：评测可用同一组 seeds 对候选 policy 做配对比较。CRN 不会消除随机性，方差收益取决于策略得分相关性，不能预设一个数量级，也不是唯一合法方式；最终报告应另留 held-out seeds。
2. **统计口径**：报告 mean / median / p25 / p2000（≥2000 分占比）/ 西瓜率 / 存活步数。**看 p25 和 mean，不看 max**（max 的方差最大、最易被运气污染）；A/B 决策要求配对差分的 bootstrap CI 不跨 0。
3. **训练期与部署期分离**：训练 rollout 的分数带 ε 噪声，**不能**当 policy 水平；policy 水平只看 evaluator 的 greedy（ε=0）固定种子套件。
4. **绝对参照系**（同环境对打后的当前数值）：

| baseline | mean | 说明 |
|---|---|---|
| 随机策略 | ~700 | 地板 |
| mattjacobs30 DQN（旧适配结果，待重评） | 593（settle）/ 1036（tempo） | 原适配器存在观测、动作和初始状态差异，旧数字不作可靠同环境 baseline |
| **我们 w2warm** | **1352（settle）/ 1654（tempo）** | 当前冠军 |
| 历史 beam 搜索教师 | 1844（旧规则，搜索/步 ~1.5s） | 下一个要超的目标 |
| oracle 上界 | 未知 | Phase B 特权 critic 测量 |

**教训记录**：跨环境比分不可直接比较。mattjacobs30 的旧适配结果曾引发误判；适配器已修正，但需要重新运行并对齐初始状态、观测和规则后，才能报告 baseline 差异。

---

## 5. 问题四：RL infra 与吞吐优化

### 5.1 架构

```
actor ×100 (CPU, ε 几何分布 0.02-0.5, n-step 组装)
   │  .npz 分片（原子 rename，4096 转移/片）
   ▼
inbox/ → learner (GPU A100, batch 8192, PER, EMA target, bf16 AMP)
   │  policy.pt（原子 publish；当前每 200 grad 才触发，约 18–20s 一次）
   ▼
actor 轮询加载          evaluator（固定种子 greedy, 每 5min）
```

代码：`suika_dqn/`（learner.py / actor.py / evaluator.py / run_arm.py / aggregate.py），实验日志 `suika_dqn/EXPERIMENT_LOG.md`。

### 5.2 当前吞吐（实测）

| 层级 | 数字 |
|---|---|
| 单核物理+编码（settle） | 原始 ~245 → Python 快扫 ~450 → **C 扫描 ~552 步/s**（见 §5.3-1、§6.4） |
| 单臂（100 actor） | **~5.5k env 步/s**（fast 路径后预计 ~2x，待重启 wave3 实测） |
| 两节点 4 臂合计 | ~23k 步/s ≈ **20 亿步/天** |
| learner | ~10-11 grad/s（max_reuse=16 主动封顶，远未吃满 A100） |

瓶颈排序（§6.4 实测分解）：**Python 逐帧检查 > CPU 物理（pymunk 单线程）> 单样本 CPU 推理 > 文件分片传输 > GPU**。每帧 ~25µs 中 C 求解器仅 7.2µs（29%），16.7µs 是逐帧游戏结束/稳定判定的 Python 开销——已于 2026-09-28 用 bit-exact 单趟扫描消除大半。

### 5.3 优化路线（按 ROI）

1. **物理 fast 路径（已落地，两层）**：① bit-exact 单趟 Python 扫描（290→620 drops/s 环境口径）；② C 扩展无状态扫描（`part2/csettle.c`，97 行，函数指针注入 Chipmunk 入口、不链接）：C 每帧只算 `(vmax_sq, min_y)`，`min_y<killy`（game-over 必要条件）时才回调 Python `_check_game_over`——谓词逐位等价，常见帧零 Python 工作。端到端 232→**552 drops/s（2.38x）**。19.3 万帧 fuzz + 30 种子全路径等价验证。远端每个 venv 跑一次 `python part2/build_csettle.py`（无编译器自动回退到 ①，行为不变）。曾试注册表镜像 `has_collided` 进 C——合并回调会重置该标志且不总经过 kill、清表破坏探针链，脏项曾改变整局走向，已回滚；教训：不镜像可变 Python 状态进 C。运行中的臂重启 actor 后生效。
2. **tempo 节拍（待接入训练）**：固定 120 帧/步会改变任务动力学；当前 +310 只是在交叉评测中观察到的整套 tempo 终止/节拍配置差异，不能单独归因于节拍。
3. **actor 批量推理**：一个 actor 进程同步驱动 8 个 env，拼 batch 前向——CPU 上单样本 torch 前向的固定开销被摊薄 ~5-8 倍。（架构 v2 放大模型的前置条件，D11）
4. **物理 dt 45fps（备选，暂不采用）**：消融显示 1/45 分布级与 60fps 不可区分（30 种子、0 穿墙、+18% 吞吐），但属于动力学改变，需训练+评测同步切换并放大种子确认；1/30 有穿墙直接排除；**求解迭代数调低已被消融否决**（收益 +2~5%，it6/it4 出现速度尖峰与穿墙）。sleeping 与当前"合并回调内 kill/加 body"实现死锁，需先重构为 post-step callback。
5. **learner 侧**：PER 抽样 ~164ms/更新是当前最大 CPU 项（可考虑优先级分桶/分段树向量化）；GPU 利用率 <20% 不是瓶颈。
6. **传输**：共享内存队列替代文件分片——目前分片 IO 占比 <1%，不做。
7. **C++ 重写引擎**：Python 开销消除后，C 求解器仅 ~7µs/帧，重写收益上限有限；暂不做。

---

## 6. 实验结果库

### 6.1 wave1（2026-09-27/28，16 臂 recipe 扫荡，9h，30-51M 步/臂）

| 排名 | 臂 | eval mean | max | 西瓜 |
|---|---|---|---|---|
| 1 | plaindqn_g7（纯 Dueling，无 QR） | 1315 | 2354 | ✅ |
| 2 | gammahi_g4（γ=0.999） | 1230 | 2251 | ✅ |
| 3 | noper_g6（无 PER） | 1152 | 1644 | — |

结论（当前一次 sweep，且旧 n-step 实现待复验）：plain Dueling 在该预算下高于 QR-DQN64；γ0.999 臂高于 γ0.995 对照；lr 1e-3 晚期下降；K=64/128/256 未见明显差异。不能据此作一般算法结论。**里程碑：在不同规则/评测设置下出现了高于历史数值 1844 的单局分数，不能称严格跨规则纪录。**

### 6.2 交叉评测（2026-09-28，32 种子/组合，我们的物理+双西瓜规则）

| policy × 环境 | mean | median | max | p2000（≥2000分） |
|---|---|---|---|---|
| 我们 w1 × settle | 1308 | 1301 | 2274 | 3% |
| 我们 w1 × tempo | **1622** | 1601 | 2175 | 12% |
| 我们 w2warm × settle | 1352 | 1334 | 1950 | 0% |
| 我们 w2warm × tempo | **1654** | 1592 | **2849** | 16% |
| mattjacobs30 × settle（旧适配） | 593 | 590 | 938 | 0% |
| mattjacobs30 × tempo（旧适配） | 1036 | 1010 | 1791 | 0% |

结论：① 同一 settle-trained policy 在整套 tempo 节拍与终止配置下均值增加约 314；这是环境设置差异，不是已验证的训练提升；② 旧 mattjacobs30 数字来自未对齐适配，不能据此归因 random_start 或宣称领先；③ 跨环境比分必须谨慎解释。

这里的 `p2000` 不是西瓜率。旧 `xeval` 只统计终局仍存在的 type-10 水果，双西瓜 poof 后会漏记；按该口径，w1 settle、w1 tempo、w2warm tempo 各为 1/32，w2warm settle 为 0。

### 6.3 wave2（进行中）

4 臂 × 100 actor：w2_warm（暖启）/ w2_fresh / b16k / γ0.9995。~1.1h 时 warm 1252 / fresh 1238，均已出西瓜。监控中。

### 6.4 物理引擎消融（2026-09-28，`suika_dqn/phy_ablation.py`，30 种子固定动作脚本重放）

前提测量（本机单核，随机策略，平均 ~13 个活果）：每帧 ~25µs 墙钟 = C 求解器 7.2µs（29%）+ Python 逐帧检查 16.7µs（68%，两趟 `space.shapes` 重建 + 每果一次 `p.pos` numpy 分配 + `.velocity.length`）+ 其他 ~3%。

| 变体 | drops/s | 分数（30种子） | 数值稳定性 | 判定 |
|---|---|---|---|---|
| base（原实现） | 290 | 580±207 | 基线（静止重叠 ~17px 是 bias=1e-5 的软堆叠特性） | — |
| **fast 单趟扫描+ffi直读** | **620** | 580±207 | 与 base **逐位一致** | **已合入** |
| fast_it8/6/4 | 631/643/653 | 494/555/597 | it6/it4：2694 px/s 速度尖峰（基线 366）、1-2 次穿墙 | 弃 |
| fast_dt45（180帧） | 733 | 566±223 | 0 穿墙，重叠/速度与基线同级 | 备选（需放大种子+CRN） |
| fast_dt30（120帧） | 845 | 653±258 | **8/30 局水果整体穿出边界** | 弃 |
| fast_check3（每3帧查） | 771 | 475±169 | 无穿墙，但每步多沉 ~2 帧 → 堆更实、局更短（-11% 步数） | 弃（改变游戏） |
| sleeping 0.5s | 死锁 | — | `cpSpaceStep` 内死循环：合并回调内 kill/加 body 违反 step 内约束 | 弃（需 post-step 重构） |

要点：① "末态一样且数值稳定即可加速"的最优解不是改物理参数，而是删 Python 开销——bit-exact fast 路径拿到 2.14x（端到端 1.95x）；② 迭代数调低在 Python 开销移除后收益只剩 +2~5% 且有穿墙风险；③ 1/30 会穿墙，1/45 是"另一个游戏"；④ tempo_env 复活前需同样修复其逐帧扫描。

---

## 7. 决策记录（ADR）

| # | 决策 | 依据 | 状态 |
|---|---|---|---|
| D1 | 双西瓜规则 = merge_disappear_score（+66） | 官方规则，用户拍板 | ✅ 已实现+测试 |
| D2 | 节拍/终止换 tempo 模式 | 交叉评测显示环境差异；w3_tf_tempo 训练臂曾接入但被用户取消（卡空出）；训练问题再度推迟 | ⏳ 搁置 |
| D3 | random_start 激进课程（3-12 果, type 0-6, 含濒死盘面） | 作为后续数据课程假设；当前 `suika_dqn` 未实现 | ⏳ wave3.5 候选 |
| D4 | 奖励 = 纯 Δscore + **γ=1**，零 shaping | §3.2：死亡机会成本以权重 1.0 内嵌回传；用户终稿 | ✅ 定稿 |
| D5 | 物理引擎 = 我们的 part2（双西瓜规则），不重写 | 保真度优先；节拍/终止只是 step 循环参数 | ✅ |
| D6 | 部署策略不看未来序列；未来信息仅用于特权 critic | 真实规则 iid 不可知；作弊不可部署 | ✅ 共识 |
| D7 | 状态架构 = Perceiver 式 set transformer（§2.4：SAB×2 → PMA → latent SAB×4，三头读出） | 排序列表非置换不变；memory bottleneck 由用户提出 | ✅ 已实现，wave3 训练中 |
| D8 | 特权 critic 见未来 50 果 + 可学位置编码（§2.3） | 用户拍板；仅训练/评测期合法；非对称落实=第二网络+π 头蒸馏（§2.3 末段） | ✅ Phase B |
| D9 | 读出含独立策略头 π（蒸馏预留），行动仍 argmax Q | §2.4 策略头段 | ✅ 定稿 |
| D10 | n-step 后继 bug：bootstrap 误用 pend[0] 的后继（s_{t+1}）而非 s_{t+n}，奖励被 bootstrap 重复计入 | review 发现；本地已修复并提交（58b7409）；wave2 已于 15:55 终止，wave3 全部修复版 | ✅ 已解决 |
| D11 | 架构 v2 允许放大：Stage 3 加深（4→6-8 层消融）、总参数量可超 ~8M 基线；MLP 对照也加深（3×2048） | 用户拍板；GPU learner 余量 >5× 非约束，约束在 actor CPU 推理（transformer 臂已上 GPU 推理服务） | ✅ wave3 试验中 |
| D12 | 物理结算加速 = bit-exact 两层：单趟 Python 扫描 + C 扩展无状态扫描（`csettle.c`，必要条件门控回调 Python 判定）；迭代数调低与 1/30 被消融否决；1/45 与 sleeping 备而不用。注册表镜像 has_collided 方案因脏项风险回滚 | §6.4：C 路径端到端 232→552 drops/s（2.38x）；19.3 万帧 fuzz + 30 种子全路径等价；it6/it4 穿墙、dt30 8/30 穿墙、check3 改变游戏、sleep 死锁 | ✅ 已合入 |
| D13 | **Qwen3.5-0.8B LLM 直接 RL（w4 臂）**：文本序列化接口（格式规范 v1 冻结）+ 128-bin dueling 头 + LoRA r64 backbone 共训（用户拍板"backbone 共享也训练 Q net"）；nothink（不走文本生成路径）；直接 RL 纯净版（无 BC/教师播种），输出小数为接口层（col↔0.XXX 双射）；图像输入为下一阶段（该 ckpt 为 VL 架构，原生 ViT 后续直接用） | 用户拍板逐条：0.8B 而非 8B；128-bin 而非连续 SAC；先 nothink；直接 RL；system prompt 冻结一句；**预注册 go/no-go（按视图数）：2e7 views eval≥850 / 1e8 ≥1100 / 3e8 ≥1300，不达即停臂** | 🚀 attempt4 训练中（09-29 02:2x，perm gpu3/6 learner + gpu7 推理/评测；实放 2 DDP 而非原定 3——egodex 占 gpu4/5）；attempts 1-3 事故实录见 EXPERIMENT_LOG w4 节（JIT 停顿杀演员→prompt 桶化+预热+看门狗；OOM×2→micro 8 + 发现 HF 梯度检查点对 GDN 混合架构无效、实测 2.55MB/sample·token 激活）；实测 0.026 grad/s ≈ 2.7M views/天 |
| D14 | **batch 65536 + lr 6e-4 + ckpt 续训协议**（wave3b）：用户拍板大 batch 降 off-policy 度；learner 支持 resume（online/EMA/opt/计数器），reuse 闸门改为**样本抽取数**记账（grad_draws，与 batch 大小无关）；replay 不持久化，resume 时 inserts=env_steps 恢复闸门；ckpt 15min/保留 4 | 两个 resume bug 实录见 EXPERIMENT_LOG wave3b 节；新 ckpt 存 grad_draws+batch 精确恢复 | ✅ wave3b 运行中 |
| D15 | **tf_xl 56M 放大 + 行为策略改 EMA 发布**（wave3c，用户拍板"切更大更深 tf、修全部已知 bug、预算一天、8 卡吃满"）：d_lat512/n_lat64/SAB16 55.88M 参数（6.8x），fresh 从零训；learner 独占 gpu0 + evaluator gpu1 + 推理 gpu2-7；梯度检查点使 micro 批 4096 可行；**publish 发 EMA 而非 online**（行为策略=目标网，消除 rollout 全局振荡）；learner 启动即播种 policy.pt + run_arm 启动闸门（杜绝随机权重出招事故）；train_step 去逐 micro 批 CPU 同步；batch 32768+lr 4.5e-4+grad_budget 6500 对应 24h 余弦（64k 经基准测试证伪：learner-bound 时 draws/天恒定，batch 减半 opt 步数翻倍） | tf_deep 收官 peak 2008@239M；振荡诊断见 EXPERIMENT_LOG（2min 桶均值 std tf=210 vs mlp=139，200 actor 同步下探）；grad_ckpt 闭包晚绑定 bug 实录同处 | 🚀 wave3c 启动 2026-09-29 00:34 |
| D16 | **BC 预热改用教师软目标 + regret 主指标**：π 头 L_pi = CE(softmax((q_T−mean_a q_T)/τ=1.0), log_softmax(π))，覆盖全部转移（不再只 91% argmax 样本）；门 = q_regret<10（常数策略基线 30.4）且 BC 后 eval ≥ 1500；agree_pi/agree_q 降为参考 | 实测：教师 top1−top2 间隔中位 1.0 Q 单位（Q~1096、f16 ULP=1.0），硬 argmax 不可达——v1 三 run 的 l_pi 停在动作边际熵 3.787、agree_pi ≈ best-constant 0.075；τ 扫描 τ=1.0 → 目标熵 1.59 nats（top5 质量 0.80）；regret 基线：常数策略 30.4、未训练网络 29.96 | 🚀 v2 训练中（t1_2 8 卡，09-29 16:0x 从 step400 温启；v2.1 = step600 起加 Q 向量软排序蒸馏 bc_qrank_tau=1.0——step600 实测 val_qd 53 但 q_regret 28.8≈常数基线、eval 523.9≈随机带，raw-Q 梯度 ~80% 花在状态水平上） |
| D17 | **BC 目标改"纯 listwise + 水平分离"，禁用一切原始尺度点式回归**：λ_pi=1.0（soft-KD τ=1，目标熵 floor 1.59）+ λ_qrank=0.5（listwise，走 Q 头）+ λ_v=1.0（水平/1000）；λ_qd=λ_td=λ_adv=0；点式 L_adv 若启用必须**按状态标准化**（固定除数=量级主导陷阱：全局 RMS 37 vs 排序值 2.6%）。副产物 1.8× 提速（3.3s/step） | 实测三探针：隐层最优线性读出 23.45 ≈ 原始输入 ridge 22.53（主干未学排序）；mirror aug 无 bug（迁移 23.4 自洽）；零预测器 0.962 vs 完美形状 0.937（点式项 2.6% 余量）→ LM 与普通 MLP 同停 adv≈0.94 | ✅ **v3.1 达标**（t1_2，09-29 18:0x→22:17，step5200/12000）：q_regret 28.7→**7.75（破 <10 门）**、val_soft_ce 4.08→2.87（floor 1.64）、agree_pi 0.09→**0.363**、eval 16 种子 572→**1500–1698**（≥1500 门达标，p2000 0→0.375）；22:17 外部清场中断（非本会话），为 SFT 让出 t1_2，step5200 ckpt 可随时续跑 |
| D18 | **SFT token-policy 设计（用户 2026-09-29 规格；policy 即 LM 的 token 概率，不设独立动作头）**：把教师 Q 归一化成动作字符串上的分布——规范化答案 `f"{(col+0.5)/128:.3f}"` 恰分解为 5 个单 token（探针 `probe_tokenizer_actions.py` 验证：3 个 digit 分布 = 128 动作分布的精确分解），soft 目标 softmax((q_T−mean)/τ) 可精确压到 token 上；格式 token（'0'、'.'、EOS）+ 全词表硬 CE 保解析；value head 共享主干回归教师 Q 水平/1000；π 头/Q 头保留辅助损失（DDP find_unused 需要）；**"后面的内容完全用 PPO 训练"**。~1h budget，warm start = v3.1 BC ckpt | 本会话 AZ 讨论结论并入：① 独立 one-hot 头本身不产生改进算子——硬标签因教师 top1−top2 gap ≈1 Q 单位不可达（v1/v2/v3.1 post-mortem）；② 把 policy 放到 token 概率上 = 用 LM 原生接口做 policy，天然适配 PPO 且免额外头；③ 若将来要"搜索式改进算子"（AZ 式 1-ply + V），可直接对同一 token 分布做，无需改架构。`sft_qwen.py` docstring 明确引用本会话 post-mortem 与 mirror 审计结论 | ⏳ 待跑（`sft_qwen.py`/`run_sft.sh` 22:23 就位，目标 t1_2 8 卡；t1_2 已腾空） |
| 已决 | **BC 门无需放宽——LM 实际破了 <10 门（7.75）**：普通 3 层 MLP 参照（同数据同目标、原始 800 维输入、1.5M 参数、400 epoch/3.2M 样本）只到 regret 18.45 / agree 0.127（常数 31.1）→ "<10" 对 MLP 难、对 LM+文本接口可达，兼作"文本接口 + 全量数据"价值的旁证 | v3.1 step5200 val；`probe_mlp_ceiling.py` 400 epoch 终值 | ✅ |
| 待决 | n-step 修复版上复验 wave1 关键结论（plain>QR、γ 方向、lr 崩溃） | §8 | ⏳ |

---

## 8. 开放问题

1. **n-step 修复版的复验（wave3 进行中）**：wave1/wave2 全部训练使用错误后继（D10）；wave3（修复版 + γ=1 + arch v2）即复验载体，plain>QR、γ 方向等旧结论按新结果改写。
2. 在修正 n-step、适配器并保留 held-out seeds 后，oracle gap 到底多大（Phase B）；它只提供信息价值线索，不自动决定搜索是否值得。
3. 终局仍存在西瓜的比例目前约为 0–3.1%；双西瓜消失会被现有终局统计漏掉，需要记录 `ever_watermelon` 后再讨论提升目标。
4. 架构 v2 的消融：memory slot 数（32 起）、Stage 1 果子级 self-attn 层数、A 头 skip path 的必要性。
5. 新规则下双西瓜 poof 的最优利用（+66 且清场——可能是超高分的关键机制）。
6. 可视化产品：物理回放 + Q/注意力热图 + 训练曲线 dashboard（引擎已具备渲染能力）。
