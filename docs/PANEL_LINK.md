# Panel 联动与协作查看指南

本文档说明 rlforge 本地面板体系的组成、它们如何与训练集群联动，以及
**协作者如何在自己的机器上搭出同一套实时面板**——只要有集群 SSH 访问权即可，
不需要在集群上部署任何东西。

## 架构总览

所有服务都跑在本地（笔记本/工作站），绑定 `127.0.0.1`，数据通过 SSH 从训练集群拉取：

```
训练集群 (360-2, 经 360-1 跳板)                 本地机器
┌─────────────────────────────┐        ssh       ┌────────────────────────────────┐
│ runs/<run_id>/              │ ◄────────────── │ scripts/panel_mirror.py --loop │
│   run.json, checkpoints     │   每 5 分钟      │   发现新 run → 抽取 metrics    │
│   rollout_samples.jsonl     │   rollout 样本   │   → 重导 SwanLab → 同步样本    │
│ logs/trainer_dp_*.log       │ ◄────────────── │ scripts/panel_mirror_remote.py │
│ evals/*_eval300_*.json      │   评测摘要/全量  │   (自动推送到远端 /tmp)        │
└─────────────────────────────┘                  └───────┬────────────────────────┘
                                                         │ 写入
              ┌──────────────────────────────────────────┼───────────────┐
              ▼                                          ▼               ▼
   artifacts/swanlab/<PROJECT>/swanlog        artifacts/samples/   artifacts/gpu_status.json
              ▼                                          ▼               ▼
   SwanLab panels (swanlab watch)             scripts/home.py (hub, 统一入口)
   63401 AIQ · 63402 Suika · 63403 SFT        63400: 面板卡片 + Rollout 样本查看器
                                              + GPU/内存占用与故障标记
```

| 端口 | 服务 | 启动方式 | 数据 |
|---|---|---|---|
| 63400 | 主站 hub | `cd scripts && python3 home.py` | 动态页 + `artifacts/samples/` + `artifacts/gpu_status.json` |
| 63401 | AIQ 面板（RL 实验） | `PROJECT=AIQ PORT=63401 bash scripts/panel.sh` | `artifacts/swanlab/AIQ/swanlog` |
| 63402 | Suika 面板 | `PROJECT=Suika PORT=63402 bash scripts/panel.sh` | `artifacts/swanlab/Suika/swanlog` |
| 63403 | SFT 面板 | `PROJECT=SFT PORT=63403 bash scripts/panel.sh` | `artifacts/swanlab/SFT/swanlog` |
| — | 镜像守护 | `python3 scripts/panel_mirror.py --loop 300` | 远端 → SwanLab + samples |
| — | 保活守护 | `bash scripts/panel_watchdog.sh` | 自动重启上述全部 |

## 协作者快速开始

前置条件：

1. **SSH 访问权**：`~/.ssh/config` 里有 `360-1` 别名（免密），且从 360-1 能
   `ssh 10.234.161.3`（360-2）。验证：
   `ssh 360-1 'ssh 10.234.161.3 hostname'` 能返回即通。
   如果训练机/跳板不同，用环境变量覆盖（见下表），不要求别名一致。
2. **本地环境**：Python ≥3.10、tmux、克隆本 repo。
3. **SwanLab venv**（只装一次）：
   ```bash
   python3 -m venv /tmp/rlforge-swanlab-preview-venv
   /tmp/rlforge-swanlab-preview-venv/bin/pip install -e ".[panel]"
   ```
   venv 路径不同没关系，用 `RLFORGE_VENV_PY=/path/to/venv/bin/python` 覆盖。

启动（每条命令一个终端，或全部交给 watchdog，见下一节）：

```bash
cd /path/to/rlforge

# 1. 主站
cd scripts && python3 home.py &

# 2. 三个 SwanLab 面板
ROOT=$PWD/.. PROJECT=AIQ   PORT=63401 VENV=/tmp/rlforge-swanlab-preview-venv bash scripts/panel.sh &
ROOT=$PWD/.. PROJECT=Suika PORT=63402 VENV=/tmp/rlforge-swanlab-preview-venv bash scripts/panel.sh &
ROOT=$PWD/.. PROJECT=SFT   PORT=63403 VENV=/tmp/rlforge-swanlab-preview-venv bash scripts/panel.sh &

# 3. 镜像守护（核心：每 5 分钟从集群拉新实验/新评测/新样本）
/tmp/rlforge-swanlab-preview-venv/bin/python scripts/panel_mirror.py --loop 300
```

打开 http://127.0.0.1:63400 即可看到所有面板入口、GPU 状态区和 Rollout 样本卡片。
首轮镜像会把集群上所有匹配 run（含历史 run 的全部阶梯评测）拉回本地，
之后只增量更新有变化的 run。

**一行懒人模式**：直接 `nohup bash scripts/panel_watchdog.sh &`，它会每 60 秒
检查主站/三面板/镜像守护，挂了就自动拉起（日志 `/tmp/panel_watchdog.log`、
`/tmp/panel_mirror.log`、`/tmp/panel_<project>.log`）。

## 联动机制说明

**实验自动出现在面板**：`panel_mirror.py` 每个周期在训练机上执行
`RLFORGE_RUN_GLOB`（默认匹配 `opus_corr.*step<N>` 的 run），对每个 run 调用
远端的 `panel_mirror_remote.py`（首次启动时自动从 `scripts/` 推送）收集
metrics / run.json / 评测摘要 / 存活状态，回本地后与缓存比对，**只有内容变化
才重导**（`import_experiments_swanlab.py --replace`，面板不闪烁）。

**全局步数对齐**：续训 run（`step400_from_ckpt200`、`step600_from_ckpt400`…）
的训练曲线按名字里的 `step<K>` 自动偏移 `max(0, K-200)`，评测按文件名里的
`_global<N>_` 落到全局步数。同一条实验线（step200→400→600）在面板里首尾相接。
新命名规律变化时改 `step_offset()` 一处即可。

**面板 → 样本页直达**：SwanLab 页面右下角有「☰ Rollout 样本」悬浮按钮
（`inject_home_link.py` 注入，`panel.sh` 启动时自动执行）。在 run 详情页，
按钮每 3 秒检测当前 experiment 名，唯一命中即变成「本 run Rollout 样本」，
直达 `http://127.0.0.1:63400/samples/<run_id>`；否则链到主站样本区。
依赖主站的 `/api/samples_runs`（已开 CORS）。

**样本查看器**：`/samples/<run_id>` 渲染训练 rollout 抽样（120 秒窗口、
每窗最多 10 条、低 reward 优先，含 task/source/n_tokens/截断）与每组评测的
10 条逐题样本（含完整思维链/回答、reward、parse/截断标记）。镜像守护负责：
拉 `rollout_samples.jsonl` 全量 → `artifacts/samples/<run>/train.jsonl`；
发现新评测 JSON → 拉全量 records 并按 `Random(42+step)` 抽 10 题 →
`eval_ckpt<N>.jsonl`（确定性抽样，重复跑结果一致）。

**GPU 状态**：`gpu_status.py` 每 60 秒轮询 7 个集群（360-1/360-2/ophis-gpu/
a100_perm/a100_t1/a100_t1_2/a100_t1_3）的 per-GPU util/mem/进程归属，写入
`artifacts/gpu_status.json`；主站 `/api/gpu_status` 提供，主页 30 秒自刷。
掉卡/ECC 不可纠正/节点不可达会滞留红色故障标记（带首次/最近时间与复发计数），
`/api/gpu_faults/clear` 消除；误清后复发会自动重挂。`private: true` 的集群
（A100 系列）不标「他人占用」。集群列表改 `gpu_status.py` 顶部 `CLUSTERS`。

## 配置与环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `RLFORGE_SSH_JUMP` | `360-1` | 跳板机 ssh 别名 |
| `RLFORGE_TRAIN_HOST` | `10.234.161.3` | 训练机（从跳板可达） |
| `RLFORGE_RUN_GLOB` | 匹配 opus_corr step run | 远端 run 发现命令（stdout 每行一个 run_id） |
| `RLFORGE_VENV_PY` | `/tmp/rlforge-swanlab-preview-venv/bin/python` | 带 swanlab 的 venv 解释器 |

排除不想展示的 run（失败/重复实验）：编辑 `artifacts/panel_mirror.json`
（不存在则用内置默认）：

```json
{"exclude": ["async_dp_..._failed_...", "async_dp_..._old_canary_..."]}
```

被排除的 run 不再镜像；若已进面板，用一次性命令删除：

```bash
PYTHONPATH=src:scripts /tmp/rlforge-swanlab-preview-venv/bin/python -c "
from pathlib import Path
from import_experiments_swanlab import delete_run
print(delete_run(Path('artifacts/swanlab/AIQ/swanlog'), 'AIQ', '<run_id>'))"
```

## 数据落盘位置（全部 gitignored）

```
artifacts/
  swanlab/{AIQ,Suika,SFT}/swanlog/   # SwanLab 离线库（面板数据源）
  samples/<run_id>/{meta,train,eval_ckpt<N>}.jsonl   # 样本查看器数据
  panel_mirror/<run_id>.json         # 镜像缓存（判定"有无变化"用）
  panel_mirror/history.json          # 最近一次导入的 history
  gpu_status.json                    # GPU 状态快照
```

## 常见问题

- **SSH 抖动**：360-1/360-2 链路偶尔超时，镜像守护会整轮报错跳过、下轮自动恢复；
  本地缓存保证面板不掉数据。远端 helper 推送失败会 fallback 用远端已有副本。
- **面板不更新**：看 `/tmp/panel_mirror.log`——`no changes` 是正常（远端没新数据）；
  `import FAILED` 才需要处理。确认 SwanLab 面板进程活着（watchdog 会自动救）。
- **新 run 不出现**：确认 run 名匹配 `RLFORGE_RUN_GLOB`，且不在 exclude 列表；
  手动触发一轮：`python3 scripts/panel_mirror.py --once`。
- **端口冲突**：四个端口（63400-63403）都只绑 localhost；换端口直接改启动参数，
  主站卡片链接在 `dashboard_home.html` 顶部 `CARDS` 常量。

相关文档：[`PANEL.md`](PANEL.md)（SwanLab 基础用法与 HTML report）。
