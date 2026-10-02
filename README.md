# rlforge

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)
![Scope](https://img.shields.io/badge/scope-single--node%20%C2%B7%20small%20models-green)

Async **GSPO/GRPO** RL post-training for small generative models (LLM/VLM), built on
[TRL](https://github.com/huggingface/trl)'s experimental `AsyncGRPOTrainer` + a
vLLM rollout server — plus the guardrails you need to run experiments unattended
on a single GPU node, fully offline.

> **Niche: small models, one machine, well optimized.** Developed and validated on an
> 8xH200 node training a 0.8B model for 1500 steps: held-out accuracy 0.11 (base) →
> 0.81, with every failure mode in [`docs/PITFALLS.md`](docs/PITFALLS.md) hit (and
> fixed) in production first.

## Feature status

| Status | Feature | Notes |
|---|---|---|
| **Algorithm** | | |
| ✅ | Async GSPO/GRPO training | TRL `AsyncGRPOTrainer` + vLLM rollout server, static GPU split (e.g. 4+4), staleness up to 3 validated |
| ✅ | Sequence-level IS with `seq_mean` normalization | The GSPO paper's objective; ratio/normalization/clip math pinned by `tests/test_gspo_core.py` |
| ✅ | Adaptive clip-fraction caps | Widens `eps` up to `GSPO_EPS_MAX` when the clipped-sequence fraction exceeds budget; `gspo/eps_*` and advantage-gated `gspo/seq_active_clip_*` metrics |
| ✅ | Length-aware truncation penalty | Reward sees token counts; `-2` on cap hit actually fires |
| ✅ | Custom reward protocol | Any `module:function`; built-in MCQ/ranking grader with per-task/source counters |
| 🔶 | Agentic ReAct + MCP harness (preview) | Bounded episodes, provenance KB, SkillBank prototype; rollout signal only, not yet a training path ([`docs/AGENTIC.md`](docs/AGENTIC.md)) |
| 🗺️ | Agentic GSPO training integration | Full ReAct action/observation trajectory into the trainer, aligned terminal reward |
| 🗺️ | Skill evolution controller | LLM skill distillation/evolution, embedding retrieval, automatic rewriting/pruning |
| 🗺️ | Async depth tuning | Map staleness vs collapse risk on more tasks; one-step-off-policy middle point |
| 🗺️ | More algorithms | DPO/PPO/KTO come from TRL; reward/watchdog/panel layers are algorithm-agnostic |
| 🗺️ | VLM/VLA | Image-input GRPO exists in TRL; missing piece is the closed-loop rollout adapter |
| **Infra** | | |
| 🔶 | Full-parameter SFT path | Qwen native-thinking, strict data/template checks, resumable Trainer checkpoints; GPU recovery smoke still required before unattended launch |
| ✅ | Precision stack | Default: **fp32 master weights + bf16 training compute + bf16 rollout inference** ([`docs/PRECISION.md`](docs/PRECISION.md)) |
| ✅ | fp8 KV cache (H200 default) | 1.82x KV capacity, accuracy-neutral in paired eval; `KV_DTYPE=auto` to opt out |
| ✅ | Watchdog for unattended runs | In-loop held-out eval, keep-best checkpoint copy, 5 auto-stop rules |
| ✅ | LoRA / adapter training | `LORA=1`: bf16 frozen base + fp32 adapter, adapter-only policy sync (25.6 MB vs 1.75 GB, 0.12-0.19 s/step) ([`docs/LORA.md`](docs/LORA.md)) |
| 🔶 | fp8 rollout weights | `ROLLOUT_QUANTIZATION=fp8` measured against a BF16 control with a fp32 LoRA adapter: sequence-level ratio deviation ~1-3%; long-run quality unproven |
| 🗺️ | FSDP path for >3B | Validated single-node config in `examples/accelerate/`; LoRA arms stay on DDP |
| **Panel** | | |
| ✅ | Experiment tracking & panels | SwanLab local mode by default, unified local Home, RL-specific auto HTML report |
| ✅ | Algorithm-neutral RL trials dashboard | Serves DQN/Suika trial history alongside LLM RL; per-run notes, 37-trial bundled export |

✅ implemented · 🔶 preview/partial · 🗺️ planned — see [`docs/ROADMAP.md`](docs/ROADMAP.md)
for scope and explicitly out-of-scope items.

## Why not just TRL / verl?

| | rlforge | TRL stock | verl |
|---|---|---|---|
| GSPO sequence-level IS | yes, with `seq_mean` normalization | sync only, token-mean normalization | yes (set `loss_agg_mode=seq-mean-token-mean`) |
| Async rollout (staleness) | yes, static GPU split (e.g. 4+4) | yes (experimental) | colocate or separate_async |
| Reward truncation signal | length-aware (`-2` on cap hit) | n/a | reward managers don't pass lengths by default |
| Unattended runs | watchdog (5 stop rules) + in-loop held-out eval + keep-best | no | no |
| Run tracking | SwanLab (local default) + RL-specific HTML report | HF integrations | configurable logger backends |
| Scale | single node, full-DP, ~0.5–3B (FSDP config for beyond) | single/multi node | multi-node, FSDP, 8B–70B+ |

Scope: RL post-training of generative models at single-node scale. For classic
control (DQN etc.) use Stable-Baselines3/CleanRL; for ≥8B or multi-node use verl.

## Documentation

| Doc | Contents |
|---|---|
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | Install, single-node quickstart, adaptive clip caps, custom rewards |
| [`docs/SFT.md`](docs/SFT.md) | Qwen native-thinking SFT, checkpoint/resume guardrails, recovery smoke, held-out checkpoint ladder SOP |
| [`docs/AGENTIC.md`](docs/AGENTIC.md) | Agentic ReAct + MCP harness, SkillBank prototype, current limitations |
| [`docs/GSPO.md`](docs/GSPO.md) | The GSPO contract — normalization, clip calibration, what we got wrong |
| [`docs/PITFALLS.md`](docs/PITFALLS.md) | Six production failure modes this framework guards against |
| [`docs/PRECISION.md`](docs/PRECISION.md) | bf16/fp32-master recipes, fp8 path, measured train/rollout logprob gap |
| [`docs/LORA.md`](docs/LORA.md) | LoRA recipe (fp32 adapter on bf16 base), adapter-only sync, measured FP8 counterpart |
| [`docs/PANEL.md`](docs/PANEL.md) | SwanLab local tracking, unified Home, RL-specific reports |
| [`docs/PANEL_LINK.md`](docs/PANEL_LINK.md) | 面板联动与协作查看：镜像守护、样本查看器、GPU 状态、协作者接入指南 |
| [`docs/rl_trials.md`](docs/rl_trials.md) | Algorithm-neutral RL trials panel, bundled Suika history |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Where this is going and what is explicitly out of scope |
| [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md) | Backbone checklist for plugging in external small-RL projects |

## Repo layout

```
src/rlforge/trainer.py    GSPOAsyncGRPOTrainer + CLI (python -m rlforge.trainer)
src/rlforge/gspo.py       naive reference math + adaptive clip controller
src/rlforge/rewards/      reward protocol + built-in MCQ/ranking grader
src/rlforge/agentic.py    bounded ReAct episodes + group-relative terminal rewards
src/rlforge/knowledge.py  provenance-aware persistent cross-task KB
src/rlforge/agent_tools.py + agent_policy.py  tool contracts + OpenAI-compatible policy
src/rlforge/agent_cli.py  API-backed agent harness CLI
src/rlforge/eval_mcq.py   held-out evaluator (vLLM), shared scoring path
src/rlforge/watchdog.py   in-loop eval / keep-best / auto-stop
src/rlforge/report.py     auto HTML report (RL-specific panel)
scripts/run_async_dp.sh   single-node launcher (vLLM server + DP trainer)
scripts/sft/              Qwen native-thinking SFT, recovery smoke, held-out checkpoint ladder
scripts/panel.sh          SwanLab local dashboard over tracked runs
scripts/home.py           unified local Home for all panels
tests/                    GSPO core math, reward/parser, SFT recovery guardrail tests
examples/aiq_mcq/         data format + a runnable example config
examples/custom_reward/   minimal annotated custom reward
examples/suika_trials/    bundled 37-trial DQN history for the trials panel
examples/accelerate/      FSDP single-node config (for >3B full-parameter)
docs/                     quickstart / GSPO / agentic / precision / panel / roadmap
```

## Citation

GSPO: [arXiv 2507.18071](https://arxiv.org/abs/2507.18071). Built on TRL and vLLM.

## License

Apache-2.0
