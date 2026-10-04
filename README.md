# PoorRL

**High-performance reinforcement learning on a small GPU budget.**

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
| 🔶 | Adaptive clip-fraction caps | Widens `eps` up to `GSPO_EPS_MAX` when the clipped-sequence fraction exceeds budget; math + tests in `src/rlforge/gspo.py` / `tests/test_gspo_core.py`. Not yet wired into the v3.2 production trainer (fixed eps there) |
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
| 🔶 | Native FP8 training + rollout | FP32 masters/gradients/Adam; shared E4M3 CUTLASS forward, E5M2 gradient GEMMs with FP32 output. Optional exact conv compilation improves the aligned FP8 batch 1.435x; it still trails fused BF16. Four prefill completions pass alignment, one unique decode case fails. Total RL speed and long-run quality remain unvalidated ([`docs/RECIPE_FP8.md`](docs/RECIPE_FP8.md)) |
| ✅ | fp8 KV cache | 1.82x KV capacity, accuracy-neutral in paired eval. **Not the 0.8B RL SOP**: at 0.8B the KV is never the bottleneck (7-51% used), fp8 KV buys no throughput and adds logprob mismatch, so production runs use `KV_DTYPE=auto` (bf16) ([`docs/SOP_0.8B.md`](docs/SOP_0.8B.md)) |
| ✅ | Production v3.2 trainer path | Merged into `src/rlforge/` ([`docs/RECIPE_H200.md`](docs/RECIPE_H200.md)): prefix sharing with token-budget sub-batches, fused fast_logprob (5.4x fwd), rank load balancing, non-blocking judge scorer, stale-drop audit, group-affinity routing. v3.2 gate: dlog-ratio p99 <= 1.1e-7, grad cos >= 0.99994; 65 -> 37 s/step. Byte-identical v3.1 as-run snapshot in [`production/v3_1/`](production/v3_1/) |
| ✅ | A100 27B LoRA e2e track | FSDP2/HSDP trainer + sm80 prefix share + multi-backend rollout router + adapter-only LoRA sync ([`docs/RECIPE_A100.md`](docs/RECIPE_A100.md)). CPU e2e verified; GPU smoke pending |
| ✅ | Cross-host rollout | One vLLM DP server across two nodes (head + `--headless`), weight sync over IB 0.145 s/step for 0.8B, mandatory watchdog ([`tools/cross_node/INTEGRATION.md`](tools/cross_node/INTEGRATION.md)) |
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

| | PoorRL | TRL stock | verl |
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
| [`docs/RECIPE_H200.md`](docs/RECIPE_H200.md) | H200 (sm90) recipe: the production v3.2 stack, launch, measured speedups and low-precision boundaries |
| [`docs/RECIPE_FP8.md`](docs/RECIPE_FP8.md) | Experimental native FP8 full FT + rollout: FP32 master/gradient/optimizer contract, launch, fixed-batch performance and prefill alignment limits |
| [`docs/RECIPE_A100.md`](docs/RECIPE_A100.md) | A100 (sm80) recipe: 27B LoRA e2e track — FSDP2/HSDP trainer, sm80 prefix share, rollout router, adapter-only sync |
| [`docs/INFRA_HANDOFF.md`](docs/INFRA_HANDOFF.md) | 当前 RL infra 交接：架构、v2→v3 瓶颈链与实测提速、踩过的坑（症状→根因→修复→证据）、27B 经验、下一批杠杆 |
| [`docs/SOP_0.8B.md`](docs/SOP_0.8B.md) | 0.8B 异步 GSPO 运行 SOP：布局、参数、在途/队列定量、judge 定量、预检、冒烟、监控、停止、从 checkpoint 重启 |
| [`production/v3_1/README.md`](production/v3_1/README.md) | The production code snapshot: what each file does, md5s, how to overlay it, why it is not merged yet |
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | Install, single-node quickstart, adaptive clip caps, custom rewards |
| [`docs/SFT.md`](docs/SFT.md) | Qwen native-thinking SFT, checkpoint/resume guardrails, recovery smoke, held-out checkpoint ladder SOP |
| [`docs/AGENTIC.md`](docs/AGENTIC.md) | Agentic ReAct + MCP harness, SkillBank prototype, current limitations |
| [`docs/GSPO.md`](docs/GSPO.md) | The GSPO contract — normalization, clip calibration, what we got wrong |
| [`docs/PITFALLS.md`](docs/PITFALLS.md) | Six production failure modes this framework guards against |
| [`docs/PACKING.md`](docs/PACKING.md) | Padding-free packing on hybrid GatedDeltaNet backbones: the silent cross-sample leakage, its measured size, and the boundary-aware fix |
| [`docs/PRECISION.md`](docs/PRECISION.md) | bf16/fp32-master recipes, fp8 path, measured train/rollout logprob gap |
| [`docs/LORA.md`](docs/LORA.md) | LoRA recipe (fp32 adapter on bf16 base), adapter-only sync, measured FP8 counterpart |
| [`docs/PANEL.md`](docs/PANEL.md) | SwanLab local tracking, unified Home, RL-specific reports |
| [`docs/PANEL_LINK.md`](docs/PANEL_LINK.md) | 面板联动与协作查看：镜像守护、样本查看器、GPU 状态、协作者接入指南 |
| [`docs/rl_trials.md`](docs/rl_trials.md) | Algorithm-neutral RL trials panel, bundled Suika history |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Where this is going and what is explicitly out of scope |
| [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md) | Backbone checklist for plugging in external small-RL projects |

## Repo layout

```
src/rlforge/trainer.py    GSPOAsyncGRPOTrainer + CLI (python -m rlforge.trainer), v3.2 throughput path
src/rlforge/gspo.py       naive reference math + adaptive clip controller
src/rlforge/prefix_share.py + fast_logprob.py + dp_route.py + score_loop.py + drop_audit.py  H200 production infra
src/rlforge/fp8.py + fp8_serving.py + fp8_alignment.py  experimental native FP8 trainer/rollout
src/rlforge/prefix_share_sm80.py + rollout/  A100 track: sm80 prefix share, multi-backend rollout router
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
