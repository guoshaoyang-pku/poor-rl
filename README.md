# rlforge

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)
![Scope](https://img.shields.io/badge/scope-single--node%20%C2%B7%20small%20models-green)

Async **GSPO/GRPO** RL post-training for small generative models (LLM/VLM), built on
[TRL](https://github.com/huggingface/trl)'s experimental `AsyncGRPOTrainer` + a
vLLM rollout server — plus the guardrails you need to run experiments unattended
on a single GPU node, fully offline.

**Niche: small models, one machine, well optimized.** Developed and validated on an
8xH200 node training a 0.8B model for 1500 steps: held-out accuracy 0.11 (base) →
0.81, with every failure mode listed below hit (and fixed) in production first.

## Contents

- [Why not just TRL / verl?](#why-not-just-trl--verl)
- [Install](#install)
- [Quickstart](#quickstart-single-node-8-gpus)
- [Custom rewards](#custom-rewards)
- [Agentic RL harness](#agentic-rl-harness-preview)
- [The GSPO contract](#the-gspo-contract-what-we-got-wrong-before-you)
- [Precision recipes](#precision-recipes)
- [The RL panel](#the-rl-panel)
- [Pitfalls this framework guards against](#pitfalls-this-framework-guards-against)
- [Repo layout](#repo-layout)
- [Docs](#docs)

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

## Install

```bash
pip install -e ".[panel]"
```

## Quickstart (single node, 8 GPUs)

```bash
export ROOT=/path/to/project   # will hold data/ logs/ runs/ evals/
export VENV=/path/to/venv
export MODEL=/path/to/base/model
export RLFORGE_BASE_MODEL=$MODEL

# train.jsonl rows: {"prompt": [...chat...] or "...", "answer": "C" | "A<B<C<D<E", "source": "..."}
ROOT=$ROOT VENV=$VENV MODEL=$MODEL \
GSPO=1 GSPO_EPS_LOW=0.007 GSPO_EPS_HIGH=0.008 MAX_STEPS=1500 \
REPORT_TO=swanlab \
bash scripts/run_async_dp.sh full _run1 &

# watchdog: in-loop held-out eval + keep-best + auto-stop
python -m rlforge.watchdog --run $ROOT/runs/async_dp_run1 \
    --trainer-log $ROOT/logs/trainer_dp_run1.log --launcher-pid $! \
    --eval-gpu 7 --eval-data data/eval.jsonl &

# panel 1 (general): SwanLab comparison dashboard
ROOT=$ROOT VENV=$VENV bash scripts/panel.sh  # http://127.0.0.1:5092

# panel 2 (RL-specific): auto HTML report every 10 min
python -m rlforge.report --run $ROOT/runs/async_dp_run1 \
    --trainer-log $ROOT/logs/trainer_dp_run1.log \
    --eval-history $ROOT/evals/async_dp_run1/history.jsonl \
    --out $ROOT/runs/async_dp_run1/report
```

## Custom rewards

Any `module:function` with the TRL signature works:

```bash
python -m rlforge.trainer --reward my_project.rewards:my_fn ...
```

See [`examples/custom_reward/arith.py`](examples/custom_reward/arith.py) for a
minimal annotated example (truncation detection, task-log counters, dataset-column
pass-through). The built-in grader (`rlforge.rewards.mcq`) scores MCQ letters and
5-choice ranking chains, and writes per-task / per-source reward/accuracy/
truncation counters to `$RLFORGE_TASK_LOG` — the report turns them into curves.
Eval and training import the same grader code, so numbers are comparable by
construction.

## Agentic RL harness (preview)

The first agent harness now follows the Qwen3.5-friendly **text ReAct + MCP** path:
the model emits a bounded `Action` / `Action Input`, rlforge calls the registered MCP
server, and returns an `Observation` for the next ReAct step. It does not send OpenAI
`tools` schemas on this default path. Local KB lookup and arithmetic remain built-in
tools. MCP uses the official [Python SDK](https://github.com/modelcontextprotocol/python-sdk)
and accepts the stdio `mcpServers` config shape shown by [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent).
MCP server entries launch local processes, so use only trusted configs. A legacy
`--protocol openai-tools` mode remains available for endpoints that expose native tool calls.

```bash
pip install -e '.[agent-mcp]'
```

MCP config example (`mcp.json`):

```json
{
  "mcpServers": {
    "time": {
      "command": "uvx",
      "args": ["mcp-server-time", "--local-timezone=Asia/Shanghai"]
    }
  }
}
```

Run against a Qwen3.5 OpenAI-compatible endpoint (non-thinking defaults match the
model-card text-task recipe; thinking is opt-in because 0.8B can loop):

```bash
export RLFORGE_API_KEY=EMPTY
rlforge-agentic --data data/train.jsonl --model "Qwen/Qwen3.5-0.8B" \
  --base-url http://localhost:8000/v1 --mcp-config mcp.json \
  --import-kb /path/to/kb_swarm.json --kb runs/agentic/knowledge.sqlite3 \
  --out runs/agentic/episodes.jsonl --split train --num-generations 8 \
  --max-tool-calls 4 --max-rounds 6 --max-tool-result-chars 4000 \
  --max-context-tokens 24576 --max-completion-tokens 16384
```

To turn on the persistent file SkillBank prototype, create files under `general/`,
`task_specific/<family>/`, and `common_mistakes/`, then pass `--memory-files`:

```bash
rlforge-agentic --data data/train.jsonl --model "Qwen/Qwen3.5-0.8B" \
  --base-url http://localhost:8000/v1 --mcp-config mcp.json \
  --memory-files runs/agentic/skillbank --memory-evidence-db runs/agentic/skillbank_evidence.sqlite3 \
  --kb runs/agentic/knowledge.sqlite3 --out runs/agentic/episodes.jsonl \
  --split train --num-generations 8 --min-skill-support 2
```

The model can list, read, create/edit, and rename `.md` files by filename. Each
rollout gets an isolated copy; an edit reaches the shared SkillBank only after a
successful training reward and matching content from at least two distinct task/pair
keys. Held-out splits never receive pending failure candidates and never promote edits.
The tool is confined to Markdown inside the SkillBank, enforces path/symlink/size limits,
requires the last-read hash for edits, and uses conflict checks before promotion. Changes,
rewards, provenance keys, promotion decisions, and before/after bank hashes are logged
per episode. Task-specific retrieval uses `task_specific/<family>/`; `general/` and
`common_mistakes/` are cross-task categories.

This is a **SkillRL-inspired prototype**, not a reproduction of the paper: it has
lexical top-k retrieval, per-episode file proposals, reward/support-gated promotion,
and retrieval of relevant unvalidated failed edits as candidates. It does not yet
implement an LLM skill-distillation/evolution job, embedding retrieval, category-level
validation-accuracy triggers, or automatic rewriting/pruning of existing skills. A
proper evolution controller should consume train/dev failures only and keep final
held-out evaluation sealed.

The harness keeps a provenance-aware SQLite knowledge base, applies terminal-loss
rewards, logs bounded episode traces and context estimates, and can export group-relative
terminal advantages. Evaluation episodes cannot write to the KB; learned claims require
successful training outcomes and support from distinct task/pair provenance. Context
budgets are harness-side gates, not model context extensions or measurements of actual
KV-cache allocation. FP8 KV can reduce cache memory but does not reduce the token cost of
ReAct observations. Historical ArchitectureIQ KB v4 JSON (`{"claims":[...]}`) imports
directly; its curation/leakage audit remains the experiment owner's responsibility.

This is a tested agent/tool orchestration harness, **not yet a model-training path**:
`gspo_advantage` is an auditable rollout signal, not a trainer input. It does not compute
actor token log-probabilities or update weights. GSPO training integration must preserve
the full ReAct action/observation token trajectory and align its terminal reward before
it can train the policy; do not treat the current API-generated episodes as an actual
GSPO run.


## The GSPO contract (what we got wrong before you)

- **Normalization matters more than the ratio.** Averaging the loss over the
  global token count weights every sequence by its length; combined with a `-2`
  truncation penalty, the longest (truncated) sequences dominate the gradient and
  the policy collapsed ~12x faster than token-level GRPO at the same lr. Use
  `seq_mean` (uniform per sequence) — it's also the paper's objective.
- **Token-level clip values are a no-op at sequence level.** eps=0.2/0.28 never
  engages on a per-sequence ratio. The paper's 3e-4/4e-4 is right for on-policy;
  with staleness >1, calibrate empirically (we run 0.007/0.008 at staleness 3)
  and watch `gspo/seq_clip_low_frac` — sustained ≥0.5 is the collapse signature.
- **Off-policy depth pushes ρ below 1 systematically.** With staleness ≥2 the
  low-side clip does real work; that's the safe direction, but read it together
  with the held-out curve, not alone.

`tests/test_gspo_core.py` pins all of this: ratio, both normalizations, clip
engagement, gradient direction, and zero-gradient-outside-clip, against a naive
reference implementation.

## Precision recipes

RL updates are tiny; in pure bf16 they can be rounded away entirely. The launcher
exposes both knobs (`DTYPE` × `MIXED_PRECISION`); for low-precision work use
**fp32 master weights + bf16 compute** (`DTYPE=none MIXED_PRECISION=bf16`).
Full rationale and verification procedure: [`docs/PRECISION.md`](docs/PRECISION.md).

## The RL panel

SwanLab is the default local experiment tracker (`REPORT_TO=swanlab`) for RL,
SFT, and custom PyTorch runs; `scripts/panel.sh` opens its local dashboard. The
built-in HTML report provides RL-specific breakdowns generic trackers cannot
infer (per-task reward vs accuracy vs truncation, held-out ladder, GSPO health).
Details: [`docs/PANEL.md`](docs/PANEL.md).

## Pitfalls this framework guards against

1. **TRL's 120s request timeout** kills slow-but-fine long completions forever →
   set `--request-timeout` for the worst case (we use 3600).
2. **Reward parsers that test one separator** (`">" in gold`) silently mis-score
   every ranking answer when the key format changes → shape-based detection.
3. **Truncated completions scored as "unparseable"** → the reward sees token
   counts, and `-2` actually fires.
4. **Eval numbers without generation conditions** → the eval report records
   decoding params, truncation rate, per-task breakdown and a constant baseline.
5. **save_total_limit deleting your peak** → watchdog copies the best checkpoint
   to `keep_best/` the moment a new best eval lands.
6. **Missing tokenizer files in checkpoints** → the evaluator builds a shim from
   the base model (never weights).

## Repo layout

```
src/rlforge/trainer.py    GSPOAsyncGRPOTrainer + CLI (python -m rlforge.trainer)
src/rlforge/gspo.py       naive reference math for the GSPO loss
src/rlforge/rewards/      reward protocol + built-in MCQ/ranking grader
src/rlforge/agentic.py    bounded ReAct episodes + group-relative terminal rewards
src/rlforge/knowledge.py  provenance-aware persistent cross-task KB
src/rlforge/agent_tools.py + agent_policy.py  tool contracts + OpenAI-compatible policy
src/rlforge/agent_cli.py  API-backed agent harness CLI
src/rlforge/eval_mcq.py   held-out evaluator (vLLM), shared scoring path
src/rlforge/watchdog.py   in-loop eval / keep-best / auto-stop
src/rlforge/report.py     auto HTML report (RL-specific panel)
scripts/run_async_dp.sh   single-node launcher (vLLM server + DP trainer)
scripts/panel.sh          SwanLab local dashboard over tracked runs
tests/                    GSPO core math + reward/parser tests
examples/aiq_mcq/         data format + a runnable example config
examples/custom_reward/   minimal annotated custom reward
examples/accelerate/      FSDP single-node config (for >3B full-parameter)
docs/                     ROADMAP / PRECISION / PANEL / COMPATIBILITY
```

## Docs

- [`docs/ROADMAP.md`](docs/ROADMAP.md) — where this is going (low-precision,
  KV cache, FSDP, VLA) and what is explicitly out of scope.
- [`docs/PRECISION.md`](docs/PRECISION.md) — bf16/fp32-master recipes, fp8 path.
- [`docs/PANEL.md`](docs/PANEL.md) — SwanLab local tracking plus RL-specific reports.
- [`docs/rl_trials.md`](docs/rl_trials.md) — algorithm-neutral RL trials, Suika history import, and the localhost panel.
- [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md) — backbone checklist and notes
  for plugging in external small-RL projects (incl. Qwen3.5-0.8B game-RL arms).

## Citation

GSPO: [arXiv 2507.18071](https://arxiv.org/abs/2507.18071). Built on TRL and vLLM.

## License

Apache-2.0
