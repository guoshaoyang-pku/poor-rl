# Model & project compatibility

## Backbones

Any HF causal LM with a chat template works; validated so far:

| backbone | size | notes |
|---|---|---|
| Qwen3.5-0.8B (thinking) | 0.8B | the reference model. `--max-model-len 24576`; keep prompt+completion within it |
| Qwen3.5-0.8B-ms (merged/specialized) | 0.8B | the AIQ runs; same template |
| other Qwen / Llama-style | <=3B | pass `--no-thinking` if the template has no `enable_thinking` flag |

Checklist for a new backbone:

1. Chat template exists (`tokenizer.chat_template is not None`).
2. Context: prompt_max + `--max-completion` <= `--max-model-len` (edit the
   launcher's `--max-model-len` if the model's context differs).
3. If the model is a *base* (non-instruct) model, expect low initial parse rates;
   the reward's `-0.5` unparseable path is designed for exactly this phase.
4. bf16 weights load fine on the rollout server; the trainer-side precision is a
   separate choice (`docs/PRECISION.md`).

## Using rlforge with the Suika (合成大西瓜) Qwen 3.5 0.8B backbone

That project trains the same 0.8B class backbone two ways:

- **its own DQN/AlphaZero arms** (`suika_dqn/learner_qwen.py` etc.): value-based,
  discrete-action, replay-buffer RL. This is deliberately **not** what rlforge
  reimplements (scope: generative RL). The reusable pieces from rlforge there are
  the watchdog pattern (auto-stop rules over a live run), the panel
  (SwanLab experiment tracking), and the run-manifest contract.
- **GRPO/GSPO on the same backbone** (e.g. reasoning arms on game traces rendered
  as text): fully supported -- point `MODEL` at the backbone directory, pass
  `--no-thinking` if the checkpoint's template lacks the thinking flag, and use a
  task reward like `examples/custom_reward/arith.py` as the template.

## Other small-RL projects

The framework assumptions are deliberately thin:

- dataset = jsonl with `prompt` (+ task-specific gold column),
- reward = one importable function,
- everything else (launcher env, watchdog, panel, report) is task-agnostic.

If your task's rollouts need an environment (games, tools, simulators), keep the
environment outside the trainer: generate prompts/responses against the vLLM
server yourself, or produce a static jsonl offline and train on it. A proper
async environment-adapter layer is on the roadmap (`docs/ROADMAP.md`, VLA item).
