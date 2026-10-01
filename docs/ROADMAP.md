# Roadmap

rlforge's niche: **small generative models (<= a few B), one node, fast iteration**.
Everything below is scoped by that -- we are not rebuilding verl's multi-node stack.

## In flight

- **Precision recipes** (`docs/PRECISION.md`): fp32 master weights + bf16 compute
  (`MIXED_PRECISION=bf16 DTYPE=none`) is wired into the launcher; bf16-pure remains
  the fast default. Next: fp8 experiments on Hopper once the reward signal justifies
  the engineering (watch update-to-weight magnitude ratios).
- **Experiment tracking** (`docs/PANEL.md`): SwanLab local mode is the default
  cross-workload tracker for RL, SFT, and custom PyTorch runs; `scripts/panel.sh`
  starts its local dashboard. The built-in HTML report provides RL-specific
  curves; RL-Insight is an optional verl runtime-observability companion.

## Planned

- **KV-cache optimization**: `KV_DTYPE=fp8` is exposed in the launcher; measure its
  effect on logprob drift between sampler and trainer (fp8 KV changes sampled
  logprobs slightly -> importance-ratio shift -> more clipping). Quantify before
  making it the default.
- **Async depth tuning**: staleness=3 works at 8xH200/0.8B. Systematically map
  staleness vs collapse risk (`gspo/seq_clip_low_frac`) on more tasks; consider
  one-step-off-policy (verl-style) as the middle point.
- **FSDP path** (`examples/accelerate/fsdp_single_node.yaml`): validated config for
  the day we train >3B full-parameter on one node. LoRA arms stay on plain DDP.
- **More small-RL algorithms**: the trainer is GRPO/GSPO today. TRL gives DPO/PPO/
  KTO for free; the pluggable reward + watchdog + panel layers are algorithm-
  agnostic, so adding an algorithm is a config-level change, not new infra.
- **VLM/VLA**: TRL's GRPO supports image inputs; the missing piece is an
  environment/rollout adapter for VLA-style closed loops. Candidate design:
  reuse the vLLM-server split and let a simulator process drive prompts.

## Explicitly out of scope

- Multi-node training (use verl).
- Classic control value-based RL (DQN/AlphaZero -- see the note in
  `docs/COMPATIBILITY.md` on how such projects reuse rlforge's watchdog/panel).
- Replacing established experiment trackers with a bespoke general-purpose UI.
