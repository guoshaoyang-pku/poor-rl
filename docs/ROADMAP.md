# Roadmap

rlforge's niche: **small generative models (<= a few B), one node, fast iteration**.
Everything below is scoped by that -- we are not rebuilding verl's multi-node stack.

## In flight

- **Precision recipes** (`docs/PRECISION.md`): fp32 master weights + bf16 compute
  (`MIXED_PRECISION=bf16 DTYPE=none`) is wired into the launcher; bf16-pure remains
  the fast default. FP8 rollout weights (`ROLLOUT_QUANTIZATION=fp8`) and FP8 KV are
  both validated on H200, including against a fp32 LoRA adapter (see below); the
  measured sequence-level ratio deviation stays within ~1-3% of 1.0.
- **LoRA / adapter training** (`docs/LORA.md`): `LORA=1` freezes the base, trains a
  fp32 adapter on a bf16 base, and syncs adapter-only each step (25.6 MB vs 1.75 GB,
  0.12-0.19 s/step). Validated end to end at 0.8B with FP8 rollout. This is the
  intended shape for the 27B stage, where rollout capacity and policy-sync cost, not
  parameter count, are the binding constraints.
- **Packed-forward correctness for hybrid models** (highest-priority open infra
  item): the async trainer packs a rank's sequences into one row with per-sequence
  `position_ids`. That is exact for pure transformers, but Qwen3.5's
  `linear_attention` (GatedDeltaNet) layers carry conv/recurrent state that does not
  reset on a `position_ids` boundary, so packed logprobs drift by -0.10 to -0.62 per
  token depending on pack position (measured; `docs/LORA.md`). This -- not FP8 and
  not bf16 -- is what drives `gspo/seq_clip_low_frac` toward 1.0. Fix by restoring
  boundaries for hybrid architectures (`cu_seqlens` plus `causal_conv1d`, or one
  sequence per forward).
- **Experiment tracking** (`docs/PANEL.md`): SwanLab local mode is the default
  cross-workload tracker for RL, SFT, and custom PyTorch runs; `scripts/panel.sh`
  starts its local dashboard. The built-in HTML report provides RL-specific
  curves; RL-Insight is an optional verl runtime-observability companion.
- **KV-cache serving**: H200 FP8 KV is operationally validated and the launcher
  default (`KV_DTYPE=fp8`); use `KV_DTYPE=auto` for compatibility or diagnosis.
  Native CPU KV offload is also validated as a capacity-oriented SOP for
  long-context/high-concurrency serving. In the tested vLLM 0.30 setup, a 16-GiB
  CPU offload tier completed 512/512 requests without a discernible throughput
  penalty. Validate backend and load on each target; this is not a universal
  no-cost guarantee.
- **CPU weight offload**: FSDP2 `fsdp_offload_params=true` is implemented and
  completed a 5-step training smoke with parameter updates. It remains opt-in
  and disabled by default: the measured steady-step median rose from 4.57 s to
  7.49 s (+64%), while observed trainer peak memory fell from 25.2 to 18.6 GiB.
  Use only when memory capacity is worth the transfer overhead; the smoke is
  not a long-run stability qualification.

## Planned

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
