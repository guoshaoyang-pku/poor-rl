# Prefix-sharing numerics gate

`gate_v3_prefix.py` compares two paths on real GRPO groups, using production numerics (FA3, bf16, the trainer's model construction):
- the shared-prompt forward (`prefix_share.py`);
- the production per-sequence forward.

Rerun it whenever you change the model, `prefix_share.py`, or the torch / transformers / FLA / FA3 versions. The launcher only accepts the `prefix_share.py` md5 that passed this gate (`PREFIX_SHARE_MD5`).

Criteria (all must pass):
1. Sequence-level GSPO log-ratio, shared vs. clean: |mean| < 2e-4 and p99 < 1e-3.
2. Mean per-token |Δlogp|, reported next to the vLLM-vs-trainer gap.
3. Gradient check: cosine(grad_shared, grad_clean) > 0.99, with the norm ratio within 2%.
4. Ablation: detaching the prompt state must clearly fail (3).
5. Per-position bias: bucket completions by their position within the group. No monotone drift, and every bucket |mean| < 3e-4.
6. Leakage: replace another completion of the same group with random tokens. The probed completion's logprobs must not change.

There is also a 2-rank DDP check (equal micro-batch counts, no hang) and a speed and memory measurement for one G=32 group.

`results/` holds the 2026-10-03 run on 360-2. Two versions were tested:
- `prefix_share` a679 used SDPA for the prefix attention. It failed (1) with p99 1.75e-3 and (5) with a bucket max of 6.5e-4.
- Patch `prefix_share_a679_to_ede53a03.patch` switches to the model's FA3 and aligns the shared prefix to the 64-token FLA chunk. With it the result is bit-exact (log-ratio 0.0, grad cosine 0.99996, leakage 18/18) and 2.10× faster per group (`gate_speed_patch.json`).

Paths default to the reference deployment. Override them with `GATE_CKPT`, `GATE_BASE_MODEL`, `GATE_WORKDIR` and `GATE_TRAIN_JSONL`. The group IDs in `GROUPS` refer to that deployment's probe data.
