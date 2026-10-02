# The GSPO contract (what we got wrong before you)

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
- **Clip fractions are more tunable than clip values.** When the clipped
  fraction drifts above budget, the adaptive controller
  (`ADAPT_CLIP_LOW_MAX_FRAC` / `ADAPT_CLIP_HIGH_MAX_FRAC`, see
  [`QUICKSTART.md`](QUICKSTART.md)) widens `eps` just enough to bring it back,
  up to `GSPO_EPS_MAX`. Track `gspo/eps_low` / `gspo/eps_high` and the
  advantage-gated `gspo/seq_active_clip_*_frac` metrics to see what the clip is
  actually doing to the gradient.

`tests/test_gspo_core.py` pins all of this: ratio, both normalizations, clip
engagement, gradient direction, and zero-gradient-outside-clip, against a naive
reference implementation.
