# Pitfalls this framework guards against

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
