# LoRA (PEFT adapter) training

Adapter training for the case where full fine-tuning costs more than it buys: a
27B-class policy where gradients, optimizer state and per-step weight sync are all
competing with rollout for the same H200s. LoRA keeps the base frozen, trains a
small fp32 adapter, and lets the rollout server pull only that adapter each step.

At 0.8B the trade-off does not pay: full FT's optimizer state is affordable and
LoRA does not save forward FLOPs. The recipe below is aimed at the 27B stage, and
is validated end to end at 0.8B on 360-1 (H200, vLLM 0.30).

## Launcher

```bash
LORA=1 LORA_R=16 LORA_ALPHA=32 \
ROLLOUT_QUANTIZATION=fp8 KV_DTYPE=fp8 \
DTYPE=bfloat16 bash scripts/run_async_dp.sh full <tag>
```

| env | default | meaning |
|---|---|---|
| `LORA` | `0` | `1` = train a PEFT adapter, base frozen |
| `LORA_R` | `16` | adapter rank |
| `LORA_ALPHA` | `0` | `0` = `2 x r` |
| `LORA_DROPOUT` | `0.0` | adapter dropout |
| `LORA_TARGET_MODULES` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | comma-separated suffixes |

`LORA=1` adds `--enable-lora --max-lora-rank <r> --max-loras <staleness+2>` on the
server side and `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`, then hands `--lora ...` to the
trainer. Nothing else in the launch path changes: the same launcher, panel and
watchdog run, and `run.json` records the LoRA config next to the precision fields.

## Precision: base bf16, adapter fp32

`DTYPE=bfloat16` loads the frozen base in bf16 and PEFT keeps the adapter itself in
**fp32** (`get_peft_model` casts lora_A/lora_B to fp32 when the base is bf16), so
the adapter gets the fp32-master property that matters at lr=2e-6: a 2e-6 update
lands instead of being rounded away at the bf16 mantissa floor. The trained
adapter is saved in fp32 (`adapter_model.safetensors`, 25.6 MB at r=16 on
Qwen3.5-0.8B) and cast to bf16 by vLLM at load time (`lora_dtype=torch.bfloat16`).

That asymmetry (adapter computed in fp32, served in bf16) is the one place where
the trainer and the rollout see different numbers, so it is measured rather than
assumed -- see below.

## What is verified (360-1, 2 rollout + 2 trainer H200, 20 steps)

- **Training runs.** GSPO + LoRA + FP8 weights + FP8 KV completed 20/20 steps;
  `grad_norm` 0.09-0.25 and a moving loss show the adapter is actually learning
  (a fresh adapter starts at `lora_B = 0`).
- **Sync is adapter-only.** Each policy version is written to
  `<out>/.vllm_lora/trl-policy-v<N>/` (25.6 MB) and loaded by the server over
  `load_lora_adapter`; `perf/weight_sync_s` was **0.12-0.19 s/step** against 1.75 GB
  of full weights for the same model (68x fewer bytes). Stale versions are pruned
  to `max_staleness + 2` slots, matching `--max-loras`.
- **Step cost.** 4.1-4.5 s/step at 4096-token completions, `fwd_bwd` 3.7-4.0 s,
  `mfu_fwd_bwd` 16-18%%.
- **Final adapter round-trips.** `trainer.save_model(<out>/final)` writes a
  reloadable PEFT directory; `PeftModel.from_pretrained` + `AutoModelFor...`
  reloads it and reproduces the trained logprobs.

## Train/rollout logprob gap with FP8 rollout

Same adapter, same token ids, scored three ways: vLLM server (three precision
configs) vs the trainer-side model. `delta = logp_trainer - logp_rollout` over 5
fixed sequences / 1044 continuation tokens.

| rollout config | per-token `abs_mean` | sequence-level `abs_mean` | sequence rho range |
|---|---|---|---|
| BF16 weights + BF16 KV | 0.0104 | 0.0019 | 0.9955 - 1.0013 |
| FP8 weights + BF16 KV | 0.0482 | 0.0087 | 0.9923 - 1.0271 |
| FP8 weights + FP8 KV | 0.0533 | 0.0091 | 0.9936 - 1.0230 |

Read this as: FP8 rollout multiplies per-token logprob noise by ~5x, but the
**sequence-level** ratio (what GSPO clips on) stays within about 1-3% of 1.0, and
the adapter's own contribution to the gap is ~1e-3 (measured directly, against a
2.7e-5 adapter weight delta). FP8 weights cost slightly more than FP8 KV, and the
two are close to additive. An independent check agrees: vLLM's generation-time
logprobs and its prefill-scoring logprobs differ by -0.016/token (rho 0.984) on the
same server and ids, i.e. the plumbing noise floor is already comparable to the
whole FP8 contribution.

Practical consequence for the 27B plan: FP8 rollout + fp32 LoRA adapter is a sound
default from the precision side. It is not the thing to blame when ratio/clip looks
unhealthy -- see the next section.

## Known limits

- **Single adapter.** TRL's adapter-only sync serves one adapter; a PEFT config
  with `modules_to_save`, `use_dora`, `bias != "none"`, `target_parameters`, or
  multiple active adapters is rejected at startup, deliberately, because vLLM could
  not reproduce that policy. `target_modules` must not include `lm_head` or
  `embed_tokens`; the launcher rejects those up front.
- **`--max-lora-rank` must be one of vLLM's stacked ranks** (1, 8, 16, 32, 64, 128,
  256, 320, 512) and at least `LORA_R`; the launcher rounds up.
- **Long-run adapter stability is not yet validated** -- 20 steps is a plumbing
  smoke, not a learning result. Reward in that smoke was degenerate (-1.85..-2.0)
  because the base model rambles past a 4096-token cap (99% of completions
  truncated); that is a `MAX_COMPLETION` config choice, not a LoRA defect.
- **LoRA costs ~15% on rollout** (9,376 -> 7,954 tok/s at 256 concurrency, p95
  6.92 -> 8.17 s) and **0.74x on the trainer** at 0.8B. Both are acceptable only
  because the system is rollout-bound (a 1.54x rollout deficit), so trainer slack
  hides the trainer tax; the payoff is sync bytes and memory headroom, which matter
  at 27B rather than at 0.8B.
- **An FP4 (MXFP4/Marlin) base works with LoRA on Hopper.** Verified on 360-1:
  kernel stays `MarlinMxFp4LinearKernel`, the adapter registers as a servable model,
  and a synthetic adapter changed 3/3 test prompts. See
  `docs/PRECISION.md` ("FP4 on H200") for the invocation and the full serving
  matrix. Note the base is weight-only A16, so no activation-side FP4 speedup.

## Do not pack a hybrid model's sequences into one forward

The async trainer packs every sequence on a rank into a single row with per-
sequence-reset `position_ids`. For a pure transformer that is exact: transformers
derives flash-attention `cu_seqlens` from those resets. Qwen3.5-0.8B is a **hybrid**
(`layer_types` = 3x `linear_attention` + 1x `full_attention`), and its
GatedDeltaNet layers carry a causal conv + recurrent state that does not reset on a
`position_ids` boundary. Measured with `attn_implementation=kernels-community/flash-attn3`:

| pack position | per-token `mean(packed - clean)` |
|---|---|
| 0 (first) | -0.0986 |
| 1 | -0.2254 |
| 2 | -0.4350 |
| 3 | -0.5935 |
| 4 (last) | -0.6202 |

Same prompt repeated 5x (the shape a training group actually has): -0.099, -0.169,
-0.263, -0.317, -0.292. And the cleanest proof that this is cross-sequence bleed
rather than a boundary artifact: scoring one fixed target sequence after three
different unrelated prefixes gives -0.105 / -0.437 / -0.181.

This bias is one-sided and grows with pack position, so `old_log_probs` from the
rollout are systematically **higher** than the trainer's recomputation. It is why
`gspo/seq_clip_low_frac` sits near 1.0 with the paper's eps and why that was
misread as train/rollout precision drift: with long (~4k token) completions the
same effect integrates to the ~-0.08 log-ratio seen in the launcher logs. It is
unrelated to FP8, to LoRA, and to bf16. A 2D block-diagonal `attention_mask` is not
a workaround -- the model scores -15.9/token under it.

Options, in order of preference:

1. Install `causal_conv1d` (missing in the current venv) so the conv path can honor
   `cu_seqlens`, and pass sequence boundaries through to the model for hybrid
   architectures.
2. Do not cross-pack sequences for models with `linear_attention` / `mamba` /
   `ssm` layer types -- one sequence per forward.
3. If neither is available, treat ratio/clip metrics on hybrid models as
   biased-but-monotone signals and widen eps accordingly; do not read them as
   policy drift.
