# Padding-free packing correctness (hybrid GatedDeltaNet backbones)

Padding-free packing is on the critical path for throughput: it concatenates several rollout
samples into one row per DP rank instead of padding each to `max_length`, which is what makes a
deeply skewed completion-length distribution affordable. The contract is that segment boundaries
are expressed by **resetting `position_ids` to 0** at the start of every sample.

That contract is only honoured by layers that are handed the boundaries. On Qwen3.5/Qwen3.8
(3:1 GatedDeltaNet / full-attention hybrid) two of the three moving parts were broken, silently.

## What was wrong

**Attention: correct, but only under a narrow contract.** `masking_utils` detects packed rows with
`find_packed_sequence_indices` and ANDs a block-diagonal mask into the causal mask -- but only when

```
position_ids is not None  and  attention_mask is None  and  past_key_values is None
```

`Qwen3_5TextModel.forward` builds a `DynamicCache` whenever `use_cache` is truthy, which defeats
that detection. Any `attention_mask` -- a plain all-ones tensor is enough -- defeats it too.
`varlen` support in `modeling_flash_attention_utils` (`_is_packed_sequence`) only applies on the
Flash Attention path, which is not what we run.

**GatedDeltaNet: never correct.** `Qwen3_5GatedDeltaNet.forward` forwards the boundaries as
`cu_seqlens=kwargs.pop("cu_seq_lens_q", None)`, but nothing upstream ever sets `cu_seq_lens_q`, so
the delta rule is always called with `cu_seqlens=None`. The conv call site forwards `**kwargs` to
`causal_conv1d_fn`, whose reference PyTorch fallback (`causal_conv1d` is not installed) accepts
`**kwargs` and ignores it outright.

Measured on 360-1 (transformers 5.17, Qwen3.5-0.8B, 18 DeltaNet + 6 attention layers): the delta
rule received a non-`None` `cu_seqlens` on **0 of 72 calls**. The conv window and the recurrent
state of sample *i* therefore leak into sample *i+1*.

## Measured impact

| Scenario | seg 0 | seg 1 | seg 2 |
|---|---:|---:|---:|
| Un-patched, trainer-style forward | 0.0000 | **+4.60** | **+4.71** |
| `install()` + `boundary_aware_packing` | 0.0000 | 0.0000 | 0.0000 |

(max abs per-token logprob deviation vs scoring each sample alone, 3-way packed row, nats.)

The deviation is **positive**: the trainer's packed forward reports *higher* logprobs than the
rollout server saw. That pushes `ratio > 1` and drives `gspo/seq_clip_low_frac` to ~1.0 -- all
mass in the low-side clip, which is what destroys sample efficiency. It also breaks the GSPO
sequence-ratio assumption outright, since the polluted segments are scored against a different
distribution than the one that produced them.

## The fix

`rlforge.hybrid_packing` patches the three module-level ops of every hybrid backbone it finds
(`qwen3_5`, `qwen3_5_moe`, `qwen3_next`, `qwen4_exp`):

```python
from rlforge.hybrid_packing import install, boundary_aware_packing

install(model)                      # once, before the first packed forward
...
with boundary_aware_packing(position_ids):   # around the training forward
    outputs = model(input_ids=..., position_ids=position_ids, use_cache=False)
```

`rlforge/trainer.py` does both for you: `install()` after the trainer is constructed, and the
context manager around the forward inside `compute_loss` (it already has `position_ids` in hand,
which is more reliable than guessing the boundary source by walking the module tree).

* **Delta rule** -- hand the boundaries to the implementation as `cu_seqlens`. The resolved
  implementation (`kernels.layer.layer.Func`, FLA-backed) honours them exactly: verified
  bit-identical to a per-segment split, and it is **one call instead of N** (1.58 ms vs 4.34 ms
  for a 6144-token row). If an implementation does not accept `cu_seqlens`, the wrapper falls back
  to splitting per segment, which is exact but pays N fixed overheads.
* **Conv** -- run the full conv once and overwrite only the first `kernel_size - 1` positions of
  each segment, recomputed with the previous segment's tokens zeroed. Those are exactly the
  positions whose receptive field straddles a boundary. The recomputation goes back through the
  real conv rather than re-deriving the taps with elementwise math: the real conv is bit-identical
  to an un-packed forward, whereas a re-derivation only matches to within bf16 rounding (0.17 nats
  of drift measured on a packed probe row, against an exactly-0 residual for the conv path).

When a row is a single sequence both wrappers delegate untouched, so the un-packed path stays
bit-identical (verified: 0.0000e+00).

## Cost

Interleaved A/B (same process, same thermal state, median of per-iteration CUDA events, 20
warmup iterations so autotune settles -- a cold baseline overstates the ratio by ~1.3x):

| Row | Stock | Patched | Overhead |
|---|---:|---:|---:|
| 3 x 2048 | 60.6 ms | 90.6 ms | 1.49x |
| 8 x 768 | 60.7 ms | 92.0 ms | 1.52x |
| 3 x 512 | 53.6 ms | 74.3 ms | 1.39x |

Body-only forward, no LM head. The cost is dominated by **per-call fixed overhead, not tokens**:
8x768 and 3x2048 are both 6144 tokens and land within 2% of each other. It is paid only on the
driver rank's gradient-accumulation forwards, and the trainer is currently *not* the bottleneck
(rollout produces 14,040 tok/s against a 21,628 tok/s consumption rate), so the tax is absorbed by
idle capacity. **Correctness is worth more than 1.5x on the faster of the two sides.**

## Guards

`assert_packing_contract(position_ids, use_cache=..., attention_mask=...)` raises if a packed row
is forwarded on a contract that cannot be correct -- the two conditions above are silent failure
modes otherwise. `tests/test_hybrid_packing.py` pins the boundary derivation, the context manager
semantics, the contract assertion, and the end-to-end equality; it also asserts the un-patched
forward *does* leak, so that if upstream ever fixes this the test fails loudly rather than passing
for the wrong reason and silently making the wrapper unnecessary.

## When this can be retired

Delete `rlforge/hybrid_packing.py` and its wiring once upstream (a) builds no `DynamicCache` for
packed rows, (b) propagates `cu_seqlens` into the GatedDeltaNet conv and delta rule, and (c) the
`test_unpatched_forward_does_leak` test flips to passing. Until then this is load-bearing.
