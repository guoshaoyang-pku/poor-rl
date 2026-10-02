"""Boundary-aware padding-free packing for hybrid GatedDeltaNet models (Qwen3.5 / Qwen3.8).

Why this exists
---------------
The trainer packs a micro-batch into one row per DP rank: several rollout samples are
concatenated along the sequence dimension and ``position_ids`` are reset at every sample
boundary. That contract is only honoured by layers that receive the segment boundaries. On a
hybrid GatedDeltaNet / full-attention backbone two things go wrong.

**Attention** is handled, but only under a narrow contract. ``masking_utils._preprocess_mask_arguments``
detects packed rows via ``find_packed_sequence_indices`` and ANDs a block-diagonal mask into the
causal mask -- but only when ``position_ids is not None and attention_mask is None and
past_key_values is None``. ``Qwen3_5TextModel.forward`` builds a ``DynamicCache`` whenever
``use_cache`` is truthy, which silently defeats that detection, and any ``attention_mask``
(a plain ones tensor is enough) defeats it too. Measured leakage on an identical-segment probe:
18.29 with the default ``use_cache``, 9.05 with ``use_cache=False``, and 0.81 even with
``use_cache=False`` when an all-ones mask is passed.

**The GatedDeltaNet layers** (18 of 24 in Qwen3.5-0.8B, 48 of 64 in Qwen3.8-27B) never honour it
at all:

* ``Qwen3_5GatedDeltaNet.forward`` forwards the boundaries as
  ``cu_seqlens=kwargs.pop("cu_seq_lens_q", None)`` -- but nothing upstream ever sets
  ``cu_seq_lens_q``, so the delta rule is always called with ``cu_seqlens=None``.
* The conv call site forwards ``**kwargs`` to ``causal_conv1d_fn``, whose reference PyTorch
  fallback (``causal_conv1d`` is not installed) accepts ``**kwargs`` and ignores it outright.

So the recurrent state and the conv window of sample *i* leak into sample *i+1*. Measured on
360-1 (transformers 5.17, Qwen3.5-0.8B, 3-way packed row): the delta rule received a non-None
``cu_seqlens`` on 0/72 calls, and the packed-vs-clean per-token logprob deviation grew with
segment index (+0.000 / +0.489 / +1.142 for segments 0/1/2). A positive deviation means the
trainer's packed forward reports *higher* logprobs than the rollout server saw, so ``ratio > 1``
and ``gspo/seq_clip_low_frac`` pins near 1.0 -- all mass in the low-side clip, which is what
destroys sample efficiency.

What this does
--------------
* **Delta rule**: derive the boundaries and hand them to the underlying implementation as
  ``cu_seqlens``. The resolved implementation (``kernels.layer.layer.Func``, FLA-backed) honours
  them exactly -- verified bit-identical to a per-segment split, and it costs one call instead of
  N. If the implementation does not accept ``cu_seqlens`` (the pure-torch reference path), fall
  back to splitting per segment, which is exact but pays N fixed overheads.
* **Conv**: run the full conv once and overwrite only the first ``kernel_size - 1`` positions of
  each segment, recomputed with the previous segment's tokens zeroed. Those are exactly the
  positions whose receptive field straddles a boundary; everything else is already correct.

When a row is a single sequence the wrappers delegate untouched, so the un-packed path stays
bit-identical.

Contract this module depends on
-------------------------------
Packed rows must be forwarded with ``use_cache=False`` and ``attention_mask=None``, otherwise
attention itself leaks and no amount of work here helps. ``assert_packing_contract`` checks it.

Usage
-----
    from rlforge.hybrid_packing import install
    install(model)   # after the model is built, before the first forward

Only the training model needs it. The rollout server (vLLM) never packs.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import logging
import threading

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "install",
    "packed_segments",
    "segments",
    "boundary_aware_packing",
    "assert_packing_contract",
    "supported_modules",
]

# Segment list for the forward currently running on this thread: [(start, end), ...] or None.
_CTX = threading.local()
_WARNED: set[str] = set()
_FLAGS: dict[str, bool] = {}


def segments():
    """Segments of the row being forwarded on this thread, or None for a single sequence."""
    return getattr(_CTX, "segments", None)


def _set_segments(value):
    _CTX.segments = value


def _warn_once(key: str, message: str):
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(message)


def packed_segments(position_ids):
    """Segment boundaries of a packed row, derived from per-sample-reset ``position_ids``.

    A ``0`` marks the start of a segment. Returns a list of ``(start, end)`` half-open ranges, or
    ``None`` when the row is a single sequence (nothing to split) or when the boundaries cannot be
    established -- in which case the caller keeps today's behaviour rather than guessing.
    """
    if position_ids is None or not torch.is_tensor(position_ids) or position_ids.dim() != 2:
        return None
    batch, length = position_ids.shape
    if batch != 1:
        # Packed rows are always batch 1 (the collator concatenates along the sequence dim).
        _warn_once(
            "batch>1",
            f"hybrid_packing: packed row has batch={batch}, expected 1; not splitting, so packing "
            "stays boundary-blind for this call",
        )
        return None
    starts = (position_ids[0] == 0).nonzero(as_tuple=False).flatten().tolist()
    if len(starts) <= 1:
        return None
    if starts[0] != 0:
        _warn_once(
            "no-zero",
            "hybrid_packing: position_ids of a packed row must reset to 0 at each segment start; "
            "not splitting",
        )
        return None
    ends = starts[1:] + [length]
    return list(zip(starts, ends))


def _cu_seqlens(segs, device) -> torch.Tensor:
    """``[0, e0, e1, ...]`` cumulative boundaries, the layout the delta-rule kernels expect."""
    return torch.tensor([segs[0][0]] + [end for _, end in segs], dtype=torch.int32, device=device)


def _accepted_kwargs(func, values: dict) -> dict:
    """Drop keys `func` does not declare, so one wrapper serves both delta-rule variants."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return values
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return values
    return {k: v for k, v in values.items() if k in params}


def _upstream_has_boundaries(kwargs) -> bool:
    """True when the caller supplied real boundaries -- then upstream owns the split, not us."""
    return kwargs.get("cu_seq_lens_q") is not None or kwargs.get("cu_seqlens") is not None


def _accepts_cu_seqlens(func) -> bool:
    """Does `func` take (and honour) a `cu_seqlens` argument?

    The resolved implementation is usually a `kernels.layer.layer.Func`, which is opaque to
    `inspect`, so fall back to a one-off functional probe on tiny tensors. Result is cached: this
    runs at most once per process per target.
    """
    key = f"cu_seqlens:{id(func)}"
    if key in _FLAGS:
        return _FLAGS[key]
    verdict = False
    try:
        verdict = "cu_seqlens" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        pass
    if not verdict and torch.cuda.is_available():
        # A `kernels.layer.layer.Func` reports its signature as `(*args, **kwargs)`, so
        # introspection can never rule this in -- ask the function itself with a tiny call.
        # Cost is irrelevant: the answer is cached and this runs once per target per process.
        try:
            b, length, heads, dim = 1, 2, 2, 8
            shape = (b, length, heads, dim)
            t = torch.zeros(shape, dtype=torch.float32, device="cuda")
            scal = torch.zeros(b, length, heads, dtype=torch.float32, device="cuda")
            func(t, t, t, g=scal, beta=scal,
                 cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"))
            verdict = True
        except Exception:  # noqa: BLE001 - a probe failure just means "do not use it"
            verdict = False
    _FLAGS[key] = verdict
    return verdict


def _make_conv(orig):
    def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        segs = segments()
        if segs is None or _upstream_has_boundaries(kwargs):
            return orig(hidden_states, weight, bias, activation, **kwargs)
        out = orig(hidden_states, weight, bias, activation, **kwargs)

        taps = weight.shape[-1]
        pad = taps - 1
        length = hidden_states.shape[-1]
        if pad <= 0 or len(segs) <= 1:
            return out

        # Only the first `taps - 1` positions of a segment have a receptive field reaching into
        # the previous one. They are the only wrong ones, so rebuild each such window as
        # `[zeros(pad), this segment's tokens]` and convolve them in a single batched call.
        #
        # This deliberately goes back through the real conv rather than re-deriving the taps with
        # elementwise math: the real conv is bit-identical to what an un-packed forward would
        # produce, whereas a re-derivation only matches to within bf16 rounding (measured 0.17
        # nats of drift on a packed probe row, against an exactly-0 residual for this path).
        groups, spans = [], []
        for start, _ in segs[1:]:
            span = min(pad, length - start)
            if span <= 0:
                continue
            zeros = torch.zeros(hidden_states.shape[0], hidden_states.shape[1], pad,
                                dtype=hidden_states.dtype, device=hidden_states.device)
            groups.append(torch.cat([zeros, hidden_states[:, :, start:start + span]], dim=-1))
            spans.append((start, span))
        if not groups:
            return out

        widest = max(span for _, span in spans)
        windows = [g if g.shape[-1] == pad + widest
                   else torch.nn.functional.pad(g, (0, pad + widest - g.shape[-1]))
                   for g in groups]
        fixed = orig(torch.cat(windows, dim=0), weight, bias, activation, **kwargs)
        # `out` is a fresh tensor from the underlying conv, so splicing in place is safe and
        # avoids a full-size copy per layer.
        for i, (start, span) in enumerate(spans):
            out[:, :, start:start + span] = fixed[i, :, pad:pad + span]
        return out

    return causal_conv1d_fn


def _make_delta_rule(orig):
    use_cu = _accepts_cu_seqlens(orig)

    def gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=64,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **kwargs,
    ):
        segs = segments()
        # A cache / carried state means decode, not packing: never touch it.
        delegate = initial_state is not None or output_final_state or _upstream_has_boundaries(kwargs)
        if segs is None or delegate:
            return orig(
                query, key, value, g=g, beta=beta,
                **_accepted_kwargs(orig, dict(
                    chunk_size=chunk_size, initial_state=initial_state,
                    output_final_state=output_final_state,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)),
            )

        extra = dict(chunk_size=chunk_size, initial_state=None, output_final_state=False,
                     use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)

        if use_cu:
            # Native variable-length path: one call, exact (verified bit-identical to splitting).
            return orig(
                query, key, value, g=g, beta=beta,
                cu_seqlens=_cu_seqlens(segs, query.device), **extra,
            )

        # No varlen support: run each segment from a fresh zero state and stitch the outputs.
        # [batch, length, heads, dim] -> split on the sequence dim.
        parts = [
            orig(query[:, start:end], key[:, start:end], value[:, start:end],
                 g=g[:, start:end], beta=beta[:, start:end],
                 **_accepted_kwargs(orig, extra))[0]
            for start, end in segs
        ]
        return torch.cat(parts, dim=1), None

    return gated_delta_rule


_TARGETS = (
    ("causal_conv1d_fn", _make_conv),
    ("torch_chunk_gated_delta_rule", _make_delta_rule),
    ("torch_recurrent_gated_delta_rule", _make_delta_rule),
)


def supported_modules():
    """transformers modules that define the hybrid GatedDeltaNet ops (qwen3_5, qwen3_8, ...).

    `dir(transformers.models)` yields subpackage names like `qwen3_5`, not `modeling_qwen3_5`, so
    the modeling module has to be imported through its subpackage.
    """
    import transformers.models as tmodels

    found = []
    for name in sorted(dir(tmodels)):
        if not name.startswith("qwen"):
            continue
        try:
            package = importlib.import_module(f"transformers.models.{name}")
        except Exception:  # noqa: BLE001 - optional subpackages may fail to import for any reason
            continue
        for attr in sorted(dir(package)):
            if not attr.startswith("modeling_"):
                continue
            try:
                module = importlib.import_module(f"transformers.models.{name}.{attr}")
            except Exception:  # noqa: BLE001
                continue
            if module not in found and any(hasattr(module, target) for target, _ in _TARGETS):
                found.append(module)
    return found


@contextlib.contextmanager
def boundary_aware_packing(position_ids):
    """Tell the GatedDeltaNet wrappers where this packed row's segment boundaries are.

    Use around the training forward that consumes a packed row::

        with boundary_aware_packing(position_ids):
            outputs = model(input_ids=..., position_ids=position_ids, use_cache=False)

    The trainer already holds `position_ids`, so this is more reliable than guessing the boundary
    source by walking the module tree -- and it is explicit about which forwards are packed.
    Re-entrant: nesting only restores the previous value on exit.

    Yields the segment list (``None`` when the row is a single sequence).
    """
    previous = segments()
    found = packed_segments(position_ids)
    _set_segments(found)
    try:
        yield found
    finally:
        _set_segments(previous)


def assert_packing_contract(position_ids, *, use_cache=None, attention_mask=None) -> None:
    """Fail loudly if a packed row is forwarded on a contract that cannot be correct.

    Only rows that actually look packed are checked, so this is free for un-packed work.
    """
    if not torch.is_tensor(position_ids) or packed_segments(position_ids) is None:
        return
    problems = []
    if use_cache:
        problems.append("use_cache must be False on a packed row (a DynamicCache disables the "
                        "block-diagonal attention mask)")
    if attention_mask is not None:
        problems.append("attention_mask must be None on a packed row (any mask disables the "
                        "block-diagonal attention mask)")
    if problems:
        raise ValueError("packed row forwarded on a broken contract: " + "; ".join(problems))


def install(model=None, verbose: bool = True) -> list[str]:
    """Make the hybrid GatedDeltaNet ops boundary-aware. Returns what was patched.

    Patches the module-level ops, so it is process-wide and must be called before the first
    packed forward. Safe to call more than once, and safe on a non-hybrid model (then it patches
    nothing). `model` is accepted for symmetry and is only used to name the model in logs.
    """
    patched = []
    for module in supported_modules():
        for target, make in _TARGETS:
            original = getattr(module, target, None)
            if original is None or getattr(original, "_rlforge_boundary_aware", False):
                continue
            wrapper = make(original)
            wrapper._rlforge_boundary_aware = True
            wrapper.__wrapped__ = original
            wrapper.__doc__ = original.__doc__
            setattr(module, target, wrapper)
            patched.append(f"{module.__name__}.{target}")

    if verbose:
        label = getattr(model, "name_or_path", None) or (type(model).__name__ if model else "-")
        if patched:
            logger.info("hybrid_packing: patched %s (model=%s); boundaries arrive via "
                        "boundary_aware_packing()", ", ".join(patched), label)
        else:
            logger.info("hybrid_packing: nothing to patch (model=%s is not a hybrid "
                        "GatedDeltaNet backbone)", label)
    return patched
