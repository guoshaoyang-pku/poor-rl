"""Shared-prompt-prefix forward/backward for GRPO/GSPO groups on hybrid GatedDeltaNet models, sm80 / FSDP2 port.

Port of ``production/v3_1/src/rlforge/prefix_share.py`` (H200, FA3, DDP, one fused forward per group) to:

* Qwen3.8-27B text model (qwen3_5_text): attn_output_gate, 24 q heads / 4 kv heads, head_dim 256, partial RoPE
  (mrope rows), 48 GatedDeltaNet layers (16 key / 48 value heads, conv kernel 4) + 16 full-attention layers;
* A100 attention: the prompt pass calls the model's own attention interface (same kernel as the per-seq path);
  branch rows attend to [prompt K/V ; own K/V] with a bottom-right-aligned causal mask through
  flash-attn 2 (``flash_attn_func(causal=True)``, when the model is loaded with flash_attention_2) or PyTorch's
  built-in FA2 kernel (``torch.nn.attention.bias.causal_lower_right`` under SDPA, GQA native). ``sdpa_mask``
  (materialised boolean mask, mem-efficient kernel) is the fallback and the CPU path;
* peft LoRA wrapping and FSDP2 ``fully_shard`` per decoder layer: every decoder layer is entered through its
  ``nn.Module.__call__`` (an instance-level forward override dispatches on a context object), so FSDP2's
  unshard/reshard and pre/post-backward hooks fire exactly as in the normal forward. State tensors travel in a
  context object, never as module args, so FSDP2's ``cast_forward_inputs`` never downcasts the fp32 recurrent state;
* MEMORY: chunked branches. ``prefix_group_backward`` runs the prompt once, cuts its outgoing state (per layer:
  GatedDeltaNet recurrent state + conv tail, attention post-RoPE K/V) into leaf tensors, runs the G branches in
  chunks of k rows with one forward+backward each (grads accumulate into the leaves -- fp32 accumulators -- and the
  LoRA params), then ONE backward through the prompt with the accumulated state grads. By the chain rule this is
  the same gradient as the unchunked shared path (which in turn is the per-sequence gradient), up to fp summation
  order. ``chunk=None`` runs the unchunked shared path (no cut) for comparison.

Activation memory: every decoder layer is checkpointed with a reentrant-style function whose saved inputs can be
offloaded to pinned CPU memory (``offload=True``).

Alignment: the prompt is shared up to P = floor((p-1)/ALIGN)*ALIGN tokens (ALIGN=64 = FLA chunk size); the prompt tail
[P, p) is duplicated into every branch, so branch GatedDeltaNet chunks fall on the same absolute positions as in the
per-sequence forward and the prompt pass ends on a chunk boundary.
"""

from __future__ import annotations

import contextlib
import math
import os
import types

import torch
import torch.nn.functional as F

__all__ = ["install", "prefix_group_backward", "prefix_group_logprobs", "perseq_group_backward",
           "perseq_group_logprobs", "chunk_logprobs", "split_point"]

ALIGN = int(os.environ.get("RLFORGE_PREFIX_ALIGN", "64"))
# Branch rows per attention call (bounds the [rows, Hkv, P+Lb, D] concat working set).
ATTN_ROWS = int(os.environ.get("RLFORGE_PS_ATTN_ROWS", "8"))
# auto | flash_attn | torch_flash | sdpa_mask
ATTN_BACKEND = os.environ.get("RLFORGE_PS_ATTN", "auto")

# Ablation switch for the gate only: drop the prompt-state gradients (no prompt backward). Must fail the grad check.
_DETACH_PROMPT = False


# ----------------------------------------------------------------------------------------------------------------- utils
def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _causal_lm(model):
    """peft PeftModel -> LoraModel -> HF CausalLM; plain HF CausalLM -> itself."""
    m = _unwrap(model)
    if hasattr(m, "get_base_model"):  # peft PeftModel: its __getattr__ forwards .model/.lm_head, so the check
        m = m.get_base_model()          # below would accept the PeftModel itself (and .model = the CausalLM)
    for _ in range(3):
        if hasattr(m, "lm_head") and hasattr(m, "model") and hasattr(_backbone(m), "layers"):
            return m
        m = getattr(m, "base_model", None) or getattr(m, "model", None)
        if m is None:
            break
    raise TypeError("prefix_share_sm80: cannot locate the causal LM (lm_head/model)")


def _backbone(lm):
    inner = lm.model
    return getattr(inner, "language_model", inner)


def split_point(p: int, align: int = ALIGN) -> int:
    """Shared prefix length for a prompt of p tokens (>= 0, multiple of align, <= p-1)."""
    return ((p - 1) // align) * align


def _attn_mod():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    return mq


def _backend(cfg, device):
    impl = getattr(cfg, "_attn_implementation", None) or ""
    if ATTN_BACKEND != "auto":
        return ATTN_BACKEND
    if device.type != "cuda":
        return "sdpa_mask"
    if "flash" in impl:
        return "hf_flash"
    return "torch_flash"


# ------------------------------------------------------------------------------------------------------ attention pieces
def _attn_project(attn, x, cos, sin):
    mq = _attn_mod()
    shp = x.shape[:-1]
    hs = (*shp, -1, attn.head_dim)
    q, gate = torch.chunk(attn.q_proj(x).view(*shp, -1, attn.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*shp, -1)
    q = attn.q_norm(q.view(hs)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(x).view(hs)).transpose(1, 2)
    v = attn.v_proj(x).view(hs).transpose(1, 2)
    q, k = mq.apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v, gate


def _attn_finish(attn, o, gate, shp):
    # identical op order to Qwen3_5Attention.forward: reshape(...).contiguous() * sigmoid(gate) -> o_proj
    return attn.o_proj(o.reshape(*shp, -1).contiguous() * torch.sigmoid(gate))


def _hf_attention(attn, q, k, v):
    """The model's own attention interface (sdpa / flash_attention_2 / eager), causal, no mask -> [B, L, H, D].
    Used for the prompt pass (Lq == Lk), i.e. exactly what the per-sequence forward runs."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    mq = _attn_mod()
    fn = ALL_ATTENTION_FUNCTIONS.get_interface(attn.config._attn_implementation, mq.eager_attention_forward)
    o, _ = fn(attn, q, k, v, None, dropout=0.0, scaling=attn.scaling, is_causal=True)
    return o


def _branch_attention(attn, q, K, V, P, backend):
    """q [n,H,Lb,D]; K/V [n,Hkv,P+Lb,D]; bottom-right causal (query i sees keys 0..P+i) -> [n, Lb, H, D]."""
    Lb = q.shape[2]
    if backend == "hf_flash":
        return _hf_attention(attn, q, K, V)  # FA2/FA3 causal with Lq < Lk is bottom-right aligned
    if backend == "flash_attn":
        from flash_attn import flash_attn_func
        return flash_attn_func(q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2),
                               softmax_scale=attn.scaling, causal=True)
    if backend == "torch_flash":
        from torch.nn.attention.bias import causal_lower_right
        o = F.scaled_dot_product_attention(q, K, V, attn_mask=causal_lower_right(Lb, P + Lb),
                                           scale=attn.scaling, enable_gqa=True)
        return o.transpose(1, 2)
    if backend == "sdpa_mask":
        rep = attn.num_key_value_groups
        j = torch.arange(Lb, device=q.device)[:, None]
        i = torch.arange(P + Lb, device=q.device)[None]
        mask = (i < P) | (i - P <= j)
        o = F.scaled_dot_product_attention(q, K.repeat_interleave(rep, 1), V.repeat_interleave(rep, 1),
                                           attn_mask=mask, scale=attn.scaling)
        return o.transpose(1, 2)
    raise ValueError(f"unknown attention backend {backend!r}")


# ------------------------------------------------------------------------------------------------- GatedDeltaNet cache
class _GDNCache:
    """Duck-typed cache for ONE Qwen3_5GatedDeltaNet call (transformers 5.17 API: has_previous_state,
    update_conv_state, layers[idx].recurrent_states[0], update_recurrent_state, layers[idx].record_past).

    prompt mode: records the final recurrent state and the last k-1 pre-conv inputs.
    branch mode: serves them (expanded to n rows) as the initial state / conv window prefix."""

    record_past = False

    def __init__(self, branch, state=None, conv_tail=None, n=1):
        self.branch = branch
        self.state = state
        self.conv_tail = conv_tail
        self.n = n
        self.layers = self

    def __getitem__(self, idx):
        return self

    @property
    def recurrent_states(self):
        st = self.state
        return [st.expand(self.n, *st.shape[1:]).contiguous()]

    def has_previous_state(self, layer_idx, state_idx=0):
        return self.branch

    def update_conv_state(self, mixed_qkv, layer_idx, conv_kernel_size):
        k1 = conv_kernel_size - 1
        if not self.branch:
            self.conv_tail = mixed_qkv[:, :, -k1:]
            return mixed_qkv
        tail = self.conv_tail.expand(self.n, -1, -1).to(mixed_qkv.dtype)
        return torch.cat([tail, mixed_qkv], dim=-1)

    def update_recurrent_state(self, state, layer_idx):
        if not self.branch:
            self.state = state


# ----------------------------------------------------------------------------------------------- per-layer bodies
def _body_prompt(layer, backend, h, cos, sin):
    x = layer.input_layernorm(h)
    if layer.block_type == "linear_attention":
        cache = _GDNCache(branch=False)
        y = layer.linear_attn(hidden_states=x, cache_params=cache, attention_mask=None)
        s1, s2 = cache.state, cache.conv_tail.contiguous()
    elif layer.block_type == "full_attention":
        attn = layer.self_attn
        q, k, v, g = _attn_project(attn, x, cos, sin)
        y = _attn_finish(attn, _hf_attention(attn, q, k, v), g, x.shape[:-1])
        s1, s2 = k, v
    else:
        raise NotImplementedError(layer.block_type)
    h = h + y
    h = h + layer.mlp(layer.post_attention_layernorm(h))
    return h, s1, s2


def _body_branch(layer, backend, P, h, cos, sin, s1, s2):
    x = layer.input_layernorm(h)
    n = x.shape[0]
    if layer.block_type == "linear_attention":
        assert x.shape[1] >= 2, "branch length must be >= 2 (seq_len==1 takes the decode path)"
        cache = _GDNCache(branch=True, state=s1, conv_tail=s2, n=n)
        y = layer.linear_attn(hidden_states=x, cache_params=cache, attention_mask=None)
    else:
        attn = layer.self_attn
        q, k, v, g = _attn_project(attn, x, cos, sin)
        outs = []
        for s in range(0, n, ATTN_ROWS):
            e = min(n, s + ATTN_ROWS)
            K = torch.cat([s1.to(k.dtype).expand(e - s, -1, -1, -1), k[s:e]], dim=2)
            V = torch.cat([s2.to(v.dtype).expand(e - s, -1, -1, -1), v[s:e]], dim=2)
            outs.append(_branch_attention(attn, q[s:e], K, V, P, backend))
        y = _attn_finish(attn, torch.cat(outs, 0), g, x.shape[:-1])
    h = h + y
    h = h + layer.mlp(layer.post_attention_layernorm(h))
    return h


class _Ckpt(torch.autograd.Function):
    """Reentrant-style activation checkpoint of one decoder layer. Inputs are saved (optionally offloaded to pinned
    CPU memory); backward recomputes the layer under grad and back-propagates into the inputs (hidden + incoming
    prompt states) and the layer's (LoRA) params. Tensor inputs are explicit, so gradients w.r.t. prompt states flow
    out through this node whether they are cut leaves or live prompt outputs."""

    @staticmethod
    def forward(ctx, fn, offload, *inputs):
        ctx.fn = fn
        ctx.offload = offload
        ctx.req = [isinstance(t, torch.Tensor) and t.requires_grad for t in inputs]
        if offload:
            cpu = []
            for t in inputs:
                c = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
                c.copy_(t, non_blocking=True)
                cpu.append(c)
            ctx.cpu = cpu
            ctx.dev = inputs[0].device
        else:
            ctx.save_for_backward(*inputs)
        with torch.no_grad():
            out = fn(*inputs)
        return out

    @staticmethod
    def backward(ctx, *gouts):
        if ctx.offload:
            inputs = [c.to(ctx.dev, non_blocking=True) for c in ctx.cpu]
            ctx.cpu = None
        else:
            inputs = ctx.saved_tensors
        det = [t.detach().requires_grad_(r) for t, r in zip(inputs, ctx.req)]
        with torch.enable_grad():
            out = ctx.fn(*det)
        if not isinstance(out, tuple):
            out = (out,)
        outs, grads = [], []
        for o, g in zip(out, gouts):
            if g is not None and o.requires_grad:
                outs.append(o)
                grads.append(g)
        if outs:
            torch.autograd.backward(outs, grads)
        return (None, None, *[d.grad if r else None for d, r in zip(det, ctx.req)])


# ------------------------------------------------------------------------------------- decoder-layer forward override
class _PSCall:
    """Context object handed to a decoder layer through its __call__ (FSDP2 hooks run around it)."""

    def __init__(self, mode, backend, ckpt, offload, P=0, states=None):
        self.mode, self.backend, self.ckpt, self.offload, self.P = mode, backend, ckpt, offload, P
        self.states = states  # branch mode: (s1, s2) for this layer


def _layer_forward(self, *args, _ps=None, **kwargs):
    if _ps is None:
        return self._ps_orig_forward(*args, **kwargs)
    if _ps.mode == "prompt":
        h, cos, sin = args
        fn = lambda h, cos, sin: _body_prompt(self, _ps.backend, h, cos, sin)  # noqa: E731
        ins = (h, cos, sin)
    else:
        h, cos, sin = args
        s1, s2 = _ps.states
        P = _ps.P
        fn = lambda h, cos, sin, s1, s2: _body_branch(self, _ps.backend, P, h, cos, sin, s1, s2)  # noqa: E731
        ins = (h, cos, sin, s1, s2)
    if _ps.ckpt and torch.is_grad_enabled():
        return _Ckpt.apply(fn, _ps.offload, *ins)
    return fn(*ins)


def _layer_call(layer, ps, *args):
    """Enter the decoder layer through nn.Module.__call__ (FSDP2 hooks) but bypass HF's GradientCheckpointingLayer
    wrapper (we checkpoint inside, after FSDP2 has unsharded the params)."""
    return torch.nn.Module.__call__(layer, *args, _ps=ps)


# ----------------------------------------------------------------------------------------------------- model passes
def _rope(bb, h, start, L):
    pos = (start + torch.arange(L, device=h.device))[None, None].expand(3, 1, -1)
    return bb.rotary_emb(h, pos)


def _ckpt_on(lm):
    bb = _backbone(lm)
    return lm.training and any(getattr(m, "gradient_checkpointing", False) for m in (bb, *bb.layers)) \
        or getattr(lm, "_ps_force_ckpt", False)


def _prompt_pass(lm, prefix_ids, offload):
    """prefix_ids [P] -> list of per-layer (s1, s2) prompt states (live, with graph when grad is enabled)."""
    bb = _backbone(lm)
    backend = _backend(bb.config, prefix_ids.device)
    h = bb.embed_tokens(prefix_ids[None])
    if torch.is_grad_enabled() and not h.requires_grad:
        h.requires_grad_(True)  # frozen embedding: make layer-0's checkpoint node part of the graph
    cos, sin = _rope(bb, h, 0, prefix_ids.numel())
    ps = _PSCall("prompt", backend, _ckpt_on(lm), offload)
    states = []
    for layer in bb.layers[: bb.config.num_hidden_layers]:
        h, s1, s2 = _layer_call(layer, ps, h, cos, sin)
        states.append((s1, s2))
    return states


def _branch_hidden(lm, states, br, P, offload):
    """br [n, Lb] right-padded branch ids continuing a prefix of length P -> final-norm hidden [n, Lb, H]."""
    bb = _backbone(lm)
    backend = _backend(bb.config, br.device)
    h = bb.embed_tokens(br)
    if torch.is_grad_enabled() and not h.requires_grad:
        h.requires_grad_(True)
    cos, sin = _rope(bb, h, P, br.shape[1])
    ck = _ckpt_on(lm)
    for layer, st in zip(bb.layers[: bb.config.num_hidden_layers], states):
        h = _layer_call(layer, _PSCall("branch", backend, ck, offload, P=P, states=st), h, cos, sin)
    return bb.norm(h)


def chunk_logprobs(lm, hidden, targets, temperature=1.0, chunk=1024, with_entropy=False):
    """Token log-probs of targets under lm_head(hidden)/T, computed in checkpointed vocab chunks through the lm_head
    module (FSDP2-safe). hidden [N, H], targets [N] -> logp [N] fp32 (and entropy [N])."""
    head = lm.lm_head
    lps, ents = [], []

    def f(hc, tc):
        lg = head(hc).float()
        if temperature != 1.0:
            lg = lg / temperature
        lse = torch.logsumexp(lg, -1)
        lp = lg.gather(1, tc[:, None]).squeeze(1) - lse
        if with_entropy:
            p = torch.softmax(lg, -1)
            return lp, lse - (p * lg).sum(-1)
        return lp, lp.new_zeros(())

    for s in range(0, hidden.shape[0], chunk):
        hc, tc = hidden[s: s + chunk], targets[s: s + chunk]
        if torch.is_grad_enabled():
            lp, en = torch.utils.checkpoint.checkpoint(f, hc, tc, use_reentrant=False)
        else:
            lp, en = f(hc, tc)
        lps.append(lp)
        ents.append(en.detach())
    lp = torch.cat(lps)
    return (lp, torch.cat(ents)) if with_entropy else lp


# --------------------------------------------------------------------------------------------- group-level drivers
class _Payload:
    """Opaque (non-pytree) carrier for the pass arguments: FSDP2's root pre-forward casts every floating tensor it
    finds in args/kwargs to param_dtype (cast_forward_inputs); the fp32 prompt states must not be touched."""

    def __init__(self, d):
        self.d = d


def _forward_dispatch(self, *args, prefix_share=None, **kwargs):
    """Root-module forward: every pass enters through lm(...) so the FSDP2 root hooks run each time."""
    if prefix_share is None:
        return self._ps_orig_forward(*args, **kwargs)
    prefix_share = prefix_share.d
    op = prefix_share["op"]
    if op == "prompt":
        return _prompt_pass(self, prefix_share["prefix_ids"], prefix_share.get("offload", False))
    if op == "branch":
        h = _branch_hidden(self, prefix_share["states"], prefix_share["br"], prefix_share["P"],
                           prefix_share.get("offload", False))
        sel_r, sel_t, tgt = prefix_share["sel"]
        return chunk_logprobs(self, h[sel_r, sel_t], tgt, prefix_share.get("temperature", 1.0),
                              prefix_share.get("lm_chunk", 1024))
    if op == "perseq":
        ids = prefix_share["ids"]  # [1, L]
        p = prefix_share["p"]
        bb = _backbone(self)
        out = bb(input_ids=ids, position_ids=torch.arange(ids.shape[1], device=ids.device)[None], use_cache=False)
        h = out.last_hidden_state[0, p - 1: -1]
        return chunk_logprobs(self, h, ids[0, p:], prefix_share.get("temperature", 1.0),
                              prefix_share.get("lm_chunk", 1024))
    raise ValueError(op)


def install(model):
    """Patch the causal LM's forward (dispatch on ``prefix_share=``) and every decoder layer's forward (dispatch on
    ``_ps=``). Instance-level attributes: FSDP2's dynamic subclass and peft wrapping keep them; idempotent.
    Call before or after fully_shard (both work: hooks are registered on the module, not on forward)."""
    lm = _causal_lm(model)
    if getattr(lm, "_ps_orig_forward", None) is None:
        lm._ps_orig_forward = lm.forward
        lm.forward = types.MethodType(_forward_dispatch, lm)
    for layer in _backbone(lm).layers:
        if getattr(layer, "_ps_orig_forward", None) is None:
            layer._ps_orig_forward = layer.forward
            layer.forward = types.MethodType(_layer_forward, layer)
    return lm


def _call(model, **ps):
    return model(prefix_share=_Payload(ps))


def _buckets(lens, k):
    """Length-sorted chunks of at most k branches (longest first, so padding inside a chunk is small)."""
    order = sorted(range(len(lens)), key=lambda i: -lens[i])
    return [order[i: i + k] for i in range(0, len(order), k)]


def _pack(ids_list, idx, P, p):
    """Right-padded branch block for branches idx: rows = seq[P:], targets = completion tokens."""
    rows = [ids_list[i][P:] for i in idx]
    Lb = max(len(r) for r in rows)
    dev = rows[0].device
    br = rows[0].new_zeros((len(rows), Lb))
    sel_r, sel_t = [], []
    for r, t in enumerate(rows):
        br[r, : len(t)] = t
        n_c = len(t) - (p - P)  # completion tokens of this row
        # slot j predicts row token j+1; completion tokens sit at row positions p-P .. len-1
        j = torch.arange(p - P - 1, len(t) - 1, device=dev)
        sel_r.append(torch.full_like(j, r))
        sel_t.append(j)
        assert j.numel() == n_c
    sel_r, sel_t = torch.cat(sel_r), torch.cat(sel_t)
    return br, (sel_r, sel_t, br[sel_r, sel_t + 1])


def prefix_group_backward(model, prompt_ids, completions, loss_fn, chunk=8, temperature=1.0, offload=False,
                          lm_chunk=1024, fsdp_modules=None, n_chunks_sync=None, perm=None, final_sync=True):
    """Forward+backward of ONE GRPO group with the prompt computed once.

    prompt_ids: LongTensor [p] (device); completions: list of LongTensor [c_i] (c_i >= 1).
    loss_fn(i, logp_i) -> scalar loss contribution of branch i (its global normaliser already applied), so the
    group loss is sum_i loss_fn(i, logp_i) and chunking is exact.
    chunk: branches per forward/backward (None -> unchunked shared path, prompt states not cut).
    fsdp_modules: FSDP2 root module(s); when given, intermediate backwards skip gradient reduce-scatter and are not
    the last backward (set_requires_gradient_sync / set_is_last_backward); the prompt backward syncs.
    n_chunks_sync: run dummy chunks up to this many (all FSDP ranks must issue the same collectives).
    perm: optional branch order (for the leakage test). Returns dict(logps=[...detached...], loss=float, stats).
    final_sync: False keeps gradient sync off for the prompt backward too (several groups per rank per optimizer
    step: only the rank's LAST group syncs, so the intra-host reduce-scatter / cross-host all-reduce runs once).
    """
    lm = _causal_lm(model)
    dev = prompt_ids.device
    p = int(prompt_ids.numel())
    P = split_point(p)
    seqs = [torch.cat([prompt_ids, c.to(dev)]) for c in completions]
    G = len(seqs)
    lens = [len(s) - P for s in seqs]
    stats = {"G": G, "p": p, "P": P, "forward_tokens": P + sum(lens), "unshared_tokens": sum(len(s) for s in seqs)}
    logps = [None] * G
    total = 0.0

    def _sync(on):
        for m in (fsdp_modules or []):
            m.set_requires_gradient_sync(on)
            m.set_is_last_backward(on)

    if P == 0:  # nothing to share: per-seq
        return perseq_group_backward(model, prompt_ids, completions, loss_fn, temperature, lm_chunk, fsdp_modules,
                                     final_sync=final_sync)

    if chunk is None:  # unchunked shared path: one forward over all branches, states live (no cut)
        states = _call(model, op="prompt", prefix_ids=prompt_ids[:P], offload=offload)
        idx = list(range(G))
        br, sel = _pack(seqs, idx, P, p)
        lp = _call(model, op="branch", states=states, br=br, P=P, sel=sel, temperature=temperature,
                   lm_chunk=lm_chunk, offload=offload)
        loss = 0.0
        off = 0
        for i in idx:
            n = len(completions[i])
            logps[i] = lp[off: off + n]
            loss = loss + loss_fn(i, logps[i])
            off += n
        _sync(final_sync)
        loss.backward()
        return {"logps": [x.detach() for x in logps], "loss": float(loss), "stats": stats}

    # chunked: prompt once (live graph), states cut into leaves
    states = _call(model, op="prompt", prefix_ids=prompt_ids[:P], offload=offload)
    flat = [t for st in states for t in st]
    # fp32 leaves: exact upcast of the bf16 K/V / conv tail (forward unchanged, the branch casts back), so the
    # gradients of all chunks accumulate in fp32 in leaf.grad.
    leaves = [t.detach().float().requires_grad_(t.requires_grad) for t in flat]
    lstates = [(leaves[2 * i], leaves[2 * i + 1]) for i in range(len(states))]
    groups = _buckets(lens, chunk) if perm is None else [perm[i: i + chunk] for i in range(0, G, chunk)]
    n_run = max(len(groups), n_chunks_sync or 0)
    stats["chunks"] = len(groups)
    for c in range(n_run):
        _sync(False)
        if c < len(groups):
            idx = groups[c]
            br, sel = _pack(seqs, idx, P, p)
            lp = _call(model, op="branch", states=lstates, br=br, P=P, sel=sel, temperature=temperature,
                       lm_chunk=lm_chunk, offload=offload)
            loss = 0.0
            off = 0
            for i in idx:
                n = len(completions[i])
                logps[i] = lp[off: off + n].detach()
                loss = loss + loss_fn(i, lp[off: off + n])
                off += n
        else:  # dummy chunk: same collectives, zero gradient
            br = prompt_ids.new_zeros((1, 2))
            sel = (torch.zeros(1, dtype=torch.long, device=dev), torch.zeros(1, dtype=torch.long, device=dev),
                   prompt_ids.new_zeros(1))
            lp = _call(model, op="branch", states=lstates, br=br, P=P, sel=sel, temperature=temperature,
                       lm_chunk=lm_chunk, offload=offload)
            loss = lp.sum() * 0.0
        loss.backward()
        total += float(loss)
    # one backward through the prompt with the accumulated state grads
    _sync(final_sync)
    outs, grads = [], []
    for t, l in zip(flat, leaves):
        if t.requires_grad and l.grad is not None:
            outs.append(t)
            grads.append(l.grad.to(t.dtype) if not _DETACH_PROMPT else torch.zeros_like(t))
    stats["state_grad_norm"] = float(torch.sqrt(sum((l.grad.double() ** 2).sum() for l in leaves if l.grad is not None)))
    torch.autograd.backward(outs, grads)
    del leaves, lstates, states, flat
    return {"logps": logps, "loss": total, "stats": stats}


def prefix_group_logprobs(model, prompt_ids, completions, chunk=8, temperature=1.0, lm_chunk=1024,
                          n_chunks_sync=None):
    """No-grad shared-prefix log-probs of one group (KL reference / old-logp recompute)."""
    p = int(prompt_ids.numel())
    P = split_point(p)
    dev = prompt_ids.device
    seqs = [torch.cat([prompt_ids, c.to(dev)]) for c in completions]
    out = [None] * len(seqs)
    with torch.no_grad():
        if P == 0:
            return perseq_group_logprobs(model, prompt_ids, completions, temperature, lm_chunk)
        states = _call(model, op="prompt", prefix_ids=prompt_ids[:P])
        lens = [len(s) - P for s in seqs]
        groups = _buckets(lens, chunk or len(seqs))
        for c in range(max(len(groups), n_chunks_sync or 0)):
            if c >= len(groups):
                br = prompt_ids.new_zeros((1, 2))
                z = torch.zeros(1, dtype=torch.long, device=dev)
                _call(model, op="branch", states=states, br=br, P=P, sel=(z, z, prompt_ids.new_zeros(1)))
                continue
            idx = groups[c]
            br, sel = _pack(seqs, idx, P, p)
            lp = _call(model, op="branch", states=states, br=br, P=P, sel=sel, temperature=temperature,
                       lm_chunk=lm_chunk)
            off = 0
            for i in idx:
                n = len(completions[i])
                out[i] = lp[off: off + n]
                off += n
    return out


def perseq_group_backward(model, prompt_ids, completions, loss_fn, temperature=1.0, lm_chunk=1024,
                          fsdp_modules=None, final_sync=True):
    """Reference: every sequence forwarded alone (batch 1, the model's standard forward) + its own backward."""
    dev = prompt_ids.device
    p = int(prompt_ids.numel())
    logps, total = [], 0.0
    G = len(completions)
    for i, c in enumerate(completions):
        last = (i == G - 1) and final_sync
        for m in (fsdp_modules or []):
            m.set_requires_gradient_sync(last)
            m.set_is_last_backward(last)
        ids = torch.cat([prompt_ids, c.to(dev)])[None]
        lp = _call(model, op="perseq", ids=ids, p=p, temperature=temperature, lm_chunk=lm_chunk)
        loss = loss_fn(i, lp)
        loss.backward()
        total += float(loss)
        logps.append(lp.detach())
    return {"logps": logps, "loss": total, "stats": {"G": G, "p": p, "forward_tokens": sum(p + len(c) for c in completions)}}


def perseq_group_logprobs(model, prompt_ids, completions, temperature=1.0, lm_chunk=1024):
    dev = prompt_ids.device
    p = int(prompt_ids.numel())
    out = []
    with torch.no_grad():
        for c in completions:
            ids = torch.cat([prompt_ids, c.to(dev)])[None]
            out.append(_call(model, op="perseq", ids=ids, p=p, temperature=temperature, lm_chunk=lm_chunk))
    return out
