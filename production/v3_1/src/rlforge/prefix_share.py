"""Shared-prompt-prefix forward for GRPO/GSPO groups on hybrid GatedDeltaNet models (Qwen3.5 / 3.8).

Why
---
A micro-batch is one GRPO group: G completions of the SAME prompt. The per-sequence forward
(``trainer.per_seq_logprobs``) recomputes that prompt G times. At 5-6k prompt / 2k completion /
G=32 that is 32 x 7.5k = 240k forward tokens of which 172k are 31 redundant copies of the prompt.

What
----
The prompt is forwarded once; the G completions are forwarded as a batch that *continues* from it:

* **GatedDeltaNet layers**: the prompt pass records the final recurrent state and the last
  ``kernel_size - 1`` pre-conv inputs; the branch pass starts from them (``initial_state`` of the
  delta rule, conv window prefilled). Done through the layer's own ``forward`` with a duck-typed
  cache object, so the exact same kernels (FLA chunk rule, conv) run as in the unshared path.
* **Full-attention layers**: the prompt's post-norm/post-RoPE K/V are concatenated in front of each
  branch's own K/V; branch queries see all prompt keys plus their own causal prefix.
* **Positions** continue from the prompt length, exactly as in the unshared sequence.

Nothing is detached: the prompt's K/V and recurrent/conv state feed every branch through autograd,
so the backward sums the G branches' gradients into the single prompt computation. This is the same
function as the unshared forward, only computed once -- equal up to bf16 summation order.

Each branch starts at the LAST prompt token (duplicated G times, negligible), so the first
completion token is predicted inside the branch and the prompt pass needs no logits at all.

Usage
-----
    from rlforge.prefix_share import install
    install(trainer_model, temperature)       # wraps the (TRL-patched) forward, before accelerate.prepare
    out = model(prefix_share=dict(input_ids=..., position_ids=..., completion_mask=...))

Calling through ``model(...)`` keeps DDP's forward bookkeeping and accelerate's autocast in place.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ["install", "plan_groups", "prefix_shared_logprobs", "GroupRowBatcher"]

# Branch rows per SDPA call in the full-attention layers: bounds the [rows, heads, Lb, P+Lb] score
# working set when SDPA falls back to a kernel that materializes it.
ATTN_ROWS = 8

# [v3-gate patch] Split point alignment. The prompt is shared up to the largest multiple of ALIGN that is <= p-1;
# the remaining prompt tail (< ALIGN tokens) is duplicated into every branch. With ALIGN = 64 = FLA's
# chunk_gated_delta_rule block size, the branch's GatedDeltaNet chunks fall on exactly the same absolute positions
# as in the unshared forward and the prompt pass ends on a chunk boundary, so the delta-rule numerics match the
# per-sequence path instead of differing by a re-chunking. ALIGN=1 restores the original split (prefix = p-1).
ALIGN = int(os.environ.get("RLFORGE_PREFIX_ALIGN", "64"))


def _use_flash(cfg):
    """[v3-gate patch] True when the model was loaded with a FlashAttention implementation (production: TRL loads
    kernels-community/flash-attn3). The prefix path then calls the model's own attention interface, i.e. the same FA
    kernel the per-sequence path uses, instead of torch SDPA with a boolean mask."""
    impl = getattr(cfg, "_attn_implementation", None) or ""
    return "flash" in impl and os.environ.get("RLFORGE_PREFIX_SDPA", "0") != "1"


def _flash(attn, q, k, v):
    """q [B,H,Lq,D], k/v [B,Hkv,Lk,D] (GQA native) -> [B,Lq,H,D]. Lq < Lk with causal=True is bottom-right aligned in
    FA>=2.1 / FA3: query i sees keys 0..Lk-Lq+i = whole prompt + own causal prefix."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_5.modeling_qwen3_5 import eager_attention_forward

    fn = ALL_ATTENTION_FUNCTIONS.get_interface(attn.config._attn_implementation, eager_attention_forward)
    o, _ = fn(attn, q, k, v, None, dropout=0.0, scaling=attn.scaling, is_causal=True)
    return o


def _unwrap(model):
    while hasattr(model, "module"):  # DDP / accelerate wrappers
        model = model.module
    return model


def _backbone(lm):
    """The text decoder stack (embed_tokens / layers / norm / rotary_emb) of a causal LM or a VLM."""
    inner = lm.model
    return getattr(inner, "language_model", inner)


def plan_groups(input_ids, position_ids, completion_mask):
    """Split a packed (1, T) row into groups of segments sharing an identical prompt.

    A segment's prompt is everything before its first completion token. Returns a list of
    ``(prompt_len, [(start, end), ...])``; segments without a completion token, or with a prompt
    shorter than 2 tokens, come back as singleton groups with ``prompt_len=None`` (no sharing).
    """
    T = input_ids.shape[1]
    starts = (position_ids[0] == 0).nonzero().flatten().tolist()
    bounds = list(zip(starts, starts[1:] + [T]))
    cm = completion_mask[0]
    ids = input_ids[0]
    groups: dict = {}
    order = []
    for a, e in bounds:
        nz = cm[a:e].nonzero()
        p = int(nz[0]) if nz.numel() else None
        if p is None or p < 2 or e - a - p < 1:
            key = ("solo", a)
        else:
            key = (p, ids[a : a + p].cpu().numpy().tobytes())
        if key not in groups:
            groups[key] = (p if key[0] != "solo" else None, [])
            order.append(key)
        groups[key][1].append((a, e))
    return [groups[k] for k in order]


class _GDNPrefixCache:
    """Duck-typed cache for ONE GatedDeltaNet layer: records the prompt's final state on the first
    forward, then serves it as the initial state (expanded to the branch batch) on later forwards."""

    record_past = False

    def __init__(self):
        self.n = 0
        self.conv_tail = None   # [1, C, k-1] pre-conv inputs of the prompt's last k-1 tokens
        self.state = None       # [1, H, Dk, Dv] recurrent state after the prompt
        self.branch = False
        self.layers = self      # layer.forward reads cache_params.layers[idx].recurrent_states[0]

    def __getitem__(self, idx):
        return self

    @property
    def recurrent_states(self):
        st = self.state.detach() if _DETACH else self.state
        return [st.expand(self.n, *st.shape[1:]).contiguous()]

    def has_previous_state(self, layer_idx, state_idx=0):
        return self.branch

    def update_conv_state(self, mixed_qkv, layer_idx, conv_kernel_size):
        k1 = conv_kernel_size - 1
        if not self.branch:
            self.conv_tail = mixed_qkv[:, :, -k1:]
            return mixed_qkv
        # Prefill the conv window with the prompt's last k-1 inputs; the layer slices the outputs
        # back to the branch length after the conv.
        tail = self.conv_tail.detach() if _DETACH else self.conv_tail
        return torch.cat([tail.expand(self.n, -1, -1).to(mixed_qkv.dtype), mixed_qkv], dim=-1)

    def update_recurrent_state(self, state, layer_idx):
        if not self.branch:
            self.state = state


# Ablation switch for the exactness test only: cut the prompt->branch gradient path. With it on,
# the prompt receives no gradient from the completions and the grad check must fail loudly.
_DETACH = False


def _attn_project(attn, x, pe):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    shp = x.shape[:-1]
    hs = (*shp, -1, attn.head_dim)
    q, gate = torch.chunk(attn.q_proj(x).view(*shp, -1, attn.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*shp, -1)
    q = attn.q_norm(q.view(hs)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(x).view(hs)).transpose(1, 2)
    v = attn.v_proj(x).view(hs).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *pe)
    return q, k, v, gate


def _attn_finish(attn, o, gate, shp):
    o = o.transpose(1, 2).reshape(*shp, -1)
    return attn.o_proj(o * torch.sigmoid(gate))


def _attn_group(attn, x_p, x_bs, pe_p, pe_bs, masks):
    rep = attn.num_key_value_groups
    flash = _use_flash(attn.config)
    q_p, k_p, v_p, g_p = _attn_project(attn, x_p, pe_p)
    if flash:
        # identical op order to Qwen3_5Attention.forward: reshape(...).contiguous() * sigmoid(gate) -> o_proj
        o_p = _flash(attn, q_p, k_p, v_p)
        y_p = attn.o_proj(o_p.reshape(*x_p.shape[:-1], -1).contiguous() * torch.sigmoid(g_p))
        k_pr, v_pr = k_p, v_p
    else:
        k_pr, v_pr = k_p.repeat_interleave(rep, 1), v_p.repeat_interleave(rep, 1)
        o_p = F.scaled_dot_product_attention(q_p, k_pr, v_pr, is_causal=True, scale=attn.scaling)
        y_p = _attn_finish(attn, o_p, g_p, x_p.shape[:-1])
    if _DETACH:
        k_pr, v_pr = k_pr.detach(), v_pr.detach()
    y_bs = []
    for x_b, pe_b, mask in zip(x_bs, pe_bs, masks):
        q_b, k_b, v_b, g_b = _attn_project(attn, x_b, pe_b)
        outs = []
        for s in range(0, x_b.shape[0], ATTN_ROWS):
            e = min(x_b.shape[0], s + ATTN_ROWS)
            n = e - s
            if flash:
                K = torch.cat([k_pr.expand(n, -1, -1, -1), k_b[s:e]], dim=2)
                V = torch.cat([v_pr.expand(n, -1, -1, -1), v_b[s:e]], dim=2)
                outs.append(_flash(attn, q_b[s:e], K, V))
            else:
                K = torch.cat([k_pr.expand(n, -1, -1, -1), k_b[s:e].repeat_interleave(rep, 1)], dim=2)
                V = torch.cat([v_pr.expand(n, -1, -1, -1), v_b[s:e].repeat_interleave(rep, 1)], dim=2)
                outs.append(F.scaled_dot_product_attention(q_b[s:e], K, V, attn_mask=mask, scale=attn.scaling))
        o = torch.cat(outs, 0)
        if flash:
            y_bs.append(attn.o_proj(o.reshape(*x_b.shape[:-1], -1).contiguous() * torch.sigmoid(g_b)))
        else:
            y_bs.append(_attn_finish(attn, o, g_b, x_b.shape[:-1]))
    return y_p, y_bs


def _layer(layer, n_b, h_p, *rest):
    """One decoder layer over the prompt and every branch bucket. ``rest`` = n_b branch hidden
    states, then n_b (cos, sin) pairs flattened, then the prompt (cos, sin), then n_b masks; flat
    tensor varargs so non-reentrant checkpointing sees every input."""
    h_bs = list(rest[:n_b])
    pe_bs = [(rest[n_b + 2 * i], rest[n_b + 2 * i + 1]) for i in range(n_b)]
    pe_p = (rest[3 * n_b], rest[3 * n_b + 1])
    masks = list(rest[3 * n_b + 2 :])
    x_p = layer.input_layernorm(h_p)
    x_bs = [layer.input_layernorm(h) for h in h_bs]
    if layer.block_type == "linear_attention":
        cache = _GDNPrefixCache()
        y_p = layer.linear_attn(hidden_states=x_p, cache_params=cache, attention_mask=None)
        cache.branch = True
        y_bs = []
        for x in x_bs:
            cache.n = x.shape[0]
            y_bs.append(layer.linear_attn(hidden_states=x, cache_params=cache, attention_mask=None))
    elif layer.block_type == "full_attention":
        y_p, y_bs = _attn_group(layer.self_attn, x_p, x_bs, pe_p, pe_bs, masks)
    else:
        raise NotImplementedError(f"prefix_share: unknown block type {layer.block_type!r}")
    h_p = h_p + y_p
    h_p = h_p + layer.mlp(layer.post_attention_layernorm(h_p))
    out = [h_p]
    for h, y in zip(h_bs, y_bs):
        h = h + y
        out.append(h + layer.mlp(layer.post_attention_layernorm(h)))
    return tuple(out)


def _group_hidden(lm, prefix_ids, buckets):
    """Final-norm hidden states of right-padded branch buckets ([n_i, L_i] each) that continue a
    shared prefix. Returns one [n_i, L_i, H] tensor per bucket."""
    bb = _backbone(lm)
    dev = prefix_ids.device
    P = prefix_ids.numel()
    h_p = bb.embed_tokens(prefix_ids[None])
    pe_p = bb.rotary_emb(h_p, torch.arange(P, device=dev)[None, None].expand(3, -1, -1))
    h_bs, pe_bs, masks = [], [], []
    for br in buckets:
        Lb = br.shape[1]
        h = bb.embed_tokens(br)
        pos_b = (P + torch.arange(Lb, device=dev))[None, None].expand(3, -1, -1)
        h_bs.append(h)
        pe_bs.extend(bb.rotary_emb(h, pos_b))  # batch 1, broadcasts over the bucket rows
        if _use_flash(bb.config):
            masks.append(None)  # FA: causal bottom-right alignment encodes the same mask, nothing materialised
        else:
            j = torch.arange(Lb, device=dev)[:, None]
            i = torch.arange(P + Lb, device=dev)[None]
            masks.append((i < P) | (i - P <= j))  # [Lb, P+Lb]: whole prompt + own causal prefix
    n_b = len(buckets)
    # HF sets the flag on the GradientCheckpointingLayer modules; we call their sub-modules directly
    # (so they never checkpoint themselves) and checkpoint the prompt+branches per layer instead.
    ckpt = (lm.training and torch.is_grad_enabled()
            and any(getattr(m, "gradient_checkpointing", False) for m in (bb, *bb.layers)))
    for layer in bb.layers[: bb.config.num_hidden_layers]:
        args = (h_p, *h_bs, *pe_bs, *pe_p, *masks)
        if ckpt:
            out = checkpoint(_layer, layer, n_b, *args, use_reentrant=False)
        else:
            out = _layer(layer, n_b, *args)
        h_p, h_bs = out[0], list(out[1:])
    return [bb.norm(h) for h in h_bs]


def _buckets(lens, ratio=0.8):
    """Length-sorted branch buckets: a branch joins the current bucket while it is at least
    ``ratio`` x the bucket's longest branch, bounding right-padding waste to (1 - ratio) per row
    (one bucket when completions are all near the cap, the truncation-heavy regime)."""
    order = sorted(range(len(lens)), key=lambda k: -lens[k])
    out, cur = [], []
    for k in order:
        if cur and lens[k] < ratio * lens[cur[0]]:
            out.append(cur)
            cur = []
        cur.append(k)
    out.append(cur)
    return out


def _logprob_fn(lm, hidden, targets, temperature, chunk_size):
    from trl.trainer.utils import _ChunkedLogProbFunction

    tc = lm.config.get_text_config()
    softcap = getattr(tc, "final_logit_softcapping", None)
    scale = getattr(tc, "logit_scale", None)
    if scale is None:
        scale = getattr(tc, "output_multiplier", None)
    scale = 1.0 if scale is None else scale
    head = lm.lm_head
    return _ChunkedLogProbFunction.apply(hidden, head.weight, head.bias, targets, temperature,
                                         chunk_size, softcap, scale)


def prefix_shared_logprobs(lm, input_ids, position_ids, completion_mask, temperature=1.0,
                           chunk_size=8192):
    """Per-token (log_probs, entropy) in the packed (1, T-1) layout, prompts forwarded once per group.

    Same contract as ``trainer.per_seq_logprobs``: slot t scores input_ids[t+1]; slots whose target
    is not a completion token are 0. Returns (log_probs, entropy, stats).
    """
    lm = _unwrap(lm)
    T = input_ids.shape[1]
    ids = input_ids[0]
    cm = completion_mask[0]
    dev = ids.device
    hid, tgt, slot_idx = [], [], []
    stats = {"groups": 0, "seqs": 0, "forward_tokens": 0, "padded_tokens": 0, "unshared_tokens": 0}
    for p, segs in plan_groups(input_ids, position_ids, completion_mask):
        stats["groups"] += 1
        stats["seqs"] += len(segs)
        stats["unshared_tokens"] += sum(e - a for a, e in segs)
        if p is None:  # no completion / no prompt: score it as its own 1-token-prefix group
            p = 1
        a0 = segs[0][0]
        P = ((p - 1) // ALIGN) * ALIGN  # shared prefix length; ALIGN=1 -> p-1 (original split)
        prefix = ids[a0 : a0 + P]
        # Branch k = prompt tail [P, p) (>= the last prompt token) + completion, so the first completion token is
        # predicted in-branch.
        lens = [e - (a + P) for a, e in segs]
        buckets = _buckets(lens)
        brs = []
        for bk in buckets:
            br = ids.new_zeros((len(bk), lens[bk[0]]))
            for r, k in enumerate(bk):
                a, e = segs[k]
                br[r, : lens[k]] = ids[a + P : e]
            brs.append(br)
        stats["forward_tokens"] += prefix.numel() + sum(lens)
        stats["padded_tokens"] += prefix.numel() + sum(b.numel() for b in brs)
        if prefix.numel():
            hs = _group_hidden(lm, prefix, brs)
        else:
            hs = [_plain_hidden(lm, br) for br in brs]
        for bk, br, h in zip(buckets, brs, hs):
            for r, k in enumerate(bk):
                a, e = segs[k]
                j = torch.arange(lens[k] - 1, device=dev)
                keep = cm[a + P + 1 : e].bool()
                j = j[keep]
                hid.append(h[r, j])
                tgt.append(br[r, j + 1])
                slot_idx.append(a + P + j)
    lp, ent = _logprob_fn(lm, torch.cat(hid), torch.cat(tgt), temperature, chunk_size)
    slots = torch.cat(slot_idx)
    log_probs = torch.zeros(T - 1, device=dev, dtype=lp.dtype).index_put((slots,), lp)
    entropy = torch.zeros(T - 1, device=dev, dtype=ent.dtype).index_put((slots,), ent)
    return log_probs[None], entropy[None], stats


def _plain_hidden(lm, br):
    bb = _backbone(lm)
    pos = torch.arange(br.shape[1], device=br.device)[None].expand(br.shape[0], -1)
    return bb(input_ids=br, position_ids=pos, use_cache=False).last_hidden_state


def _forward(self, *args, prefix_share=None, **kwargs):
    if prefix_share is None:
        return self._prefix_share_orig_forward(*args, **kwargs)
    lp, ent, stats = prefix_shared_logprobs(self, temperature=self._prefix_share_temperature,
                                            chunk_size=self._prefix_share_chunk, **prefix_share)
    return {"log_probs": lp, "entropy": ent, "prefix_share_stats": stats}


def install(model, temperature: float, chunk_size: int = 8192):
    """Route ``model(prefix_share={...})`` to the shared-prefix path; every other call is untouched.

    Must run before ``accelerator.prepare`` so accelerate's autocast/fp32-output wrappers and DDP
    wrap this forward like the original one. Everything is stored as bound methods/attributes on
    the module, so ``copy.deepcopy`` (the KL reference model) rebinds to the copy instead of
    silently calling the policy.
    """
    import types

    lm = _unwrap(model)
    if getattr(lm, "_prefix_share_orig_forward", None) is None:
        lm._prefix_share_orig_forward = lm.forward
    lm._prefix_share_temperature = float(temperature)
    lm._prefix_share_chunk = int(chunk_size)
    lm.forward = types.MethodType(_forward, lm)
    return lm


class GroupRowBatcher(torch.utils.data.IterableDataset):
    """Drop-in for TRL's ``FixedCountBatcher`` that keeps prompt groups together.

    Same sample count per micro-batch (``microbatch_size = per_device_train_batch_size x ranks``),
    so optimizer-step semantics are unchanged; but rows are filled with whole groups (consecutive
    samples sharing a ``group_id`` -- the scorer pushes a group contiguously) instead of being
    Sum-L^2 balanced sample by sample, which would scatter a group over every rank and leave each
    row with only a few copies of the prompt to share. With pdb = num_generations this puts exactly
    one group in each row, which is also what ``GSPOAsyncGRPOTrainer``'s seq_mean normalisation
    assumes. Partial groups (stale drops, a group straddling two micro-batches) become their own
    units; units are placed longest-first into the row with the smallest shared-token load, and the
    largest unit is split if there are fewer units than rows (every rank must get a non-empty row).
    """

    def __init__(self, dataset, num_processes: int, microbatch_size: int, max_row_tokens: int | None = None):
        import os

        self.dataset = dataset
        self.num_processes = num_processes
        self.microbatch_size = microbatch_size
        # Memory guard: a unit (group) whose shared-forward cost exceeds this is split in halves
        # (each half pays the prompt once) before placement. Sample count per micro-batch is
        # unchanged, so step semantics are too. Measured on 0.8B + grad-ckpt + KL ref: 25.4 GB peak at 71k tokens.
        self.max_row_tokens = max_row_tokens or int(os.environ.get("RLFORGE_PREFIX_MAX_ROW_TOKENS", 196608))

    @staticmethod
    def _cost(unit):
        cm = unit[0]["completion_mask"]
        p = next((i for i, m in enumerate(cm) if m), len(cm))
        return p + sum(len(s["input_ids"]) - p for s in unit)

    def _partition(self, batch):
        units = []
        for s in batch:
            if units and units[-1][0]["group_id"] == s["group_id"]:
                units[-1].append(s)
            else:
                units.append([s])
        split = []
        while units:
            u = units.pop()
            if len(u) > 1 and self._cost(u) > self.max_row_tokens:
                h = len(u) // 2
                units += [u[:h], u[h:]]
            else:
                split.append(u)
        units = split
        while len(units) < self.num_processes:
            units.sort(key=len, reverse=True)
            big = units.pop(0)
            if len(big) < 2:
                units.append(big)
                break
            h = len(big) // 2
            units += [big[:h], big[h:]]
        rows = [[] for _ in range(self.num_processes)]
        loads = [0] * self.num_processes
        for u in sorted(units, key=self._cost, reverse=True):
            i = min(range(self.num_processes), key=lambda j: loads[j])
            rows[i].extend(u)
            loads[i] += self._cost(u)
        return rows

    def __iter__(self):
        batch = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) == self.microbatch_size:
                yield self._partition(batch)
                batch = []


def _prompt_len(sample):
    cm = sample["completion_mask"]
    return next((i for i, m in enumerate(cm) if m), len(cm))


class GroupTokenBudgetBatcher(torch.utils.data.IterableDataset):
    """Drop-in for TRL's ``TokenBudgetBatcher`` that budgets *shared-forward* tokens.

    TRL streams each sample into the Sum-L^2-lightest row that fits ``token_budget`` real tokens,
    which scatters one group's G samples over all ranks and charges every copy of the prompt. Here a
    sample joins the row that already holds its group (cost = its completion only, the prompt is
    already paid) while that row fits; otherwise it opens its group in the lightest row that fits
    (cost = prompt + completion). Rows therefore stay within ``token_budget`` *forwarded* tokens --
    the same peak-memory bound as before -- but carry several times more samples. When no row fits,
    the micro-batch is emitted, exactly like TRL. Every emitted row is non-empty.
    """

    def __init__(self, dataset, num_processes: int, token_budget: int, metrics: dict):
        self.dataset = dataset
        self.num_processes = num_processes
        self.token_budget = token_budget
        self.metrics = metrics

    def __iter__(self):
        R = self.num_processes
        rows = [[] for _ in range(R)]
        loads = [0] * R
        groups = [set() for _ in range(R)]
        for sample in self.dataset:
            n = len(sample["input_ids"])
            if n > self.token_budget:
                self.metrics["batch/dropped_oversize_total"].append(1.0)
                continue
            g = sample["group_id"]
            tail = n - _prompt_len(sample)
            home = [i for i in range(R) if g in groups[i] and loads[i] + tail <= self.token_budget]
            if home:
                i = home[0]
                loads[i] += tail
            else:
                empty = [i for i in range(R) if not rows[i]]
                fits = [i for i in range(R) if loads[i] + n <= self.token_budget]
                if not fits:
                    yield rows
                    rows = [[] for _ in range(R)]
                    loads = [0] * R
                    groups = [set() for _ in range(R)]
                    fits = empty = list(range(R))
                # Prefer an empty row so every rank gets work before any row takes a second group.
                i = min(empty or fits, key=lambda j: loads[j])
                loads[i] += n
                groups[i].add(g)
            rows[i].append(sample)
