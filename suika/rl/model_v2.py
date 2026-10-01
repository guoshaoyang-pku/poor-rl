"""Architecture v2: Perceiver-style set-transformer Q network (D7/D11).

Input: flat [B, T*5] token set from encoding.encode_tokens
(row 0 = current fruit, row 1 = next fruit, rows 2.. = board fruits;
padded rows have type = -1).

Pipeline (design doc section 2.4):
  Stage 0  per-fruit embedding: type_emb(16d) + radius + Fourier(x,y) + [vx,vy]
           -> Linear -> d_tok, plus learnable role embedding (current/next/board)
  Stage 1  fruit-level self-attention x stage1 (pairwise geometry before pooling)
  Stage 2  cross-attention: n_lat learnable latent slots (query) read fruits (K/V)
  Stage 3  latent self-attention x stage3 (heavy compute on n_lat slots only)
  Readout  V: pool(latents) -> MLP -> scalar
           A: n_actions learnable column queries cross-attend latents
              + linear skip from pooled latents
           pi: pool -> MLP -> n_actions logits (aux losses only, D9)
  Q = V + A - mean(A), returned as [B, A, 1] to match the quantile API.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _cp

from paths import setup_engine_path
setup_engine_path()

from config import config  # noqa: E402

_TOK_FEATS = 5
# geo block layout (encoding.geo_vector): [(W-448)/75, ((bot-killy)-505)/110,
# ((killy-top)-85)/85] = [width, floor distance, headroom]. Everything the
# per-token features need (container height, death-line y) is derived from
# these live, so a moving floor is handled correctly.
_GEO_W_REF, _GEO_W_SCALE = 448.0, 75.0
_GEO_H_REF, _GEO_H_SCALE = 505.0, 110.0   # bot - killy
_GEO_A_REF, _GEO_A_SCALE = 85.0, 85.0     # killy - top
_TOP_REF = 85.0              # stock container top (pinned)
_GEO_LEN_NORM = 250.0          # px scale of the derived per-token features
_GEO_EXTRA = 3                 # left gap, right gap, depth below death line
_RADIUS_TABLE = [float(config[t, "radius"]) for t in range(11)]
_MAX_RADIUS = _RADIUS_TABLE[10]
_PAD_TYPE = 11


class Block(nn.Module):
    """Pre-LN attention block: self-attn when kv is None, cross-attn otherwise."""

    def __init__(self, d, heads, ff, kv_dim=None):
        super().__init__()
        self.cross = kv_dim is not None
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(kv_dim if self.cross else d)
        self.attn = nn.MultiheadAttention(
            d, heads, batch_first=True,
            kdim=kv_dim if self.cross else d,
            vdim=kv_dim if self.cross else d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, kv=None, pad_mask=None):
        h = self.ln_q(x)
        if self.cross:
            k = self.ln_kv(kv)
            a, _ = self.attn(h, k, k, key_padding_mask=pad_mask,
                             need_weights=False)
        else:
            a, _ = self.attn(h, h, h, key_padding_mask=pad_mask,
                             need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class SDPABlock(nn.Module):
    """Same pre-LN block, but attention runs through fused SDPA.

    nn.MultiheadAttention materialises the [B, H, T, T] score matrix and has
    no flash/mem-efficient path, which is what forces grad_ckpt at large
    micro-batches. Parameter names differ from Block (q/k/v/out projections
    instead of packed in_proj_weight), so `to_sdpa_state_dict` converts.
    """

    def __init__(self, d, heads, ff, kv_dim=None):
        super().__init__()
        self.cross = kv_dim is not None
        self.heads = int(heads)
        self.dh = d // self.heads
        kvd = kv_dim if self.cross else d
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(kvd)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(kvd, d)
        self.v_proj = nn.Linear(kvd, d)
        self.out_proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def _split(self, p, dim):
        B = p.shape[0]
        return p.view(B, -1, self.heads, self.dh).transpose(1, 2)

    def forward(self, x, kv=None, pad_mask=None):
        h = self.ln_q(x)
        s = self.ln_kv(kv) if self.cross else h
        q = self._split(self.q_proj(h), self.dh)
        k = self._split(self.k_proj(s), self.dh)
        v = self._split(self.v_proj(s), self.dh)
        am = None
        if pad_mask is not None:
            # nn.MultiheadAttention semantics: key_padding_mask True = ignore.
            # SDPA bool masks are the opposite: True = keep.
            am = (~pad_mask)[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        a = a.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        x = x + self.out_proj(a)
        x = x + self.ff(self.ln2(x))
        return x


def _block(d, heads, ff, kv_dim, impl):
    return SDPABlock(d, heads, ff, kv_dim) if impl == "sdpa" else \
        Block(d, heads, ff, kv_dim)


def to_sdpa_state_dict(sd):
    """Map a Block-keyed state_dict onto SDPABlock keys (same shapes).

    Block indices are inferred from the `.attn.in_proj_weight` keys present,
    so this works for any stage1/stage3 depth.
    """
    import re
    pat = re.compile(r"^((?:stage1|stage3)\.\d+|cross)\.attn\.in_proj_weight$")
    names = [m.group(1) for k in sd if (m := pat.match(k))]
    out = {}
    for p in names:
        # everything not under `.attn.` (LayerNorms, feed-forward) keeps keys
        for k, val in sd.items():
            if k.startswith(f"{p}.") and ".attn." not in k:
                out[k] = val
        w = sd[f"{p}.attn.in_proj_weight"]                     # [3d, d]
        b = sd[f"{p}.attn.in_proj_bias"]
        d8 = w.shape[0] // 3
        for j, proj in enumerate(("q_proj", "k_proj", "v_proj")):
            out[f"{p}.{proj}.weight"] = w[j * d8:(j + 1) * d8]
            out[f"{p}.{proj}.bias"] = b[j * d8:(j + 1) * d8]
        out[f"{p}.out_proj.weight"] = sd[f"{p}.attn.out_proj.weight"]
        out[f"{p}.out_proj.bias"] = sd[f"{p}.attn.out_proj.bias"]
    for k, val in sd.items():
        if not k.startswith(("stage1.", "cross.", "stage3.")):
            out[k] = val
    return out


def adapt_state_dict(sd, model):
    """Fit a checkpoint state_dict onto `model` (warm start across arch edits).

    Handles: eager->sdpa attention key renaming, and adding board-geometry
    conditioning (`geo_*` params missing from the ckpt keep their zero init;
    `in_proj.weight` gains zero columns for the derived geometry features).
    Anything else that does not line up raises instead of loading silently.
    """
    msd = model.state_dict()
    if getattr(model, "attn_impl", "eager") == "sdpa" and any(
            k.endswith(".attn.in_proj_weight") for k in sd):
        sd = to_sdpa_state_dict(sd)
    out = {}
    for k, v in sd.items():
        if k not in msd:
            raise KeyError(f"ckpt key {k!r} not in model")
        if v.shape != msd[k].shape:
            if (k == "in_proj.weight" and v.shape[0] == msd[k].shape[0]
                    and v.shape[1] < msd[k].shape[1]):
                w = torch.zeros_like(msd[k])
                w[:, :v.shape[1]] = v
                v = w
            else:
                raise ValueError(f"{k}: ckpt {tuple(v.shape)} vs model "
                                 f"{tuple(msd[k].shape)}")
        out[k] = v
    missing = [k for k in msd if k not in out]
    bad = [k for k in missing if not k.startswith("geo_")]
    if bad:
        raise KeyError(f"model keys missing from ckpt: {bad[:5]}")
    for k in missing:
        out[k] = msd[k]
    return out


class SetTransformerQ(nn.Module):
    def __init__(self, n_actions, T=160, d_tok=128, d_lat=256, n_lat=32,
                 stage1=2, stage3=4, heads=8, type_dim=16, head_dim=512,
                 fourier_freqs=4, grad_ckpt=False, attn_impl="eager",
                 geo_dim=0):
        super().__init__()
        self.T = int(T)
        self.geo_dim = int(geo_dim)
        self.n_actions = int(n_actions)
        self.n_quant = 1
        self.grad_ckpt = bool(grad_ckpt)
        self.attn_impl = str(attn_impl)
        self.type_emb = nn.Embedding(12, type_dim)  # 11 types + pad slot
        self.role_emb = nn.Parameter(torch.zeros(3, d_tok))
        self.register_buffer(
            "radius_tab", torch.tensor(_RADIUS_TABLE + [0.0]) / _MAX_RADIUS)
        self.register_buffer("freqs", 2.0 ** torch.arange(fourier_freqs) * np.pi)
        in_dim = type_dim + 1 + (2 * fourier_freqs + 1) * 2 + 2
        if self.geo_dim:
            in_dim += _GEO_EXTRA
        self.in_proj = nn.Linear(in_dim, d_tok)
        self.stage1 = nn.ModuleList(
            _block(d_tok, heads, d_tok * 4, None, self.attn_impl)
            for _ in range(stage1))
        self.tok_to_lat = nn.Linear(d_tok, d_lat)
        self.latents = nn.Parameter(torch.randn(n_lat, d_lat) * 0.02)
        self.cross = _block(d_lat, heads, d_lat * 4, d_lat, self.attn_impl)
        self.stage3 = nn.ModuleList(
            _block(d_lat, heads, d_lat * 4, None, self.attn_impl)
            for _ in range(stage3))
        self.v_head = nn.Sequential(
            nn.Linear(d_lat, head_dim), nn.SiLU(), nn.Linear(head_dim, 1))
        self.col_queries = nn.Parameter(torch.randn(n_actions, d_lat) * 0.02)
        self.a_attn = nn.MultiheadAttention(d_lat, heads, batch_first=True)
        self.a_out = nn.Linear(d_lat, 1)
        self.a_skip = nn.Linear(d_lat, n_actions)
        self.pi_head = nn.Sequential(
            nn.Linear(d_lat, head_dim), nn.SiLU(),
            nn.Linear(head_dim, n_actions))
        if self.geo_dim:
            # Board-geometry conditioning, appended LAST so the parameter order
            # of a geo-less checkpoint is a prefix of this model's (optimizer
            # state maps by index). Zero-init + zero-init new in_proj columns
            # make a warm start from a geo-less ckpt exactly function-preserving.
            self.geo_tok = nn.Linear(self.geo_dim, d_tok)
            self.geo_lat = nn.Linear(self.geo_dim, d_lat)
            for m in (self.geo_tok, self.geo_lat):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
            with torch.no_grad():
                self.in_proj.weight[:, -_GEO_EXTRA:].zero_()

    def _split(self, x):
        n = self.T * _TOK_FEATS
        if self.geo_dim:
            return x[:, :n], x[:, n:n + self.geo_dim]
        return x, None

    def _embed(self, x):
        B = x.shape[0]
        x, geo = self._split(x)
        tok = x.reshape(B, self.T, _TOK_FEATS)
        tids = tok[..., 0].long()
        pad = tids < 0
        tids_c = torch.where(pad, torch.full_like(tids, _PAD_TYPE), tids)
        cont = tok[..., 1:].float()                    # x, y, vx, vy
        xy = cont[..., :2]
        ang = xy.unsqueeze(-1) * self.freqs            # [B,T,2,F]
        four = torch.cat([xy, ang.sin().flatten(-2), ang.cos().flatten(-2)],
                         dim=-1)
        rad = self.radius_tab[tids_c].unsqueeze(-1)
        parts = [self.type_emb(tids_c), rad, four, cont[..., 2:]]
        if geo is not None:
            geo = geo.float()
            w = (_GEO_W_REF + geo[:, 0] * _GEO_W_SCALE).unsqueeze(1)   # [B,1]
            h_below = _GEO_H_REF + geo[:, 1] * _GEO_H_SCALE            # bot-killy
            h_above = _GEO_A_REF + geo[:, 2] * _GEO_A_SCALE            # killy-top
            play_h = (h_below + h_above).unsqueeze(1)                  # [B,1]
            killy = (_TOP_REF + h_above).unsqueeze(1)                  # [B,1]
            xn, yn = cont[..., 0], cont[..., 1]
            gap_l = xn * w / _GEO_LEN_NORM
            gap_r = (1.0 - xn) * w / _GEO_LEN_NORM
            # game over tests the fruit CENTRE against killy (p.pos[1] < killy),
            # so depth below the line is measured centre-to-line; y must be
            # denormalised with the LIVE container height (the floor moves).
            y_phys = (killy - h_above.unsqueeze(1)) + yn * play_h
            below = (y_phys - killy) / _GEO_LEN_NORM
            extra = torch.stack([gap_l, gap_r, below], dim=-1)
            # rows 0/1 (current/next) carry no position; mask them and padding
            board = torch.ones(self.T, dtype=extra.dtype, device=x.device)
            board[:2] = 0.0
            parts.append(extra * ((~pad).unsqueeze(-1) * board.unsqueeze(-1)))
        feats = torch.cat(parts, dim=-1)
        h = self.in_proj(feats)
        role = torch.ones(self.T, dtype=torch.long, device=x.device)
        role[0] = 0
        h = h + self.role_emb[role]
        geo_lat = None
        if geo is not None:
            h = h + self.geo_tok(geo).unsqueeze(1)
            geo_lat = self.geo_lat(geo)
        return h, pad, geo_lat

    def _trunk(self, x):
        h, pad, geo_lat = self._embed(x)
        ckpt = self.grad_ckpt and torch.is_grad_enabled()
        for blk in self.stage1:
            if ckpt:
                h = _cp.checkpoint(lambda t, b=blk: b(t, pad_mask=pad), h,
                                   use_reentrant=False)
            else:
                h = blk(h, pad_mask=pad)
        kl = self.tok_to_lat(h)
        lat = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)
        if geo_lat is not None:
            lat = lat + geo_lat.unsqueeze(1)
        if ckpt:
            # cross attends over the full T-token set; its kv activations are
            # the memory hog at large micro-batches, so checkpoint it too
            lat = _cp.checkpoint(lambda l: self.cross(l, kv=kl, pad_mask=pad),
                                 lat, use_reentrant=False)
        else:
            lat = self.cross(lat, kv=kl, pad_mask=pad)
        for blk in self.stage3:
            if ckpt:
                lat = _cp.checkpoint(blk, lat, use_reentrant=False)
            else:
                lat = blk(lat)
        return lat

    def forward(self, x):
        lat = self._trunk(x)
        pooled = lat.mean(dim=1)
        v = self.v_head(pooled)                                   # [B,1]
        q = self.col_queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        a_tok, _ = self.a_attn(q, lat, lat, need_weights=False)
        a = self.a_out(a_tok).squeeze(-1) + self.a_skip(pooled)   # [B,A]
        qvals = v + a - a.mean(dim=1, keepdim=True)
        return qvals.unsqueeze(-1)                                # [B,A,1]

    def q_values(self, x):
        return self.forward(x).mean(dim=-1)

    def pi_logits(self, x):
        return self.pi_head(self._trunk(x).mean(dim=1))
