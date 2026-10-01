"""Qwen3.5-0.8B backbone Q-network (w4: direct RL, nothink, 128-bin dueling).

Design (RL设计与分析.md D12, format spec v1):
- input: compact token obs [B, T*5] float32 (identical to settf v2 arms),
  so actor shards / replay / mirror aug are unchanged;
- serialization to the frozen text prompt happens INSIDE forward (qwen_text),
  one shared tokenization path for learner / inference server / evaluator;
- trunk: Qwen3.5-0.8B (VL ckpt). Only the language-model path runs in this
  arm; the ViT is dropped at load time (it will serve the image arm later).
  LoRA (peft) on attention/MLP projections, everything else frozen;
  lora_r=0 -> full fine-tune;
- gradient checkpointing always enabled (only matters for backward passes);
- readout: last non-pad prompt position hidden (causal: it attends to the
  whole serialized board) -> V head + 128-bin A head (dueling) + pi head
  (aux, D9);
- output API matches DuelingQ: forward -> [B, A, 1]; q_values -> [B, A].
"""
import numpy as np
import torch
import torch.nn as nn

import qwen_text


def _load_language_model(model_path, dtype):
    """Load the VL ckpt, return just the text decoder stack (ViT freed)."""
    from transformers import AutoModelForImageTextToText
    full = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=dtype, low_cpu_mem_usage=True)
    core = getattr(full, "model", full)
    lang = getattr(core, "language_model", None)
    if lang is None:
        lang = core          # text-only checkpoint or flat layout
    # detach lang from the VL wrapper so ViT/connector can be freed
    lang = lang._orig_mod if hasattr(lang, "_orig_mod") else lang
    import gc
    del full, core
    gc.collect()
    return lang


class QwenQ(nn.Module):
    def __init__(self, n_actions, T=160, model_path="", n_text=96, max_len=768,
                 lora_r=64, lora_alpha=128, head_dim=512,
                 dtype=torch.float32):
        super().__init__()
        from transformers import AutoTokenizer
        self.T = int(T)
        self.n_actions = int(n_actions)
        self.n_quant = 1
        self.n_text = int(n_text)
        self.max_len = int(max_len)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        trunk = _load_language_model(model_path, dtype)
        if int(lora_r) > 0:
            from peft import LoraConfig, get_peft_model
            lcfg = LoraConfig(
                r=int(lora_r), lora_alpha=int(lora_alpha), lora_dropout=0.0,
                bias="none", task_type=None,   # bare decoder: no generate() API
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"])
            trunk = get_peft_model(trunk, lcfg)
        if hasattr(trunk, "enable_input_require_grads"):
            trunk.enable_input_require_grads()
        if hasattr(trunk, "gradient_checkpointing_enable"):
            try:
                trunk.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                trunk.gradient_checkpointing_enable()
        self.trunk = trunk

        hid = self._hidden_size()
        self.v_head = nn.Sequential(
            nn.Linear(hid, head_dim), nn.SiLU(), nn.Linear(head_dim, 1))
        self.a_head = nn.Sequential(
            nn.Linear(hid, head_dim), nn.SiLU(),
            nn.Linear(head_dim, self.n_actions))
        self.pi_head = nn.Sequential(
            nn.Linear(hid, head_dim), nn.SiLU(),
            nn.Linear(head_dim, self.n_actions))

    def _hidden_size(self):
        p = next(self.trunk.parameters())
        # final norm output == hidden size; use config when available
        cfg = getattr(self.trunk, "config", None)
        for attr in ("hidden_size",):
            if cfg is not None and getattr(cfg, attr, None):
                return int(cfg.hidden_size)
        return int(p.shape[-1])

    def _serialize(self, x):
        """[B, T*5] float tensor (any device) -> (ids, mask) on x.device.

        Padded length is rounded UP to the next multiple of 128 (capped at
        max_len) so the trunk only ever sees a small fixed set of sequence
        lengths — Triton autotune fires once per shape, and unbounded dynamic
        padding caused multi-second JIT stalls that blew the actors' 10s zmq
        timeout on the first real batches (attempt 1 incident).
        """
        arr = x.detach().to("cpu", torch.float32).numpy()
        enc = qwen_text.encode_prompt_batch(
            arr, self.tokenizer, n_text=self.n_text, max_len=self.max_len)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        b, n = ids.shape
        bucket = min(((n + 127) // 128) * 128, self.max_len)
        if bucket > n:
            ids = torch.nn.functional.pad(ids, (0, bucket - n))
            mask = torch.nn.functional.pad(mask, (0, bucket - n))
        return (ids.to(x.device), mask.to(x.device))

    def _hidden(self, x):
        ids, mask = self._serialize(x)
        out = self.trunk(input_ids=ids, attention_mask=mask, use_cache=False)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        last = (mask.sum(dim=1) - 1).clamp(min=0)      # right padding
        b = torch.arange(h.shape[0], device=h.device)
        return h[b, last]                               # [B, hid]

    def forward(self, x, return_pi=False, seq=None):
        """x: [B,T*5] compact obs. seq: optional (ids, mask, last, npos) to run
        the token-policy SFT path (prompt + answer tokens in ONE sequence).
        Kept inside forward() on purpose: DDP grad sync must cover every
        trainable param, and calling a submodule directly would bypass the
        wrapper's forward bookkeeping. Returns (q, [pi], ha) with
        ha = hidden at npos, [B, npos, hid], for the tying-projected logits.
        """
        if seq is not None:
            ids, mask, last, npos = seq
            out = self.trunk(input_ids=ids, attention_mask=mask,
                             use_cache=False)
            h = (out.last_hidden_state if hasattr(out, "last_hidden_state")
                 else out[0])
            b = torch.arange(h.shape[0], device=h.device)
            z = h[b, last]
            v = self.v_head(z)
            a = self.a_head(z)
            q = v + a - a.mean(dim=1, keepdim=True)
            ha = h[b[:, None], npos]
            if return_pi:
                return q.unsqueeze(-1), self.pi_head(z), ha
            return q.unsqueeze(-1), ha
        z = self._hidden(x)
        v = self.v_head(z)                                   # [B,1]
        a = self.a_head(z)                                   # [B,A]
        q = v + a - a.mean(dim=1, keepdim=True)
        if return_pi:
            # combined path (BC/distill): all trainable params participate in
            # ONE forward so DDP grad sync covers the pi head too
            return q.unsqueeze(-1), self.pi_head(z)
        return q.unsqueeze(-1)                               # [B,A,1]

    def q_values(self, x):
        return self.forward(x).mean(dim=-1)                  # [B,A]

    def pi_logits(self, x):
        return self.pi_head(self._hidden(x))

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def warmup(model, batches=(64, 16), lengths=None, log=print):
    """Pre-trigger Triton/cuBLAS JIT for every bucketed shape the trunk can
    see (bf16 autocast, no grad). Call before serving traffic; otherwise the
    first batch at a new shape stalls for seconds and clients time out.
    """
    if lengths is None:
        lengths = list(range(128, model.max_len + 1, 128))
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        for b in batches:
            for n in lengths:
                ids = torch.zeros((b, n), dtype=torch.long, device=dev)
                mask = torch.ones((b, n), dtype=torch.long, device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=dev.type == "cuda"):
                    model.trunk(input_ids=ids, attention_mask=mask,
                                use_cache=False)
                log(f"[warmup] b={b} len={n} done")
    if dev.type == "cuda":
        torch.cuda.synchronize()
    model.train(was_training)
