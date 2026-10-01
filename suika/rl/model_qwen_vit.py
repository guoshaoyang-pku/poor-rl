"""Qwen3.5-0.8B VL vision-arm Q-network (w6: image input, VLA prototype).

Design (RL设计与分析.md):
- input: board render 288x416 RGB (vis_render.py; multiples of 32 so the ViT
  token grid is exact: 9x13 = 117 image tokens after 16px patch + 2x2 merge);
- trunk: FULL Qwen3.5 VL ckpt. The ViT + merger run FROZEN under no_grad
  (BC stage 1: visual features come from pretraining; only the LM side
  learns the physics); LoRA (peft) on the language model's attention/MLP
  projections, everything else frozen; lora_r=0 -> full LM fine-tune;
- merge: image embeddings are spliced into the LM input embedding sequence
  manually (ids == image_token_id positions), then the peft-wrapped language
  model runs with plain 1-D positions (M-RoPE bypassed on purpose: the readout
  heads are trained from scratch anyway, and this keeps the forward identical
  in spirit to model_qwen.QwenQ);
- readout: last position hidden -> V head + A head (dueling) + pi head,
  same API as QwenQ: forward -> [B, A, 1] (+ pi), q_values -> [B, A].
"""
import numpy as np
import torch
import torch.nn as nn


class QwenViTQ(nn.Module):
    def __init__(self, n_actions, model_path="", lora_r=64, lora_alpha=128,
                 head_dim=512, dtype=torch.float32, freeze_visual=True):
        super().__init__()
        from transformers import (AutoModelForImageTextToText,
                                  AutoImageProcessor, AutoTokenizer)
        self.n_actions = int(n_actions)
        self.n_quant = 1
        tok = AutoTokenizer.from_pretrained(model_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.tokenizer = tok
        self.image_processor = AutoImageProcessor.from_pretrained(model_path)

        full = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=dtype, low_cpu_mem_usage=True)
        core = getattr(full, "model", full)
        # unregistered back-reference for get_image_features (visual+merger);
        # bypasses nn.Module.__setattr__ so params are not double-registered
        self.__dict__["_core_ref"] = core
        self.visual = getattr(core, "visual", None)
        if self.visual is None:
            raise RuntimeError("VL ckpt has no .visual module")
        lang = getattr(core, "language_model", None)
        if lang is None:
            lang = core
        cfg = getattr(full, "config", None)
        self.image_token_id = int(getattr(cfg, "image_token_id", 248056))
        vc = getattr(cfg, "vision_config", None)
        self.merge_size = int(getattr(vc, "spatial_merge_size", 2))
        self.temporal_patch = int(getattr(vc, "temporal_patch_size", 2))

        if freeze_visual:
            for p in self.visual.parameters():
                p.requires_grad_(False)
        if hasattr(self.visual, "gradient_checkpointing_disable"):
            self.visual.gradient_checkpointing_disable()
        self.visual.eval()

        if int(lora_r) > 0:
            from peft import LoraConfig, get_peft_model
            lcfg = LoraConfig(
                r=int(lora_r), lora_alpha=int(lora_alpha), lora_dropout=0.0,
                bias="none", task_type=None,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"])
            lang = get_peft_model(lang, lcfg)
        if hasattr(lang, "enable_input_require_grads"):
            lang.enable_input_require_grads()
        if hasattr(lang, "gradient_checkpointing_enable"):
            try:
                lang.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                lang.gradient_checkpointing_enable()
        self.trunk = lang

        hid = self._hidden_size()
        self.v_head = nn.Sequential(
            nn.Linear(hid, head_dim), nn.SiLU(), nn.Linear(head_dim, 1))
        self.a_head = nn.Sequential(
            nn.Linear(hid, head_dim), nn.SiLU(),
            nn.Linear(head_dim, self.n_actions))
        self.pi_head = nn.Sequential(
            nn.Linear(hid, head_dim), nn.SiLU(),
            nn.Linear(head_dim, self.n_actions))
        # keep references so the frozen visual travels with .to(device)
        self._freeze_visual = bool(freeze_visual)

    def _hidden_size(self):
        cfg = getattr(self.trunk, "config", None)
        if cfg is not None and getattr(cfg, "hidden_size", None):
            return int(cfg.hidden_size)
        return int(next(self.trunk.parameters()).shape[-1])

    def n_image_tokens(self, grid_thw):
        """Post-merge token count per image: t * (h/m) * (w/m).

        transformers 5.x already reports grid_t AFTER temporal merging
        (single image -> t=1), so no temporal division here.
        """
        t, h, w = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
        return t * (h // self.merge_size) * (w // self.merge_size)

    def images_to_inputs(self, images_uint8):
        """[B, H, W, 3] uint8 numpy -> (pixel_values, grid_thw, ids, mask).

        Runs the ckpt's own image processor (CPU); images are pre-sized to a
        multiple of 32 so no rescale happens, only normalize + patchify.
        """
        enc = self.image_processor(images=list(images_uint8),
                                   return_tensors="pt")
        pv = enc["pixel_values"]
        grid = enc["image_grid_thw"]            # [B, 3]
        counts = [self.n_image_tokens(g) for g in grid]
        n = max(counts)
        b = len(counts)
        ids = torch.zeros((b, n), dtype=torch.long)
        mask = torch.zeros((b, n), dtype=torch.long)
        for i, c in enumerate(counts):
            ids[i, :c] = self.image_token_id
            mask[i, :c] = 1
        return pv, grid, ids, mask

    def _hidden(self, pixel_values, grid_thw, ids, mask):
        dev = ids.device
        with torch.no_grad():
            out = self.__dict__["_core_ref"].get_image_features(
                pixel_values.to(dev), grid_thw.to(dev), return_dict=True)
            vis = out.pooler_output             # list of [tokens_i, hid]
            if isinstance(vis, (tuple, list)):
                vis = torch.cat(list(vis), dim=0)
        emb = self.trunk.get_input_embeddings()(ids)
        emb = emb.clone()
        pos = 0
        for i in range(ids.shape[0]):
            c = int(mask[i].sum())
            emb[i, :c] = vis[pos:pos + c].to(emb.dtype)
            pos += c
        out = self.trunk(inputs_embeds=emb, attention_mask=mask,
                         use_cache=False)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") \
            else out[0]
        last = (mask.sum(dim=1) - 1).clamp(min=0)
        b = torch.arange(h.shape[0], device=h.device)
        return h[b, last]                          # [B, hid]

    def forward(self, batch, return_pi=False):
        """batch = (pixel_values, grid_thw, ids, mask) from images_to_inputs
        (already on device). Returns q [B,A,1] (+ pi [B,A])."""
        pv, grid, ids, mask = batch
        z = self._hidden(pv, grid, ids, mask)
        v = self.v_head(z)
        a = self.a_head(z)
        q = v + a - a.mean(dim=1, keepdim=True)
        if return_pi:
            # combined path: all trainable params in ONE forward so DDP grad
            # sync covers the pi head too
            return q.unsqueeze(-1), self.pi_head(z)
        return q.unsqueeze(-1)

    def q_values(self, batch):
        return self.forward(batch).mean(dim=-1)

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg, _obs_dim_unused):
    """Drop-in for model.build_model in the BC learner (arch: qwen_vit)."""
    return QwenViTQ(
        int(cfg["K"]), model_path=cfg["model_path"],
        lora_r=int(cfg.get("lora_r", 64)),
        lora_alpha=int(cfg.get("lora_alpha", 128)),
        head_dim=int(cfg.get("head_dim", 512)),
        freeze_visual=bool(cfg.get("freeze_visual", True)))
