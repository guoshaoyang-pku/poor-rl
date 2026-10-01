"""Dueling (optionally quantile) Q-network with a LayerNorm MLP trunk."""
import torch
import torch.nn as nn


def _mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.LayerNorm(dims[i + 1]),
                   nn.SiLU()]
    return nn.Sequential(*layers)


class DuelingQ(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(1024, 1024), n_quant=1,
                 head_dim=512):
        super().__init__()
        self.n_actions = int(n_actions)
        self.n_quant = int(n_quant)
        self.trunk = _mlp([obs_dim, *hidden])
        self.v_head = nn.Sequential(
            nn.Linear(hidden[-1], head_dim), nn.SiLU(), nn.Linear(head_dim, n_quant))
        self.a_head = nn.Sequential(
            nn.Linear(hidden[-1], head_dim), nn.SiLU(),
            nn.Linear(head_dim, n_actions * n_quant))

    def forward(self, x):
        """x: [B, obs_dim] -> quantiles [B, n_actions, n_quant]."""
        h = self.trunk(x)
        v = self.v_head(h).unsqueeze(1)                      # [B, 1, Q]
        a = self.a_head(h).view(-1, self.n_actions, self.n_quant)
        return v + a - a.mean(dim=1, keepdim=True)

    def q_values(self, x):
        return self.forward(x).mean(dim=-1)                  # [B, n_actions]


def build_model(cfg, obs_dim):
    if cfg.get("arch") == "qwen":
        from model_qwen import QwenQ
        return QwenQ(
            n_actions=cfg["K"],
            T=int(cfg.get("T", 160)),
            model_path=cfg["model_path"],
            n_text=int(cfg.get("n_text", 96)),
            max_len=int(cfg.get("max_len", 640)),
            lora_r=int(cfg.get("lora_r", 64)),
            lora_alpha=int(cfg.get("lora_alpha", 128)),
            head_dim=int(cfg.get("head_dim", 512)),
        )
    if cfg.get("arch") == "settf":
        from model_v2 import SetTransformerQ
        return SetTransformerQ(
            n_actions=cfg["K"],
            T=int(cfg.get("T", 160)),
            d_tok=int(cfg.get("d_tok", 128)),
            d_lat=int(cfg.get("d_lat", 256)),
            n_lat=int(cfg.get("n_lat", 32)),
            stage1=int(cfg.get("stage1", 2)),
            stage3=int(cfg.get("stage3", 4)),
            heads=int(cfg.get("tf_heads", 8)),
            head_dim=int(cfg.get("head_dim", 512)),
            grad_ckpt=bool(cfg.get("grad_ckpt", False)),
            attn_impl=str(cfg.get("attn_impl", "eager")),
            geo_dim=int(cfg.get("geo_dim", 0)),
        )
    return DuelingQ(
        obs_dim=obs_dim,
        n_actions=cfg["K"],
        hidden=tuple(cfg.get("hidden", (1024, 1024))),
        n_quant=int(cfg.get("n_quant", 64)),
        head_dim=int(cfg.get("head_dim", 512)),
    )


def param_count(model):
    return sum(p.numel() for p in model.parameters())
