"""Generate per-arm yaml configs into configs/ (run locally before rsync)."""
import os
import yaml

BASE = dict(
    name="anchor",
    K=128, max_fruits=80, boundary=True, reward_scale=1.0,
    gamma=0.995, n_step=3, n_quant=64, double=True,
    hidden=[1024, 1024], head_dim=512,
    batch=8192, lr=3.0e-4, lr_warmup=2000, lr_floor_frac=0.1,
    weight_decay=1.0e-5, ema_tau=0.003,
    per_alpha=0.6, per_beta0=0.4, per_beta_grad_steps=2_000_000,
    replay_capacity=2_000_000, min_replay=50_000, max_reuse=16.0,
    mirror_aug=True, eps_min=0.02, eps_max=0.5,
    seed=0, grad_budget=120_000, ckpt_interval_s=600,
    actor_shard=4096,
)

ARMS = {
    # ---- node_d (gpu 0-7) ----
    "anchor_g0":      {},
    "bigbatch_g1":    {"batch": 16384, "lr": 4.0e-4},
    "lrhi_g2":        {"lr": 1.0e-3},
    "bigmodel_g3":    {"hidden": [2048, 2048]},
    "gammahi_g4":     {"gamma": 0.999},
    "nstep5_g5":      {"n_step": 5, "gamma": 0.997},
    "noper_g6":       {"per_alpha": 0.0},
    "plaindqn_g7":    {"n_quant": 1},
    # ---- node_c (gpu 0-7) ----
    "anchor_s1_g0":   {"seed": 1},
    "k64_g1":         {"K": 64},
    "k256_g2":        {"K": 256},
    "epslow_g3":      {"eps_min": 0.005, "eps_max": 0.15},
    "lrlo_g4":        {"lr": 1.0e-4},
    "nstep1_g5":      {"n_step": 1},
    "small_g6":       {"hidden": [512, 512], "batch": 4096},
    "reuse32_g7":     {"max_reuse": 32.0},
}

NODE_OF = {
    "anchor_g0": "node_d", "bigbatch_g1": "node_d", "lrhi_g2": "node_d",
    "bigmodel_g3": "node_d", "gammahi_g4": "node_d", "nstep5_g5": "node_d",
    "noper_g6": "node_d", "plaindqn_g7": "node_d",
    "anchor_s1_g0": "node_c", "k64_g1": "node_c", "k256_g2": "node_c",
    "epslow_g3": "node_c", "lrlo_g4": "node_c", "nstep1_g5": "node_c",
    "small_g6": "node_c", "reuse32_g7": "node_c",
}


WAVE2_BASE = dict(
    K=128, max_fruits=80, boundary=True, reward_scale=1.0,
    gamma=0.999, n_step=3, n_quant=1, double=True,
    hidden=[1024, 1024], head_dim=512,
    batch=8192, lr=3.0e-4, lr_warmup=0, lr_floor_frac=0.1,
    weight_decay=1.0e-5, ema_tau=0.003,
    per_alpha=0.6, per_beta0=0.4, per_beta_grad_steps=600_000,
    replay_capacity=4_000_000, min_replay=50_000, max_reuse=16.0,
    mirror_aug=True, eps_min=0.02, eps_max=0.5,
    seed=0, grad_budget=600_000, ckpt_interval_s=900,
    actor_shard=4096,
)

WAVE2_ARMS = {
    # node_d gpu0: warm-start from plaindqn_g7 step98k ckpt (path on node)
    "w2_warm_g999": {
        "init_from": "/data/user/suika-dqn/runs/wave1_20260928/"
                     "plaindqn_g7/checkpoints/step98000_env50175837.pt",
    },
    # node_d gpu1: fresh control, same recipe
    "w2_fresh_g999": {},
    # node_c gpu0: bigger batch variant
    "w2_fresh_g999_b16k": {"batch": 16384, "lr": 4.0e-4},
    # node_c gpu1: longer horizon
    "w2_fresh_g9995": {"gamma": 0.9995},
}

# wave3: n-step successor fix everywhere, fresh start, gamma=1 (D4), arch v2 (D7).
# 4 arms x 100 actors; perm: mlp_deep(gpu0)+tf_base(gpu1), t1: tf_deep(gpu0)+tf_tempo(gpu1)
WAVE3_BASE = dict(
    arch="mlp", obs_format="flat", infer="local", obs_dim=333,
    K=128, max_fruits=80, boundary=True, reward_scale=1.0,
    gamma=1.0, n_step=3, n_quant=1, double=True,
    hidden=[1024, 1024], head_dim=512,
    batch=8192, lr=3.0e-4, lr_warmup=2000, lr_floor_frac=0.1,
    weight_decay=1.0e-5, ema_tau=0.003,
    per_alpha=0.6, per_beta0=0.4, per_beta_grad_steps=600_000,
    replay_capacity=4_000_000, min_replay=50_000, max_reuse=16.0,
    mirror_aug=True, eps_min=0.02, eps_max=0.5,
    seed=0, grad_budget=600_000, ckpt_interval_s=900,
    actor_shard=4096, tempo=False,
)

# set-transformer arms: token obs (T*5=800), GPU inference server for actors,
# smaller replay (token obs is 2.4x larger per transition).
# batch 2048 (not 8192): Stage-1 activations on 160 tokens OOM a 40G card at
# 8192 (~37GB for SAB8); 2048 leaves headroom for infer server + evaluator.
_TF = dict(arch="settf", obs_format="tokens", infer="server", obs_dim=800,
           T=160, d_tok=128, d_lat=256, n_lat=32, stage1=2, stage3=4,
           tf_heads=8, batch=2048, replay_capacity=2_000_000)

WAVE3_ARMS = {
    # deep MLP anchor: 3x2048 hidden + 1024 head (~13M params), settle
    "w3_mlp_deep": {"hidden": [2048, 2048, 2048], "head_dim": 1024},
    # arch v2 base (SAB2 -> PMA 32x256 -> latent SAB4), settle;
    # inference servers on perm gpu1 (with learner) + gpu2 (spare)
    "w3_tf_base": dict(_TF, infer_gpus=[1, 2]),
    # arch v2 deep: latent self-attn x8 (user's deeper-Stage3 bet);
    # inference servers on t1 gpu0 (with learner) + gpu1 (spare)
    "w3_tf_deep": dict(_TF, stage3=8, infer_gpus=[0, 1]),
    # (w3_tf_tempo cancelled by user 16:35; D2 deferred; w3_tf_base killed 18:0x)
}

# wave3b: C-extension physics + ckpt resume + batch 64k + more infer GPUs.
# 2 arms x 200 actors, one arm per node (user 18:02 directives).
WAVE3B_BASE = dict(
    arch="mlp", obs_format="flat", infer="server", obs_dim=333,
    K=128, max_fruits=80, boundary=True, reward_scale=1.0,
    gamma=1.0, n_step=3, n_quant=1, double=True,
    hidden=[1024, 1024], head_dim=512,
    batch=65536, grad_accum=1, lr=6.0e-4, lr_warmup=500, lr_floor_frac=0.1,
    weight_decay=1.0e-5, ema_tau=0.003,
    per_alpha=0.6, per_beta0=0.4, per_beta_grad_steps=600_000,
    replay_capacity=4_000_000, min_replay=50_000, max_reuse=16.0,
    mirror_aug=True, eps_min=0.02, eps_max=0.5,
    seed=0, grad_budget=600_000, ckpt_interval_s=900, ckpt_keep=4,
    actor_shard=4096, tempo=False,
)

WAVE3B_ARMS = {
    # perm: deep MLP control; learner gpu0 (+eval), infer servers gpu1,2
    "w3b_mlp_deep": dict(
        hidden=[2048, 2048, 2048], head_dim=1024, infer_gpus=[1, 2],
        resume_from_dir="/data/user/suika-dqn/runs/"
                        "wave3_20260928/w3_mlp_deep"),
    # t1: set-transformer SAB8 champion; learner gpu0 (+eval+1 server),
    # infer servers gpu0-3; 64k batch via 32x2048 grad accumulation
    "w3b_tf_deep": dict(
        arch="settf", obs_format="tokens", obs_dim=800,
        T=160, d_tok=128, d_lat=256, n_lat=32, stage1=2, stage3=8,
        tf_heads=8, grad_accum=32, infer_gpus=[0, 1, 2, 3],
        # note: live arm runs replay_capacity=4M (base value; 2M override was
        # edited in after the yaml had already been generated and shipped —
        # 4M token-obs replay is 25.6GB and the node handles it fine)
        resume_from_dir="/path/to/suika-dqn/runs/"
                        "wave3_20260928/w3_tf_deep"),
}


# wave3c: scaled-up set transformer, fresh from scratch (D15, 20260929).
# 24h budget; learner gpu0 dedicated, evaluator gpu1, infer servers gpu2-7.
WAVE3C_BASE = dict(WAVE3B_BASE)
WAVE3C_ARMS = {
    # t1: ~56M-param set transformer (d_lat 512, 64 latents, SAB16).
    # grad_ckpt lets micro-batch stay 8192 on the 40G card.
    "w3c_tf_xl": dict(
        arch="settf", obs_format="tokens", obs_dim=800,
        T=160, d_tok=128, d_lat=512, n_lat=64, stage1=2, stage3=16,
        tf_heads=8, grad_ckpt=True, grad_accum=8, torch_compile=True,
        # 24h budget is learner-compute-bound (~13s/step at 32k batch):
        # draws/day ~214M regardless of batch, so 32k doubles opt steps vs 64k.
        batch=32768, lr=4.5e-4,
        replay_capacity=8_000_000,
        grad_budget=6_000, per_beta_grad_steps=6_000,
        # t1 container rebuilt 0929 ~11:00; resume from the frozen 10:33 ckpt
        resume_from_dir="/path/to/suika-dqn/runs/"
                        "wave3c_20260929/w3c_tf_xl",
        infer_gpus=[2, 3, 4, 5, 6, 7],
    ),
}


# wave4 (D16, 20260929): kill-line ablation. Same tf_xl recipe as wave3c so the
# curves are step-for-step comparable with the killy=170 baseline; only the
# death line moves (170 -> 200 via SUIKA_KILLY=200, see part2/config.py).
# killy is a y-down screen coordinate: 200 puts the line 30px LOWER, so the
# over-the-line band grows from [85,170] to [85,200] and the usable stack
# height shrinks 505 -> 475 px. That is the harder direction.
# The set-transformer token obs does not encode killy, so obs are identical
# and a wave3c ckpt loads into the new env without any shape/feature change.
WAVE4_BASE = dict(WAVE3C_BASE)
WAVE4_ARMS = {
    # t1_3 (fully idle): from scratch. learner gpu0, eval gpu1, infer gpu2-7.
    "w4_k200_scratch": dict(
        arch="settf", obs_format="tokens", obs_dim=800,
        T=160, d_tok=128, d_lat=512, n_lat=64, stage1=2, stage3=16,
        tf_heads=8, grad_ckpt=True, grad_accum=8, torch_compile=True,
        batch=32768, lr=4.5e-4,
        replay_capacity=8_000_000,
        grad_budget=6_000, per_beta_grad_steps=6_000,
        # points at its own run dir: empty on first launch (starts fresh),
        # correct resume of its own progress if the arm is ever restarted
        resume_from_dir="/path/to/suika-dqn/runs/"
                        "wave4_20260929/w4_k200_scratch",
        infer_gpus=[2, 3, 4, 5, 6, 7],
    ),
    # t1 (coexists with the running wave3c arm): continue from the wave3c
    # killy=170 ckpt, seeded into this run dir before launch. Learner on gpu1
    # (gpu0 is wave3c's learner); infer servers share gpu2-7 with wave3c
    # (~40% -> ~80% each). budget 8000 = 3801 (seed) + ~4200 new steps.
    "w4_k200_cont": dict(
        arch="settf", obs_format="tokens", obs_dim=800,
        T=160, d_tok=128, d_lat=512, n_lat=64, stage1=2, stage3=16,
        tf_heads=8, grad_ckpt=True, grad_accum=8, torch_compile=True,
        batch=32768, lr=4.5e-4,
        replay_capacity=8_000_000,
        grad_budget=8_000, per_beta_grad_steps=6_000,
        resume_from_dir="/path/to/suika-dqn/runs/"
                        "wave4_20260929/w4_k200_cont",
        infer_gpus=[2, 3, 4, 5, 6, 7],
    ),
}


# wave5 (D19, 20260929): tall-board tf_xl, two arms, both warm-started from the
# best killy=200 ckpt (w4_k200_cont). New tf default from the throughput
# ablation: sdpa attention + no grad-ckpt + micro-batch 1024 (3013 vs 2594
# samples/s on the A100-40G). Board geometry is now a config knob:
#   geometry: {width: int|[lo,hi], killy: int|[lo,hi]}  drawn per episode seed
# killy is a y-DOWN coordinate: larger = death line lower = LESS room = harder
# (170 stock, 505px stack height; 220 -> 455px; 250 -> 425px).
# Every training episode's full action sequence is written to actions_a*.jsonl
# and every greedy-eval episode to eval_actions.jsonl (seed+board+actions
# replay the game exactly; see replay_actions.py).
_SEED_CKPT = ("/path/to/suika-dqn/runs/wave5_20260929/"
              "seed_w4_k200_cont.pt")
WAVE5_BASE = dict(WAVE3B_BASE)
_TFXL5 = dict(
    arch="settf", obs_format="tokens", T=160, d_tok=128, d_lat=512, n_lat=64,
    stage1=2, stage3=16, tf_heads=8, attn_impl="sdpa", grad_ckpt=False,
    torch_compile=True, batch=32768, grad_accum=32,          # micro 1024
    lr=2.0e-4, lr_warmup=300, lr_floor_frac=0.25,          # fresh AdamW state: peak below the w4 4.5e-4
    replay_capacity=8_000_000, grad_budget=8_000, per_beta_grad_steps=8_000,
    init_from=_SEED_CKPT, log_actions=True, eval_max_moves=1500,
)
WAVE5_ARMS = {
    # node_a (6 GPUs): fixed tall board, killy=220 (455px stack height).
    # learner gpu0, evaluator gpu1, infer servers gpu1-5.
    "w5_k220_fixed": dict(
        _TFXL5, obs_dim=800, geo_dim=0, seed=5,
        geometry={"killy": 220},
        eval_grid=[[448, 220]], eval_grid_seeds=12,
        resume_from_dir="/path/to/suika-dqn/runs/"
                        "wave5_20260929/w5_k220_fixed",
        infer_gpus=[1, 2, 3, 4, 5],
    ),
    # node_b (8 GPUs, gpu7 reserved for the LLM job): variable board,
    # width 352..544 (stock 448), killy 170..250, drawn per episode. The model
    # sees the board via a 3-dim geo block (zero-init => warm start exact).
    # learner gpu0, evaluator gpu1, infer servers gpu1-6.
    "w5_var": dict(
        _TFXL5, obs_dim=803, geo_dim=3, seed=6,
        geometry={"width": [352, 544], "killy": [170, 250]},
        eval_grid=[[w, k] for k in (170, 210, 250) for w in (352, 448, 544)],
        eval_grid_seeds=4,
        resume_from_dir="/path/to/suika-dqn/runs/"
                        "wave5_20260929/w5_var",
        infer_gpus=[1, 2, 3, 4, 5, 6],
    ),
}


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    os.makedirs(out, exist_ok=True)
    for arm, ov in ARMS.items():
        cfg = dict(BASE)
        cfg.update(ov)
        cfg["name"] = arm
        path = os.path.join(out, f"{arm}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=True)
        print("wrote", path)
    for arm, ov in WAVE2_ARMS.items():
        cfg = dict(WAVE2_BASE)
        cfg.update(ov)
        cfg["name"] = arm
        path = os.path.join(out, f"{arm}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=True)
        print("wrote", path)
    for arm, ov in WAVE3_ARMS.items():
        cfg = dict(WAVE3_BASE)
        cfg.update(ov)
        cfg["name"] = arm
        path = os.path.join(out, f"{arm}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=True)
        print("wrote", path)
    for arm, ov in WAVE3B_ARMS.items():
        cfg = dict(WAVE3B_BASE)
        cfg.update(ov)
        cfg["name"] = arm
        path = os.path.join(out, f"{arm}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=True)
        print("wrote", path)
    for arm, ov in WAVE3C_ARMS.items():
        cfg = dict(WAVE3C_BASE)
        cfg.update(ov)
        cfg["name"] = arm
        path = os.path.join(out, f"{arm}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=True)
        print("wrote", path)
    for arm, ov in WAVE4_ARMS.items():
        cfg = dict(WAVE4_BASE)
        cfg.update(ov)
        cfg["name"] = arm
        path = os.path.join(out, f"{arm}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=True)
        print("wrote", path)
    for arm, ov in WAVE5_ARMS.items():
        cfg = dict(WAVE5_BASE)
        cfg.update(ov)
        cfg["name"] = arm
        path = os.path.join(out, f"{arm}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=True)
        print("wrote", path)


if __name__ == "__main__":
    main()
