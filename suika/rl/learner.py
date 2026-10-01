"""Learner process: QR/dueling double-DQN with PER, EMA target, bf16 AMP."""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from model import build_model, param_count
from replay import PrioritizedReplay


def quantile_huber_loss(pred, target, weights, kappa=1.0):
    """pred: [B, A, Q] chosen-action quantiles [B, Q]; target: [B, Q']."""
    # pred [B,Q,1], target [B,1,Q'] -> td [B,Q,Q']
    td = target.unsqueeze(1) - pred.unsqueeze(2)
    abs_td = td.abs()
    huber = torch.where(abs_td <= kappa, 0.5 * td ** 2, kappa * (abs_td - 0.5 * kappa))
    tau = (torch.arange(pred.shape[1], device=pred.device, dtype=torch.float32)
           + 0.5) / pred.shape[1]
    loss = (tau.view(1, -1, 1) - (td.detach() < 0).float()).abs() * huber / kappa
    per_sample = loss.sum(dim=2).mean(dim=1)          # [B]
    return (per_sample * weights).mean(), per_sample.detach()


class Learner:
    def __init__(self, cfg, obs_dim, run_dir, device):
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = device
        self.batch = int(cfg["batch"])
        self.gamma = float(cfg["gamma"])
        self.K = int(cfg["K"])
        self.max_fruits = int(cfg["max_fruits"])
        self.boundary = bool(cfg.get("boundary", True))
        self.n_quant = int(cfg.get("n_quant", 64))
        self.mirror_aug = bool(cfg.get("mirror_aug", True))
        self.tokens = cfg.get("obs_format") == "tokens"
        self.T = int(cfg.get("T", 160))
        self.beta0 = float(cfg.get("per_beta0", 0.4))
        self.beta_frames = float(cfg.get("per_beta_grad_steps", 2_000_000))
        self.max_reuse = float(cfg.get("max_reuse", 16.0))
        self.total_grad = int(cfg.get("grad_budget", 2_000_000))

        self.online = build_model(cfg, obs_dim).to(device)
        self.target = build_model(cfg, obs_dim).to(device)
        # compiled twins used only for forward calls; state_dict/EMA stay on
        # the raw modules so ckpt keys and publish payloads are unaffected.
        self.online_c = self.online
        self.target_c = self.target
        init_from = cfg.get("init_from")
        ema_loaded = False
        if init_from:
            ck = torch.load(init_from, map_location="cpu", weights_only=False)
            if cfg.get("arch") == "settf":
                # warm start across arch edits (eager->sdpa, +geo conditioning);
                # optimizer state is NOT carried over (fresh AdamW + lr warmup),
                # target/EMA comes from the ckpt's EMA when it has one.
                from model_v2 import adapt_state_dict
                self.online.load_state_dict(
                    adapt_state_dict(ck["state_dict"], self.online))
                if "ema" in ck:
                    self.target.load_state_dict(
                        adapt_state_dict(ck["ema"], self.target))
                    ema_loaded = True
            else:
                self.online.load_state_dict(ck["state_dict"])
            print(f"[learner] warm-start from {init_from} "
                  f"(src_grad_steps={ck.get('grad_steps')}, "
                  f"src_env_steps={ck.get('env_steps')})", flush=True)
        if not ema_loaded:
            self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        # geometry-conditioning params were zero-init at warm start (identity
        # at stock geometry); after ~650 steps they were still ~0.004 while the
        # token pathway sits at ~0.17, i.e. the policy was functionally
        # geometry-blind. Give them an LR multiplier to catch up.
        geo_mult = float(cfg.get("geo_lr_mult", 1.0))
        is_geo = lambda n: "geo_tok" in n or "geo_lat" in n  # noqa: E731
        named = list(self.online.named_parameters())
        geo_p = [p for n, p in named if is_geo(n)]
        other = [p for n, p in named if not is_geo(n)]
        if geo_mult != 1.0 and geo_p:
            groups = [{"params": other, "lr_scale": 1.0},
                      {"params": geo_p, "lr_scale": geo_mult}]
            print(f"[learner] geo_lr_mult={geo_mult} on "
                  f"{sum(p.numel() for p in geo_p)} geo params", flush=True)
        else:
            groups = [{"params": other + geo_p, "lr_scale": 1.0}]
        self.opt = torch.optim.AdamW(
            groups, lr=float(cfg["lr"]),
            weight_decay=float(cfg.get("weight_decay", 1e-5)),
            betas=(0.9, 0.95))
        if cfg.get("torch_compile") and device == "cuda":
            self.online_c = torch.compile(self.online, dynamic=False)
            self.target_c = torch.compile(self.target, dynamic=False)
        self.warmup = int(cfg.get("lr_warmup", 2000))
        self.replay = PrioritizedReplay(
            int(cfg.get("replay_capacity", 2_000_000)), obs_dim,
            alpha=float(cfg.get("per_alpha", 0.6)))
        self.tau = float(cfg.get("ema_tau", 0.003))
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)) + 777)
        self.grad_steps = 0
        self.grad_draws = 0     # sum of batch over steps; reuse gate unit
        self.policy_path = os.path.join(run_dir, "policy.pt")
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self._last_publish = 0.0
        self._last_ckpt = 0.0
        self._last_ckpt_step = 0
        self._watch_hits = 0
        self._loss_ema = None
        self._q_ema = None

    def lr_at(self, step):
        base = float(self.cfg["lr"])
        if step < self.warmup:
            return base * (step + 1) / self.warmup
        t = (step - self.warmup) / max(1, self.total_grad - self.warmup)
        t = min(1.0, t)
        floor = float(self.cfg.get("lr_floor_frac", 0.1))
        return base * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * t)))

    def _mirror(self, obs, act):
        """Horizontally mirror obs (in place on a clone) and actions."""
        B = obs.shape[0]
        if self.tokens:
            # a trailing geo block (geo_dim cols, left/right symmetric) sits
            # after the T*5 token columns and is left untouched
            tok = obs[:, :self.T * 5].view(B, self.T, 5)
            valid = tok[:, :, 0] >= 0
            valid[:, :2] = False          # current/next rows carry no position
            x = tok[:, :, 1]
            vx = tok[:, :, 3]
            tok[:, :, 1] = torch.where(valid, 1.0 - x, x)
            tok[:, :, 3] = torch.where(valid, -vx, vx)
            return obs, self.K - 1 - act
        gdim = 9 + (4 if self.boundary else 0)
        tok = obs[:, gdim:].view(B, self.max_fruits, 4)
        x = tok[:, :, 0]
        valid = tok[:, :, 2] > 0
        tok[:, :, 0] = torch.where(valid, 1.0 - x, torch.zeros_like(x))
        if self.boundary:
            tmp = obs[:, 9].clone()
            obs[:, 9] = obs[:, 10]
            obs[:, 10] = tmp
        return obs, self.K - 1 - act

    def train_step(self):
        beta = min(1.0, self.beta0 + (1.0 - self.beta0)
                   * self.grad_steps / max(1.0, self.beta_frames))
        accum = int(self.cfg.get("grad_accum", 1))
        micro = self.batch // accum
        # one PER sample for the whole batch; forward/backward in micro-batches
        (idx, obs, act, rew, nobs, done, gam, w) = self.replay.sample(
            self.batch, beta, self.rng)
        obs = torch.from_numpy(obs).to(self.device)
        nobs = torch.from_numpy(nobs).to(self.device)
        act = torch.from_numpy(act).to(self.device)
        rew = torch.from_numpy(rew).to(self.device)
        done = torch.from_numpy(done).to(self.device)
        gam = torch.from_numpy(gam).to(self.device)
        w = torch.from_numpy(w).to(self.device)

        if self.mirror_aug:
            m = torch.rand(obs.shape[0], device=self.device) < 0.5
            om, am = self._mirror(obs[m].clone(), act[m].clone())
            obs[m], act[m] = om, am
            nm, _ = self._mirror(nobs[m].clone(), act[m].clone())
            nobs[m] = nm

        lr = self.lr_at(self.grad_steps)
        for g in self.opt.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0)

        self.opt.zero_grad(set_to_none=True)
        td_parts = []
        # keep scalars on GPU through the accum loop; a single sync at the end
        # (per-micro .cpu() calls used to drain the pipeline 3x per micro-batch)
        l_t = torch.zeros((), device=self.device)
        q_t = torch.zeros((), device=self.device)
        for i in range(0, self.batch, micro):
            sl = slice(i, i + micro)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(self.device == "cuda")):
                z = self.online_c(obs[sl])                     # [B, A, Q]
                pred = z.gather(1, act[sl].view(-1, 1, 1)
                                .expand(-1, 1, self.n_quant)).squeeze(1)
                with torch.no_grad():
                    if self.cfg.get("double", True):
                        a_star = self.online_c(nobs[sl]).mean(-1).argmax(1)
                    else:
                        a_star = self.target_c(nobs[sl]).mean(-1).argmax(1)
                    zt = self.target_c(nobs[sl])
                    nxt = zt.gather(1, a_star.view(-1, 1, 1)
                                    .expand(-1, 1, self.n_quant)).squeeze(1)
                    target = rew[sl].unsqueeze(1) + (gam[sl] * (1 - done[sl])).unsqueeze(1) * nxt
                    # gamma=1 gives the Bellman backup no contraction, so an
                    # unbounded bootstrap can ratchet Q upward without ever
                    # violating a non-terminal sample (wave5 day1 divergence).
                    # Rewards here are non-negative and real returns top out
                    # far below q_target_max, so clamping only cuts the
                    # runaway chain, never a legitimate target.
                    q_max = float(self.cfg.get("q_target_max", 0.0))
                    if q_max > 0:
                        target = target.clamp(0.0, q_max)
                loss, per_sample = quantile_huber_loss(pred, target, w[sl])
            (loss / accum).backward()
            td_parts.append((target.mean(1) - pred.mean(1)).abs()
                            .detach().float())
            l_t += loss.detach() / accum
            q_t += pred.mean().detach() / accum
        gn = torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        if not torch.isfinite(gn):
            # One pathological batch (e.g. a physics hiccup producing inf
            # grads) must not poison AdamW moments or the EMA target:
            # clip_grad_norm_ scales inf grads by 10/inf = 0 and inf*0 = NaN
            # weights (w6_var_xl died this way at step 69). Drop the whole
            # step instead: no opt.step, no EMA, no priority update.
            self.opt.zero_grad(set_to_none=True)
            self._nan_skips = getattr(self, "_nan_skips", 0) + 1
            print(f"[learner] non-finite grad norm ({float(gn)}); step "
                  f"skipped (total {self._nan_skips})", flush=True)
            return float(l_t.cpu()), float(q_t.cpu()), float(gn), lr
        self.opt.step()
        with torch.no_grad():
            pos = list(self.online.parameters())
            pts = list(self.target.parameters())
            torch._foreach_mul_(pts, 1 - self.tau)
            torch._foreach_add_(pts, pos, alpha=self.tau)
        td_np = torch.cat(td_parts).cpu().numpy()
        td_np = np.nan_to_num(td_np, nan=1.0, posinf=1e4, neginf=1e4)
        l, q = float(l_t.cpu()), float(q_t.cpu())
        self.replay.update_priorities(idx, td_np)
        self.grad_steps += 1
        self.grad_draws += self.batch
        self._loss_ema = l if self._loss_ema is None else 0.99 * self._loss_ema + 0.01 * l
        self._q_ema = q if self._q_ema is None else 0.99 * self._q_ema + 0.01 * q
        return l, q, float(gn), lr

    def publish(self, env_steps):
        if time.time() - self._last_publish < 2.0:
            return
        self._last_publish = time.time()
        payload = {
            # serve the EMA (target) weights, not the online net: the online
            # net injects per-step gradient noise straight into 200 actors'
            # behavior and made train rollouts oscillate (D15).
            "state_dict": {k: v.detach().cpu() for k, v in self.target.state_dict().items()},
            "grad_steps": self.grad_steps,
            "env_steps": int(env_steps),
            "cfg": {k: self.cfg[k] for k in ("K", "max_fruits", "boundary", "n_quant")
                    if k in self.cfg},
        }
        tmp = self.policy_path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, self.policy_path)

    def checkpoint(self, env_steps, force=False):
        # ckpt_interval_steps (grad-step interval) overrides the legacy
        # wall-clock interval when set: prod ~100 steps (~30min), debug ~20.
        step_iv = int(self.cfg.get("ckpt_interval_steps", 0))
        if step_iv:
            if not force and self.grad_steps - self._last_ckpt_step < step_iv:
                return
            self._last_ckpt_step = self.grad_steps
        else:
            now = time.time()
            if not force and now - self._last_ckpt < float(
                    self.cfg.get("ckpt_interval_s", 600)):
                return
            self._last_ckpt = now
        path = os.path.join(
            self.ckpt_dir, f"step{self.grad_steps}_env{int(env_steps)}.pt")
        torch.save({"state_dict": self.online.state_dict(),
                    "ema": self.target.state_dict(),
                    "opt": self.opt.state_dict(),
                    "grad_steps": self.grad_steps,
                    "grad_draws": self.grad_draws,
                    "batch": self.batch,
                    "env_steps": int(env_steps)}, path)
        keep = int(self.cfg.get("ckpt_keep", 4))
        cks = sorted(glob.glob(os.path.join(self.ckpt_dir, "step*.pt")),
                     key=os.path.getmtime)
        for old in cks[:-keep]:
            try:
                os.remove(old)
            except OSError:
                pass

    def resume(self, resume_dir):
        """Restore online/EMA/optimizer + counters from newest ckpt in dir."""
        import glob as g
        cks = sorted(g.glob(os.path.join(resume_dir, "checkpoints", "step*.pt")),
                     key=os.path.getmtime)
        if not cks:
            print(f"[learner] resume_from_dir {resume_dir}: no ckpt, fresh",
                  flush=True)
            return 0
        ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
        self.online.load_state_dict(ck["state_dict"])
        self.target.load_state_dict(ck.get("ema", ck["state_dict"]))
        if "opt" in ck:
            try:
                self.opt.load_state_dict(ck["opt"])
                for st in self.opt.state.values():
                    for k, v in st.items():
                        if torch.is_tensor(v):
                            st[k] = v.to(self.device)
            except (ValueError, KeyError):
                print("[learner] opt state skipped (param groups changed); "
                      "fresh AdamW on resume", flush=True)
        self.grad_steps = int(ck.get("grad_steps", 0))
        env_steps = int(ck.get("env_steps", 0))
        # replay is not persisted; without this the reuse gate
        # (draws < inserts * max_reuse) demands ~100M fresh inserts before
        # training and the learner sleeps forever.
        self.replay.inserts = env_steps
        # reuse accounting is in sample-draws, not steps: old ckpts ran
        # batch 8192, so restore draws = steps * old_batch (exact for new
        # ckpts via the saved grad_draws field).
        self.grad_draws = int(ck.get(
            "grad_draws",
            self.grad_steps * int(self.cfg.get("resume_old_batch", 8192))))
        print(f"[learner] resumed {cks[-1]} grad={self.grad_steps} "
              f"draws={self.grad_draws/1e6:.0f}M "
              f"env={env_steps} opt={'opt' in ck}", flush=True)
        return env_steps

    def watchdog(self, eval_path):
        """Detect gamma=1 value runaway: Q drifting far above realized eval
        returns. Returns True to request a hard stop (caller exits nonzero).
        Needs 3 consecutive minute-spaced hits to fire, so a single stale or
        unlucky eval row cannot trip it."""
        try:
            with open(eval_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 8192))
                tail = f.read().decode().strip().splitlines()
            em = float(json.loads(tail[-1])["mean"])
        except Exception:
            self._watch_hits = 0
            return False
        if (self._q_ema is not None and em > 0
                and self._q_ema > max(3.0 * em, 1000.0)):
            self._watch_hits += 1
        else:
            self._watch_hits = 0
        if self._watch_hits >= 3:
            print(f"[learner] WATCHDOG q_ema={self._q_ema:.0f} "
                  f">> eval_mean={em:.0f} x3 sustained; halting", flush=True)
            return True
        return False

    def log(self, rec):
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--obs-dim", type=int, required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config)) if args.config.endswith(".json") else \
        __import__("yaml").safe_load(open(args.config))

    torch.manual_seed(int(cfg.get("seed", 0)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # SDPA backend selection: flash/cudnn attention on torch>=2.13 (H200)
    # emit inf grads in bf16 for this model (~56% of steps on w6_var_xl);
    # sdp_backends: [mem_efficient, math] in the config avoids them.
    backs = cfg.get("sdp_backends")
    if backs:
        for name in ("flash", "cudnn", "mem_efficient"):
            setter = getattr(torch.backends.cuda, f"enable_{name}_sdp", None)
            if setter is not None:
                setter(name in backs)
            elif hasattr(torch.backends.cuda, f"{name}_sdp_enabled"):
                getattr(torch.backends.cuda, f"{name}_sdp_enabled")(name in backs)
        print(f"[learner] sdp_backends={backs} flash="
              f"{torch.backends.cuda.flash_sdp_enabled()}", flush=True)
    print(f"[learner] device={device} cfg={cfg.get('name')}", flush=True)

    learner = Learner(cfg, args.obs_dim, args.run_dir, device)
    print(f"[learner] params={param_count(learner.online)/1e6:.2f}M", flush=True)

    # Transitions arrive as atomically-renamed .npz shards in run_dir/inbox/.
    inbox = os.path.join(args.run_dir, "inbox")
    os.makedirs(inbox, exist_ok=True)
    seen = set()

    env_steps = 0
    resume_dir = cfg.get("resume_from_dir")
    if resume_dir:
        env_steps = learner.resume(resume_dir)
    # seed policy.pt BEFORE entering the loop: run_arm gates actor/infer-server
    # startup on this file, so the served policy is never random init weights
    # (the 19:20 wave3b incident served random weights for ~25min).
    learner.publish(env_steps)
    t0 = time.time()
    last_log = 0.0
    last_pub = 0.0
    last_watch = 0.0
    min_replay = int(cfg.get("min_replay", 50_000))
    while learner.grad_steps < learner.total_grad:
        # ingest shard files written by actors
        for fn in sorted(os.listdir(inbox)):
            if fn in seen or not fn.endswith(".npz"):
                continue
            seen.add(fn)
            try:
                d = np.load(os.path.join(inbox, fn))
                learner.replay.add_batch(
                    d["obs"], d["act"], d["rew"], d["nobs"], d["done"], d["gam"])
                env_steps += len(d["obs"])
            except Exception as e:  # partial write; retry next round
                seen.discard(fn)
                continue
            try:
                os.remove(os.path.join(inbox, fn))
            except OSError:
                pass
        if learner.replay.size < min_replay:
            time.sleep(0.5)
            continue
        # flow control: cap sample reuse (draw-based, batch-size invariant)
        if learner.grad_draws >= learner.replay.inserts * learner.max_reuse:
            time.sleep(0.05)
            continue
        loss, q, gn, lr = learner.train_step()
        now = time.time()
        if now - last_watch >= 60.0:
            last_watch = now
            if learner.watchdog(os.path.join(args.run_dir, "eval.jsonl")):
                learner.checkpoint(env_steps, force=True)
                import sys
                sys.exit(3)
        if now - last_pub >= 15.0:
            last_pub = now
            learner.publish(env_steps)
        learner.checkpoint(env_steps)
        if now - last_log >= 5.0:
            last_log = now
            learner.log({
                "t": round(now - t0, 1),
                "grad_steps": learner.grad_steps,
                "env_steps": int(env_steps),
                "replay": learner.replay.size,
                "loss": learner._loss_ema,
                "q_mean": learner._q_ema,
                "grad_norm": gn,
                "lr": lr,
                "grad_sps": learner.grad_steps / max(1e-9, now - t0),
                "env_sps": env_steps / max(1e-9, now - t0),
            })
    learner.publish(env_steps)
    learner.checkpoint(env_steps, force=True)
    print("[learner] done", flush=True)


if __name__ == "__main__":
    main()
