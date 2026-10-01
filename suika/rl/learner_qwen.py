"""DDP learner for the Qwen LLM arm (w4). Fork of learner.py — shared file untouched.

Differences vs learner.py:
- launched by torchrun (RANK / LOCAL_RANK / WORLD_SIZE env); rank 0 owns the
  replay, samples the batch and broadcasts tensors; every rank computes
  gradients on its shard; DDP allreduces. Control flow is decided on rank 0
  and broadcast as a flag so all ranks stay in lockstep (no NCCL deadlock);
- micro-batching with gradient accumulation (0.8B text fwd+bwd does not fit
  at full batch on 40G cards);
- tokenization happens inside QwenQ.forward (compact obs -> prompt ids);
- optimizer with two lr groups: LoRA params (lora_lr) and heads (lr);
- EMA target over trainable params only (identical on all ranks by
  construction, updated locally — no sync needed);
- double DQN + PER + n-step(3, fixed) + gamma from cfg (1.0) + mirror aug
  on compact obs before serialization (same semantics as learner.py).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from model import build_model, param_count
from replay import PrioritizedReplay


def huber_loss_half(pred, target, weights, kappa=1.0):
    """Per-sample 0.5*huber (== quantile_huber_loss at n_quant=1). All [B,1]."""
    td = target - pred
    abs_td = td.abs()
    h = torch.where(abs_td <= kappa, 0.5 * td ** 2,
                    kappa * (abs_td - 0.5 * kappa))
    per = 0.5 * h.mean(dim=-1)                          # [B]
    return (per * weights).mean(), per.detach()


class QwenLearner:
    def __init__(self, cfg, obs_dim, run_dir, device, rank, world):
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = device
        self.rank = rank
        self.world = world
        self.obs_dim = int(obs_dim)
        self.base_lr = float(cfg["lr"])
        self.batch = int(cfg["batch"])
        assert self.batch % world == 0, "batch must divide by world size"
        self.micro = int(cfg.get("micro_bs", 96))
        self.gamma = float(cfg["gamma"])
        self.K = int(cfg["K"])
        self.n_quant = 1
        self.mirror_aug = bool(cfg.get("mirror_aug", True))
        self.T = int(cfg.get("T", 160))
        self.beta0 = float(cfg.get("per_beta0", 0.4))
        self.beta_frames = float(cfg.get("per_beta_grad_steps", 2_000_000))
        self.max_reuse = float(cfg.get("max_reuse", 8.0))
        self.total_grad = int(cfg.get("grad_budget", 200_000))
        self.publish_every = int(cfg.get("publish_every", 200))
        self.tau = float(cfg.get("ema_tau", 0.003))
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)) + 777)
        self.grad_steps = 0
        self._loss_ema = None
        self._q_ema = None
        self._kl_ema = None

        raw = build_model(cfg, obs_dim).to(device)
        init_from = cfg.get("init_from")
        if init_from:
            ck = torch.load(init_from, map_location="cpu", weights_only=False)
            sd = ck.get("state_dict", ck)
            missing, unexpected = raw.load_state_dict(sd, strict=False)
            if rank == 0:
                print(f"[learner] warm-start from {init_from} "
                      f"(src_grad={ck.get('grad_steps')}, loaded {len(sd)} "
                      f"keys, missing {len(missing)}, "
                      f"unexpected {len(unexpected)})", flush=True)
                assert not unexpected, f"unexpected keys: {unexpected[:5]}"
        if rank == 0:
            n_tr = raw.trainable_param_count()
            print(f"[learner] total={param_count(raw)/1e6:.1f}M "
                  f"trainable={n_tr/1e6:.2f}M", flush=True)

        # two lr groups: heads (and full-FT params) vs LoRA params
        head_p, lora_p = [], []
        for n, p in raw.named_parameters():
            if not p.requires_grad:
                continue
            (lora_p if "lora_" in n else head_p).append(p)
        groups = [{"params": head_p, "lr": float(cfg["lr"])}]
        if lora_p:
            groups.append({"params": lora_p, "lr": float(cfg.get("lora_lr", 1e-4))})
        self.opt = torch.optim.AdamW(
            groups, weight_decay=float(cfg.get("weight_decay", 1e-5)),
            betas=(0.9, 0.95))
        self.warmup = int(cfg.get("lr_warmup", 500))

        if world > 1:
            self.online = torch.nn.parallel.DistributedDataParallel(
                raw, device_ids=[device.index] if device.index is not None
                else None, find_unused_parameters=True)
        else:
            self.online = raw
        self.target = build_model(cfg, obs_dim).to(device)
        self.target.load_state_dict(self._module().state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)

        # frozen BC teacher for listwise anchoring ("teacher warmup" phase):
        # soft-KD on the action distribution preserves the BC policy's ranking
        # while TD recalibrates values; lambda decays over anchor_decay_steps.
        anchor_from = cfg.get("anchor_from")
        self.anchor = None
        if anchor_from:
            self.anchor = build_model(cfg, obs_dim).to(device)
            ck = torch.load(anchor_from, map_location="cpu",
                            weights_only=False)
            missing, unexpected = self.anchor.load_state_dict(
                ck.get("state_dict", ck), strict=False)
            assert not unexpected, f"anchor unexpected: {unexpected[:5]}"
            for p in self.anchor.parameters():
                p.requires_grad_(False)
            self.anchor.eval()
            if rank == 0:
                print(f"[learner] anchor teacher from {anchor_from} "
                      f"({len(missing)} frozen keys skipped)", flush=True)
        self.anchor_l0 = float(cfg.get("anchor_lambda", 0.0))
        self.anchor_l1 = float(cfg.get("anchor_lambda_end",
                                       self.anchor_l0))
        self.anchor_decay = float(cfg.get("anchor_decay_steps", 4000))
        self.anchor_tau = float(cfg.get("anchor_tau", 1.0))
        self._anchor_path = anchor_from
        self._anchor_mtime = (os.path.getmtime(anchor_from)
                              if anchor_from else 0.0)

        # rank-0-only state
        self.replay = (PrioritizedReplay(
            int(cfg.get("replay_capacity", 2_000_000)), obs_dim,
            alpha=float(cfg.get("per_alpha", 0.6))) if rank == 0 else None)
        self.env_steps = 0
        self.policy_path = os.path.join(run_dir, "policy.pt")
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        if rank == 0:
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self._last_publish = 0.0
        self._last_ckpt = 0.0

    def maybe_reload_anchor(self):
        """Hot-swap the frozen teacher when the anchor file is atomically
        replaced (expert-iteration cycle writes a better teacher)."""
        if self.anchor is None or not self._anchor_path:
            return
        try:
            mt = os.path.getmtime(self._anchor_path)
        except OSError:
            return
        if mt <= self._anchor_mtime:
            return
        try:
            ck = torch.load(self._anchor_path, map_location="cpu",
                            weights_only=False)
            missing, unexpected = self.anchor.load_state_dict(
                ck.get("state_dict", ck), strict=False)
            assert not unexpected
            for p in self.anchor.parameters():
                p.requires_grad_(False)
            self.anchor.eval()
            self._anchor_mtime = mt
            if self.rank == 0:
                print(f"[learner] anchor teacher HOT-SWAPPED "
                      f"({self._anchor_path})", flush=True)
        except Exception as e:      # partially-written file: try next time
            if self.rank == 0:
                print(f"[learner] anchor reload deferred: {e}",
                      flush=True)

    def _module(self):
        return self.online.module if hasattr(self.online, "module") else self.online

    def anchor_lambda(self):
        t = min(1.0, self.grad_steps / max(1.0, self.anchor_decay))
        return self.anchor_l1 + (self.anchor_l0 - self.anchor_l1) * (1 - t)

    def lr_at(self, step):
        base = float(self.cfg["lr"])
        if step < self.warmup:
            return base * (step + 1) / self.warmup
        t = (step - self.warmup) / max(1, self.total_grad - self.warmup)
        t = min(1.0, t)
        floor = float(self.cfg.get("lr_floor_frac", 0.1))
        return base * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * t)))

    def _mirror(self, obs, act):
        """Mirror compact token obs in place; action -> K-1-a."""
        B = obs.shape[0]
        tok = obs.view(B, self.T, 5)
        valid = tok[:, :, 0] >= 0
        valid[:, :2] = False
        x = tok[:, :, 1]
        vx = tok[:, :, 3]
        tok[:, :, 1] = torch.where(valid, 1.0 - x, x)
        tok[:, :, 3] = torch.where(valid, -vx, vx)
        return obs, self.K - 1 - act

    def train_step(self, obs, act, rew, nobs, done, gam, w):
        """All ranks: full-batch cuda tensors; each rank works on its shard."""
        shard = self.batch // self.world
        lo, hi = self.rank * shard, (self.rank + 1) * shard
        o, a, r, no, d, g, ww = (obs[lo:hi], act[lo:hi], rew[lo:hi],
                                 nobs[lo:hi], done[lo:hi], gam[lo:hi],
                                 w[lo:hi])

        # ---- targets (no grad, micro-chunked) ----
        a_star = torch.empty(shard, dtype=torch.long, device=self.device)
        nxt = torch.empty(shard, 1, device=self.device)
        with torch.no_grad():
            for s in range(0, shard, self.micro):
                e = min(s + self.micro, shard)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    a_star[s:e] = self.online(no[s:e]).mean(-1).argmax(1)
                    zt = self.target(no[s:e])
                nxt[s:e] = zt.gather(
                    1, a_star[s:e].view(-1, 1, 1).expand(-1, 1, 1)).squeeze(1)
            target = r.unsqueeze(1) + (g * (1 - d)).unsqueeze(1) * nxt  # [shard,1]
            qtmax = float(self.cfg.get("q_target_max", 0))
            if qtmax > 0:
                target = target.clamp(0.0, qtmax)

        # ---- online pass with grad accumulation ----
        lr = self.lr_at(self.grad_steps)
        scale = lr / self.base_lr
        for gi, gp in enumerate(self.opt.param_groups):
            gp["lr"] = lr if gi == 0 else \
                float(self.cfg.get("lora_lr", 1e-4)) * scale
        lam = self.anchor_lambda() if self.anchor is not None else 0.0
        self.opt.zero_grad(set_to_none=True)
        preds = torch.empty(shard, 1, device=self.device)
        for s in range(0, shard, self.micro):
            e = min(s + self.micro, shard)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = self.online(o[s:e])
                pred = z.gather(1, a[s:e].view(-1, 1, 1).expand(-1, 1, 1)).squeeze(1)
            loss, _ = huber_loss_half(pred, target[s:e], ww[s:e])
            if lam > 0:
                with torch.no_grad(), torch.autocast("cuda",
                                                     dtype=torch.bfloat16):
                    qa = self.anchor(o[s:e]).mean(-1).float()      # [mb,A]
                zq = z.squeeze(-1).float()
                logp = torch.log_softmax(zq / self.anchor_tau, dim=1)
                pt = torch.softmax(qa / self.anchor_tau, dim=1)
                kl = (pt * (pt.clamp(min=1e-8).log() - logp)).sum(1)
                loss = loss + lam * (kl * ww[s:e]).mean()
                self._kl_ema = (float(kl.mean()) if self._kl_ema is None
                                else 0.99 * self._kl_ema
                                + 0.01 * float(kl.mean()))
            (loss * (e - s) / shard).backward()
            preds[s:e] = pred.detach()

        gn = torch.nn.utils.clip_grad_norm_(
            (p for p in self.online.parameters() if p.requires_grad), 10.0)
        self.opt.step()

        # ---- EMA target over trainable params (identical across ranks) ----
        with torch.no_grad():
            src = dict(self._module().named_parameters())
            for n, pt in self.target.named_parameters():
                po = src[n]
                if po.requires_grad:
                    pt.mul_(1 - self.tau).add_(po.detach(), alpha=self.tau)

        td = (target - preds).abs().mean(dim=1)          # [shard]
        abs_td = (target - preds).abs().clamp(min=1e-8)
        hub = torch.where(abs_td <= 1.0, 0.5 * abs_td ** 2, abs_td - 0.5)
        l = float((0.5 * hub).mean())
        q = float(preds.mean())
        if self._loss_ema is None:
            self._loss_ema, self._q_ema = l, q
        else:
            self._loss_ema = 0.99 * self._loss_ema + 0.01 * l
            self._q_ema = 0.99 * self._q_ema + 0.01 * q

        # ---- gather priorities (rank order == sample order) ----
        if self.world > 1:
            parts = [torch.empty_like(td) for _ in range(self.world)]
            dist.all_gather(parts, td.contiguous())
            td_full = torch.cat(parts)
        else:
            td_full = td
        return td_full, float(gn), lr, q

    def publish(self):
        if time.time() - self._last_publish < 2.0:
            return
        self._last_publish = time.time()
        payload = {
            "state_dict": {k: v.detach().cpu() for k, v in
                           self._module().state_dict().items()},
            "grad_steps": self.grad_steps,
            "env_steps": int(self.env_steps),
        }
        tmp = self.policy_path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, self.policy_path)

    def checkpoint(self):
        now = time.time()
        if now - self._last_ckpt < float(self.cfg.get("ckpt_interval_s", 1800)):
            return
        self._last_ckpt = now
        path = os.path.join(self.ckpt_dir,
                            f"step{self.grad_steps}_env{int(self.env_steps)}.pt")
        torch.save({"state_dict": self._module().state_dict(),
                    "grad_steps": self.grad_steps,
                    "env_steps": int(self.env_steps)}, path)
        keep = int(self.cfg.get("ckpt_keep", 4))
        cks = sorted((f for f in os.listdir(self.ckpt_dir)
                      if f.startswith("step") and f.endswith(".pt")),
                     key=lambda f: os.path.getmtime(
                         os.path.join(self.ckpt_dir, f)))
        for old in cks[:-keep] if keep > 0 else cks:
            try:
                os.remove(os.path.join(self.ckpt_dir, old))
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--obs-dim", type=int, required=True)
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
    torch.manual_seed(int(cfg.get("seed", 0)) + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available()
                          else "cpu")
    print(f"[learner] rank={rank}/{world} device={device} "
          f"cfg={cfg.get('name')}", flush=True)

    learner = QwenLearner(cfg, args.obs_dim, args.run_dir, device, rank, world)
    if rank == 0 and cfg.get("init_from"):
        # publish the warm-started weights immediately so the evaluator can
        # measure the BC baseline before any RL update
        learner.publish()
    t0 = time.time()
    last_log = 0.0
    min_replay = int(cfg.get("min_replay", 50_000))
    seen = set()
    inbox = os.path.join(args.run_dir, "inbox")
    if rank == 0:
        os.makedirs(inbox, exist_ok=True)

    flag = torch.zeros(2, dtype=torch.long, device=device)
    bt = {k: None for k in ("obs", "act", "rew", "nobs", "done", "gam", "w")}

    while True:
        # ---- rank 0 decides, broadcast ----
        if rank == 0:
            for fn in sorted(os.listdir(inbox)):
                if fn in seen or not fn.endswith(".npz"):
                    continue
                seen.add(fn)
                try:
                    dd = np.load(os.path.join(inbox, fn))
                    learner.replay.add_batch(
                        dd["obs"], dd["act"], dd["rew"], dd["nobs"],
                        dd["done"], dd["gam"])
                    learner.env_steps += len(dd["obs"])
                except Exception:
                    seen.discard(fn)
                    continue
                try:
                    os.remove(os.path.join(inbox, fn))
                except OSError:
                    pass
            if learner.grad_steps >= learner.total_grad:
                flag[0] = 2
            elif (learner.replay.size < min_replay or
                  learner.grad_steps >= learner.replay.inserts
                  * learner.max_reuse / learner.batch):
                flag[0] = 0
            else:
                flag[0] = 1
            flag[1] = min(learner.env_steps, 2 ** 30)
        if world > 1:
            dist.broadcast(flag, 0)
        learner.env_steps = int(flag[1].item())
        if flag[0].item() == 2:
            break
        if flag[0].item() == 0:
            time.sleep(0.5)
            continue

        # ---- rank 0 samples, broadcast batch ----
        beta = min(1.0, learner.beta0 + (1.0 - learner.beta0)
                   * learner.grad_steps / max(1.0, learner.beta_frames))
        if rank == 0:
            (idx, obs, act, rew, nobs, done, gam, w) = learner.replay.sample(
                learner.batch, beta, learner.rng)
            if learner.mirror_aug:
                m = learner.rng.random(obs.shape[0]) < 0.5
                mo = torch.from_numpy(obs[m].copy())
                ma = torch.from_numpy(act[m].copy())
                mo, ma = learner._mirror(mo, ma)
                obs[m], act[m] = mo.numpy(), ma.numpy()
                mno = torch.from_numpy(nobs[m].copy())
                mno, _ = learner._mirror(mno, ma)
                nobs[m] = mno.numpy()
            bt["obs"] = torch.from_numpy(obs).to(device)
            bt["act"] = torch.from_numpy(act).to(device)
            bt["rew"] = torch.from_numpy(rew).to(device)
            bt["nobs"] = torch.from_numpy(nobs).to(device)
            bt["done"] = torch.from_numpy(done).to(device)
            bt["gam"] = torch.from_numpy(gam).to(device)
            bt["w"] = torch.from_numpy(w).to(device)
            learner._idx = idx
        else:
            for k, dt in (("obs", torch.float32), ("nobs", torch.float32),
                          ("rew", torch.float32), ("done", torch.float32),
                          ("gam", torch.float32), ("w", torch.float32),
                          ("act", torch.int64)):
                if k in ("obs", "nobs"):
                    bt[k] = torch.zeros(learner.batch, learner.obs_dim,
                                        dtype=dt, device=device)
                elif k == "act":
                    bt[k] = torch.zeros(learner.batch, dtype=dt, device=device)
                else:
                    bt[k] = torch.zeros(learner.batch, dtype=dt, device=device)
        if world > 1:
            for k in ("obs", "act", "rew", "nobs", "done", "gam", "w"):
                dist.broadcast(bt[k], 0)

        t_step = time.time()
        td_full, gn, lr, q = learner.train_step(
            bt["obs"], bt["act"], bt["rew"], bt["nobs"], bt["done"],
            bt["gam"], bt["w"])
        if rank == 0:
            learner.replay.update_priorities(
                learner._idx, td_full.float().cpu().numpy())
        learner.grad_steps += 1
        if learner.grad_steps % 25 == 0:
            learner.maybe_reload_anchor()

        if rank == 0:
            if learner.grad_steps % learner.publish_every == 0:
                learner.publish()
            learner.checkpoint()
            now = time.time()
            if now - last_log >= 30.0:
                last_log = now
                with open(learner.metrics_path, "a") as f:
                    f.write(json.dumps({
                        "t": round(now - t0, 1),
                        "grad_steps": learner.grad_steps,
                        "env_steps": learner.env_steps,
                        "replay": learner.replay.size,
                        "loss": learner._loss_ema,
                        "q_mean": learner._q_ema,
                        "grad_norm": gn,
                        "lr": lr,
                        "anchor_lam": (learner.anchor_lambda()
                                       if learner.anchor is not None else 0.0),
                        "anchor_kl": learner._kl_ema,
                        "grad_sps": learner.grad_steps / max(1e-9, now - t0),
                        "env_sps": learner.env_steps / max(1e-9, now - t0),
                        "step_s": round(now - t_step, 2),
                    }) + "\n")
                print(f"[learner] g={learner.grad_steps} env={learner.env_steps}"
                      f" loss={learner._loss_ema:.1f} q={learner._q_ema:.1f}",
                      flush=True)
    if rank == 0:
        learner.publish()
        learner.checkpoint()
        print("[learner] done", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
