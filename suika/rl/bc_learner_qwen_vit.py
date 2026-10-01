"""Vision-arm BC pretrainer (w6): image + V/pi heads, v3.1 listwise recipe.

Fork of bc_learner_qwen.py for the vision model (model_qwen_vit.QwenViTQ).
Differences vs the text arm:
- data: bc_collector_vis.py shards (JPEG board renders + qteach + act/rew/
  done/gam + argmax). Images are decoded lazily per batch from a small LRU
  shard cache (the full image set is ~500 GB; the small fields ride along
  in the same npz, so nothing else needs preloading);
- batch = n samples drawn from ONE random shard per step (cache-friendly);
  shard-level train/val split as before;
- no nobs -> TD term unavailable (v3.1 has lam_td=0 anyway); no mirror aug
  (the next-fruit preview stamp is not mirror-symmetric);
- losses identical to v3.1: L_pi (soft KD tau=1) + 0.5*L_qrank (listwise)
  + L_v (level/1000); raw qd / TD / adv all off.
"""
import argparse
import contextlib
import hashlib
import json
import os
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist

from model_qwen_vit import build_model  # noqa: E402


def smooth_l1(x, y):
    return torch.nn.functional.smooth_l1_loss(x, y, reduction="none")


class BCVisData:
    """Lazy shard cache over bc_collector_vis shards; shard-level val split."""

    def __init__(self, data_dir, val_frac=0.02, seed=0, cache_shards=16,
                 verbose=True):
        files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))
        if not files:
            raise RuntimeError(f"no shards in {data_dir}")
        rng = np.random.default_rng(seed)
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_frac))
        self.val_f = [os.path.join(data_dir, f) for f in files[:n_val]]
        self.train_f = [os.path.join(data_dir, f) for f in files[n_val:]]
        self.cache = {}
        self.cache_cap = int(cache_shards)
        self.rng = np.random.default_rng(seed + 99)
        if verbose:
            print(f"[bcvis-data] train_shards={len(self.train_f)} "
                  f"val_shards={len(self.val_f)}", flush=True)

    def _load(self, path):
        d = self.cache.get(path)
        if d is None:
            raw = np.load(path)
            d = {k: raw[k] for k in raw.files}
            if len(self.cache) >= self.cache_cap:
                self.cache.pop(next(iter(self.cache)))
            self.cache[path] = d
        return d

    def _decode(self, d, idx):
        import io
        from PIL import Image
        img, off = d["img"], d["img_off"]
        out = np.empty((len(idx), *self._hw(d), 3), dtype=np.uint8)
        for j, i in enumerate(idx):
            out[j] = np.asarray(Image.open(
                io.BytesIO(img[off[i]:off[i + 1]].tobytes())))
        return out

    @staticmethod
    def _hw(d):
        return (int(d["img_h"]), int(d["img_w"])) if "img_h" in d \
            else (416, 288)

    def sample(self, n, split="train"):
        flist = self.train_f if split == "train" else self.val_f
        path = flist[int(self.rng.integers(len(flist)))]
        d = self._load(path)
        m = len(d["act"])
        idx = self.rng.integers(0, m, size=min(n, m))
        imgs = self._decode(d, idx)
        return (imgs,
                torch.from_numpy(np.asarray(d["act"][idx])),
                torch.from_numpy(np.asarray(d["rew"][idx])),
                torch.from_numpy(np.asarray(d["done"][idx])),
                torch.from_numpy(np.asarray(d["gam"][idx])),
                torch.from_numpy(np.asarray(d["qteach"][idx])),
                torch.from_numpy(np.asarray(d["argmax"][idx])))


class BCVitLearner:
    def __init__(self, cfg, run_dir, device, rank, world):
        self.cfg, self.rank, self.world, self.device = cfg, rank, world, device
        self.batch = int(cfg["batch"])
        self.micro = int(cfg.get("micro_bs", 32))
        self.K = int(cfg["K"])
        self.total_grad = int(cfg.get("grad_budget", 12000))
        self.lam_pi = float(cfg.get("bc_lambda_pi", 1.0))
        self.pi_tau = float(cfg.get("bc_pi_tau", 1.0))
        self.qrank_tau = float(cfg.get("bc_qrank_tau", 1.0))
        self.lam_qrank = float(cfg.get("bc_lambda_qrank", 0.5))
        self.lam_v = float(cfg.get("bc_lambda_v", 1.0))
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)) + 4242 + rank)
        self.grad_steps = 0

        raw = build_model(cfg, 0).to(device)
        if rank == 0:
            print(f"[bc] trainable={raw.trainable_param_count()/1e6:.2f}M "
                  f"pi_tau={self.pi_tau} qrank_tau={self.qrank_tau}",
                  flush=True)
        head_p, lora_p = [], []
        for n, p in raw.named_parameters():
            if p.requires_grad:
                (lora_p if "lora_" in n else head_p).append(p)
        groups = [{"params": head_p, "lr": float(cfg["lr"])}]
        if lora_p:
            groups.append({"params": lora_p,
                           "lr": float(cfg.get("lora_lr", 1e-4))})
        self.lr_fracs = [1.0]
        if lora_p:
            self.lr_fracs.append(
                float(cfg.get("lora_lr", 1e-4)) / float(cfg["lr"]))
        self.opt = torch.optim.AdamW(
            groups, weight_decay=float(cfg.get("weight_decay", 1e-5)),
            betas=(0.9, 0.95))
        self.warmup = int(cfg.get("lr_warmup", 200))
        if world > 1:
            self.online = torch.nn.parallel.DistributedDataParallel(
                raw, device_ids=[device.index] if device.index is not None
                else None, find_unused_parameters=False)
        else:
            self.online = raw

        self.policy_path = os.path.join(run_dir, "policy.pt")
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        if rank == 0:
            os.makedirs(self.ckpt_dir, exist_ok=True)

    def _module(self):
        return self.online.module if hasattr(self.online, "module") \
            else self.online

    def lr_at(self, step):
        base = float(self.cfg["lr"])
        if step < self.warmup:
            return base * (step + 1) / self.warmup
        t = min(1.0, (step - self.warmup) / max(1, self.total_grad
                                                - self.warmup))
        floor = float(self.cfg.get("lr_floor_frac", 0.1))
        return base * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * t)))

    def train_step(self, batch):
        imgs, a, r, d, g, q16, am = batch
        dev = self.device
        shard = imgs.shape[0]
        a = a.to(dev)
        am = am.to(dev)
        qt = q16.float().to(dev)
        mod = self._module()

        self.opt.zero_grad(set_to_none=True)
        stats = [0.0] * 5      # loss, l_pi, l_pi_hard, l_qrank, l_v
        nchunks = (shard + self.micro - 1) // self.micro
        for ci, s in enumerate(range(0, shard, self.micro)):
            e = min(s + self.micro, shard)
            last = ci == nchunks - 1
            cm = (self.online.no_sync() if (self.world > 1 and not last)
                  else contextlib.nullcontext())
            pv, grid, ids, mask = mod.images_to_inputs(imgs[s:e])
            seq = (pv.to(dev, non_blocking=True), grid.to(dev),
                   ids.to(dev), mask.to(dev))
            with cm:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z, pi = self.online(seq, return_pi=True)
                z = z.squeeze(-1).float()
                pi_f = pi.float()
                qc = qt[s:e] - qt[s:e].mean(1, keepdim=True)
                p_t = torch.softmax(qc / self.pi_tau, dim=1)
                l_pi = -(p_t * torch.log_softmax(pi_f, dim=1)).sum(1).mean()
                m = am[s:e].float()
                l_pi_hard = (torch.nn.functional.cross_entropy(
                    pi_f, a[s:e], reduction="none") * m
                    ).sum() / m.sum().clamp(min=1.0)
                p_r = torch.softmax(qc / self.qrank_tau, dim=1)
                l_qrank = -(p_r * torch.log_softmax(
                    z / self.qrank_tau, dim=1)).sum(1).mean()
                l_v = smooth_l1(z.mean(1) / 1000.0,
                                qt[s:e].mean(1) / 1000.0).mean()
                loss = (self.lam_pi * l_pi + self.lam_qrank * l_qrank
                        + self.lam_v * l_v) * (e - s) / shard
            loss.backward()
            stats[0] += float(loss)
            stats[1] += float(l_pi) * (e - s) / shard
            stats[2] += float(l_pi_hard) * (e - s) / shard
            stats[3] += float(l_qrank) * (e - s) / shard
            stats[4] += float(l_v) * (e - s) / shard
        self.opt.step()
        return stats

    @torch.no_grad()
    def validate(self, data, n_batches=8, chunk=None):
        agree_pi = agree_q = regret = soft_ce = tgt_ent = qrank = 0.0
        n = 0
        chunk = int(chunk or max(self.micro, 32))
        mod = self._module()
        for _ in range(n_batches):
            imgs, a, _, _, _, q16, am = data.sample(512, split="val")
            m = am > 0
            if int(m.sum()) < 8:
                continue
            imgs, a, q16 = imgs[m.numpy()], a[m], q16[m].float()
            a = a.to(self.device)
            q16 = q16.to(self.device)
            zs, pis = [], []
            for s in range(0, len(imgs), chunk):
                e = min(s + chunk, len(imgs))
                pv, grid, ids, mask = mod.images_to_inputs(imgs[s:e])
                seq = (pv.to(self.device), grid.to(self.device),
                       ids.to(self.device), mask.to(self.device))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z, pi = self.online(seq, return_pi=True)
                zs.append(z.squeeze(-1).float())
                pis.append(pi.float())
            zf = torch.cat(zs)
            pi_f = torch.cat(pis)
            agree_pi += float((pi_f.argmax(1) == a).float().mean())
            za = zf.argmax(1)
            agree_q += float((za == a).float().mean())
            regret += float((q16.max(1).values
                             - q16.gather(1, za.view(-1, 1)).squeeze(1)
                             ).mean())
            pt = torch.softmax(
                (q16 - q16.mean(1, keepdim=True)) / self.pi_tau, dim=1)
            soft_ce += float(-(pt * torch.log_softmax(pi_f, dim=1)
                               ).sum(1).mean())
            tgt_ent += float(-(pt * pt.clamp_min(1e-12).log()
                               ).sum(1).mean())
            pr = torch.softmax(
                (q16 - q16.mean(1, keepdim=True)) / self.qrank_tau, dim=1)
            qrank += float(-(pr * torch.log_softmax(
                zf / self.qrank_tau, dim=1)).sum(1).mean())
            n += 1
        n = max(n, 1)
        return (agree_pi / n, agree_q / n, regret / n, soft_ce / n,
                tgt_ent / n, qrank / n)

    def _trainable_sd(self):
        names = {n for n, p in self._module().named_parameters()
                 if p.requires_grad}
        return {k: v for k, v in self._module().state_dict().items()
                if k in names}

    def save(self, name):
        torch.save({"state_dict": self._trainable_sd(),
                    "trainable_only": True,
                    "grad_steps": self.grad_steps},
                   os.path.join(self.ckpt_dir, name))
        cks = sorted((f for f in os.listdir(self.ckpt_dir)
                      if f.endswith(".pt")),
                     key=lambda f: os.path.getmtime(
                         os.path.join(self.ckpt_dir, f)))
        keep = int(self.cfg.get("ckpt_keep", 4))
        for old in (cks[:-keep] if keep > 0 else cks):
            try:
                os.remove(os.path.join(self.ckpt_dir, old))
            except OSError:
                pass

    def publish(self):
        tmp = self.policy_path + ".tmp"
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in
                                   self._module().state_dict().items()},
                    "grad_steps": self.grad_steps}, tmp)
        os.replace(tmp, self.policy_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--grad-budget", type=int, default=None)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    if args.grad_budget:
        cfg["grad_budget"] = int(args.grad_budget)

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available()
                          else "cpu")
    os.makedirs(args.run_dir, exist_ok=True)
    if rank == 0:
        fp = hashlib.md5(open(__file__, "rb").read()).hexdigest()
        print(f"[code] bc_learner_qwen_vit.py md5={fp}", flush=True)

    data = BCVisData(args.data_dir, verbose=(rank == 0))
    if world > 1:
        dist.barrier()

    learner = BCVitLearner(cfg, args.run_dir, device, rank, world)
    if args.resume_from:
        ck = torch.load(args.resume_from, map_location="cpu")
        strict = not ck.get("trainable_only", False)
        learner._module().load_state_dict(ck["state_dict"], strict=strict)
        learner.grad_steps = int(ck.get("grad_steps", 0))
        if rank == 0:
            print(f"[bc] resumed from {args.resume_from} "
                  f"at step {learner.grad_steps}", flush=True)
    metrics = (open(learner.metrics_path, "a") if rank == 0 else None)
    t0 = time.time()

    while learner.grad_steps < learner.total_grad:
        ts = time.time()
        per_rank = learner.batch // world
        batch = data.sample(per_rank)
        td = time.time() - ts
        stats = learner.train_step(batch)
        learner.grad_steps += 1

        base = learner.lr_at(learner.grad_steps)
        for gp, fr in zip(learner.opt.param_groups, learner.lr_fracs):
            gp["lr"] = base * fr

        if rank == 0 and learner.grad_steps % 50 == 0:
            print(f"[bc] step {learner.grad_steps}/{learner.total_grad} "
                  f"loss={stats[0]:.4f} pi={stats[1]:.4f} "
                  f"pi_hard={stats[2]:.4f} qrank={stats[3]:.4f} "
                  f"v={stats[4]:.4f} {time.time()-ts:.1f}s "
                  f"data={td:.2f}s", flush=True)

        if rank == 0 and learner.grad_steps % args.val_every == 0:
            api, aq, regret, sce, tent, qrk = learner.validate(data)
            rec = {"t": round(time.time() - t0, 1),
                   "grad_steps": learner.grad_steps,
                   "loss": stats[0], "l_pi": stats[1],
                   "l_pi_hard": round(stats[2], 4),
                   "l_qrank": round(stats[3], 4), "l_v": round(stats[4], 4),
                   "agree_pi": round(api, 4), "agree_q": round(aq, 4),
                   "q_regret": round(regret, 3),
                   "val_soft_ce": round(sce, 4),
                   "val_tgt_ent": round(tent, 4),
                   "val_qrank": round(qrk, 4),
                   "step_s": round(time.time() - ts, 2)}
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
            print(f"[bc-val] agree_pi={api:.3f} agree_q={aq:.3f} "
                  f"q_regret={regret:.2f} soft_ce={sce:.3f} "
                  f"(floor {tent:.3f}) qrank={qrk:.3f}", flush=True)
            learner.save(f"step{learner.grad_steps}.pt")
            learner.publish()

    if rank == 0:
        learner.save(f"step{learner.grad_steps}_final.pt")
        learner.publish()
        metrics.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
