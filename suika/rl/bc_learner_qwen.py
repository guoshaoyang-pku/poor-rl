"""Supervised BC + Q-distillation pretrainer for the Qwen arm (w4 warm start).

Data: shards from bc_collector.py (obs/act/rew/nobs/done/gam + qteach f16 +
argmax flag). Read-only preload into RAM; shard-level train/val split.

Losses (lambda-configurable):
  L_pi  = KD(pi_head(s), softmax((q_T - mean_a q_T) / bc_pi_tau))  [tau > 0]
        = CE(pi_head(s), a_teacher)                [fallback, expert only]
  L_qd  = smooth_l1(Q(s, .), q_teacher(s, .))      [full 128-vector distill]
  L_td  = smooth_l1(Q(s,a), Rn + gam*(1-done)*Q_t(s', a*))   [double, EMA target]
Mirror aug applies consistently to obs/nobs/act/qteach.

Output: policy.pt + checkpoints/ in the SAME format as learner_qwen
(loadable by inference_server_qwen / evaluator_qwen unchanged).
"""
import argparse
import contextlib
import json
import os
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist

from model import build_model, param_count  # noqa: E402


def smooth_l1(x, y):
    return torch.nn.functional.smooth_l1_loss(x, y, reduction="none")


class BCLearner:
    def __init__(self, cfg, run_dir, device, rank, world, obs_dim):
        self.cfg, self.rank, self.world, self.device = cfg, rank, world, device
        self.batch = int(cfg["batch"])
        self.micro = int(cfg.get("micro_bs", 8))
        self.T = int(cfg.get("T", 160))
        self.K = int(cfg["K"])
        self.total_grad = int(cfg.get("grad_budget", 12000))
        self.tau = float(cfg.get("ema_tau", 0.003))
        self.lam_pi = float(cfg.get("bc_lambda_pi", 1.0))
        self.lam_qd = float(cfg.get("bc_lambda_qd", 1.0))
        self.lam_td = float(cfg.get("bc_lambda_td", 0.5))
        # >0: pi trains on soft teacher targets softmax((q_T - mean q_T)/tau).
        # Hard-argmax CE is unattainable here: the teacher's top1-top2 margin
        # is ~1 Q unit on a ~1100 scale (fp16 ULP), so l_pi saturates at the
        # action-marginal entropy (3.787) and agree_pi at 0.075 (best constant).
        self.pi_tau = float(cfg.get("bc_pi_tau", 0.0))
        # >0: soft ranking distill on the Q vector itself,
        # CE(softmax((q_T-mean)/tau), log_softmax(z/tau)). smooth_l1 on raw Q
        # is ~80% level error (a state scalar; qd 53 vs per-state contrast 12)
        # so the action contrast that decides argmax got no dedicated gradient:
        # q_regret sat at the constant-policy baseline (28.8 vs 30.4) while
        # qd kept falling 437->97->53.
        self.qrank_tau = float(cfg.get("bc_qrank_tau", 0.0))
        self.lam_qrank = float(cfg.get("bc_lambda_qrank", 1.0))
        # v3 normalized-advantage mode. Measured (probe_bc_features.py, step
        # 800): the optimal LINEAR readout of the very features the heads see
        # scores regret 23.5 vs 22.5 for a ridge on the raw 800d board input --
        # i.e. the trunk had learned the Q LEVEL (~1100, range 0-2064) and
        # nothing about action ranking (~12 units, gaps ~1). Raw-scale losses
        # explain it: l_qd ~44 and l_td ~42 vs a bounded soft-CE gradient ->
        # ~99% of the gradient was the state scalar. Fix: supervise the dueling
        # parts separately, in scale-free units -- since q = v + a - mean(a),
        # mean_a q(s,.) == v exactly and q - mean_a q == a - mean(a), so
        #   L_v    = smooth_l1(v/1000, level/1000)          [easy scalar]
        #   L_adv  = smooth_l1(c, (q_T - level)/std_T)      [all the ranking]
        # with std_T the per-state teacher contrast (detached, clamp >= 1).
        self.adv_norm = bool(cfg.get("bc_adv_norm", False))
        self.lam_adv = float(cfg.get("bc_lambda_adv", 1.0))
        self.lam_v = float(cfg.get("bc_lambda_v", 1.0))
        # reference contrast scale (median per-state std of q_T in the shards:
        # 12.4). Both prediction and target are divided by it so the loss is
        # O(1) and gradient magnitude stops scaling with the ~1100 Q level.
        self.adv_ref = float(cfg.get("bc_adv_ref", 12.0))
        self.mirror_aug = bool(cfg.get("mirror_aug", True))
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)) + 4242 + rank)
        self.grad_steps = 0

        raw = build_model(cfg, obs_dim).to(device)
        if rank == 0:
            print(f"[bc] total={param_count(raw)/1e6:.1f}M "
                  f"trainable={raw.trainable_param_count()/1e6:.2f}M "
                  f"lambdas pi/qd/td={self.lam_pi}/{self.lam_qd}/{self.lam_td}"
                  f" pi_tau={self.pi_tau}",
                  flush=True)
        head_p, lora_p = [], []
        for n, p in raw.named_parameters():
            if p.requires_grad:
                (lora_p if "lora_" in n else head_p).append(p)
        groups = [{"params": head_p, "lr": float(cfg["lr"])}]
        if lora_p:
            groups.append({"params": lora_p,
                           "lr": float(cfg.get("lora_lr", 1e-4))})
        # per-group lr multipliers vs schedule base (kept identical on every
        # rank — optimizer state must stay in sync across DDP ranks)
        self.lr_fracs = [1.0]
        if lora_p:
            self.lr_fracs.append(
                float(cfg.get("lora_lr", 1e-4)) / float(cfg["lr"]))
        self.opt = torch.optim.AdamW(
            groups, weight_decay=float(cfg.get("weight_decay", 1e-5)),
            betas=(0.9, 0.95))
        self.warmup = int(cfg.get("lr_warmup", 200))
        if world > 1:
            # all trainable params (trunk lora + v/a/pi heads) participate in
            # the combined forward -> find_unused not needed (faster, and
            # no_sync accumulation is safe)
            self.online = torch.nn.parallel.DistributedDataParallel(
                raw, device_ids=[device.index] if device.index is not None
                else None, find_unused_parameters=False)
        else:
            self.online = raw
        self.target = build_model(cfg, obs_dim).to(device)
        self.target.load_state_dict(self._module().state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.policy_path = os.path.join(run_dir, "policy.pt")
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        self.run_dir = run_dir
        if rank == 0:
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self._last_ckpt = 0.0

    def _module(self):
        return self.online.module if hasattr(self.online, "module") else self.online

    def lr_at(self, step):
        base = float(self.cfg["lr"])
        if step < self.warmup:
            return base * (step + 1) / self.warmup
        t = (step - self.warmup) / max(1, self.total_grad - self.warmup)
        t = min(1.0, t)
        floor = float(self.cfg.get("lr_floor_frac", 0.1))
        return base * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * t)))

    def _mirror(self, obs, act, qteach):
        B = obs.shape[0]
        tok = obs.view(B, self.T, 5)
        valid = tok[:, :, 0] >= 0
        valid[:, :2] = False
        x, vx = tok[:, :, 1], tok[:, :, 3]
        tok[:, :, 1] = torch.where(valid, 1.0 - x, x)
        tok[:, :, 3] = torch.where(valid, -vx, vx)
        if qteach is not None:
            qteach = qteach.flip(1)
        return obs, self.K - 1 - act, qteach

    def train_step(self, batch):
        o, a, r, no, d, g, q16, am = batch
        dev = self.device
        shard = o.shape[0]      # this rank's own slice of the global batch
        qt = q16.float()

        # TD targets (no grad); skipped when the TD term is switched off (v3)
        if self.lam_td > 0:
            a_star = torch.empty(shard, dtype=torch.long, device=dev)
            nxt = torch.empty(shard, 1, device=dev)
            with torch.no_grad():
                for s in range(0, shard, self.micro):
                    e = min(s + self.micro, shard)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        a_star[s:e] = self.online(no[s:e]).mean(-1).argmax(1)
                        zt = self.target(no[s:e])
                    nxt[s:e] = zt.gather(
                        1, a_star[s:e].view(-1, 1, 1).expand(-1, 1, 1)
                    ).squeeze(1)
            tgt = (r.unsqueeze(1)
                   + g.unsqueeze(1) * (1 - d.unsqueeze(1)) * nxt)

        # online pass with grad accumulation; allreduce only on last chunk
        self.opt.zero_grad(set_to_none=True)
        stats = [0.0] * 8      # loss, l_pi, l_qd, l_td, l_pi_hard, l_qrank,
                               # l_adv, l_v
        nchunks = (shard + self.micro - 1) // self.micro
        for ci, s in enumerate(range(0, shard, self.micro)):
            e = min(s + self.micro, shard)
            last = ci == nchunks - 1
            cm = (self.online.no_sync() if (self.world > 1 and not last)
                  else contextlib.nullcontext())
            with cm:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z, pi = self.online(o[s:e], return_pi=True)  # [m,K,1],[m,K]
                z = z.squeeze(-1)
                zi = z.gather(1, a[s:e].view(-1, 1)).squeeze(1).float()
                zf = z.float()
                m = am[s:e].float()
                mse = m.sum().clamp(min=1.0)
                pi_f = pi.float()
                l_pi_hard = (torch.nn.functional.cross_entropy(
                    pi_f, a[s:e], reduction="none") * m).sum() / mse
                if self.pi_tau > 0 or self.qrank_tau > 0:
                    # centred per state: the ~1100 Q level is state-scalar
                    qc = qt[s:e] - qt[s:e].mean(1, keepdim=True)
                if self.pi_tau > 0:
                    # Soft-target KD over ALL samples (teacher Q exists for
                    # every transition).
                    p_t = torch.softmax(qc / self.pi_tau, dim=1)
                    l_pi = -(p_t * torch.log_softmax(pi_f, dim=1)).sum(
                        1).mean()
                else:
                    l_pi = l_pi_hard
                l_qrank = torch.zeros((), device=zf.device)
                if self.qrank_tau > 0:
                    # Rank the Q vector against the teacher's soft target (see
                    # bc_qrank_tau in __init__). softmax is invariant to the
                    # per-state level, so this supervises contrast only.
                    p_r = torch.softmax(qc / self.qrank_tau, dim=1)
                    l_qrank = -(p_r * torch.log_softmax(
                        zf / self.qrank_tau, dim=1)).sum(1).mean()
                l_qd = smooth_l1(zf, qt[s:e]).mean(1).mean()
                l_td = smooth_l1(zi, tgt[s:e].squeeze(1)).mean() \
                    if self.lam_td > 0 else torch.zeros((), device=zf.device)
                l_adv = l_v = torch.zeros((), device=zf.device)
                if self.adv_norm:
                    # dueling split: mean_a q == v (level), q-mean_a q == a-mean(a)
                    lvl_t = qt[s:e].mean(1)
                    c_t = qt[s:e] - lvl_t.unsqueeze(1)
                    lvl_p = zf.mean(1)
                    # PER-STATE standardization. A fixed divisor (12) makes the
                    # term magnitude-dominated: the target's global RMS is 37
                    # (per-state std spans 6-22), so the zero predictor scores
                    # 0.962 and the perfect shape only 0.937 -- a 2.6% margin,
                    # which is why both the LM and a plain MLP plateaued at
                    # adv=0.937-0.94 (measured on the audit split). Dividing
                    # both sides by sigma_t makes the shape worth 100% (zero
                    # predictor 0.45, perfect 0). Note: with a fixed divisor
                    # this term also fought the listwise terms by shrinking the
                    # contrast, so v3 sets bc_lambda_adv: 0.0.
                    std = c_t.std(1, keepdim=True).clamp(min=1.0)
                    l_adv = smooth_l1((zf - lvl_p.unsqueeze(1)) / std,
                                      c_t / std).mean(1).mean()
                    l_v = smooth_l1(lvl_p / 1000.0, lvl_t / 1000.0).mean()
                loss = (self.lam_pi * l_pi + self.lam_qd * l_qd
                        + self.lam_td * l_td
                        + self.lam_qrank * l_qrank
                        + self.lam_adv * l_adv + self.lam_v * l_v
                        ) * (e - s) / shard
            loss.backward()
            stats[0] += float(loss)
            stats[1] += float(l_pi) * (e - s) / shard
            stats[2] += float(l_qd) * (e - s) / shard
            stats[3] += float(l_td) * (e - s) / shard
            stats[4] += float(l_pi_hard) * (e - s) / shard
            stats[5] += float(l_qrank) * (e - s) / shard
            stats[6] += float(l_adv) * (e - s) / shard
            stats[7] += float(l_v) * (e - s) / shard
        self.opt.step()
        # EMA target
        with torch.no_grad():
            for pt, po in zip(self.target.parameters(),
                              self._module().parameters()):
                pt.mul_(1 - self.tau).add_(po, alpha=self.tau)
        return stats

    @torch.no_grad()
    def validate(self, vb, n_batches=8):
        agree_pi = agree_q = qd = regret = agree_qd = 0.0
        soft_ce = tgt_ent = qrank = 0.0
        n = 0
        for i in range(n_batches):
            rng = np.random.default_rng(777000 + i)
            o, a, _, _, _, _, q16, am = vb.sample(min(512, vb.n), rng)
            m = am > 0
            if int(m.sum()) < 8:
                continue
            o, a, q16 = o[m], a[m], q16[m]
            o = o.to(self.device)
            a = a.to(self.device)
            q16 = q16.float().to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z, pi = self.online(o, return_pi=True)
            agree_pi += float((pi.argmax(1) == a).float().mean())
            zf = z.squeeze(-1).float()
            za = zf.argmax(1)
            agree_q += float((za == a).float().mean())
            qd += float(smooth_l1(zf, q16).mean())
            # Value regret: the Q the teacher assigns to the student's action
            # vs the teacher's own best. The two argmax-match metrics above are
            # decided by teacher decision margins that are far smaller than any
            # achievable regression error (median top1-top2 gap is 1.0 on
            # Q~1096, i.e. 0.09%, while val_qd is ~100), so they are
            # unattainable by construction and say nothing about policy
            # quality. Regret is the control-relevant quantity.
            regret += float((q16.max(1).values
                             - q16.gather(1, za.view(-1, 1)).squeeze(1)).mean())
            # agree_q restricted to states whose teacher margin is well above
            # the fp16 storage ULP (>= 1% of the per-state bin spread), where
            # exact-action agreement is actually meaningful.
            srt = q16.sort(1).values
            det = (srt[:, -1] - srt[:, -2]) >= 0.01 * (srt[:, -1] - srt[:, 0])
            if int(det.sum()) >= 8:
                agree_qd += float((za[det] == a[det]).float().mean())
            if self.pi_tau > 0:
                # soft-target CE and its floor (entropy of the target itself)
                pt = torch.softmax(
                    (q16 - q16.mean(1, keepdim=True)) / self.pi_tau, dim=1)
                lp = torch.log_softmax(pi.float(), dim=1)
                soft_ce += float(-(pt * lp).sum(1).mean())
                tgt_ent += float(-(pt * pt.clamp_min(1e-12).log()
                                   ).sum(1).mean())
            if self.qrank_tau > 0:
                # ranking fidelity of the (argmax-deployed) Q vector
                pr = torch.softmax(
                    (q16 - q16.mean(1, keepdim=True)) / self.qrank_tau,
                    dim=1)
                lz = torch.log_softmax(zf / self.qrank_tau, dim=1)
                qrank += float(-(pr * lz).sum(1).mean())
            n += 1
        n = max(n, 1)
        return (agree_pi / n, agree_q / n, qd / n, regret / n, agree_qd / n,
                soft_ce / n, tgt_ent / n, qrank / n)

    def _trainable_sd(self):
        """LoRA + heads only (~110 MB) instead of the full 3.1 GB trunk.

        The frozen trunk is reproducible from the pretrained checkpoint, so
        re-writing it every val_every steps was pure disk churn (~350 GB/day)
        and a plausible source of the writeback stall that tripped the NCCL
        watchdog at step 200.
        """
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


class BCData:
    """Read-only preload of collector shards; shard-level val split."""

    def __init__(self, data_dir, max_transitions, val_frac=0.02, seed=0,
                 verbose=True):
        files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))
        rng = np.random.default_rng(seed)
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_frac))
        val_f, train_f = set(files[:n_val]), files[n_val:]
        self.fields = {}
        total = 0
        t0 = time.time()
        for tag, flist in (("train", train_f), ("val", list(val_f))):
            arrs = {k: [] for k in ("obs", "act", "rew", "nobs", "done",
                                    "gam", "qteach", "argmax")}
            for fn in flist:
                d = np.load(os.path.join(data_dir, fn))
                take = len(d["obs"])
                if tag == "train" and total + take > max_transitions:
                    take = max(0, max_transitions - total)
                    if take == 0:
                        break
                    d = {k: v[:take] for k, v in d.items()}
                for k in arrs:
                    arrs[k].append(d[k])
                if tag == "train":
                    total += take
            self.fields[tag] = {
                k: np.concatenate(v) if v else np.zeros(0)
                for k, v in arrs.items()}
        self.n = total
        if verbose:
            print(f"[bc-data] train={total:,} "
                  f"val={len(self.fields['val']['act']):,}"
                  f" from {len(files)} shards in {time.time()-t0:.0f}s",
                  flush=True)

    def sample(self, n, rng):
        i = rng.integers(0, self.n, size=n)
        f = self.fields["train"]
        return (torch.from_numpy(f["obs"][i]), torch.from_numpy(f["act"][i]),
                torch.from_numpy(f["rew"][i]), torch.from_numpy(f["nobs"][i]),
                torch.from_numpy(f["done"][i]), torch.from_numpy(f["gam"][i]),
                torch.from_numpy(f["qteach"][i]),
                torch.from_numpy(f["argmax"][i]))

    def sample_val(self, n, rng):
        f = self.fields["val"]
        nv = len(f["act"])
        i = rng.integers(0, nv, size=min(n, nv))
        return (torch.from_numpy(f["obs"][i]), torch.from_numpy(f["act"][i]),
                torch.zeros(min(n, nv)), torch.zeros(min(n, nv)),
                torch.zeros(min(n, nv)), torch.zeros(min(n, nv)),
                torch.from_numpy(f["qteach"][i]),
                torch.from_numpy(f["argmax"][i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-transitions", type=int, default=12_000_000)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--resume-from", default=None,
                    help="checkpoint from learner.save(); continues grad_steps")
    ap.add_argument("--init-from", default=None,
                    help="weights-only warm start (e.g. RL ckpt); "
                         "grad_steps stays 0, fresh optimizer")
    ap.add_argument("--grad-budget", type=int, default=None,
                    help="override cfg grad_budget (smoke tests)")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    if args.grad_budget:
        cfg["grad_budget"] = int(args.grad_budget)
    obs_dim = int(cfg.get("T", 160)) * 5

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        # 60 min: a 20 h run must not die because one rank is briefly slow.
        # The default 10 min watchdog killed a healthy run at step 200.
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available()
                          else "cpu")
    os.makedirs(args.run_dir, exist_ok=True)

    # Every rank loads the shards and samples its own slice of the global
    # batch. The previous rank-0-sample-then-broadcast design left 7 ranks
    # parked in a collective for the whole sample+H2D window, so any rank-0
    # hiccup aborted the job. RAM is ample (1.9 TB), so 8 local copies are
    # cheaper than the serialization.
    data = BCData(args.data_dir, args.max_transitions,
                  verbose=(rank == 0))
    if world > 1:
        dist.barrier()

    learner = BCLearner(cfg, args.run_dir, device, rank, world, obs_dim)
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu",
                        weights_only=False)
        sd = ck.get("state_dict", ck)
        missing, unexpected = learner._module().load_state_dict(
            sd, strict=False)
        learner.target.load_state_dict(learner._module().state_dict())
        assert not unexpected, f"init-from unexpected: {unexpected[:5]}"
        if rank == 0:
            print(f"[bc] init-from {args.init_from} "
                  f"(weights only, {len(missing)} frozen keys skipped)",
                  flush=True)
    if args.resume_from:
        ck = torch.load(args.resume_from, map_location="cpu")
        sd = ck["state_dict"]
        # trainable-only ckpts leave the (frozen, reproducible) trunk at its
        # pretrained init, which is exactly where it should be
        strict = not ck.get("trainable_only", False)
        learner._module().load_state_dict(sd, strict=strict)
        learner.target.load_state_dict(sd, strict=strict)
        learner.grad_steps = int(ck.get("grad_steps", 0))
        if rank == 0:
            print(f"[bc] resumed from {args.resume_from} "
                  f"at step {learner.grad_steps} "
                  f"(trainable_only={ck.get('trainable_only', False)})",
                  flush=True)
    metrics = (open(learner.metrics_path, "a") if rank == 0 else None)
    t0 = time.time()

    while learner.grad_steps < learner.total_grad:
        ts = time.time()
        per_rank = learner.batch // world
        o, a, r, no, d, g, q16, am = [
            t.to(device, non_blocking=True)
            for t in data.sample(per_rank, learner.rng)]
        am = am.long()
        if learner.mirror_aug:
            m = torch.rand(o.shape[0], device=device) < 0.5
            oo, aa, qq = learner._mirror(
                o[m].clone(), a[m].clone(), q16[m].clone())
            o[m], a[m], q16[m] = oo, aa, qq
            no[m], _, _ = learner._mirror(no[m].clone(), a[m].clone(),
                                          None)
        t_data = time.time() - ts
        stats = learner.train_step((o, a, r, no, d, g, q16, am))
        learner.grad_steps += 1

        # lr schedule: identical on every rank (optimizer sync requirement)
        base = learner.lr_at(learner.grad_steps)
        for gp, fr in zip(learner.opt.param_groups, learner.lr_fracs):
            gp["lr"] = base * fr

        if rank == 0 and learner.grad_steps % 50 == 0:
            print(f"[bc] step {learner.grad_steps}/{learner.total_grad} "
                  f"loss={stats[0]:.4f} pi={stats[1]:.4f} "
                  f"pi_hard={stats[4]:.4f} qd={stats[2]:.4f} "
                  f"qrank={stats[5]:.4f} adv={stats[6]:.4f} v={stats[7]:.4f} "
                  f"td={stats[3]:.4f} {time.time()-ts:.1f}s "
                  f"data={t_data:.2f}s", flush=True)

        if rank == 0 and learner.grad_steps % args.val_every == 0:
            vb = _VBWrapper(data)
            api, aq, qd, regret, aqd, sce, tent, qrk = learner.validate(vb)
            rec = {"t": round(time.time() - t0, 1),
                   "grad_steps": learner.grad_steps,
                   "loss": stats[0], "l_pi": stats[1], "l_qd": stats[2],
                   "l_td": stats[3], "l_pi_hard": round(stats[4], 4),
                   "l_qrank": round(stats[5], 4),
                   "agree_pi": round(api, 4),
                   "agree_q": round(aq, 4), "val_qd": round(qd, 2),
                   "q_regret": round(regret, 3),
                   "agree_q_wellsep": round(aqd, 4),
                   "val_soft_ce": round(sce, 4), "val_tgt_ent": round(tent, 4),
                   "val_qrank": round(qrk, 4),
                   "step_s": round(time.time() - ts, 2)}
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
            print(f"[bc-val] agree_pi={api:.3f} agree_q={aq:.3f} "
                  f"val_qd={qd:.1f} q_regret={regret:.2f} "
                  f"agree_q_wellsep={aqd:.3f} soft_ce={sce:.3f} "
                  f"(floor {tent:.3f}) qrank={qrk:.3f}", flush=True)
            learner.save(f"step{learner.grad_steps}.pt")
            learner.publish()

    if rank == 0:
        learner.save(f"step{learner.grad_steps}_final.pt")
        learner.publish()
        metrics.close()
    if world > 1:
        dist.destroy_process_group()


class _VBWrapper:
    def __init__(self, data):
        self.data = data
        self.n = len(data.fields["val"]["act"])

    def sample(self, n, rng):
        return self.data.sample_val(n, rng)


if __name__ == "__main__":
    main()
