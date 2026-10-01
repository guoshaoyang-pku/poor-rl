"""SFT warm start for the LLM arm: the policy IS the token distribution.

User spec (2026-09-29): ~1 h of SFT on teacher rollouts to (a) learn the output
format, (b) train the policy as token probabilities = a normalization of the
teacher Q ("让它输出的token的概率就是Q的某种正规化"), (c) train a value head on
the shared backbone -> "后面的内容就完全用PPO来训练".

Mechanics (verified by probe_tokenizer_actions.py, run on t1_2):
  * the canonical answer for action col is f"{(col+0.5)/128:.3f}" and ALL 128
    strings tokenize to exactly 5 single tokens ['0'=15, '.'=13, d1, d2, d3];
    digits '0'..'9' are single tokens with ids 15..24. So the action policy is
    exactly the product of three 10-way digit distributions, and the Q target
    can be pushed onto tokens exactly (no head needed):
        P(a) = softmax((q_T(a) - mean_a q_T)/tau)
      pos-d1 <- marginal of P over d1        (all 128 actions contribute)
      pos-d2 <- conditional of P given the mode's d1
      pos-d3 <- conditional of P given the mode's (d1,d2)
    Teacher forcing on the mode path -> ONE trunk forward per state.
    (Soft target now, "sharpen later" in PPO -- the v1/v2/v3.1 BC post-mortem
    showed the teacher's top1-top2 gap is ~1 Q unit, so hard labels are
    unattainable and always saturate at the marginal entropy.)
  * format tokens '0', '.', EOS get hard full-vocab CE so the output parses.
  * value head (shared backbone) regresses the teacher's Q level / 1000.
  * aux losses keep the pi head and the Q head alive: DDP with
    find_unused_parameters=False requires every trainable param in the graph,
    and the Q head is the future critic's raw material.

Reuses (no duplication): model._serialize / model.trunk / model heads,
eval_qwen_policy._lm_head|_digit_logits|DIGIT_IDS|DOT_ID, bc_learner_qwen.BCData.
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
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bc_learner_qwen import BCData                        # noqa: E402
from eval_qwen_policy import DOT_ID, DIGIT_IDS, _lm_head  # noqa: E402
from model import build_model, param_count                # noqa: E402
from model_qwen import warmup                             # noqa: E402

TOK0 = 15            # token id of the leading '0'
EOS_ID = 248046      # <|im_end|>


def smooth_l1(x, y):
    return torch.nn.functional.smooth_l1_loss(x, y, reduction="none")


def mirror_batch(obs, qteach, K, T):
    """x -> 1-x on valid fruits (provenance: BCLearner._mirror, audited clean in
    audit_mirror_headroom.py: mirror->orig transfer 23.4 vs 22.5 reference)."""
    B = obs.shape[0]
    tok = obs.view(B, T, 5)
    valid = tok[:, :, 0] >= 0
    valid[:, :2] = False
    x, vx = tok[:, :, 1].clone(), tok[:, :, 3].clone()
    tok[:, :, 1] = torch.where(valid, 1.0 - x, x)
    tok[:, :, 3] = torch.where(valid, -vx, vx)
    if qteach is not None:
        qteach = qteach.flip(1)
    return obs, qteach


def digit_table(K):
    """col -> (d1,d2,d3) of its canonical '0.XXX' string; bijection check."""
    triples = np.array([[int(c) for c in f"{(c + 0.5) / K:.3f}"[2:]]
                        for c in range(K)], dtype=np.int64)
    uniq = np.unique(triples, axis=0)
    assert len(uniq) == K, f"digit mapping not a bijection: {len(uniq)} != {K}"
    return triples


class SFTLearner:
    def __init__(self, cfg, run_dir, device, rank, world, obs_dim):
        self.cfg, self.rank, self.world, self.device = cfg, rank, world, device
        self.batch = int(cfg["batch"])
        self.micro = int(cfg.get("micro_bs", 8))
        self.T = int(cfg.get("T", 160))
        self.K = int(cfg["K"])
        self.total_grad = int(cfg.get("grad_budget", 2000))
        self.tau = float(cfg.get("sft_tau", 1.0))
        self.lam_fmt = float(cfg.get("sft_lambda_fmt", 0.5))
        self.lam_d = [float(cfg.get("sft_lambda_d1", 1.0)),
                      float(cfg.get("sft_lambda_d2", 1.0)),
                      float(cfg.get("sft_lambda_d3", 1.0))]
        self.lam_v = float(cfg.get("sft_lambda_v", 1.0))
        self.lam_pi = float(cfg.get("sft_lambda_pi", 0.2))
        self.lam_qrank = float(cfg.get("sft_lambda_qrank", 0.2))
        self.v_ref = float(cfg.get("sft_v_ref", 1000.0))
        self.mirror_aug = bool(cfg.get("mirror_aug", True))
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)) + 777 + rank)
        self.grad_steps = 0

        dt = torch.tensor(digit_table(self.K), device=device)
        self.D1, self.D2, self.D3 = dt[:, 0], dt[:, 1], dt[:, 2]  # [K] int64

        raw = build_model(cfg, obs_dim).to(device)
        if rank == 0:
            print(f"[sft] total={param_count(raw)/1e6:.1f}M "
                  f"trainable={raw.trainable_param_count()/1e6:.2f}M "
                  f"tau={self.tau} lam_fmt={self.lam_fmt} "
                  f"lam_d={self.lam_d} lam_v={self.lam_v} "
                  f"lam_pi={self.lam_pi} lam_qrank={self.lam_qrank}",
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
            self.online = DDP(
                raw, device_ids=[device.index] if device.index is not None
                else None, find_unused_parameters=False)
        else:
            self.online = raw
        self.policy_path = os.path.join(run_dir, "policy.pt")
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        self.run_dir = run_dir
        if rank == 0:
            os.makedirs(self.ckpt_dir, exist_ok=True)

    def _module(self):
        return self.online.module if hasattr(self.online, "module") else self.online

    def lr_at(self, step):
        base = float(self.cfg["lr"])
        if step < self.warmup:
            return base * (step + 1) / self.warmup
        t = min(1.0, (step - self.warmup) / max(1, self.total_grad - self.warmup))
        floor = float(self.cfg.get("lr_floor_frac", 0.1))
        return base * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * t)))

    # ---------------- targets ----------------
    def targets(self, q16):
        """Q vector -> (P, p1, p2, p3, md1, md2, md3, level)."""
        q = q16.float()
        lvl = q.mean(1)
        P = torch.softmax((q - lvl[:, None]) / self.tau, dim=1)     # [B,K]
        B = q.shape[0]
        p1 = torch.zeros(B, 10, device=q.device).scatter_add_(
            1, self.D1[None, :].expand(B, -1), P)
        mode = P.argmax(1)
        md1, md2, md3 = self.D1[mode], self.D2[mode], self.D3[mode]
        m1 = (self.D1[None, :] == md1[:, None]).float()
        Q1 = P * m1
        p2 = torch.zeros_like(p1).scatter_add_(
            1, self.D2[None, :].expand(B, -1),
            Q1 / Q1.sum(1, keepdim=True).clamp_min(1e-12))
        m2 = m1 * (self.D2[None, :] == md2[:, None]).float()
        Q2 = P * m2
        p3 = torch.zeros_like(p1).scatter_add_(
            1, self.D3[None, :].expand(B, -1),
            Q2 / Q2.sum(1, keepdim=True).clamp_min(1e-12))
        return P, p1, p2, p3, md1, md2, md3, lvl

    def build_seq(self, x, md1, md2, md3):
        """prompt ++ ['0','.',d1m,d2m,d3m]; supervised logit positions are
        last_prompt_pos + 0..5 (predicting '0','.',d1,d2,d3,EOS)."""
        ids_p, mask_p = self._module()._serialize(x)
        B = ids_p.shape[0]
        dig = torch.stack([md1 + DIGIT_IDS[0], md2 + DIGIT_IDS[0],
                           md3 + DIGIT_IDS[0]], 1)              # digit token ids
        ans = torch.cat([torch.full((B, 1), TOK0, dtype=ids_p.dtype,
                                    device=ids_p.device),
                         torch.full((B, 1), DOT_ID, dtype=ids_p.dtype,
                                    device=ids_p.device),
                         dig], 1)                                # [B,5]
        ids = torch.cat([ids_p, ans], 1)
        mask = torch.cat([mask_p, torch.ones_like(ans)], 1)
        last = (mask_p.sum(1) - 1).clamp(min=0)                  # [B] last prompt
        return ids, mask, last

    # ---------------- one optimization step ----------------
    def train_step(self, batch):
        o, a, r, no, d, g, q16, am = batch
        shard = o.shape[0]
        P, p1, p2, p3, md1, md2, md3, lvl = self.targets(q16)
        ids, mask, last = self.build_seq(o, md1, md2, md3)
        W = _lm_head(self._module())                       # [V,H] frozen (tied)
        Wd = W[torch.tensor(DIGIT_IDS, device=W.device)]   # [10,H]
        npos = last[:, None] + torch.arange(6, device=last.device)[None, :]
        fmt_tgt = torch.stack([torch.full_like(md1, TOK0),
                               torch.full_like(md1, DOT_ID),
                               torch.full_like(md1, EOS_ID)], 1)   # [B,3]

        self.opt.zero_grad(set_to_none=True)
        stats = [0.0] * 8     # loss, fmt, d1, d2, d3, v, pi, qrank
        nchunks = (shard + self.micro - 1) // self.micro
        for ci, s in enumerate(range(0, shard, self.micro)):
            e = min(s + self.micro, shard)
            last_ck = ci == nchunks - 1
            cm = (self.online.no_sync() if (self.world > 1 and not last_ck)
                  else contextlib.nullcontext())
            with cm:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    q1, pi, ha = self.online(
                        None, return_pi=True,
                        seq=(ids[s:e], mask[s:e], last[s:e], npos[s:e]))
                ha = ha.float()
                q = q1.squeeze(-1).float()                 # [m,K]
                v = q.mean(1, keepdim=True)                # dueling: mean_a q
                dl = ha[:, 2:5] @ Wd.float().t()           # [m,3,10]
                fl = ha[:, [0, 1, 5]] @ W.float().t()      # [m,3,V]
                fmt = torch.nn.functional.cross_entropy(
                    fl.transpose(1, 2), fmt_tgt[s:e])
                qc = q16[s:e].float() - q16[s:e].float().mean(1, keepdim=True)
                l_d = [-(p * torch.log_softmax(dl[:, i], dim=1)).sum(1).mean()
                       for i, p in enumerate((p1[s:e], p2[s:e], p3[s:e]))]
                l_v = smooth_l1(v.squeeze(1) / self.v_ref,
                                lvl[s:e] / self.v_ref).mean()
                pi = self._module().pi_head(z)             # aux (DDP coverage)
                p_t = torch.softmax(qc / self.tau, dim=1)
                l_pi = -(p_t * torch.log_softmax(pi.float(), dim=1)).sum(
                    1).mean()
                l_qrank = -(p_t * torch.log_softmax(
                    q / float(self.cfg.get("bc_qrank_tau", 1.0) or 1.0),
                    dim=1)).sum(1).mean()
                loss = (self.lam_fmt * fmt
                        + self.lam_d[0] * l_d[0] + self.lam_d[1] * l_d[1]
                        + self.lam_d[2] * l_d[2]
                        + self.lam_v * l_v + self.lam_pi * l_pi
                        + self.lam_qrank * l_qrank) * (e - s) / shard
            loss.backward()
            stats[0] += float(loss)
            for i, x in enumerate([fmt, l_d[0], l_d[1], l_d[2], l_v, l_pi,
                                   l_qrank]):
                stats[i + 1] += float(x) * (e - s) / shard
        torch.nn.utils.clip_grad_norm_(
            [p for p in self._module().parameters() if p.requires_grad],
            float(self.cfg.get("grad_clip", 1.0)))
        self.opt.step()
        return stats

    # ---------------- validation ----------------
    @torch.no_grad()
    def greedy_cols(self, x):
        """Exact 3-step greedy digit decode (same path as eval_qwen_policy)."""
        m = self._module()
        ids, mask = m._serialize(x)
        W = _lm_head(m)
        Wd = W[torch.tensor(DIGIT_IDS, device=W.device)].float()
        cols = torch.zeros(ids.shape[0], dtype=torch.long, device=ids.device)
        digs = torch.zeros(ids.shape[0], 3, dtype=torch.long, device=ids.device)
        pos = (mask.sum(1) - 1).clamp(min=0)
        for step in range(3):
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(self.device.type == "cuda")):
                out = m.trunk(input_ids=ids, attention_mask=mask,
                              use_cache=False)
            h = out.last_hidden_state.float()
            b = torch.arange(h.shape[0], device=h.device)
            lg = h[b, pos] @ Wd.t()
            dg = lg.argmax(1)
            digs[:, step] = dg
            ids = torch.cat([ids, (dg + DIGIT_IDS[0]).unsqueeze(1)], 1)
            mask = torch.cat([mask, torch.ones((ids.shape[0], 1),
                                               dtype=mask.dtype,
                                               device=mask.device)], 1)
            pos = torch.full((ids.shape[0],), ids.shape[1] - 1,
                             dtype=torch.long, device=ids.device)
        dv = digs.float()
        cols = (dv[:, 0] * 0.1 + dv[:, 1] * 0.01 + dv[:, 2] * 0.001) * 128.0
        return cols.floor().clamp(0, 127).long()

    @torch.no_grad()
    def validate(self, vb, n_batches=4):
        """Token-policy readouts on held-out shards. `val_regret_tok` (teacher
        Q of the greedily decoded action vs the teacher's best) is the
        control-relevant number; mode_match is the exact-string rate; the pi
        head and the Q head are scored on the same states for an A/B against
        the v3.1 head-BC (regret 25.6, pi_ce 3.8 / floor 1.6)."""
        m = self._module()
        acc = np.zeros(9)
        n = 0
        for i in range(n_batches):
            rng = np.random.default_rng(31337 + i)
            o, a, _, _, _, _, q16, am = vb.sample(min(256, vb.n), rng)
            o = o.to(self.device)
            q16 = q16.to(self.device)
            am = am.to(self.device).long()
            q = q16.float()
            lvl = q.mean(1, keepdim=True)
            mode = q.argmax(1)
            cols = self.greedy_cols(o)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(self.device.type == "cuda")):
                z = m._hidden(o)
                v = m.v_head(z).squeeze(1).float()
                aa = m.a_head(z).float()
                ppi = m.pi_head(z).float()
            qh = v[:, None] + aa - aa.mean(1, keepdim=True)
            pt = torch.softmax((q - lvl) / self.tau, dim=1)
            tgt_ent = -(pt * pt.clamp_min(1e-12).log()).sum(1).mean()
            pi_ce = -(pt * torch.log_softmax(ppi, dim=1)).sum(1).mean()
            acc += np.array([
                float((cols == mode).float().mean()),
                float((q.max(1).values
                       - q.gather(1, cols[:, None]).squeeze(1)).mean()),
                float((v - lvl.squeeze(1)).abs().mean() / self.v_ref),
                float((ppi.argmax(1) == mode).float().mean()),
                float((q.max(1).values - qh.gather(
                    1, qh.argmax(1)[:, None]).squeeze(1)).mean()),
                float(pi_ce),
                float(tgt_ent),
                float(pt.gather(1, mode[:, None]).mean()),
                float((cols == am).float().mean()),
            ])
            n += 1
        return acc / max(n, 1)

    # ---------------- persistence ----------------
    def _trainable_sd(self):
        mod = self._module()
        keep = {n for n, p in mod.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in mod.state_dict().items()
                if k in keep}

    def save(self, name):
        torch.save({"state_dict": self._trainable_sd(),
                    "trainable_only": True,
                    "grad_steps": self.grad_steps},
                   os.path.join(self.ckpt_dir, name))
        cks = sorted((f for f in os.listdir(self.ckpt_dir) if f.endswith(".pt")),
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
    ap.add_argument("--max-transitions", type=int, default=12_000_000)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--grad-budget", type=int, default=None)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    if args.grad_budget:
        cfg["grad_budget"] = int(args.grad_budget)
    obs_dim = int(cfg.get("T", 160)) * 5

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available()
                          else "cpu")
    os.makedirs(args.run_dir, exist_ok=True)

    data = BCData(args.data_dir, args.max_transitions, verbose=(rank == 0))
    if world > 1:
        dist.barrier()
    learner = SFTLearner(cfg, args.run_dir, device, rank, world, obs_dim)
    if args.resume_from:
        ck = torch.load(args.resume_from, map_location="cpu")
        sd = ck["state_dict"]
        strict = not ck.get("trainable_only", False)
        learner._module().load_state_dict(sd, strict=strict)
        learner.grad_steps = int(ck.get("grad_steps", 0))
        if rank == 0:
            print(f"[sft] resumed {args.resume_from} at step "
                  f"{learner.grad_steps} (trainable_only="
                  f"{ck.get('trainable_only', False)})", flush=True)
    if rank == 0:
        # every sequence this run can build: prompt bucket + 5 answer tokens
        # (train), and +1/+2/+3 (3-step greedy decode in validate/eval)
        max_len = learner._module().max_len
        warmup(learner._module(), batches=(8,),
               lengths=[l + k for l in range(128, max_len + 1, 128)
                        for k in (1, 2, 3, 5)],
               log=lambda s: print(s, flush=True))
        print("[sft] warmup done (prompt+1/2/3/5 shapes)", flush=True)

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
            oo, qq = mirror_batch(o[m].clone(), q16[m].clone(), learner.K,
                                  learner.T)
            o[m], q16[m] = oo, qq
        stats = learner.train_step((o, a, r, no, d, g, q16, am))
        learner.grad_steps += 1
        base = learner.lr_at(learner.grad_steps)
        for gp, fr in zip(learner.opt.param_groups, learner.lr_fracs):
            gp["lr"] = base * fr

        if rank == 0 and learner.grad_steps % 10 == 0:
            print(f"[sft] step {learner.grad_steps}/{learner.total_grad} "
                  f"loss={stats[0]:.4f} fmt={stats[1]:.4f} "
                  f"d1={stats[2]:.4f} d2={stats[3]:.4f} d3={stats[4]:.4f} "
                  f"v={stats[5]:.4f} pi={stats[6]:.4f} qr={stats[7]:.4f} "
                  f"{time.time()-ts:.1f}s", flush=True)

        if rank == 0 and learner.grad_steps % args.val_every == 0:
            vb = _VBWrapper(data)
            st = learner.validate(vb)
            rec = {"t": round(time.time() - t0, 1),
                   "grad_steps": learner.grad_steps,
                   "loss": round(stats[0], 4),
                   "l_fmt": round(stats[1], 4), "l_d1": round(stats[2], 4),
                   "l_d2": round(stats[3], 4), "l_d3": round(stats[4], 4),
                   "l_v": round(stats[5], 4), "l_pi": round(stats[6], 4),
                   "l_qrank": round(stats[7], 4),
                   "val_mode_match": round(float(st[0]), 4),
                   "val_regret_tok": round(float(st[1]), 3),
                   "val_v_mae": round(float(st[2]), 4),
                   "val_pi_agree": round(float(st[3]), 4),
                   "val_qhead_regret": round(float(st[4]), 3),
                   "val_pi_ce": round(float(st[5]), 4),
                   "val_tgt_ent": round(float(st[6]), 4),
                   "val_mass_mode": round(float(st[7]), 4),
                   "val_act_match": round(float(st[8]), 4),
                   "step_s": round(time.time() - ts, 2)}
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
            print(f"[sft-val] mode_match={rec['val_mode_match']:.3f} "
                  f"regret_tok={rec['val_regret_tok']:.2f} "
                  f"qhead_regret={rec['val_qhead_regret']:.2f} "
                  f"pi_agree={rec['val_pi_agree']:.3f} "
                  f"pi_ce={rec['val_pi_ce']:.2f}/{rec['val_tgt_ent']:.2f} "
                  f"v_mae={rec['val_v_mae']:.4f} "
                  f"(bc-era reference: regret 25.6, pi_ce 3.8, floor 1.6)",
                  flush=True)
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
