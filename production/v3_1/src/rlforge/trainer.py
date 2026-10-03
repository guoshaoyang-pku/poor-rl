#!/usr/bin/env python3
"""Async GSPO/GRPO trainer for generative-model RL post-training (TRL + vLLM server).

Run under `accelerate launch` (see scripts/run_async_dp.sh for the full single-node
layout: vLLM rollout server on N GPUs + data-parallel trainer on the rest).

The reward function is pluggable: `--reward module:function` (default
`rlforge.rewards.mcq:mcq_reward`, an MCQ/ranking grader with per-task accounting).
The evaluator (rlforge.eval_mcq) imports the same grader code, so an eval number and
a training reward mean the same thing.
"""

import argparse
import json
import os
import time
from functools import partial

import torch
from datasets import Dataset

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.trainer.utils import nanmax, nanmin

from rlforge.rewards import load_reward_fn


def _has_linear_attention(model) -> bool:
    """True for hybrid models whose config lists linear-attention (GatedDeltaNet/Mamba-like) layers."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return False
    tc = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    layer_types = getattr(tc, "layer_types", None) or []
    return any("linear" in str(t) or "mamba" in str(t) for t in layer_types)


def per_seq_logprobs(model, input_ids, position_ids, completion_mask):
    """Run the packed (1, T) row as ONE right-padded (n_seq, L_max) batch so every sequence
    is computed in isolation, then scatter (log_probs, entropy) back into the packed
    (1, T-1) layout. The boundary slot (last token of sequence k predicting the first token
    of k+1) is a prompt position and gets 0.

    One forward per micro-batch (not one per sequence): under DDP every forward issues
    collectives (buffer broadcast), so a per-sequence loop deadlocks as soon as ranks hold
    rows with different sequence counts (2026-10-03, step 3 hang). Right padding is exact
    for causal models: pad tokens sit after every real token and are masked."""
    T = input_ids.shape[1]
    starts = (position_ids[0] == 0).nonzero().flatten().tolist()
    bounds = list(zip(starts, starts[1:] + [T]))
    n = len(bounds)
    L = max(e - a for a, e in bounds)
    ids = input_ids.new_zeros((n, L))
    attn = input_ids.new_zeros((n, L))
    cmask = completion_mask.new_zeros((n, L))
    for i, (a, e) in enumerate(bounds):
        ids[i, : e - a] = input_ids[0, a:e]
        attn[i, : e - a] = 1
        cmask[i, : e - a] = completion_mask[0, a:e]
    pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(n, -1)
    out = model(input_ids=ids, attention_mask=attn, position_ids=pos, labels=ids,
                completion_mask=cmask, use_cache=False)
    lp_pad, ent_pad = out["log_probs"], out["entropy"]
    log_probs = lp_pad.new_zeros((1, T - 1))
    entropy = ent_pad.new_zeros((1, T - 1))
    for i, (a, e) in enumerate(bounds):
        log_probs[0, a : e - 1] = lp_pad[i, : e - a - 1]
        entropy[0, a : e - 1] = ent_pad[i, : e - a - 1]
    return log_probs, entropy, n


class GSPOAsyncGRPOTrainer(AsyncGRPOTrainer):
    """AsyncGRPO with GSPO-style sequence-level importance sampling.

    Identical to the parent's compute_loss except the per-token ratio
    exp(log_ratio_t) is replaced by the per-sequence geometric-mean ratio
    exp(mean_t log_ratio_t) (GSPO, arXiv 2507.18071; mirrors TRL sync
    importance_sampling_level="sequence"). Token-level stats are kept for
    logging only.

    Two knobs (see --gspo-norm / --gspo-eps-*):
      * norm: the parent divides the loss by the global completion-token count, which
        weights every sequence by its length. Arm C (2026-09-29) collapsed ~12x faster
        than the token-level arm at the same lr, and the longest sequences were exactly
        the ones carrying the -2 truncation penalty. `seq_mean` instead averages the
        per-token losses within each sequence first and then averages over sequences
        (uniform per sequence, the GSPO-paper objective); `token` reproduces the arm-C
        layout.
      * eps: the clip range for the *sequence* ratio. Token-level values (0.2/0.28)
        made the sequence clip a no-op; the paper uses ~3e-4/4e-4. The values are read
        in __init__ rather than in compute_loss so a sweep only touches the launcher.
    """

    def __init__(self, *args, gspo_norm: str = "seq_mean",
                 gspo_dynamic_low_frac: float = 0.0, per_seq_forward: str = "auto",
                 kl_beta: float = 0.0, prefix_share: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # KL-to-reference (frozen copy of the starting policy): k3 estimator on completion
        # tokens, seq-mean then mean over sequences; added to the GSPO loss with weight
        # kl_beta. Off (0) by default, which keeps the previous behaviour bit-for-bit.
        self._kl_beta = float(kl_beta)
        self._ref_model = None
        if self._kl_beta > 0:
            import copy
            self._ref_model = copy.deepcopy(self.accelerator.unwrap_model(self.model)).to(torch.bfloat16).eval()
            self._ref_model.requires_grad_(False)
            print(f"[rlforge] kl_beta={self._kl_beta} (frozen reference = start policy)", flush=True)
        # Shared-prompt forward (rlforge.prefix_share): each group's prompt is forwarded once and its
        # completions branch from it. Installed AFTER the reference deepcopy and on each model
        # separately; the forward is a bound method, so a later deepcopy would also stay correct.
        self._prefix_share = bool(prefix_share)
        if self._prefix_share:
            from rlforge.prefix_share import install
            install(self.model, temperature=self.temperature)
            if self._ref_model is not None:
                install(self._ref_model, temperature=self.temperature)
            print("[rlforge] prefix_share=on (prompt forwarded once per group; overrides per_seq_forward)",
                  flush=True)
        hybrid = _has_linear_attention(self.model)
        if per_seq_forward == "auto":
            self._per_seq_forward = hybrid
        else:
            self._per_seq_forward = per_seq_forward == "on"
        self._per_seq_calls = 0
        print(f"[rlforge] per_seq_forward={self._per_seq_forward} "
              f"(mode={per_seq_forward}, linear_attention_layers={hybrid})", flush=True)
        if hybrid and not self._per_seq_forward:
            print("[rlforge] WARNING: hybrid linear-attention model trained with packed rows; "
                  "recurrent/conv state leaks across sequence boundaries", flush=True)
        if gspo_norm not in ("token", "seq_mean"):
            raise ValueError(f"unknown gspo_norm {gspo_norm!r}")
        self._gspo_norm = gspo_norm
        # Dynamic low clip: bound = exp(EMA of this quantile of per-seq log-ratio),
        # updated AFTER each micro-batch, so the bound in effect is always the
        # previous average (first micro-batch falls back to the fixed eps_low).
        self._dyn_low_frac = float(gspo_dynamic_low_frac)
        self._dyn_low_ema = None
        # v3 audits (rlforge_v3, 2026-10-03), no collectives:
        #  * mb_audit: every rank prints its own micro-batch / row-sequence count per optimizer
        #    step (first RLFORGE_MB_AUDIT_STEPS steps, then every 50) -- DDP needs equal counts.
        #  * poslog (gate 5): per-sequence mean log-ratio with the sequence's position inside its
        #    prompt group, for the first RLFORGE_POSLOG_STEPS optimizer steps, one jsonl per rank.
        self._mb_audit_steps = int(os.environ.get("RLFORGE_MB_AUDIT_STEPS", "20"))
        self._audit_local_seqs = 0
        self._poslog_dir = os.environ.get("RLFORGE_POSLOG_DIR") or None
        self._poslog_steps = int(os.environ.get("RLFORGE_POSLOG_STEPS", "0"))
        self._poslog_mb = 0

    def _write_poslog(self, input_ids, position_ids, completion_mask, seq_mean_lr, seq_n_tok):
        """Gate 5 (per-position bias): one line per sequence of this rank's row with its index in
        row order inside its prompt group (k_row) and in the order the shared path processes it
        (k_proc: length-descending, as prefix_share._buckets). Best effort, never raises."""
        try:
            from rlforge.prefix_share import plan_groups
            rank = self.accelerator.process_index
            T = input_ids.shape[1]
            starts = (position_ids[0] == 0).nonzero().flatten().tolist()
            seq_index = {a: j for j, a in enumerate(starts)}
            lr = seq_mean_lr.detach().float().cpu().tolist()
            nt = seq_n_tok.detach().float().cpu().tolist()
            lines = []
            for gi, (p, segs) in enumerate(plan_groups(input_ids, position_ids, completion_mask)):
                lens = [e - a for a, e in segs]
                proc = sorted(range(len(segs)), key=lambda k: -lens[k])
                k_proc = {k: r for r, k in enumerate(proc)}
                for k, (a, e) in enumerate(segs):
                    j = seq_index[a]
                    lines.append(json.dumps({
                        "step": int(self.state.global_step), "mb": self._poslog_mb, "rank": rank,
                        "group": gi, "group_size": len(segs), "prompt_len": p, "k_row": k,
                        "k_proc": k_proc[k], "seq_len": e - a, "n_tok": nt[j], "log_rho": lr[j]}))
            os.makedirs(self._poslog_dir, exist_ok=True)
            with open(os.path.join(self._poslog_dir, f"poslog_rank{rank}.jsonl"), "a") as f:
                f.write("\n".join(lines) + "\n")
            self._poslog_mb += 1
        except Exception as e:  # noqa: BLE001
            print(f"[rlforge] poslog skipped: {type(e).__name__}: {e}", flush=True)

    def _log_step_metrics(self):
        step = int(self.state.global_step)
        if step <= self._mb_audit_steps or step % 50 == 0:
            print(f"[rlforge][mb_audit] rank={self.accelerator.process_index} step={step} "
                  f"microbatches={self._step_microbatches} local_seqs={self._audit_local_seqs} "
                  f"global_samples={self._step_samples:.0f}", flush=True)
        self._audit_local_seqs = 0
        super()._log_step_metrics()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        mask_bool = inputs["attention_mask"].bool()
        input_ids = inputs["input_ids"][mask_bool].unsqueeze(0)
        completion_mask = inputs["completion_mask"][mask_bool].unsqueeze(0)
        old_log_probs = inputs["old_log_probs"][mask_bool].unsqueeze(0)
        position_ids = inputs["position_ids"][mask_bool].unsqueeze(0)
        advantages = inputs["advantages"][mask_bool].unsqueeze(0)

        forward_start = time.perf_counter()
        if self._prefix_share:
            outputs = model(prefix_share=dict(input_ids=input_ids, position_ids=position_ids,
                                              completion_mask=completion_mask))
            log_probs, entropy = outputs["log_probs"], outputs["entropy"]
            ps = outputs["prefix_share_stats"]
            m = self._metrics["train"]
            m["prefix_share/forward_token_frac"].append(ps["forward_tokens"] / max(ps["unshared_tokens"], 1))
            m["prefix_share/pad_frac"].append(1 - ps["forward_tokens"] / max(ps["padded_tokens"], 1))
            m["prefix_share/seqs_per_group"].append(ps["seqs"] / max(ps["groups"], 1))
            if self.aux_loss_enabled:
                raise NotImplementedError("prefix-share forward does not aggregate MoE aux loss")
        elif self._per_seq_forward:
            # Hybrid models (Qwen3.5 GatedDeltaNet: causal conv + recurrent state) do NOT
            # reset state at packed-sequence boundaries without fla/causal_conv1d varlen
            # kernels, so a packed row leaks state from sequence i into i+1 (2026-10-03:
            # packed-vs-clean logprob gap -0.1..-0.6, growing with pack position). Run each
            # sequence alone and stitch the outputs back into the packed layout: the
            # boundary slot (last token of i predicting first token of i+1) is a prompt
            # position (completion_mask 0) and is filled with 0.
            log_probs, entropy, n_calls = per_seq_logprobs(model, input_ids, position_ids, completion_mask)
            self._per_seq_calls += n_calls
            outputs = {"log_probs": log_probs, "entropy": entropy}
            if self.aux_loss_enabled:
                raise NotImplementedError("per-seq forward does not aggregate MoE aux loss")
        else:
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                labels=input_ids,
                completion_mask=completion_mask,
                use_cache=False,
            )
            log_probs, entropy = outputs["log_probs"], outputs["entropy"]
        self._last_forward_time_s = time.perf_counter() - forward_start

        completion_mask = completion_mask[:, 1:]
        old_log_probs = old_log_probs[:, 1:]
        advantages = advantages[:, 1:]
        log_ratio = log_probs - old_log_probs

        # ---- packed-sequence segmentation (mirrors parent's stats block) ----
        seq_ids = (position_ids[0] == 0).cumsum(0)[
            1:
        ] - 1  # (T-1,) sample idx per shifted token
        num_seq = int((position_ids == 0).sum())
        valid = completion_mask[0] > 0  # (T-1,)

        # ---- GSPO: per-sequence mean log-ratio over completion tokens ----
        zeros = torch.zeros(num_seq, device=log_ratio.device, dtype=log_ratio.dtype)
        seq_lr_sum = zeros.index_add(0, seq_ids, log_ratio[0] * valid)
        seq_n_tok = zeros.index_add(0, seq_ids, valid.to(log_ratio.dtype))
        seq_mean_lr = seq_lr_sum / seq_n_tok.clamp(min=1.0)
        self._audit_local_seqs += num_seq
        if self._poslog_dir and self.state.global_step < self._poslog_steps:
            self._write_poslog(input_ids, position_ids, inputs["completion_mask"][mask_bool].unsqueeze(0),
                               seq_mean_lr, seq_n_tok)
        rho = torch.exp(seq_mean_lr)  # (num_seq,) sequence-level IS ratio
        low_bound = 1 - self.epsilon_low
        if self._dyn_low_frac > 0:
            if self._dyn_low_ema is not None:
                low_bound = float(torch.exp(self._dyn_low_ema))
            with torch.no_grad():
                q = torch.quantile(seq_mean_lr.detach().float(), self._dyn_low_frac)
                self._dyn_low_ema = (
                    q if self._dyn_low_ema is None
                    else 0.9 * self._dyn_low_ema + 0.1 * q
                )
        rho_clipped = torch.clamp(rho, low_bound, 1 + self.epsilon_high)

        rho_tok = rho[seq_ids]  # (T-1,) broadcast to tokens
        rho_clip_tok = rho_clipped[seq_ids]
        adv = advantages[0]
        per_token_loss = -torch.min(rho_tok * adv, rho_clip_tok * adv)

        global_n_tokens = inputs["global_n_tokens"][0]
        if self._gspo_norm == "seq_mean":
            # Uniform per sequence: mean over each sequence's completion tokens, then
            # mean over sequences. pdb is one full group, so num_seq is constant across
            # micro-batches and accumulation steps and no cross-rank correction is needed.
            seq_loss_sum = torch.zeros(
                num_seq, device=log_ratio.device, dtype=log_ratio.dtype
            ).index_add_(0, seq_ids, per_token_loss * valid)
            loss = (seq_loss_sum / seq_n_tok.clamp(min=1.0)).mean()
        else:
            loss = (per_token_loss * completion_mask).sum()
            world_size = self.accelerator.num_processes
            tokens_per_rank = (global_n_tokens / world_size).clamp(min=1.0)
            loss = loss / tokens_per_rank.to(torch.float32)
        kl_seq_mean = None
        if self._ref_model is not None:
            with torch.no_grad():
                if self._prefix_share:
                    ref_lp = self._ref_model(prefix_share=dict(
                        input_ids=input_ids, position_ids=position_ids,
                        completion_mask=inputs["completion_mask"][mask_bool].unsqueeze(0)))["log_probs"]
                elif self._per_seq_forward:
                    ref_lp, _, _ = per_seq_logprobs(self._ref_model, input_ids, position_ids,
                                                    inputs["completion_mask"][mask_bool].unsqueeze(0))
                else:
                    ref_lp = self._ref_model(input_ids=input_ids, position_ids=position_ids,
                                             labels=input_ids, completion_mask=inputs["completion_mask"][mask_bool].unsqueeze(0),
                                             use_cache=False)["log_probs"]
            d = (ref_lp.detach() - log_probs)[0]
            k3 = torch.exp(d) - d - 1.0
            kl_seq = torch.zeros(num_seq, device=k3.device, dtype=k3.dtype).index_add_(
                0, seq_ids, k3 * valid) / seq_n_tok.clamp(min=1.0)
            kl_seq_mean = kl_seq.mean()
            loss = loss + self._kl_beta * kl_seq_mean
        loss = loss / self.current_gradient_accumulation_steps

        if self.aux_loss_enabled:
            aux_loss = outputs["aux_loss"]
            loss = (
                loss
                + self.router_aux_loss_coef
                * aux_loss
                / self.current_gradient_accumulation_steps
            )

        with torch.no_grad():
            # token-level ratios for logging/metrics only (detached)
            coef_1_stat = torch.exp(log_ratio)
            valid_mask = completion_mask > 0
            local_count = valid_mask.sum().float()
            local_ratio_sum = coef_1_stat[valid_mask].sum()
            local_kl_sum = ((coef_1_stat[valid_mask] - 1) - log_ratio[valid_mask]).sum()
            local_entropy_sum = entropy[valid_mask].sum()
            is_low_clipped = (coef_1_stat < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1_stat > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped
            local_low_clip_sum = is_low_clipped[valid_mask].float().sum()
            local_high_clip_sum = is_high_clipped[valid_mask].float().sum()
            local_region_clip_sum = is_region_clipped[valid_mask].float().sum()

            stats = torch.stack(
                [
                    local_ratio_sum,
                    local_kl_sum,
                    local_entropy_sum,
                    local_low_clip_sum,
                    local_high_clip_sum,
                    local_region_clip_sum,
                    local_count,
                ]
            )
            stats = self.accelerator.reduce(stats, reduction="sum")
            (
                global_ratio_sum,
                global_kl_sum,
                global_entropy_sum,
                global_low_clip_sum,
                global_high_clip_sum,
                global_region_clip_sum,
                global_count,
            ) = stats.unbind(0)
            self._metrics["train"]["ratio"].append(
                (global_ratio_sum / global_count).item()
            )
            self._metrics["train"]["kl"].append((global_kl_sum / global_count).item())
            if kl_seq_mean is not None:
                self._metrics["train"]["kl_ref"].append(
                    self.accelerator.gather(kl_seq_mean.detach().float().reshape(1)).mean().item())
            self._metrics["train"]["entropy"].append(
                (global_entropy_sum / global_count).item()
            )
            self._metrics["train"]["clip_ratio/low_mean"].append(
                (global_low_clip_sum / global_count).item()
            )
            self._metrics["train"]["clip_ratio/high_mean"].append(
                (global_high_clip_sum / global_count).item()
            )
            self._metrics["train"]["clip_ratio/region_mean"].append(
                (global_region_clip_sum / global_count).item()
            )
            self._metrics["train"]["gspo/rho_mean"].append(rho.detach().mean().item())

            # Sequence-level clipping is what this loss actually applies; the token-level
            # clip block above reports the parent's quantities, which arm C's numbers
            # showed can look calm while the sequence clip does all the work.
            abs_log_rho = seq_mean_lr.detach().abs()
            seq_low_frac = (rho < low_bound).float().mean()
            seq_high_frac = (rho > 1 + self.epsilon_high).float().mean()
            seq_stats = torch.stack(
                [
                    seq_low_frac,
                    seq_high_frac,
                    abs_log_rho.mean(),
                    abs_log_rho.max(),
                ]
            )
            self._metrics["train"]["gspo/seq_clip_low_frac"].append(
                self.accelerator.reduce(seq_stats[0], reduction="mean").item()
            )
            self._metrics["train"]["gspo/seq_clip_high_frac"].append(
                self.accelerator.reduce(seq_stats[1], reduction="mean").item()
            )
            self._metrics["train"]["gspo/abs_log_rho_mean"].append(
                self.accelerator.reduce(seq_stats[2], reduction="mean").item()
            )
            self._metrics["train"]["gspo/abs_log_rho_max"].append(
                self.accelerator.reduce(seq_stats[3], reduction="max").item()
            )
            # signed bias + spread of the per-sequence log-ratio (packing bug showed up as a
            # negative-only shift; after the fix it should be ~symmetric around 0)
            slr = seq_mean_lr.detach().float()
            extra = torch.stack([slr.mean(), torch.quantile(slr.abs(), 0.5), torch.quantile(slr.abs(), 0.9)])
            extra = self.accelerator.reduce(extra, reduction="mean")
            self._metrics["train"]["gspo/log_rho_mean"].append(extra[0].item())
            self._metrics["train"]["gspo/abs_log_rho_p50"].append(extra[1].item())
            self._metrics["train"]["gspo/abs_log_rho_p90"].append(extra[2].item())
            if self._dyn_low_frac > 0 and self._dyn_low_ema is not None:
                self._metrics["train"]["gspo/dyn_low_bound"].append(
                    float(torch.exp(self._dyn_low_ema))
                )

            comp_mask = completion_mask[0].float()

            def seg_sum(vals):
                return torch.zeros(num_seq, device=comp_mask.device).index_add_(
                    0, seq_ids, vals
                )

            seq_tokens = seg_sum(comp_mask)
            seq_low = seg_sum(is_low_clipped[0].float() * comp_mask)
            seq_high = seg_sum(is_high_clipped[0].float() * comp_mask)
            per_seq_low = seq_low / seq_tokens
            per_seq_high = seq_high / seq_tokens
            gathered_low_min = self.accelerator.gather(nanmin(per_seq_low))
            gathered_high_max = self.accelerator.gather(nanmax(per_seq_high))
            self._metrics["train"]["clip_ratio/low_min"].append(
                nanmin(gathered_low_min).item()
            )
            self._metrics["train"]["clip_ratio/high_max"].append(
                nanmax(gathered_high_max).item()
            )

            if self.aux_loss_enabled:
                gathered_aux = self.accelerator.reduce(
                    aux_loss.detach().to(torch.float32), reduction="sum"
                )
                self._metrics["train"]["aux_loss"].append(
                    (gathered_aux / self.accelerator.num_processes).item()
                )

        n_forward_tokens = float(inputs["global_n_forward_tokens"][0])
        mean_seq_len = float(inputs["mean_seq_len"][0])
        self._step_forward_tokens += n_forward_tokens
        self._step_trained_tokens += float(global_n_tokens)
        self._step_seq_len_weighted += mean_seq_len * n_forward_tokens
        self._step_samples += n_forward_tokens / mean_seq_len
        self._step_forward_s += self._last_forward_time_s
        return loss


def _apply_liger_base_kernels(model_path: str) -> None:
    """Monkey-patch base kernels (RMSNorm/RoPE/SwiGLU) onto the model class.

    TRL's async trainer hard-blocks `use_liger_kernel` (NotImplementedError) and its
    fused-linear-CE would bypass the per-token logprob computation GRPO needs, so we
    apply only the elementwise kernels. Dispatch by config model_type.
    """
    from transformers import AutoConfig

    model_type = AutoConfig.from_pretrained(model_path).model_type
    patches = {
        "qwen2": "apply_liger_kernel_to_qwen2",
        "qwen3": "apply_liger_kernel_to_qwen3",
        "qwen3_moe": "apply_liger_kernel_to_qwen3_moe",
        "llama": "apply_liger_kernel_to_llama",
    }
    if model_type not in patches:
        raise SystemExit(f"--use-liger: no base-kernel patch registered for model_type "
                         f"{model_type!r} (known: {sorted(patches)})")
    from liger_kernel.transformers import monkey_patch as liger_mp

    getattr(liger_mp, patches[model_type])(
        rope=True, rms_norm=True, swiglu=True,
        cross_entropy=False, fused_linear_cross_entropy=False,
    )
    print(f"[rlforge] liger base kernels applied for {model_type}")


def _patch_attention_for_old_gpus() -> None:
    """A100/older compat: TRL hardcodes the flash-attn3 kernel (Hopper-only); swap to
    flash_attention_2 when the local device is pre-Hopper. No-op on H100/H200."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] >= 9:
        return
    import trl.experimental.async_grpo.async_grpo_trainer as agt

    orig = agt.create_model_from_path

    def with_fa2(*a, **kw):
        if kw.get("attn_implementation") == "kernels-community/flash-attn3":
            kw["attn_implementation"] = "flash_attention_2"
        return orig(*a, **kw)

    agt.create_model_from_path = with_fa2
    print("[rlforge] pre-Hopper GPU detected: attention implementation -> flash_attention_2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local model path or HF model id")
    ap.add_argument("--reward", default="rlforge.rewards.mcq:mcq_reward",
                    help="reward function as module:function (must be importable)")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--out", default="runs/async_v1")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--num-generations", type=int, default=16)
    ap.add_argument("--completions-per-step", type=int, default=256)
    ap.add_argument("--max-completion", type=int, default=16384)
    ap.add_argument("--max-staleness", type=int, default=4)
    ap.add_argument("--max-inflight-tasks", type=int, default=128)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--server-url", default="http://localhost:8000")
    ap.add_argument(
        "--dtype",
        choices=["none", "bfloat16"],
        default="none",
        help="none = mixed precision fp32 master weights; bfloat16 = pure bf16 (A/B test)",
    )
    ap.add_argument(
        "--gspo", action="store_true", help="GSPO sequence-level importance sampling"
    )
    ap.add_argument("--max-steps", type=int, default=0, help="0 = driven by epochs")
    # TRL's 120 s default assumes a request can finish quickly. Here every sample may run to
    # the 8k completion cap, and the server admits many sequences at once, so a slow-but-fine
    # request is killed and retried forever ("POST /v1/completions failed (TimeoutError)")
    # while the trainer waits for samples. Budget for the worst case instead:
    # max_completion / (aggregate tok/s / concurrent sequences), with margin.
    ap.add_argument("--request-timeout", type=float, default=120.0,
                    help="per-request timeout in seconds (TRL default 120)")
    ap.add_argument(
        "--gspo-norm",
        choices=["token", "seq_mean"],
        default="seq_mean",
        help="token = arm-C layout (length-weighted); seq_mean = uniform per sequence",
    )
    ap.add_argument("--gspo-eps-low", type=float, default=3e-4,
                    help="sequence-ratio lower clip (paper ~3e-4; token-level 0.2 was a no-op)")
    ap.add_argument("--gspo-eps-high", type=float, default=4e-4,
                    help="sequence-ratio upper clip (paper ~4e-4)")
    ap.add_argument("--gspo-dynamic-low-frac", type=float, default=0.0,
                    help="if >0: low clip bound = exp(EMA of this quantile of per-seq "
                         "log-ratio over micro-batches, targeting this low-side clip "
                         "fraction instead of a fixed eps_low; high side stays fixed")
    ap.add_argument("--kl-beta", type=float, default=0.0,
                    help="weight of k3 KL to a frozen copy of the start policy (0 = off)")
    ap.add_argument("--per-seq-forward", choices=["auto", "on", "off"], default="auto",
                    help="GSPO path: forward each packed sequence separately (auto = on for models "
                         "with linear-attention layers, whose state leaks across packed boundaries)")
    ap.add_argument("--prefix-share", choices=["on", "off"], default="off",
                    help="GSPO path: forward each group's prompt once and branch its completions from it "
                         "(rlforge.prefix_share; exact, hybrid Qwen3.5/3.8 only). Also swaps TRL's planners "
                         "for group-aware ones so a group's samples land in the same row.")
    ap.add_argument("--token-budget", type=int, default=None,
                    help="TRL token_budget: per-row token cap of the micro-batch planner (default: vLLM "
                         "max_model_len). 0 = fixed count (per_device_train_batch_size samples per row; "
                         "with --prefix-share on that is exactly one group per row)")
    # --- v3 infra (rlforge_v3, 2026-10-03) --------------------------------------------
    ap.add_argument("--dp-route", choices=["on", "off"], default="off",
                    help="group-affine routing for a data-parallel vLLM server (rlforge.dp_route): every "
                         "in-flight request of one prompt group is pinned to one DP replica through the "
                         "X-data-parallel-rank header (least-loaded at first request). No-op at DP == 1. "
                         "DP size from RLFORGE_DP_SIZE, else /get_world_size.")
    ap.add_argument("--queue-maxsize", type=int, default=None,
                    help="TRL rollout queue (scored samples waiting for the trainer) capacity; default "
                         "TRL's 1024. Raise to >= samples/step x max_staleness so the scorer never "
                         "backpressures (rollout/backpressure_s).")
    _env_audit = os.environ.get("RLFORGE_DROP_AUDIT", "off").strip().lower()
    if _env_audit not in ("on", "off", "1", "0", "true", "false", ""):
        raise SystemExit(f"RLFORGE_DROP_AUDIT={_env_audit!r}: expected on/off")
    ap.add_argument("--drop-audit", choices=["on", "off"],
                    default="on" if _env_audit in ("on", "1", "true") else "off",
                    help="rlforge.drop_audit (observe-only, rank 0): per optimizer step, length-bucketed stale-drop "
                         "rates, generated-vs-trained completion lengths and dropped-vs-trained group rewards, as "
                         "drop_audit/* metrics plus one line in $RLFORGE_DROP_AUDIT_PATH (default "
                         "<out>/drop_audit.jsonl). Default from env RLFORGE_DROP_AUDIT, else off.")
    ap.add_argument("--no-thinking", action="store_true",
                    help="do not pass enable_thinking=True to the chat template "
                         "(Qwen-family templates only; omit for other models)")
    ap.add_argument("--report-to", default="none",
                    help="comma-separated HF integrations: tensorboard, wandb, mlflow, "
                         "swanlab... ('none' disables). The run's own JSONL/HTML panel "
                         "(rlforge.report) is independent of this.")
    ap.add_argument("--run-name", default=None,
                    help="run name for the tracker integrations (default: run dir name)")
    # --- performance knobs (see docs/OPTIMIZATION.md) -------------------------------
    ap.add_argument("--use-liger", action="store_true",
                    help="apply liger base kernels (RMSNorm/RoPE/SwiGLU) to the model "
                         "class before init. NOT the fused-linear-CE path (the async "
                         "trainer forbids it and it conflicts with logprob scoring).")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="disable gradient checkpointing (~30%% faster, much more VRAM; "
                         "fits at <=1B on >=140GB cards, keep ON on A100-40/80G)")
    ap.add_argument("--optim", default=None,
                    help="HF optimizer name, e.g. adamw_torch_fused (default: TRL's)")
    ap.add_argument("--allow-tf32", action="store_true",
                    help="enable TF32 matmul (matters for the fp32-master recipe; "
                         "bf16 compute is unaffected)")
    # --- non-blocking scoring (rlforge/score_loop.py; all off by default = stock TRL) -------
    ap.add_argument("--score-concurrency", type=int, default=0,
                    help="score up to N rollout groups concurrently (0 = stock TRL serial score "
                         "loop; 1 = rlforge loop, serial, metric-identical to stock). A slow reward "
                         "(LLM judge) then delays only its own group, not generation. Idle slots cost "
                         "nothing: size it well above the number of judged groups in flight (e.g. 32).")
    ap.add_argument("--judged-max-staleness", type=int, default=None,
                    help="cap on the staleness of samples from groups the reward sent to the judge "
                         "(reward sets rlforge_info['judged']). Judged samples are allowed exactly the "
                         "weight syncs their own scoring took (rlforge/judge_versions), up to this cap. "
                         "Default: --max-staleness + 2 when --score-concurrency > 1, else off; "
                         "-1 = off; otherwise must be > --max-staleness.")
    ap.add_argument("--reward-early-hooks", action="store_true",
                    help="EXPERIMENTAL, not recommended: call the reward's rlforge_on_rollout hook as each "
                         "rollout finishes (aiq_think_reward_v3: submit judge calls before the group is "
                         "complete). Under judge saturation it favours short rollouts.")
    ap.add_argument("--score-task-max-s", type=float, default=None,
                    help="fail the rollout worker if one group's reward scoring takes longer than this "
                         "(hung reward/judge; TRL's check_health then stops the run). Default: env "
                         "RLFORGE_SCORE_TASK_MAX_S, else max(600, 3 x AIQ_HALLUC_TIMEOUT_S).")
    args = ap.parse_args()

    if (args.score_concurrency > 0 or args.reward_early_hooks
            or (args.judged_max_staleness is not None and args.judged_max_staleness >= 0)):
        from rlforge.score_loop import install as _install_score_loop
        if args.judged_max_staleness is None:
            _judged = "auto"
        else:
            _judged = args.judged_max_staleness if args.judged_max_staleness >= 0 else None
        _install_score_loop(
            score_concurrency=args.score_concurrency,
            judged_max_staleness=_judged,
            early_hooks=args.reward_early_hooks,
            max_staleness=args.max_staleness,
            score_task_max_s=args.score_task_max_s,
        )

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.use_liger:
        _apply_liger_base_kernels(args.model)

    _patch_attention_for_old_gpus()

    rows = [json.loads(line) for line in open(args.train)]
    ds = Dataset.from_list(rows)

    pdb = args.num_generations  # one full group per micro-batch
    gas = max(1, args.completions_per_step // pdb)

    report_to = [] if args.report_to in ("none", "") else args.report_to.split(",")

    cfg_kwargs = dict(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        per_device_train_batch_size=pdb,
        gradient_accumulation_steps=gas,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        temperature=1.0,
        top_p=1.0,
        epsilon=args.gspo_eps_low if args.gspo else 0.2,
        epsilon_high=args.gspo_eps_high if args.gspo else 0.28,
        max_staleness=args.max_staleness,
        max_inflight_tasks=args.max_inflight_tasks,
        weight_sync_steps=1,
        chat_template_kwargs={} if args.no_thinking else {"enable_thinking": True},
        vllm_server_base_url=args.server_url,
        bf16=True,
        gradient_checkpointing=not args.no_grad_ckpt,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=5,
        save_only_model=True,
        report_to=report_to,
        logging_dir=args.out + "/tb",
        run_name=args.run_name or args.out.rstrip("/").rsplit("/", 1)[-1],
        log_completions=False,
        seed=0,
        request_timeout=args.request_timeout,
    )
    if args.dtype == "bfloat16":
        cfg_kwargs["dtype"] = "bfloat16"
    if args.token_budget is not None:
        cfg_kwargs["token_budget"] = args.token_budget
    if args.queue_maxsize is not None:
        cfg_kwargs["queue_maxsize"] = args.queue_maxsize
    if args.optim:
        cfg_kwargs["optim"] = args.optim
    if args.max_steps:
        cfg_kwargs["max_steps"] = args.max_steps
        cfg_kwargs.pop("num_train_epochs")
    # TRL's experimental config is a filtered dataclass, not the full
    # TrainingArguments -- unknown kwargs raise TypeError (e.g. logging_dir on
    # TRL 1.14). Filter to declared fields; dropped keys are reported so a
    # silently-ignored option is visible in the log.
    import dataclasses
    known = {f.name for f in dataclasses.fields(AsyncGRPOConfig)}
    if report_to and "logging_dir" not in known:
        # The TB callback reads args.logging_dir; on TRL versions whose config
        # lacks the field the callback crashes. Disable trackers instead --
        # the rlforge.report HTML panel is unaffected.
        print("[rlforge] this TRL version's config has no logging_dir; "
              "report_to disabled (use the rlforge.report panel instead)")
        cfg_kwargs["report_to"] = []
    dropped = sorted(set(cfg_kwargs) - known)
    if dropped:
        print(f"[rlforge] config keys not supported by this TRL version, dropped: {dropped}")
    cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in known}
    cfg = AsyncGRPOConfig(**cfg_kwargs)

    if args.prefix_share == "on":
        if not args.gspo:
            raise SystemExit("--prefix-share requires --gspo")
        # The planners are looked up by name when the dataloader is built; the group-aware ones keep
        # TRL's contracts (same micro-batch sample count / same forwarded-token budget per row).
        import trl.experimental.async_grpo.async_grpo_trainer as agt
        from rlforge.prefix_share import GroupRowBatcher, GroupTokenBudgetBatcher
        agt.FixedCountBatcher = GroupRowBatcher
        agt.TokenBudgetBatcher = GroupTokenBudgetBatcher

    if args.dp_route == "on":
        # Must precede trainer construction: AsyncRolloutWorker._loop_cls is pickled by reference
        # into the spawned rollout child, which then imports rlforge.dp_route itself.
        from rlforge import dp_route
        dp_route.install()
    print(f"[rlforge] v3 infra: prefix_share={args.prefix_share} token_budget={args.token_budget} "
          f"dp_route={args.dp_route} (RLFORGE_DP_SIZE={os.environ.get('RLFORGE_DP_SIZE', 'auto')}) "
          f"queue_maxsize={cfg.queue_maxsize} max_inflight={cfg.max_inflight_tasks} "
          f"pdb={cfg.per_device_train_batch_size} gas={cfg.gradient_accumulation_steps}", flush=True)

    trainer_cls = GSPOAsyncGRPOTrainer if args.gspo else AsyncGRPOTrainer
    if not args.gspo:
        print("[rlforge] WARNING: non-GSPO path has no per-seq forward; hybrid models pack rows", flush=True)
    trainer_kwargs = {}
    if args.gspo:
        trainer_kwargs["gspo_norm"] = args.gspo_norm
        trainer_kwargs["gspo_dynamic_low_frac"] = args.gspo_dynamic_low_frac
        trainer_kwargs["per_seq_forward"] = args.per_seq_forward
        trainer_kwargs["kl_beta"] = args.kl_beta
        trainer_kwargs["prefix_share"] = args.prefix_share == "on"
    trainer = trainer_cls(
        model=args.model,
        reward_funcs=partial(load_reward_fn(args.reward), cap=args.max_completion),
        args=cfg,
        train_dataset=ds,
        **trainer_kwargs,
    )
    if args.drop_audit == "on":
        # Wraps get_train_dataloader (rank 0 only builds the queue dataset) and adds a step-end callback after
        # TRL's own; never touches which samples are trained or in what order (rlforge/drop_audit.py).
        # Any failure here leaves the trainer as it was (the audit is optional, training is not).
        audit_path = os.environ.get("RLFORGE_DROP_AUDIT_PATH") or os.path.join(args.out, "drop_audit.jsonl")
        try:
            from rlforge import drop_audit
            _audit = drop_audit.install(trainer, path=audit_path, max_completion=args.max_completion,
                                        num_generations=args.num_generations)
        except Exception as e:  # noqa: BLE001
            _audit = None
            print(f"[rlforge] drop_audit import/install failed: {type(e).__name__}: {e}", flush=True)
        if _audit is not None:
            print(f"[rlforge] drop_audit=on -> {audit_path} (observe-only, rank 0; drop_audit/* metrics; "
                  f"buckets {_audit.names})", flush=True)
        else:
            print("[rlforge] WARNING: --drop-audit on but the audit is NOT installed; training continues "
                  "without it", flush=True)
    trainer.train()
    trainer.save_model(args.out + "/final")
    # Rank 0 only: every rank used to write this file (a race), and only rank 0's log_history carries the
    # rank-0 metrics (TRL's reward/queue metrics, drop_audit/*).
    if trainer.accelerator.is_main_process:
        with open(args.out + "/log_history.json", "w") as f:
            json.dump(trainer.state.log_history, f, indent=1)


if __name__ == "__main__":
    main()
