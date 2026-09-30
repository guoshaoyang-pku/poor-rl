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
import time
from functools import partial

import torch
from datasets import Dataset

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.trainer.utils import nanmax, nanmin

from rlforge.rewards import load_reward_fn

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

    def __init__(self, *args, gspo_norm: str = "seq_mean", **kwargs):
        super().__init__(*args, **kwargs)
        if gspo_norm not in ("token", "seq_mean"):
            raise ValueError(f"unknown gspo_norm {gspo_norm!r}")
        self._gspo_norm = gspo_norm

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
        rho = torch.exp(seq_mean_lr)  # (num_seq,) sequence-level IS ratio
        rho_clipped = torch.clamp(rho, 1 - self.epsilon_low, 1 + self.epsilon_high)

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
            seq_low_frac = (rho < 1 - self.epsilon_low).float().mean()
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
    args = ap.parse_args()

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
    dropped = sorted(set(cfg_kwargs) - known)
    if dropped:
        print(f"[rlforge] config keys not supported by this TRL version, dropped: {dropped}")
    cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in known}
    cfg = AsyncGRPOConfig(**cfg_kwargs)

    trainer_cls = GSPOAsyncGRPOTrainer if args.gspo else AsyncGRPOTrainer
    trainer_kwargs = {}
    if args.gspo:
        trainer_kwargs["gspo_norm"] = args.gspo_norm
    trainer = trainer_cls(
        model=args.model,
        reward_funcs=partial(load_reward_fn(args.reward), cap=args.max_completion),
        args=cfg,
        train_dataset=ds,
        **trainer_kwargs,
    )
    trainer.train()
    trainer.save_model(args.out + "/final")
    with open(args.out + "/log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)


if __name__ == "__main__":
    main()
