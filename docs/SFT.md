# Supervised Fine-Tuning (SFT)

rlforge includes a full-parameter causal-LM SFT path for small models. It uses the Hugging Face `Trainer` rather than adding another training engine. The current implementation is Qwen-native-thinking oriented and is separate from the async GSPO/GRPO trainer.

## What the path guarantees

- Strict input validation for the three-message `system` / `user` / `assistant` format and native `<think>…</think><answer>…</answer>` targets. Legacy `<explanation>` targets are rejected.
- Tokenization through the model's own chat template. Prompt tokens are masked from loss; the answer/think target is supervised. A token-prefix assertion detects template-boundary drift.
- A preflight mode that validates all JSONL rows and reports sequence, prompt, and supervised-token statistics without loading model weights or touching GPUs.
- Full-precision base/master weights with bf16 training compute, gradient accumulation, optional gradient checkpointing, fused AdamW, cosine schedule, and fixed random/data seeds.
- Per-step or per-epoch checkpoint saves with optimizer, scheduler, trainer state, model weights, and tokenizer. A completion marker is written only after the checkpoint contents validate.
- An atomic run manifest and append-only event log, an exclusive run-directory lock, and refusal to overwrite a non-empty run or resume with changed model/data/script/config hashes.
- Graceful SIGTERM/SIGINT/SIGUSR1 handling: finish the current step, save, and mark the run interrupted. Resume chooses the newest complete checkpoint only.
- A checkpoint ladder evaluator that holds evaluation data, evaluator version, and decoding settings fixed; it skips incomplete checkpoints, records per-step results, and kills evaluator process groups on timeout.

## Data contract

One JSON object per line:

```json
{"question_id":"example-001","messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"<think>...</think><answer>A</answer>"}]}
```

The roles and order are exact. The assistant target must include `</think>` and `<answer>` and must not contain the removed `<explanation>` field. Targets should come from approved RFT/distillation traces; this path does not claim teacher-visible rationale is private chain-of-thought. Keep training and held-out evaluation files separate, immutable, and hashed in the run records.

## SOP

Use a fresh, unique output directory for each experiment. Keep model and data local to the compute node, pin code/data/model versions, and do not start until the chosen GPU is explicitly idle and healthy.

1. **Preflight on CPU** (no model weights or GPU allocation):

   ```bash
   python scripts/sft/train_qwen_sft.py \
     --model /models/Qwen3.5-0.8B \
     --train /data/sft/train.jsonl \
     --output /data/runs/sft-YYYYMMDD-HHMMSS \
     --max-length 4096 --preflight-only
   ```

   Review row count, maximum sequence length, mean prompt/target tokens, and prefix-match ratio (`1.0` expected). Resolve any validation error before GPU use.

2. **Run the recovery/save smoke** in a separate empty directory on one designated GPU. This deliberately trains to step 1, sends SIGTERM, resumes, and checks that it completes exactly at `--max-steps` with a valid final checkpoint:

   ```bash
   python scripts/sft/test_recovery.py \
     --trainer scripts/sft/train_qwen_sft.py \
     --model /models/Qwen3.5-0.8B \
     --train /data/sft/train.jsonl \
     --output /data/runs/sft-recovery-YYYYMMDD-HHMMSS \
     --gpu 0 --max-steps 6 --timeout 600
   ```

   Do not point this destructive smoke at a real run directory. Require `recovery_test.json` with `status: passed` before approving unattended training.

3. **Start the real run** with an explicit learning rate, batch/accumulation, epoch or step limit, save cadence, output directory, and held-out set. Example command-line shape:

   ```bash
   python scripts/sft/train_qwen_sft.py \
     --model /models/Qwen3.5-0.8B \
     --train /data/sft/train.jsonl \
     --output /data/runs/sft-YYYYMMDD-HHMMSS \
     --epochs 1 --learning-rate 5e-6 --max-length 4096 \
     --batch-size 1 --gradient-accumulation-steps 8 \
     --save-steps 20 --save-total-limit 20 --warmup-steps 4
   ```

   These are example arguments, not a claim that this LR/batch is optimal. Record the exact invocation and estimate checkpoint storage before launch. Watch `run_manifest.json` and `run_events.jsonl`; `status=running` with increasing `global_step` is expected. Stop with SIGTERM, not `kill -9`, when possible. To resume an interrupted run, repeat the exact command and add `--resume auto`. Resume is intentionally rejected if the model, data, script, or effective configuration changed.

4. **Evaluate every complete checkpoint** on the same held-out file and exact same decode settings:

   ```bash
   python scripts/sft/eval_checkpoint_ladder.py \
     --run-dir /data/runs/sft-YYYYMMDD-HHMMSS \
     --eval-data /data/sft/heldout.jsonl \
     --evaluator scripts/sft/eval_qwen_native.py \
     --output-dir /data/runs/sft-YYYYMMDD-HHMMSS/checkpoint_evals \
     --n-samples 2 --temperature 1 --top-p 1 --max-tokens 4096 \
     --timeout-seconds 3600 --stop-on-error
   ```

   Compare base model and checkpoints under the same split, sampling, and scoring contract. Select checkpoints on held-out instruction adherence, truncation, answer quality, and format compliance—not training loss alone. Preserve the ladder manifest, per-checkpoint JSON, and logs.

5. **Promote to RL only after review.** Verify checkpoint files and tokenizer, run the standard held-out suite, check truncation/format metrics, and copy the selected checkpoint to an immutable promotion path with a checksum. SFT success or falling loss by itself is not an RL readiness gate.

## Run artifacts and resume semantics

A run directory contains `run_manifest.json`, `run_events.jsonl`, `checkpoint-N/`, and the final `recovery_test.json` for the smoke. The manifest freezes effective configuration and content hashes. Resume uses Trainer optimizer/scheduler/RNG state from the latest complete checkpoint; an incomplete save is ignored and never promoted. Checkpoints beyond `save_total_limit` may be pruned by Trainer, so set retention to cover expected recovery needs and the evaluation plan.

The SFT training and evaluation scripts are independent of the RL run's background processes, but share GPUs and disk. Do not infer GPU availability from a process-name search alone: inspect per-GPU memory/utilization/process owners and confirm ownership before launching.

## Validation status and limitations

The max-step stopping bug found during recovery testing is fixed: reaching `max_steps` now requests both save and stop. CPU tests cover the stopping condition, recovery watchdog, checkpoint integrity gate, evaluator timeout cleanup, and input tokenization. A GPU interrupt/resume smoke must still pass on an idle healthy GPU before the first unattended long run; CPU preflight is not a substitute. The current recipe is validated for the Qwen 3.5 0.8B native-thinking format only. Other model families need their own template/target contract and a GPU smoke before use.
