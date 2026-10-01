"""Export a BC QwenQ checkpoint to a plain HF causal-LM dir for rlforge/vLLM.

BC saves the full QwenQ module state_dict (trunk w/ LoRA + V/A/pi heads).
rlforge's trainer and vLLM need a standard HF model directory:
  1. rebuild QwenQ, load the full state_dict;
  2. merge LoRA into the trunk weights (peft merge_and_unload);
  3. wrap the bare decoder in AutoModelForCausalLM and save_pretrained
     (+ tokenizer + chat template).

The V/A/pi heads are NOT exported (generation only needs the LM); the critic
for the reward still loads the original policy.pt via model_qwen.

Usage:
  python export_bc_to_hf.py --ckpt runs/w4bc_bc/policy.pt \
      --model-path /data/user/models/Qwen3.5-0.8B \
      --out /data/user/models/qwen_suika_bc5200_merged
"""
import argparse
import shutil

import torch

from paths import setup_engine_path
setup_engine_path()
import model_qwen  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    args = ap.parse_args()

    m = model_qwen.QwenQ(args.K, model_path=args.model_path,
                         lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    m.load_state_dict(sd, strict=True)

    trunk = m.trunk
    if hasattr(trunk, "merge_and_unload"):
        trunk = trunk.merge_and_unload()
        print("[export] LoRA merged")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    lm = AutoModelForCausalLM.from_config(trunk.config)
    lm.model = trunk
    lm.tie_weights()
    lm.save_pretrained(args.out)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    tok.save_pretrained(args.out)
    try:
        shutil.copy(f"{args.model_path}/chat_template.jinja", args.out)
    except OSError:
        pass
    print(f"[export] -> {args.out}")


if __name__ == "__main__":
    main()
