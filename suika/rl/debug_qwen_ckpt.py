"""Test whether enabling gradient checkpointing BEFORE peft wrapping makes
it actually engage (memory probe: one fwd+bwd chunk at micro 16).
"""
import sys
import time

import numpy as np
import torch
import yaml

cfg = yaml.safe_load(open("configs/w4_qwen_text.yaml"))
MODEL_PATH = cfg["model_path"]
N_ACT = 128


def build(order, ssm_bf16=False, dtype=torch.float32):
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    full = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=dtype, low_cpu_mem_usage=True)
    if ssm_bf16:
        try:
            full.config.text_config.mamba_ssm_dtype = "bfloat16"
        except Exception:
            pass
    core = getattr(full, "model", full)
    lang = getattr(core, "language_model", None) or core
    import gc
    del full, core
    gc.collect()
    if order == "ckpt_first":
        lang.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        lang = get_peft_model(lang, LoraConfig(
            r=64, lora_alpha=128, lora_dropout=0.0, bias="none", task_type=None,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
    else:
        lang = get_peft_model(lang, LoraConfig(
            r=64, lora_alpha=128, lora_dropout=0.0, bias="none", task_type=None,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        lang.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(lang, "enable_input_require_grads"):
        lang.enable_input_require_grads()
    return lang


def probe(order, ssm_bf16=False, micro=16, seq=512):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    trunk = build(order, ssm_bf16).to("cuda")
    n_flag = sum(1 for m_ in trunk.modules()
                 if getattr(m_, "gradient_checkpointing", False))
    opt = torch.optim.AdamW((p for p in trunk.parameters() if p.requires_grad),
                            lr=1e-4)
    ids = torch.randint(0, 100000, (micro, seq), device="cuda")
    mask = torch.ones_like(ids)
    t0 = time.time()
    for it in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = trunk(input_ids=ids, attention_mask=mask, use_cache=False)
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            loss = h.float().mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if it == 0:
            first = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"[{order} ssm_bf16={ssm_bf16} micro={micro} seq={seq}] "
          f"flag_modules={n_flag} peak={peak:.1f}GiB "
          f"static={sum(p.numel() * p.element_size() for p in trunk.parameters()) / 2 ** 30:.1f}GiB "
          f"first_iter={first:.1f}s iters={(time.time() - t0) / 3:.2f}s", flush=True)
    del trunk, opt
    torch.cuda.empty_cache()
    return peak


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "a"):
        probe("ckpt_first")
    if which in ("all", "b"):
        probe("peft_first")
    if which in ("all", "c"):
        probe("ckpt_first", ssm_bf16=True)
