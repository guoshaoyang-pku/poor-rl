#!/usr/bin/env python3
"""Fixed-input FP8 numerical, gradient, throughput, and vLLM reload gates."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch


def timed(fn, repeats):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def graph_timed(fn, repeats):
    torch.autograd.graph.set_override_stale_capture_stream(True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    torch.autograd.graph.set_override_stale_capture_stream(False)
    return statistics.median(samples)


def error(candidate, reference):
    a, b = candidate.detach().float().flatten(), reference.detach().float().flatten()
    return {"relative_l2": float((a - b).norm() / b.norm().clamp_min(1e-30)),
            "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
            "abs_max": float((a - b).abs().max())}


def linear_gate(args):
    from rlforge.fp8 import install, quantize_rows
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.online.fp8 import (
        _fp8_channel_scale, _fp8_quant_per_channel,
    )

    rows = []
    torch.manual_seed(17)
    for tokens, hidden, output in [(32768, 1024, 3584), (4096, 1024, 3584), (4096, 3584, 1024),
                                   (64, 1024, 3584), (4096, 1024, 16)]:
        bf16 = torch.nn.Linear(hidden, output, bias=False, device="cuda", dtype=torch.float32)
        fp8 = torch.nn.Linear(hidden, output, bias=False, device="cuda", dtype=torch.float32)
        fp8.load_state_dict(bf16.state_dict())
        install(fp8)
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        dy = torch.randn(tokens, output, device="cuda", dtype=torch.bfloat16) * 1e-4
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yb = bf16(x)
        yb.backward(dy)
        bx, bw = x.grad.detach().clone(), bf16.weight.grad.detach().clone()
        x.grad = None
        yf = fp8(x)
        yf.backward(dy)
        assert fp8.weight.dtype == fp8.weight.grad.dtype == torch.float32
        assert torch.isfinite(fp8.weight.grad).all()
        q, scale = quantize_rows(fp8.weight.detach())
        expected_scale = _fp8_channel_scale(fp8.weight.detach().abs().amax(-1, keepdim=True))
        expected_q = _fp8_quant_per_channel(fp8.weight.detach(), expected_scale)
        # Activation parity also covers exact zeros and tiny values.
        act = torch.cat([x.detach()[:32], torch.zeros_like(x[:1]), x.detach()[:1] * 1e-8])
        aq, asc = quantize_rows(act)
        vq, vs = ops.scaled_fp8_quant(act, scale=None, use_per_token_if_dynamic=True)
        parity = {"weight_bytes_equal": bool(torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))),
                  "weight_scales_equal": bool(torch.equal(scale, expected_scale)),
                  "activation_bytes_equal": bool(torch.equal(aq.view(torch.uint8), vq.view(torch.uint8))),
                  "activation_scales_equal": bool(torch.equal(asc, vs))}
        assert all(parity.values()), parity
        numerics = {"forward": error(yf, yb), "dgrad": error(x.grad, bx),
                    "wgrad": error(fp8.weight.grad, bw)}
        assert numerics["wgrad"]["cosine"] > 0.98, numerics
        assert numerics["dgrad"]["cosine"] > 0.98, numerics

        def step(model):
            model.zero_grad(set_to_none=True)
            x.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x).backward(dy)

        bf_ms = timed(lambda: step(bf16), args.repeats)
        fp_ms = timed(lambda: step(fp8), args.repeats)
        bf_graph_ms = graph_timed(lambda: step(bf16), args.repeats)
        fp_graph_ms = graph_timed(lambda: step(fp8), args.repeats)
        frozen_ms = None
        fp8.requires_grad_(False)
        frozen_ms = timed(lambda: step(fp8), args.repeats)
        fp8.requires_grad_(True)
        step(fp8)
        optimizer = torch.optim.AdamW(fp8.parameters(), lr=2e-6)
        before = fp8.weight.detach().clone()
        optimizer.step()
        assert fp8.weight.grad.dtype == torch.float32
        assert optimizer.state[fp8.weight]["exp_avg"].dtype == torch.float32
        assert optimizer.state[fp8.weight]["exp_avg_sq"].dtype == torch.float32
        assert not torch.equal(before, fp8.weight)
        row = {"shape": [tokens, hidden, output], "bf16_ms": bf_ms, "fp8_ms": fp_ms,
               "speedup": bf_ms / fp_ms, "fp8_frozen_ms": frozen_ms,
               "bf16_graph_ms": bf_graph_ms, "fp8_graph_ms": fp_graph_ms,
               "graph_speedup": bf_graph_ms / fp_graph_ms,
               "parity": parity, "numerics": numerics, "master_grad_optimizer": "float32"}
        print(json.dumps(row), flush=True)
        rows.append(row)
        del bf16, fp8, x, dy, optimizer, before, yf, yb, bx, bw
        torch.cuda.empty_cache()
    return {"linear": rows}


def load_policy(args):
    if os.environ.get("RLFORGE_FUSED_OPS"):
        from rlforge.fused_ops import install
        install()
    from transformers import AutoModelForCausalLM, PreTrainedModel
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation=args.attention, local_files_only=True,
    ).cuda()
    text_config = model.config.get_text_config()
    if text_config is not model.config:
        model.requires_grad_(False)
        text = next(m for m in model.modules() if isinstance(m, PreTrainedModel)
                    and m is not model and m.config is text_config)
        text.requires_grad_(True)
        model.get_output_embeddings().requires_grad_(True)
    return model


def fixed_sequences(args):
    data = json.loads(Path(args.tokens).read_text())
    return [torch.tensor(s["token_ids"], device="cuda", dtype=torch.long).unsqueeze(0)
            for s in data["sequences"]], data


def score(model, sequences):
    from rlforge.prefix_share import _backbone
    from rlforge.fast_logprob import logprob_entropy
    scores = []
    for ids in sequences:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = _backbone(model)(input_ids=ids[:, :-1], use_cache=False).last_hidden_state
            lp, _ = logprob_entropy(hidden[0], model.lm_head.weight, None, ids[0, 1:], 1.0)
        scores.append(lp)
    return scores


def model_gate(args):
    from rlforge.fp8 import install
    from rlforge.prefix_share import prefix_shared_logprobs
    from torch.profiler import ProfilerActivity, profile

    sequences, data = fixed_sequences(args)
    records, scores, gradients = [], {}, {}
    for mode in ("bf16", "fp8"):
        model = load_policy(args)
        names = install(model) if mode == "fp8" else ()
        if mode == "fp8":
            model.lm_head.weight._rlforge_fp8_head = True
            if args.align:
                from rlforge.fp8_alignment import install as align_forward
                align_forward(model, gdn=True)
        model.train()
        # Measured rows use actual prompt/completion tokens, with the v3.2 shared-prefix path.
        ids = torch.cat(sequences, dim=1)
        pos = torch.cat([torch.arange(s.shape[1], device="cuda")[None] for s in sequences], dim=1)
        mask = torch.cat([torch.cat([torch.zeros(s["prompt_tokens"], device="cuda", dtype=torch.long),
                                     torch.ones(len(s["token_ids"]) - s["prompt_tokens"],
                                                device="cuda", dtype=torch.long)])[None]
                          for s in data["sequences"]], dim=1)
        completion_tokens = int(mask.sum())
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        def step():
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp, _, _ = prefix_shared_logprobs(model, ids, pos, mask, temperature=1.0)
                loss = -(lp * mask[:, 1:]).sum() / completion_tokens
            loss.backward()
            return loss

        torch.cuda.reset_peak_memory_stats()
        milliseconds = timed(step, args.repeats)
        loss = step()
        selected = {n: p.grad.detach().float().cpu().flatten() for n, p in model.named_parameters()
                    if p.grad is not None and ("layers.0." in n or "layers.3." in n)}
        gradients[mode] = selected
        with torch.no_grad():
            scores[mode] = [x.detach().cpu() for x in score(model.eval(), sequences)]
        graph_setting = os.environ.get("RLFORGE_FP8_GRAPHS", "0")
        os.environ["RLFORGE_FP8_GRAPHS"] = "0"
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        os.environ["RLFORGE_FP8_GRAPHS"] = graph_setting
        prof.export_chrome_trace(str(Path(args.output).with_suffix("." + mode + ".trace.json")))
        scaled_mm_calls = sum(e.name == "aten::_scaled_mm" for e in prof.events())
        native_kernels = sorted({e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
                                 and any(k in e.name.lower() for k in ["e4m3", "e5m2", "fp8", "nvjet_sm90_q", "float_e4m3"])})
        record = {"mode": mode, "loss": float(loss.detach()), "fwd_bwd_ms": milliseconds,
                  "completion_tokens": completion_tokens,
                  "completion_token_s": completion_tokens * 1000 / milliseconds,
                  "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                  "fp8_layers": list(names), "scaled_mm_calls": scaled_mm_calls,
                  "native_fp8_kernels": native_kernels[:30]}
        record["profile_scope"] = "eager kernels; timing uses requested graph setting"
        if mode == "fp8":
            assert scaled_mm_calls and native_kernels, "No native FP8 CUDA kernel observed"
            assert all(p.grad.dtype == torch.float32 for p in model.parameters() if p.grad is not None)
        records.append(record)
        print(json.dumps(record), flush=True)
        del model, selected, loss, prof
        gc.collect()
        torch.cuda.empty_cache()
    gradient_errors = {n: error(gradients["fp8"][n], g) for n, g in gradients["bf16"].items()}
    sequence_errors = []
    for idx, sample in enumerate(data["sequences"]):
        start = sample["prompt_tokens"] - 1
        delta = scores["fp8"][idx][start:] - scores["bf16"][idx][start:]
        sequence_errors.append({"abs_mean_token_logprob": float(delta.abs().mean()),
                                "mean_log_ratio": float(delta.mean())})
    score_path = Path(args.output).with_suffix(".scores.pt")
    torch.save(scores, score_path)
    return {"model": records, "speedup": records[0]["fwd_bwd_ms"] / records[1]["fwd_bwd_ms"],
            "gradients": gradient_errors, "fp8_vs_bf16": sequence_errors, "scores_file": str(score_path)}


def generate(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = [json.loads(line) for line in Path(args.data).open()][:2]
    prompts = [tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True,
                                            enable_thinking=True) for row in rows]
    engine = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=1,
                 gpu_memory_utilization=0.35, max_model_len=8192, max_num_seqs=16,
                 enforce_eager=True, kv_cache_dtype="auto")
    outputs = engine.generate([{"prompt_token_ids": p} for p in prompts],
                              SamplingParams(n=2, temperature=1.0, top_p=1.0,
                                             max_tokens=args.completion, seed=17))
    seqs = [{"prompt_tokens": len(o.prompt_token_ids),
             "token_ids": list(o.prompt_token_ids) + list(c.token_ids)}
            for o in outputs for c in o.outputs]
    Path(args.tokens).write_text(json.dumps({"sequences": seqs, "source_data": args.data,
                                            "checkpoint": args.model, "seed": 17}))
    return {"generated_sequences": len(seqs), "tokens": args.tokens}


def reload_probe(worker, checkpoint):
    from safetensors import safe_open
    from vllm.model_executor.model_loader.reload.layerwise import (
        finalize_layerwise_reload, initialize_layerwise_reload,
    )
    from rlforge.fp8 import quantize_rows
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    model = worker.model_runner.model
    target = next(m for name, m in model.named_modules() if name.endswith("layers.0.mlp.gate_up_proj"))
    before_q, before_s = target.weight.detach().clone(), target.weight_scale.detach().clone()
    pointer_q, pointer_s = target.weight.data_ptr(), target.weight_scale.data_ptr()
    head = next(m for m in model.modules() if isinstance(m, ParallelLMHead))
    head_q, head_s = head._poor_rl_head_cache
    before_head_q, before_head_s = head_q[0].clone(), head_s[0].clone()
    head_pointers = head_q.data_ptr(), head_s.data_ptr()
    expected = {}

    def weights():
        for shard in sorted(Path(checkpoint).glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as tensors:
                for name in tensors.keys():
                    value = tensors.get_tensor(name).float().cuda()
                    if name.endswith("layers.0.mlp.gate_proj.weight"):
                        value[0, 0] = value.abs().max() * 2
                        expected["gate"] = value.clone()
                    elif name.endswith("layers.0.mlp.up_proj.weight"):
                        expected["up"] = value.clone()
                    elif name.endswith("embed_tokens.weight"):
                        value[0, 0] = value.abs().max() * 2
                        expected["embedding"] = value
                    yield name, value

    initialize_layerwise_reload(model)
    model.load_weights(weights())
    finalize_layerwise_reload(model, worker.model_runner.vllm_config.model_config)
    quantized, scales = quantize_rows(torch.cat([expected["gate"], expected["up"]]))
    result = {"weight_changed": not torch.equal(target.weight.view(torch.uint8), before_q.view(torch.uint8)),
              "scale_changed": not torch.equal(target.weight_scale, before_s),
              "weight_matches_master": torch.equal(target.weight.t().contiguous().view(torch.uint8),
                                                     quantized.view(torch.uint8)),
              "scale_matches_master": torch.equal(target.weight_scale.reshape_as(scales), scales),
              "weight_storage_preserved": pointer_q == target.weight.data_ptr(),
              "scale_storage_preserved": pointer_s == target.weight_scale.data_ptr()}
    head_quantized, head_scales = quantize_rows(expected["embedding"])
    head_q, head_s = head._poor_rl_head_cache
    result.update({
        "tied_head_master_fp32": head.weight.dtype == torch.float32,
        "tied_head_master_matches": torch.equal(head.weight, expected["embedding"]),
        "tied_head_bytes_changed": not torch.equal(head_q[0].view(torch.uint8), before_head_q.view(torch.uint8)),
        "tied_head_scale_changed": not torch.equal(head_s[0], before_head_s),
        "tied_head_bytes_match": torch.equal(head_q.view(torch.uint8), head_quantized.view(torch.uint8)),
        "tied_head_scales_match": torch.equal(head_s, head_scales),
        "tied_head_storage_preserved": head_pointers == (head_q.data_ptr(), head_s.data_ptr()),
    })
    assert all(result.values()), result
    return result


class ReloadProbeWorker:
    def poor_rl_reload_probe(self, checkpoint):
        return reload_probe(self, checkpoint)


def serving_gate(args):
    from rlforge.fp8_serving import register
    from vllm import LLM, SamplingParams
    register()
    data = json.loads(Path(args.tokens).read_text())
    engine = LLM(model=args.model, dtype="bfloat16", quantization="poor_rl_fp8",
                 tensor_parallel_size=1, gpu_memory_utilization=0.4,
                 max_model_len=8192, max_num_seqs=16, enforce_eager=True, kv_cache_dtype="auto",
                 worker_extension_cls="fp8_gate.ReloadProbeWorker",
                 additional_config={"gdn_prefill_backend": "triton"} if args.align else {})
    outputs = engine.generate([{"prompt_token_ids": s["token_ids"]} for s in data["sequences"]],
                              SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1))
    scores = []
    for o in outputs:
        scores.append([o.prompt_logprobs[i][token].logprob for i, token in enumerate(o.prompt_token_ids) if i])
    torch.save(scores, Path(args.output).with_suffix(".scores.pt"))
    decode_outputs = engine.generate(
        [{"prompt_token_ids": s["token_ids"][:s["prompt_tokens"]]} for s in data["sequences"][:4]],
        SamplingParams(temperature=0, max_tokens=args.completion, logprobs=1, seed=17),
    )
    decode_rows = []
    for output in decode_outputs:
        completion = output.outputs[0]
        decode_rows.append({
            "prompt_tokens": len(output.prompt_token_ids),
            "token_ids": list(output.prompt_token_ids) + list(completion.token_ids),
            "rollout_logprobs": [lp[token].logprob for token, lp in zip(completion.token_ids, completion.logprobs)],
        })
    decode_path = Path(args.output).with_suffix(".decode.json")
    decode_path.write_text(json.dumps({"sequences": decode_rows, "checkpoint": args.model}))
    reload_result = engine.collective_rpc("poor_rl_reload_probe", args=(args.model,))
    return {"serving": "poor_rl_fp8", "sequences": len(scores),
            "alignment": args.align, "decode_file": str(decode_path),
            "hot_reload": reload_result,
            "scores_file": str(Path(args.output).with_suffix(".scores.pt"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["linear", "generate", "model", "serving"])
    parser.add_argument("--model", default=os.environ.get("FP8_MODEL", ""))
    parser.add_argument("--data", default=os.environ.get("FP8_DATA", ""))
    parser.add_argument("--tokens", default="fp8_tokens.json")
    parser.add_argument("--attention", default="kernels-community/flash-attn3")
    parser.add_argument("--completion", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--align", action="store_true", help="apply experimental Qwen3.5 BF16 boundary alignment")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result = {"mode": args.mode, "torch": torch.__version__,
              "device": torch.cuda.get_device_name(), "checkpoint": args.model,
              "settings": {"alignment": args.align, "attention": args.attention,
                           "repeats": args.repeats,
                           "linear_graphs": os.environ.get("RLFORGE_FP8_GRAPHS", "0"),
                           "head_graphs": os.environ.get("RLFORGE_FP8_HEAD_GRAPH", "0"),
                           "graph_max_mb": os.environ.get("RLFORGE_FP8_GRAPH_MAX_MB", "1024"),
                           "fused_mlp": os.environ.get("RLFORGE_FP8_FUSE_MLP", "1"),
                           "aligned_conv_compile": os.environ.get("RLFORGE_FP8_ALIGN_COMPILE", "0"),
                           "fused_ops": os.environ.get("RLFORGE_FUSED_OPS", "")},
              "tokens_sha256": hashlib.sha256(Path(args.tokens).read_bytes()).hexdigest()
              if Path(args.tokens).exists() else None}
    result.update({"linear": linear_gate, "generate": generate,
                   "model": model_gate, "serving": serving_gate}[args.mode](args))
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
