"""Is CPU KV offloading lossless on long sequences?

For each prompt length L, builds a long prompt from real train prompts and runs a greedy
continuation (bs=1, logprobs) three ways on servers that differ only in cache tier:
  cold   - server with prefix caching disabled: every token recomputed
  gpu    - second identical request on a cache-enabled server: prefix served from GPU blocks
  cpu    - same, but GPU cache flushed by filler requests first so the prefix must come back
           from the CPU offload tier (checked via the server's offload counters)
Reports first divergent token and logprob gaps vs `cold`. Offloading copies fp8 KV blocks
byte-for-byte, so `cpu` should match `gpu` exactly; any cpu-vs-gpu gap is a real defect,
while gpu-vs-cold gaps are ordinary chunking/kernel numerics.
"""
import argparse, json, re
import httpx


def build_prompt(prompts, n_chars):
    text, i = "", 0
    while len(text) < n_chars:
        text += prompts[i % len(prompts)][-1]["content"] + "\n\n"
        i += 1
    return [{"role": "user", "content": text[:n_chars] + "\n\nSummarize every question above, one by one."}]


def gen(client, base, msgs, max_tokens):
    r = client.post(f"{base}/v1/chat/completions", json={
        "model": "t", "messages": msgs, "max_tokens": max_tokens, "temperature": 0.0,
        "logprobs": True, "top_logprobs": 1, "ignore_eos": True}, timeout=None)
    r.raise_for_status()
    d = r.json()
    lp = d["choices"][0]["logprobs"]["content"]
    return d["usage"]["prompt_tokens"], [t["token"] for t in lp], [t["logprob"] for t in lp]


def metrics(client, base, pat):
    t = client.get(f"{base}/metrics").text
    return {m.group(0).split(" ")[0]: float(m.group(0).split(" ")[-1])
            for m in re.finditer(rf"^vllm:\S*(?:{pat})\S* [0-9.eE+-]+$", t, re.M)}


def compare(ref, x):
    rt, rl = ref
    xt, xl = x
    div = next((i for i, (a, b) in enumerate(zip(rt, xt)) if a != b), None)
    upto = div if div is not None else min(len(rt), len(xt))
    gaps = [abs(a - b) for a, b in zip(rl[:upto], xl[:upto])]
    return {"first_div": div, "n": len(rt), "max_lp_gap": max(gaps) if gaps else 0.0,
            "mean_lp_gap": sum(gaps) / len(gaps) if gaps else 0.0,
            "identical": div is None and max(gaps or [0]) == 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold", required=True)   # base URL, prefix caching off
    ap.add_argument("--off", required=True)    # base URL, prefix caching + CPU offload
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--chars", type=int, nargs="+", default=[16000, 48000, 96000])
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--flush-reqs", type=int, default=64)
    ap.add_argument("--flush-chars", type=int, default=60000)
    a = ap.parse_args()
    prompts = [json.loads(l)["prompt"] for l in open(a.prompts)]
    c = httpx.Client(timeout=None)
    for n in a.chars:
        msgs = build_prompt(prompts, n)
        L, *cold = gen(c, a.cold, msgs, a.max_tokens)
        gen(c, a.off, msgs, a.max_tokens)                      # populate cache
        _, *gpu = gen(c, a.off, msgs, a.max_tokens)            # GPU prefix hit
        m0 = metrics(c, a.off, "offload|kv_transfer|external|connector")
        # Flush: distinct long prompts sent concurrently-ish so the GPU blocks of `msgs` are
        # evicted (and, with offloading, pushed to CPU) before the next identical request.
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(32) as ex:
            list(ex.map(lambda k: gen(c, a.off, [{"role": "user", "content": f"[{k}] " +
                 build_prompt(prompts[k % len(prompts):] + prompts, a.flush_chars)[0]["content"]}], 1),
                 range(a.flush_reqs)))
        _, *cpu = gen(c, a.off, msgs, a.max_tokens)
        m1 = metrics(c, a.off, "offload|kv_transfer|external|connector")
        delta = {k: m1[k] - m0.get(k, 0) for k in m1 if m1[k] != m0.get(k, 0)}
        print("RESULT", json.dumps({"prompt_tokens": L, "gen": a.max_tokens,
              "gpu_vs_cold": compare(cold, gpu), "cpu_vs_cold": compare(cold, cpu),
              "cpu_vs_gpu": compare(gpu, cpu), "offload_metric_delta": delta}), flush=True)


main()
