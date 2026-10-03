"""Closed-loop saturation benchmark for a vLLM rollout server.

Keeps exactly C requests in flight (re-issuing on completion), shares each prompt across a
GRPO group of G requests, samples at T=1 with ignore_eos so every completion runs to the cap
(matching the 96-100% truncation regime of production RL), and reads the server's own
counters (/metrics) over a steady-state window instead of timing a single gather(), which
mixes ramp-up and tail into the number.

NOTE: httpx.AsyncClient defaults to max_connections=100. The older /tmp/bench_serve.py used
that default, so its "256 / 768 concurrency" runs had at most 100 requests in flight.
"""
import argparse, asyncio, json, random, re, time
import httpx

METRICS = ["vllm:generation_tokens_total", "vllm:prompt_tokens_total",
           "vllm:num_requests_running", "vllm:num_requests_waiting",
           "vllm:kv_cache_usage_perc", "vllm:num_preemptions_total",
           "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"]


def parse_metrics(text):
    out = {}
    for name in METRICS:
        vals = [float(m.group(1)) for m in re.finditer(
            rf"^{re.escape(name)}(?:\{{[^}}]*\}})? ([0-9.eE+-]+)$", text, re.M)]
        if vals:
            out[name] = sum(vals)
    return out


async def scrape(client, base):
    r = await client.get(f"{base}/metrics", timeout=30)
    return parse_metrics(r.text)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--conc", type=int, nargs="+", required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--warmup", type=float, default=90)
    ap.add_argument("--window", type=float, default=120)
    ap.add_argument("--tag", default="")
    ap.add_argument("--prod", default="",
                    help="tokenizer path: send token-id prompts to /v1/completions with "
                         "return_token_ids + logprobs=0, exactly as TRL's async rollout worker does")
    a = ap.parse_args()

    prompts = [json.loads(l)["prompt"] for l in open(a.prompts)]
    if a.prod:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.prod)
        prompts = [tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True),
                       add_special_tokens=False)["input_ids"] for m in prompts]
    base = f"http://localhost:{a.port}"
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(limits=limits, timeout=None) as client:
        model = (await client.get(f"{base}/v1/models")).json()["data"][0]["id"]
        for C in a.conc:
            stop = False
            counter = [0]

            def next_prompt():
                i = counter[0] // a.group
                counter[0] += 1
                return prompts[i % len(prompts)]

            async def worker(first):
                # First request gets a random cap so in-flight sequences are spread across
                # context lengths, as in a long-running async rollout, not all in lockstep.
                mt = random.randint(1, a.max_tokens) if first else a.max_tokens
                while not stop:
                    body = {"model": model, "max_tokens": mt,
                            "temperature": 1.0, "top_p": 1.0, "ignore_eos": True}
                    if a.prod:
                        body.update(prompt=next_prompt(), n=1, return_token_ids=True, logprobs=0)
                        path = "/v1/completions"
                    else:
                        body["messages"] = next_prompt()
                        path = "/v1/chat/completions"
                    try:
                        await client.post(f"{base}{path}", json=body)
                    except Exception as e:  # keep the loop alive; report at the end
                        print("ERR", type(e).__name__, str(e)[:120], flush=True)
                        await asyncio.sleep(1)
                    mt = a.max_tokens
                    first = False

            tasks = [asyncio.create_task(worker(True)) for _ in range(C)]
            await asyncio.sleep(a.warmup)
            m0, t0 = await scrape(client, base), time.time()
            samples = []
            while time.time() - t0 < a.window:
                await asyncio.sleep(10)
                samples.append(await scrape(client, base))
            m1, t1 = await scrape(client, base), time.time()
            stop = True
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            dt = t1 - t0
            g = lambda k: m1.get(k, 0) - m0.get(k, 0)
            avg = lambda k: sum(s.get(k, 0) for s in samples) / max(len(samples), 1)
            q = g("vllm:prefix_cache_queries_total")
            res = {"tag": a.tag, "conc": C, "max_tokens": a.max_tokens,
                   "gen_tok_s": round(g("vllm:generation_tokens_total") / dt, 1),
                   "prompt_tok_s": round(g("vllm:prompt_tokens_total") / dt, 1),
                   "running": round(avg("vllm:num_requests_running"), 1),
                   "waiting": round(avg("vllm:num_requests_waiting"), 1),
                   "kv_usage": round(avg("vllm:kv_cache_usage_perc"), 3),
                   "preempt": g("vllm:num_preemptions_total"),
                   "prefix_hit": round(g("vllm:prefix_cache_hits_total") / q, 3) if q else None,
                   "window_s": round(dt, 1)}
            print("RESULT", json.dumps(res), flush=True)
            await asyncio.sleep(20)  # let aborted requests drain before the next level


asyncio.run(main())
