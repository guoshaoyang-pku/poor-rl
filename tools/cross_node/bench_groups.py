"""Group-burst rollout benchmark: does DP routing keep a group's shared prompt on one replica?

Mimics TRL's async rollout worker: each group = G concurrent n=1 /v1/completions requests
with the same token-id prompt (return_token_ids, logprobs=0, T=1, ignore_eos). Keeps K groups
in flight (closed loop). `--route affine` adds the X-data-parallel-rank header via
dp_route.GroupAffineRouter; `--route none` leaves placement to vLLM's internal balancer.
Throughput and prefix-cache hit rate come from the server's /metrics over a steady window.
"""
import argparse, asyncio, json, re, sys, time, os
import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dp_route import GroupAffineRouter  # noqa: E402

NAMES = ["vllm:generation_tokens_total", "vllm:prompt_tokens_total",
         "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
         "vllm:num_requests_running"]


def scrape(text):
    out = {}
    for n in NAMES:
        out[n] = sum(float(m.group(1)) for m in re.finditer(
            rf"^{re.escape(n)}(?:\{{[^}}]*\}})? ([0-9.eE+-]+)$", text, re.M))
    per_engine = {}
    for m in re.finditer(r'^vllm:generation_tokens_total\{([^}]*)\} ([0-9.eE+-]+)$', text, re.M):
        eng = re.search(r'engine="(\d+)"', m.group(1))
        per_engine[eng.group(1) if eng else "?"] = float(m.group(2))
    out["per_engine_gen"] = per_engine
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--groups-inflight", type=int, default=16)
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--route", choices=["none", "affine"], default="none")
    ap.add_argument("--dp", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=60)
    ap.add_argument("--window", type=float, default=90)
    ap.add_argument("--lead", action="store_true",
                    help="prefill the group's prompt with one max_tokens=1 request before the burst, "
                         "so the G samples hit a warm prefix instead of prefilling concurrently")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    prompts = [tok(tok.apply_chat_template(json.loads(l)["prompt"], tokenize=False,
                                           add_generation_prompt=True),
                   add_special_tokens=False)["input_ids"] for l in open(a.prompts)]
    router = GroupAffineRouter(a.dp) if a.route == "affine" else None
    conn = aiohttp.TCPConnector(limit=0)
    stop = False
    counter = [0]
    async with aiohttp.ClientSession(connector=conn) as s:
        model = (await (await s.get(f"{a.url}/v1/models")).json())["data"][0]["id"]

        async def one(p, max_tokens=None):
            body = {"model": model, "prompt": p, "max_tokens": max_tokens or a.max_tokens, "temperature": 1.0,
                    "top_p": 1.0, "n": 1, "return_token_ids": True, "logprobs": 0,
                    "ignore_eos": True}
            headers, key = {}, None
            if router:
                key = router.key(p)
                headers["X-data-parallel-rank"] = str(router.acquire(key))
            try:
                async with s.post(f"{a.url}/v1/completions", json=body, headers=headers,
                                  timeout=aiohttp.ClientTimeout(total=3600)) as r:
                    await r.read()
            finally:
                if router:
                    router.release(key)

        async def group_loop():
            while not stop:
                p = prompts[counter[0] % len(prompts)]
                counter[0] += 1
                if a.lead:
                    await one(p, max_tokens=1)
                await asyncio.gather(*[one(p) for _ in range(a.group)], return_exceptions=True)

        async def metrics():
            return scrape(await (await s.get(f"{a.url}/metrics")).text())

        tasks = [asyncio.create_task(group_loop()) for _ in range(a.groups_inflight)]
        await asyncio.sleep(a.warmup)
        m0, t0 = await metrics(), time.time()
        await asyncio.sleep(a.window)
        m1, t1 = await metrics(), time.time()
        stop = True
        for t in tasks:
            t.cancel()
        dt = t1 - t0
        d = lambda k: m1[k] - m0[k]
        q = d("vllm:prefix_cache_queries_total")
        print("RESULT", json.dumps({
            "route": a.route, "lead": a.lead, "groups_inflight": a.groups_inflight, "group": a.group,
            "max_tokens": a.max_tokens,
            "gen_tok_s": round(d("vllm:generation_tokens_total") / dt, 1),
            "prompt_tok_s": round(d("vllm:prompt_tokens_total") / dt, 1),
            "prefix_hit": round(d("vllm:prefix_cache_hits_total") / q, 4) if q else None,
            "computed_prefill_tok_s": round((q - d("vllm:prefix_cache_hits_total")) / dt, 1),
            "per_engine_gen_tok_s": {k: round((m1["per_engine_gen"].get(k, 0) - v) / dt, 1)
                                     for k, v in m0["per_engine_gen"].items()},
        }), flush=True)


asyncio.run(main())
