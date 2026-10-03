"""Group-affine routing for a data-parallel vLLM rollout server.

vLLM's internal DP load balancer (DPLBAsyncMPClient.get_core_engine_for_request) picks the
least-loaded engine per request with a rotating tie-break; it does not look at prefixes. TRL's
async rollout worker sends the G samples of a group as G separate n=1 /v1/completions
requests, so a burst of one group's requests is spread round-robin over the replicas and the
shared prompt is prefilled on (up to) every replica instead of once.

vLLM honours an `X-data-parallel-rank` request header (entrypoints/generate/base/serving.py
`_get_data_parallel_rank`), which bypasses the balancer. This module pins every in-flight
request that shares a prompt to one replica, chosen least-loaded when the prompt first
appears, and released when its last in-flight request finishes. Balance is therefore kept at
group granularity, with the client's own exact in-flight counts as the load signal.

Opt-in: `install()` swaps `AsyncRolloutWorker._loop_cls`; the class is pickled by reference
into the spawned rollout child, so the child imports this module and uses the subclass.
With DP == 1 (or the server not reporting DP) requests are sent exactly as before.
"""
import hashlib
import logging
import os

import aiohttp
import requests
from trl.experimental.async_grpo import async_rollout_worker as _arw

logger = logging.getLogger(__name__)


class GroupAffineRouter:
    """Sticky prompt -> DP rank map with least-loaded placement of new prompts."""

    def __init__(self, dp_size: int):
        self.dp_size = dp_size
        self.load = [0] * dp_size              # in-flight requests per rank (this client)
        self._pin: dict[bytes, list[int]] = {}  # key -> [rank, refcount]

    @staticmethod
    def key(prompt_ids) -> bytes:
        return hashlib.blake2b(repr(list(prompt_ids)).encode(), digest_size=16).digest()

    def acquire(self, key: bytes) -> int:
        entry = self._pin.get(key)
        if entry is None:
            rank = min(range(self.dp_size), key=lambda r: self.load[r])
            entry = self._pin[key] = [rank, 0]
        entry[1] += 1
        self.load[entry[0]] += 1
        return entry[0]

    def release(self, key: bytes) -> None:
        entry = self._pin[key]
        entry[1] -= 1
        self.load[entry[0]] -= 1
        if entry[1] == 0:
            del self._pin[key]


def server_dp_size(server_url: str) -> int:
    """DP size from /get_world_size (dev-mode endpoint; TRL's weight sync already needs it)."""
    if os.environ.get("RLFORGE_DP_SIZE"):
        return int(os.environ["RLFORGE_DP_SIZE"])
    with_dp = requests.get(f"{server_url}/get_world_size", params={"include_dp": "true"}, timeout=30)
    without = requests.get(f"{server_url}/get_world_size", params={"include_dp": "false"}, timeout=30)
    return max(1, with_dp.json()["world_size"] // without.json()["world_size"])


class GroupAffineRolloutLoop(_arw._AsyncRolloutLoop):
    _router: GroupAffineRouter | None = None

    def _get_router(self) -> GroupAffineRouter | None:
        if self._router is None:
            dp = server_dp_size(self.vllm_server_url)
            self._router = GroupAffineRouter(dp) if dp > 1 else False
            logger.warning(f"[dp_route] server DP size {dp}: "
                           + ("pinning each prompt to one replica" if dp > 1 else "routing off"))
        return self._router or None

    async def _generate_one_turn(self, prompt_ids: list[int]) -> tuple[list[int], list[float]]:
        router = self._get_router()
        if router is None:
            return await super()._generate_one_turn(prompt_ids)
        payload = {
            "model": self._request_model,
            "prompt": prompt_ids,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "n": 1,
            "return_token_ids": True,
            "logprobs": 0,
        }
        if self.min_p is not None:
            payload["min_p"] = self.min_p
        key = router.key(prompt_ids)
        rank = router.acquire(key)
        try:
            output = await self._retry(
                lambda: self._post_routed("/v1/completions", payload, self.request_timeout, rank),
                max_attempts=30,
                label="vllm /v1/completions",
            )
        finally:
            router.release(key)
        choice = output["choices"][0]
        return choice["token_ids"], choice["logprobs"]["token_logprobs"]

    async def _post_routed(self, path: str, payload: dict, timeout: float, rank: int,
                           max_retries: int = 3) -> dict:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        headers = {"X-data-parallel-rank": str(rank)}

        async def _do_post():
            async with self.session.post(f"{self.vllm_server_url}{path}", json=payload,
                                         headers=headers, timeout=client_timeout) as response:
                response.raise_for_status()
                content = await response.json()
                return content if content else {}

        return await self._retry(_do_post, label=f"POST {path}", max_attempts=max_retries)


def install() -> None:
    _arw.AsyncRolloutWorker._loop_cls = GroupAffineRolloutLoop
    logger.warning("[dp_route] AsyncRolloutWorker now uses GroupAffineRolloutLoop")
