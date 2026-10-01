from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

_REACT_ACTION_RE = re.compile(
    r"(?:^|\n)Action:\s*([A-Za-z0-9_.-]+)\s*\nAction Input:\s*(.*?)\s*(?:\nObservation:|$)",
    re.DOTALL,
)


def _normalize_react_message(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, str):
        return message
    match = _REACT_ACTION_RE.search(content)
    if match:
        raw_arguments = match.group(2).strip()
        if raw_arguments.startswith("```") and raw_arguments.endswith("```"):
            raw_arguments = raw_arguments[3:-3].strip()
            if raw_arguments.startswith("json"):
                raw_arguments = raw_arguments[4:].strip()
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return message
        if not isinstance(arguments, dict):
            return message
        normalized = dict(message)
        normalized["react_action"] = {
            "name": match.group(1),
            "arguments": arguments,
        }
        return normalized
    final = re.search(r"(?:^|\n)Final Answer:\s*(.*)$", content, re.DOTALL)
    if final:
        normalized = dict(message)
        normalized["content"] = final.group(1).strip()
        return normalized
    return message


class OpenAIChatPolicy:
    """Small OpenAI-compatible chat client with injectable transport for tests."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.7,
        transport=None,
        protocol: str = "openai-tools",
        extra_body: dict[str, Any] | None = None,
        top_p: float = 1.0,
        top_k: int | None = None,
        presence_penalty: float = 0.0,
    ):
        if protocol not in {"react", "openai-tools"}:
            raise ValueError(f"unsupported agent protocol: {protocol}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("RLFORGE_API_KEY")
        self.timeout = timeout
        self.temperature = temperature
        self.transport = transport or self._request
        self.protocol = protocol
        self.extra_body = extra_body or {}
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:1000]
            raise RuntimeError(f"chat API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"chat API request failed: {exc.reason}") from exc

    def complete(
        self, messages: list[dict], tools: list[dict], max_tokens: int
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            **self.extra_body,
        }
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if tools and self.protocol == "openai-tools":
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        response = self.transport(payload)
        choices = response.get("choices") or []
        if not choices or "message" not in choices[0]:
            raise RuntimeError("chat API response is missing choices[0].message")
        message = choices[0]["message"]
        if self.protocol == "react":
            message = _normalize_react_message(message)
        usage = response.get("usage") or {}
        try:
            token_count = int(usage.get("completion_tokens", 0))
        except (TypeError, ValueError):
            token_count = 0
        return {
            "message": message,
            "completion_tokens": max(0, token_count),
            "finish_reason": choices[0].get("finish_reason"),
        }
