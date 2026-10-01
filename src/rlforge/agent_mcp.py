from __future__ import annotations

import asyncio
import json
import threading
from contextlib import AsyncExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any


class MCPToolBridge:
    """Persistent MCP stdio clients exposed as bounded synchronous tool calls."""

    def __init__(self, config: str | Path | dict[str, Any], timeout: float = 30.0):
        loaded = (
            json.loads(Path(config).read_text(encoding="utf-8"))
            if isinstance(config, (str, Path))
            else config
        )
        if isinstance(loaded, list):
            merged = {}
            for item in loaded:
                if not isinstance(item, dict) or not isinstance(
                    item.get("mcpServers"), dict
                ):
                    raise ValueError(
                        "MCP config list entries need an mcpServers object"
                    )
                overlap = set(merged).intersection(item["mcpServers"])
                if overlap:
                    raise ValueError(f"duplicate MCP server names: {sorted(overlap)}")
                merged.update(item["mcpServers"])
            loaded = {"mcpServers": merged}
        if not isinstance(loaded, dict):
            raise ValueError("MCP config must be an object or Qwen-Agent config list")
        self.config = loaded
        servers = self.config.get("mcpServers")
        if not isinstance(servers, dict):
            raise ValueError("MCP config must contain an mcpServers object")
        self.servers = servers
        self.timeout = timeout
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.shutdown_event: asyncio.Event | None = None
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.stack: AsyncExitStack | None = None
        self.sessions: dict[str, Any] = {}
        self.tool_map: dict[str, tuple[str, str, str, dict[str, Any]]] = {}
        self.start_error: BaseException | None = None
        self.thread.start()
        if not self.ready.wait(timeout):
            self.close()
            raise TimeoutError("timed out starting MCP client loop")
        if self.start_error:
            self.close()
            raise RuntimeError(
                f"failed to start MCP servers: {self.start_error}"
            ) from self.start_error

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.shutdown_event = asyncio.Event()
        try:
            self.loop.run_until_complete(self._serve())
        except BaseException as exc:
            self.start_error = exc
            self.ready.set()
        finally:
            self.loop.close()

    async def _serve(self) -> None:
        try:
            await self._open()
            self.ready.set()
            await self.shutdown_event.wait()
        finally:
            if self.stack is not None:
                await self.stack.aclose()

    async def _open(self) -> None:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError(
                "MCP support requires `pip install rlforge[agent-mcp]`"
            ) from exc

        self.stack = AsyncExitStack()
        await self.stack.__aenter__()
        for server_name, raw_config in self.servers.items():
            if not isinstance(raw_config, dict) or not raw_config.get("command"):
                raise ValueError(f"MCP server {server_name!r} needs a command")
            params = StdioServerParameters(
                command=str(raw_config["command"]),
                args=[str(arg) for arg in raw_config.get("args", [])],
                env={
                    str(key): str(value)
                    for key, value in raw_config.get("env", {}).items()
                }
                or None,
                cwd=raw_config.get("cwd"),
            )
            read_stream, write_stream = await self.stack.enter_async_context(
                stdio_client(params)
            )
            session = await self.stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self.timeout),
                )
            )
            await session.initialize()
            self.sessions[server_name] = session
            response = await session.list_tools()
            for tool in response.tools:
                exposed_name = self._exposed_name(server_name, tool.name)
                if exposed_name in self.tool_map:
                    raise ValueError(f"duplicate MCP tool name: {exposed_name}")
                self.tool_map[exposed_name] = (
                    server_name,
                    tool.name,
                    tool.description or "MCP tool",
                    tool.inputSchema or {"type": "object", "properties": {}},
                )

    @staticmethod
    def _exposed_name(server: str, tool: str) -> str:
        def clean(value: str) -> str:
            return "".join(
                char if char.isalnum() or char == "_" else "_" for char in value
            )

        return f"mcp_{clean(server)}_{clean(tool)}"

    def schemas(self) -> list[dict[str, Any]]:
        result = []
        for name, (server, original, description, schema) in self.tool_map.items():
            result.append(
                {
                    "name": name,
                    "description": f"{description} (MCP server: {server}; tool: {original}).",
                    "parameters": schema,
                }
            )
        return result

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self.tool_map:
            raise ValueError(f"unknown MCP tool: {name}")
        server, original, _description, _schema = self.tool_map[name]
        future = asyncio.run_coroutine_threadsafe(
            self._call(server, original, arguments), self.loop
        )
        result = future.result(timeout=self.timeout)
        if result.isError:
            detail = "\n".join(
                item.text for item in result.content if getattr(item, "text", None)
            )
            raise RuntimeError(detail[:1000] or "MCP tool returned an error")
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return structured
        return [
            {"type": "text", "text": item.text}
            if getattr(item, "text", None) is not None
            else {"type": item.type}
            for item in result.content
        ]

    async def _call(self, server: str, tool: str, arguments: dict[str, Any]) -> Any:
        return await self.sessions[server].call_tool(tool, arguments)

    def close(self) -> None:
        if self.thread.is_alive() and self.loop.is_running() and self.shutdown_event:
            self.loop.call_soon_threadsafe(self.shutdown_event.set)
            self.thread.join(timeout=self.timeout)
        elif self.thread.is_alive():
            self.thread.join(timeout=self.timeout)
        elif not self.loop.is_closed():
            self.loop.close()

    def __enter__(self) -> "MCPToolBridge":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
