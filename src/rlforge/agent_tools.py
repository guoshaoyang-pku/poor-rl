from __future__ import annotations

import ast
import json
import math
import operator
from dataclasses import dataclass
from typing import Any, Callable

from rlforge.knowledge import KnowledgeBase


@dataclass
class ToolContext:
    kb: KnowledgeBase
    family: str
    task_id: str
    split: str
    exclude_keys: tuple[str, ...]
    kb_limit: int
    kb_max_chars: int
    file_workspace: Any = None


def _search_kb(arguments: dict[str, Any], context: ToolContext) -> list[dict]:
    claims = context.kb.retrieve(
        str(arguments.get("query", "")),
        context.family,
        min(max(int(arguments.get("limit", context.kb_limit)), 1), context.kb_limit),
        exclude_task_id=context.task_id,
        exclude_keys=context.exclude_keys,
    )
    selected = []
    used_chars = 0
    for claim in claims:
        summary = {key: claim[key] for key in ("id", "text", "family", "kind", "score")}
        result_chars = len(json.dumps(summary, ensure_ascii=False))
        if used_chars + result_chars > context.kb_max_chars:
            continue
        selected.append(summary)
        used_chars += result_chars
    return selected


def _memory_list(_arguments: dict[str, Any], context: ToolContext) -> list[dict]:
    if context.file_workspace is None:
        return []
    return context.file_workspace.list_files()


def _memory_read(arguments: dict[str, Any], context: ToolContext) -> dict:
    if context.file_workspace is None:
        raise ValueError("file memory is not enabled")
    return context.file_workspace.read_file(str(arguments.get("name", "")))


def _memory_write(arguments: dict[str, Any], context: ToolContext) -> dict:
    if context.file_workspace is None:
        raise ValueError("file memory is not enabled")
    return context.file_workspace.write_file(
        str(arguments.get("name", "")),
        arguments.get("content"),
        arguments.get("expected_sha256"),
    )


def _memory_rename(arguments: dict[str, Any], context: ToolContext) -> dict:
    if context.file_workspace is None:
        raise ValueError("file memory is not enabled")
    return context.file_workspace.rename_file(
        str(arguments.get("old_name", "")),
        str(arguments.get("new_name", "")),
    )


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    invoke: Callable[[dict[str, Any], ToolContext], Any]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools = {tool.name: tool for tool in (tools or [])}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def register_mcp_bridge(self, bridge) -> None:
        for exposed_name, metadata in bridge.tool_map.items():
            server, original, description, schema = metadata
            self.register(
                Tool(
                    name=exposed_name,
                    description=f"{description} (MCP server: {server}; tool: {original}).",
                    parameters=schema,
                    invoke=lambda arguments, _context, tool_name=exposed_name: (
                        bridge.call(tool_name, arguments)
                    ),
                )
            )

    def register_file_memory_tools(self) -> None:
        specs = [
            Tool(
                name="memory_list_files",
                description="List Markdown files in this episode's isolated cross-task memory workspace.",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                invoke=_memory_list,
            ),
            Tool(
                name="memory_read_file",
                description="Read a Markdown skill or note by relative filename and get its sha256.",
                parameters={
                    "type": "object",
                    "properties": {"name": {"type": "string", "maxLength": 180}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
                invoke=_memory_read,
            ),
            Tool(
                name="memory_write_file",
                description="Create or edit a Markdown memory file. To edit, first read the file and pass its expected_sha256. Files are staged per episode; only successful supported training proposals are promoted.",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "maxLength": 180},
                        "content": {"type": "string", "maxLength": 12000},
                        "expected_sha256": {"type": "string", "maxLength": 64},
                    },
                    "required": ["name", "content"],
                    "additionalProperties": False,
                },
                invoke=_memory_write,
            ),
            Tool(
                name="memory_rename_file",
                description="Rename a Markdown memory file by its relative filename. The rename is staged until promotion.",
                parameters={
                    "type": "object",
                    "properties": {
                        "old_name": {"type": "string", "maxLength": 180},
                        "new_name": {"type": "string", "maxLength": 180},
                    },
                    "required": ["old_name", "new_name"],
                    "additionalProperties": False,
                },
                invoke=_memory_rename,
            ),
        ]
        for tool in specs:
            self.register(tool)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def call(self, name: str, arguments: dict[str, Any], context: ToolContext) -> Any:
        if name not in self._tools:
            raise ValueError(f"unknown tool: {name}")
        return self._tools[name].invoke(arguments, context)

    @classmethod
    def standard(cls) -> "ToolRegistry":
        registry = cls()
        registry.register(
            Tool(
                name="kb_search",
                description="Search reusable, non-evaluation knowledge relevant to a question.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                invoke=_search_kb,
            )
        )
        registry.register(
            Tool(
                name="calculate",
                description="Evaluate a basic arithmetic expression; no names, calls, or attributes.",
                parameters={
                    "type": "object",
                    "properties": {"expression": {"type": "string", "maxLength": 256}},
                    "required": ["expression"],
                    "additionalProperties": False,
                },
                invoke=lambda args, _ctx: safe_calculate(
                    str(args.get("expression", ""))
                ),
            )
        )
        return registry


def safe_calculate(expression: str) -> int | float:
    if not expression or len(expression) > 256:
        raise ValueError("expression must contain 1-256 characters")
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.FloorDiv: operator.floordiv,
    }

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ValueError("exponent magnitude must be at most 8")
            result = operators[type(node.op)](left, right)
            if abs(result) > 1e100:
                raise ValueError("result magnitude is too large")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.operand))
        raise ValueError(
            "only numeric literals and basic arithmetic operators are allowed"
        )

    result = evaluate(ast.parse(expression, mode="eval"))
    if isinstance(result, complex) or not math.isfinite(float(result)):
        raise ValueError("result must be a finite real number")
    return result
