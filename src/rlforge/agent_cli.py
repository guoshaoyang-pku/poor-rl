from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time
from pathlib import Path

from rlforge.agent_policy import OpenAIChatPolicy
from rlforge.agent_tools import ToolRegistry
from rlforge.agent_mcp import MCPToolBridge
from rlforge.agent_file_memory import SkillFileBank
from rlforge.agentic import AgentHarness, run_jsonl
from rlforge.knowledge import KnowledgeBase
from rlforge.rewards import load_reward_fn


def _import_reward(spec: str):
    if ":" in spec:
        return load_reward_fn(spec)
    module_name, function_name = spec.rsplit(".", 1)
    fn = getattr(importlib.import_module(module_name), function_name)
    if not callable(fn):
        raise ValueError(f"reward is not callable: {spec}")
    return fn


def main():
    parser = argparse.ArgumentParser(
        description="Run bounded tool-using RL episodes with cross-task KB memory"
    )
    parser.add_argument(
        "--data", required=True, help="input JSONL with question_id,prompt,answer"
    )
    parser.add_argument("--out", default="runs/agentic/episodes.jsonl")
    parser.add_argument("--kb", default="runs/agentic/knowledge.sqlite3")
    parser.add_argument(
        "--memory-files",
        default=None,
        help="optional persistent Markdown SkillBank directory; edits are staged and promoted only from supported successful train episodes",
    )
    parser.add_argument(
        "--memory-evidence-db",
        default=None,
        help="optional SQLite evidence ledger for SkillBank promotions",
    )
    parser.add_argument(
        "--import-kb", default=None, help="optional historical KB JSON to seed"
    )
    parser.add_argument(
        "--split", choices=["train", "eval", "test", "predict"], default="train"
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("RLFORGE_API_BASE_URL", "http://localhost:8000/v1"),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="local tokenizer path for an exact chat-template context count",
    )
    parser.add_argument("--api-key-env", default="RLFORGE_API_KEY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=2.0)
    parser.add_argument(
        "--protocol", choices=["react", "openai-tools"], default="react"
    )
    parser.add_argument(
        "--mcp-config", default=None, help="Qwen-Agent-style MCP server config JSON"
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="enable model thinking; disabled by default for Qwen3.5-0.8B",
    )
    parser.add_argument("--reward", default="rlforge.rewards.mcq:mcq_reward")
    parser.add_argument("--max-tool-calls", type=int, default=4)
    parser.add_argument("--max-tool-result-chars", type=int, default=4000)
    parser.add_argument(
        "--num-generations",
        type=int,
        default=1,
        help="rollouts per task; emits group-relative terminal advantages",
    )
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)
    parser.add_argument("--max-context-tokens", type=int, default=24576)
    parser.add_argument("--max-kb-entries", type=int, default=8)
    parser.add_argument("--max-kb-chars", type=int, default=6000)
    parser.add_argument("--learn-threshold", type=float, default=1.0)
    parser.add_argument("--min-skill-support", type=int, default=2)
    args = parser.parse_args()

    output_path = Path(args.out)
    if output_path.exists():
        raise FileExistsError(
            f"refusing to append into an existing run log: {output_path}"
        )
    input_hash = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    run_manifest = {
        "status": "running",
        "started_at": time.time(),
        "data_path": str(Path(args.data).resolve()),
        "data_sha256": input_hash,
        "kb_path": str(Path(args.kb).resolve()),
        "memory_files_path": str(Path(args.memory_files).resolve())
        if args.memory_files
        else None,
        "memory_evidence_db_path": str(Path(args.memory_evidence_db).resolve())
        if args.memory_evidence_db
        else None,
        "model": args.model,
        "protocol": args.protocol,
        "mcp_config_path": str(Path(args.mcp_config).resolve())
        if args.mcp_config
        else None,
        "mcp_config_sha256": hashlib.sha256(
            Path(args.mcp_config).read_bytes()
        ).hexdigest()
        if args.mcp_config
        else None,
        "enable_thinking": args.enable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "tokenizer": args.tokenizer,
        "base_url": args.base_url,
        "split": args.split,
        "num_generations": args.num_generations,
        "max_tool_calls": args.max_tool_calls,
        "max_tool_result_chars": args.max_tool_result_chars,
        "max_rounds": args.max_rounds,
        "max_completion_tokens": args.max_completion_tokens,
        "max_context_tokens": args.max_context_tokens,
        "max_kb_entries": args.max_kb_entries,
        "max_kb_chars": args.max_kb_chars,
        "reward": args.reward,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing run manifest: {manifest_path}"
        )
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    key = os.environ.get(args.api_key_env)
    policy = OpenAIChatPolicy(
        args.base_url,
        args.model,
        api_key=key,
        timeout=args.timeout,
        temperature=args.temperature,
        protocol=args.protocol,
        extra_body=(
            {"chat_template_kwargs": {"enable_thinking": True}}
            if args.enable_thinking
            else {}
        ),
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
    )
    token_counter = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True
        )

        def token_counter(messages, tools):
            return len(
                tokenizer.apply_chat_template(
                    messages, tools=tools, tokenize=True, add_generation_prompt=True
                )
            )

    kb = KnowledgeBase(args.kb)
    mcp_bridge = None
    skill_file_bank = None
    try:
        tools = ToolRegistry.standard()
        if args.memory_files:
            skill_file_bank = SkillFileBank(
                args.memory_files, evidence_path=args.memory_evidence_db
            )
            tools.register_file_memory_tools()
            run_manifest["memory_snapshot_initial"] = skill_file_bank.snapshot()
        if args.mcp_config:
            mcp_bridge = MCPToolBridge(args.mcp_config, timeout=args.timeout)
            tools.register_mcp_bridge(mcp_bridge)
        run_manifest["mcp_tools"] = sorted(mcp_bridge.tool_map) if mcp_bridge else []
        seed_imported = 0
        seed_kb_hash = None
        if args.import_kb:
            seed_path = Path(args.import_kb)
            seed_kb_hash = hashlib.sha256(seed_path.read_bytes()).hexdigest()
            seed_imported = kb.import_json(seed_path)
            print(f"[agentic] imported {seed_imported} seed claims")
        run_manifest.update(
            {
                "seed_kb_path": str(Path(args.import_kb).resolve())
                if args.import_kb
                else None,
                "seed_kb_sha256": seed_kb_hash,
                "seed_claims_imported": seed_imported,
                "kb_snapshot_initial": kb.snapshot(),
            }
        )
        manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        harness = AgentHarness(
            policy=policy,
            kb=kb,
            tools=tools,
            reward_fn=_import_reward(args.reward),
            max_tool_calls=args.max_tool_calls,
            max_rounds=args.max_rounds,
            max_tool_result_chars=args.max_tool_result_chars,
            max_completion_tokens=args.max_completion_tokens,
            max_context_tokens=args.max_context_tokens,
            max_kb_entries=args.max_kb_entries,
            max_kb_chars=args.max_kb_chars,
            learn_threshold=args.learn_threshold,
            min_skill_support=args.min_skill_support,
            skill_file_bank=skill_file_bank,
            token_counter=token_counter,
        )
        results = run_jsonl(
            harness,
            args.data,
            args.out,
            split=args.split,
            generations=args.num_generations,
        )
        mean_reward = (
            sum(result.reward for result in results) / len(results) if results else None
        )
        run_manifest.update(
            {
                "status": "complete",
                "ended_at": time.time(),
                "episodes": len(results),
                "mean_reward": mean_reward,
                "kb_claims": len(kb),
                "kb_snapshot": kb.snapshot(),
                "memory_files": sum(
                    1 for path in Path(args.memory_files).rglob("*.md") if path.is_file()
                )
                if args.memory_files
                else 0,
                "memory_snapshot": skill_file_bank.snapshot()
                if skill_file_bank
                else None,
            }
        )
        manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "episodes": len(results),
                    "mean_reward": mean_reward,
                    "kb_claims": len(kb),
                    "kb_snapshot": kb.snapshot(),
                    "memory_snapshot": skill_file_bank.snapshot()
                    if skill_file_bank
                    else None,
                    "output": str(Path(args.out)),
                },
                ensure_ascii=False,
            )
        )
    except Exception as exc:
        run_manifest.update(
            {
                "status": "failed",
                "ended_at": time.time(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        raise
    finally:
        if mcp_bridge:
            mcp_bridge.close()
        if skill_file_bank:
            skill_file_bank.close()
        kb.close()


if __name__ == "__main__":
    main()
