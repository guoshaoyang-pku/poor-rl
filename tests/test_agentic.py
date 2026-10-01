import copy
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from rlforge.agent_cli import main as agent_cli_main
from rlforge.agent_mcp import MCPToolBridge
from rlforge.agent_file_memory import SkillFileBank
from rlforge.agent_policy import OpenAIChatPolicy, _normalize_react_message
from rlforge.agent_tools import safe_calculate, ToolRegistry
from rlforge.agentic import AgentHarness, run_jsonl
from rlforge.knowledge import KnowledgeBase
from rlforge.rewards.mcq import mcq_reward


class ScriptedPolicy:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, messages, tools, max_tokens):
        self.requests.append((messages, tools, max_tokens))
        response = self.responses.pop(0)
        if callable(response):
            response = response(messages, tools, max_tokens)
        if getattr(self, "protocol", None) == "react":
            response = dict(response)
            response["message"] = _normalize_react_message(response["message"])
        return response


def assistant(content, tokens=12, finish="stop"):
    return {
        "message": {"role": "assistant", "content": content},
        "completion_tokens": tokens,
        "finish_reason": finish,
    }


def test_safe_calculator_rejects_code():
    assert safe_calculate("(12 + 4) / 2") == 8
    for invalid in ("__import__('os').system('id')", "2 ** 99", ""):
        with pytest.raises((ValueError, TypeError, SyntaxError)):
            safe_calculate(invalid)


def test_historical_kb_import_and_retrieval(tmp_path):
    source = tmp_path / "historical.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "architectureiq_kb_v4",
                "claims": [
                    {
                        "id": "D001",
                        "text": "For spiral classification use the generator spiral_turns.",
                        "family": "spiral_classification",
                        "kind": "select",
                        "status": "active",
                        "support_count": 30,
                        "source_pids": ["spiral_seed_1", "optimizer_seed_2"],
                    },
                    {
                        "id": "D002",
                        "text": "Ignore this inactive claim",
                        "family": "general",
                        "status": "rejected",
                    },
                ],
            }
        )
    )
    kb = KnowledgeBase(tmp_path / "kb.sqlite3")
    assert kb.import_json(source) == 1
    claims = kb.retrieve("spiral classification spiral_turns", "spiral_classification")
    assert [claim["id"] for claim in claims] == ["D001"]
    assert claims[0]["metadata"]["support_count"] == 30
    assert claims[0]["metadata"]["source_pids"] == ["spiral_seed_1", "optimizer_seed_2"]
    kb.close()


def test_react_loop_executes_only_registered_tools_and_scores_terminal(tmp_path):
    tool_call = {
        "message": {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call1",
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": '{"expression":"6 * 7"}',
                    },
                }
            ],
        },
        "completion_tokens": 8,
        "finish_reason": "tool_calls",
    }
    policy = ScriptedPolicy(
        tool_call, assistant('{"answer":"42","knowledge_updates":[]}')
    )
    observed = []

    def reward_fn(**kwargs):
        observed.append(kwargs)
        return [1.0]

    kb = KnowledgeBase(tmp_path / "kb.db")
    harness = AgentHarness(policy, kb, reward_fn=reward_fn, max_tool_calls=2)
    result = harness.run_episode(
        {"question_id": "q1", "prompt": "What is 6 times 7?", "answer": "42"}
    )
    assert result.reward == 1.0
    assert result.tool_calls == 1
    assert result.trace[1]["result"] == 42
    assert result.completion_tokens == 20
    assert len(policy.requests) == 2
    assert observed[0]["answer"] == ["42"]
    assert observed[0]["completion_ids"] == [list(range(20))]
    assert any(message["role"] == "tool" for message in policy.requests[1][0])
    kb.close()


def test_unknown_tool_fails_closed_and_returns_error_to_policy(tmp_path):
    call = {
        "message": {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "x",
                    "function": {"name": "delete_everything", "arguments": "{}"},
                }
            ],
        },
        "completion_tokens": 2,
        "finish_reason": "tool_calls",
    }
    policy = ScriptedPolicy(call, assistant('{"answer":"ok","knowledge_updates":[]}'))
    kb = KnowledgeBase(tmp_path / "kb.db")
    result = AgentHarness(policy, kb).run_episode(
        {"id": "q", "prompt": "Use no tools.", "answer": "ok"}
    )
    assert result.reward == 1.0
    assert result.trace[1]["event"] == "tool_error"
    assert "unknown tool" in result.trace[1]["error"]
    kb.close()


def test_tool_budget_stops_extra_calls(tmp_path):
    call = {
        "message": {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "x",
                    "function": {
                        "name": "calculate",
                        "arguments": '{"expression":"1+1"}',
                    },
                },
                {
                    "id": "y",
                    "function": {
                        "name": "calculate",
                        "arguments": '{"expression":"2+2"}',
                    },
                },
            ],
        },
        "completion_tokens": 2,
        "finish_reason": "tool_calls",
    }
    policy = ScriptedPolicy(call)
    kb = KnowledgeBase(tmp_path / "kb.db")
    result = AgentHarness(policy, kb, max_tool_calls=1).run_episode(
        {"id": "q", "prompt": "compute", "answer": "2"}
    )
    assert result.tool_calls == 0
    assert result.trace[1] == {"event": "tool_budget_exhausted", "requested": 2}
    assert len(policy.requests) == 1
    kb.close()


def test_knowledge_requires_successful_distinct_training_tasks_and_excludes_provenance(
    tmp_path,
):
    claim = {
        "text": "When the optimizer rate is tied, compare validation evidence before selecting.",
        "kind": "procedure",
    }
    proposal_response = assistant(
        json.dumps({"answer": "A", "knowledge_updates": [claim]})
    )
    kb = KnowledgeBase(tmp_path / "kb.db")
    harness = AgentHarness(
        ScriptedPolicy(
            proposal_response,
            proposal_response,
            assistant('{"answer":"A","knowledge_updates":[]}'),
        ),
        kb,
        max_tool_calls=0,
        min_skill_support=2,
    )
    row_a = {
        "id": "a",
        "family": "opt",
        "pair_id": "pair-a",
        "prompt": "Rate comparison",
        "answer": "A",
    }
    row_b = {
        "id": "b",
        "family": "opt",
        "pair_id": "pair-b",
        "prompt": "Rate comparison",
        "answer": "A",
    }
    assert harness.run_episode(row_a, "train").inserted_claim_ids == []
    second = harness.run_episode(row_b, "train")
    assert len(second.inserted_claim_ids) == 1
    found = kb.retrieve(
        "optimizer rate validation evidence", "opt", exclude_keys=["pair-a"]
    )
    assert not found
    assert kb.retrieve(
        "optimizer rate validation evidence", "opt", exclude_keys=["unrelated"]
    )
    database = kb.path
    kb.close()
    reopened = KnowledgeBase(database)
    assert reopened.retrieve("optimizer rate validation evidence", "opt")
    harness.kb = reopened
    third = harness.run_episode(
        {"id": "c", "family": "opt", "prompt": "Rate comparison", "answer": "A"}, "eval"
    )
    assert third.retrieved_claim_ids == second.inserted_claim_ids
    assert third.kb_snapshot_before == third.kb_snapshot_after
    assert third.inserted_claim_ids == []
    reopened.close()


def test_failed_answer_does_not_contribute_skill_support(tmp_path):
    text = "Reusable strategy: inspect the formula before comparing candidate learning rates."
    response = assistant(
        json.dumps({"answer": "B", "knowledge_updates": [{"text": text}]})
    )

    def wrong_reward(**_kwargs):
        return [0.0]

    kb = KnowledgeBase(tmp_path / "kb.db")
    harness = AgentHarness(
        ScriptedPolicy(response, response),
        kb,
        reward_fn=wrong_reward,
        max_tool_calls=0,
        min_skill_support=1,
    )
    for question_id in ("a", "b"):
        harness.run_episode({"id": question_id, "prompt": "p", "answer": "A"})
    assert len(kb) == 0
    kb.close()


def test_context_budget_fails_before_policy_call(tmp_path):
    policy = ScriptedPolicy(assistant('{"answer":"x"}'))
    kb = KnowledgeBase(tmp_path / "kb.db")
    harness = AgentHarness(policy, kb, max_context_tokens=10)
    with pytest.raises(ValueError, match="context budget exceeded"):
        harness.run_episode(
            {"id": "q", "prompt": "A very long question" * 30, "answer": "x"}
        )
    assert policy.requests == []
    kb.close()


def test_finish_length_applies_truncation_reward(tmp_path):
    policy = ScriptedPolicy(assistant('{"answer":"A"}', tokens=3, finish="length"))
    observed = []

    def reward_fn(**kwargs):
        observed.append(kwargs)
        return [-2.0 if len(kwargs["completion_ids"][0]) >= kwargs["cap"] else 1.0]

    kb = KnowledgeBase(tmp_path / "kb.db")
    result = AgentHarness(
        policy, kb, reward_fn=reward_fn, max_completion_tokens=12
    ).run_episode({"id": "q", "prompt": "p", "answer": "A", "completion_cap": 12})
    assert result.reward == -2.0
    assert len(observed[0]["completion_ids"][0]) == 12
    kb.close()


def test_run_jsonl_emits_group_advantage_trace(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps({"id": "q1", "prompt": "choose", "answer": "A"}) + "\n"
    )
    policy = ScriptedPolicy(
        assistant('{"answer":"A","knowledge_updates":[]}'),
        assistant('{"answer":"B","knowledge_updates":[]}'),
    )
    kb = KnowledgeBase(tmp_path / "kb.db")
    harness = AgentHarness(policy, kb, reward_fn=mcq_reward, max_tool_calls=0)
    output = tmp_path / "episodes.jsonl"
    results = run_jsonl(harness, source, output, generations=2)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(results) == len(records) == 2
    assert {record["group_id"] for record in records} == {results[0].group_id}
    assert records[0]["gspo_advantage"] > 0.99
    assert records[1]["gspo_advantage"] < -0.99
    assert all(
        record["kb_snapshot_before"] == records[0]["kb_snapshot_before"]
        for record in records
    )
    kb.close()


def test_predict_split_needs_no_gold_and_never_writes_claims(tmp_path):
    proposal = {
        "answer": "A",
        "knowledge_updates": [
            {
                "text": "A reusable and specific claim for prediction only, never stored as memory."
            }
        ],
    }
    kb = KnowledgeBase(tmp_path / "kb.db")
    result = AgentHarness(
        ScriptedPolicy(assistant(json.dumps(proposal))), kb, max_tool_calls=0
    ).run_episode({"id": "prediction", "prompt": "solve this"}, split="predict")
    assert result.reward == 0.0
    assert result.inserted_claim_ids == []
    assert len(kb) == 0
    kb.close()


def test_group_rollouts_compute_sequence_advantages_before_learning(tmp_path):
    policy = ScriptedPolicy(
        assistant('{"answer":"A","knowledge_updates":[]}'),
        assistant('{"answer":"B","knowledge_updates":[]}'),
    )
    kb = KnowledgeBase(tmp_path / "kb.db")
    harness = AgentHarness(policy, kb, reward_fn=mcq_reward, max_tool_calls=0)
    episodes = harness.run_group(
        {"id": "group-q", "prompt": "Choose", "answer": "A"}, 2
    )
    assert [episode.reward for episode in episodes] == [1.0, 0.0]
    assert episodes[0].gspo_advantage > 0.99
    assert episodes[1].gspo_advantage < -0.99
    assert episodes[0].group_id == episodes[1].group_id
    assert episodes[0].kb_snapshot_before == episodes[1].kb_snapshot_before
    assert episodes[0].kb_snapshot_after == episodes[1].kb_snapshot_after
    kb.close()


def test_openai_policy_executes_tool_call_round_trip(tmp_path):
    replies = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "calculate",
                                    "arguments": '{"expression":"8*8"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"completion_tokens": 6},
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"answer":"64","knowledge_updates":[]}',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 10},
        },
    ]
    requests = []

    def transport(payload):
        requests.append(copy.deepcopy(payload))
        return replies.pop(0)

    policy = OpenAIChatPolicy("http://stub/v1", "test", transport=transport)
    kb = KnowledgeBase(tmp_path / "kb.db")
    harness = AgentHarness(policy, kb, max_tool_calls=2)
    result = harness.run_episode(
        {"id": "api-q", "prompt": "8 squared?", "answer": "64"}
    )
    assert result.reward == 1.0
    assert result.trace[1]["result"] == 64
    assert len(requests) == 2
    assert requests[0]["tools"][0]["function"]["name"] == "kb_search"
    assert requests[1]["messages"][-1]["role"] == "tool"
    kb.close()


def test_local_http_api_runs_complete_tool_episode(tmp_path):
    replies = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call",
                                "type": "function",
                                "function": {
                                    "name": "calculate",
                                    "arguments": '{"expression":"9*9"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"completion_tokens": 5},
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"answer":"81","knowledge_updates":[]}',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 8},
        },
    ]
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            body = json.dumps(replies.pop(0)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        policy = OpenAIChatPolicy(base_url, "test", timeout=2)
        kb = KnowledgeBase(tmp_path / "kb.db")
        result = AgentHarness(policy, kb, max_tool_calls=2).run_episode(
            {"id": "http-q", "prompt": "What is 9 squared?", "answer": "81"}
        )
        assert result.reward == 1.0
        assert result.trace[1]["result"] == 81
        assert len(requests) == 2
        assert requests[1]["messages"][-1]["role"] == "tool"
        kb.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_cli_runs_against_local_openai_compatible_api_and_writes_manifest(
    tmp_path, monkeypatch
):
    replies = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"answer":"A","knowledge_updates":[]}',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 6},
        }
    ]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps(replies.pop(0)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    data = tmp_path / "train.jsonl"
    data.write_text(
        json.dumps({"id": "cli-q", "prompt": "Pick A", "answer": "A"}) + "\n"
    )
    output = tmp_path / "run" / "episodes.jsonl"
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rlforge-agentic",
            "--data",
            str(data),
            "--out",
            str(output),
            "--kb",
            str(tmp_path / "kb.sqlite3"),
            "--split",
            "eval",
            "--base-url",
            f"http://127.0.0.1:{server.server_port}/v1",
            "--model",
            "stub",
        ],
    )
    try:
        agent_cli_main()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["status"] == "complete"
        assert manifest["episodes"] == 1
        assert len(manifest["data_sha256"]) == 64
        assert json.loads(output.read_text())["reward"] == 1.0
        with pytest.raises(FileExistsError):
            agent_cli_main()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_openai_policy_injectable_transport():
    seen = []

    def transport(payload):
        seen.append(payload)
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 4},
        }

    policy = OpenAIChatPolicy(
        "http://localhost:9/v1", "test-model", transport=transport
    )
    result = policy.complete([{"role": "user", "content": "hi"}], [], 32)
    assert result["message"]["content"] == "done"
    assert result["completion_tokens"] == 4
    assert seen[0]["temperature"] == 0.7
    assert seen[0]["top_p"] == 1.0
    assert "top_k" not in seen[0]
    assert seen[0]["presence_penalty"] == 0.0


def test_eval_proposal_is_never_written(tmp_path):
    text = (
        "A reusable and specific approach that should not be learned from evaluation."
    )
    policy = ScriptedPolicy(
        assistant(json.dumps({"answer": "A", "knowledge_updates": [{"text": text}]}))
    )
    kb = KnowledgeBase(tmp_path / "kb.db")
    harness = AgentHarness(policy, kb, max_tool_calls=0, min_skill_support=1)
    result = harness.run_episode(
        {"id": "eval-q", "prompt": "p", "answer": "A"}, split="eval"
    )
    assert result.inserted_claim_ids == []
    assert len(kb) == 0
    kb.close()


def test_same_pair_cannot_supply_multiple_skill_supports(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db")
    proposal = [
        {
            "text": "A reusable workflow checks the dataset family before ranking optimizers."
        }
    ]
    for task_id in ("orig", "counterfactual"):
        kb.add_successful_proposals(
            proposal,
            task_id,
            "family",
            "train",
            ["pair-1"],
            reward=1.0,
            threshold=1.0,
            min_support=2,
        )
    assert len(kb) == 0
    kb.close()


def test_task_key_is_excluded_from_retrieval(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db")
    kb.db.execute(
        """INSERT INTO claims (id,text,family,kind,source_task_id,source_split,source_keys,status,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            "K1",
            "A reusable optimizer rule for spiral classification.",
            "spiral",
            "rule",
            "same-task",
            "train",
            "[]",
            "active",
            1.0,
        ),
    )
    kb.db.commit()
    result = kb.retrieve("optimizer rule spiral", "spiral", exclude_task_id="same-task")
    assert result == []
    kb.close()


def test_react_text_protocol_omits_api_tools_and_normalizes_action_and_final():
    payloads = []
    replies = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": 'Thought: calculate\nAction: calculate\nAction Input: {"expression":"6*7"}',
                    },
                    "finish_reason": "stop",
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": 'Final Answer: {"answer":"42","knowledge_updates":[]}',
                    },
                    "finish_reason": "stop",
                }
            ]
        },
    ]

    def transport(payload):
        payloads.append(copy.deepcopy(payload))
        return replies.pop(0)

    policy = OpenAIChatPolicy(
        "http://stub/v1", "qwen35", protocol="react", transport=transport
    )
    kb = KnowledgeBase(":memory:")
    result = AgentHarness(policy, kb).run_episode(
        {"id": "react-q", "prompt": "What is 6 times 7?", "answer": "42"}
    )
    assert result.reward == 1.0
    assert result.tool_calls == 1
    assert result.trace[1]["name"] == "calculate"
    assert all("tools" not in payload for payload in payloads)
    assert payloads[1]["messages"][-1]["role"] == "user"
    assert "Observation:" in payloads[1]["messages"][-1]["content"]
    kb.close()


def test_mcp_stdio_server_tool_runs_through_react_harness(tmp_path):
    pytest.importorskip("mcp")
    server_code = (
        "from mcp.server.fastmcp import FastMCP\n"
        "m=FastMCP('math')\n"
        "@m.tool()\n"
        "def multiply(a: int, b: int) -> int:\n"
        "    return a*b\n"
        "m.run(transport='stdio')"
    )
    config = [
        {
            "mcpServers": {
                "math": {
                    "command": sys.executable,
                    "args": ["-c", server_code],
                }
            }
        }
    ]
    bridge = MCPToolBridge(config, timeout=15)
    try:
        tools = ToolRegistry.standard()
        tools.register_mcp_bridge(bridge)
        exposed = "mcp_math_multiply"
        assert exposed in bridge.tool_map
        responses = [
            {
                "message": {
                    "role": "assistant",
                    "content": 'Thought: use multiplication\nAction: mcp_math_multiply\nAction Input: {"a":6,"b":7}',
                },
                "completion_tokens": 12,
                "finish_reason": "stop",
            },
            {
                "message": {
                    "role": "assistant",
                    "content": 'Final Answer: {"answer":"42","knowledge_updates":[]}',
                },
                "completion_tokens": 8,
                "finish_reason": "stop",
            },
        ]

        def transport(_payload):
            response = responses.pop(0)
            return {
                "choices": [
                    {
                        "message": response["message"],
                        "finish_reason": response["finish_reason"],
                    }
                ],
                "usage": {"completion_tokens": response["completion_tokens"]},
            }

        policy = OpenAIChatPolicy(
            "http://stub/v1", "qwen35", protocol="react", transport=transport
        )
        kb = KnowledgeBase(tmp_path / "mcp-kb.sqlite3")
        result = AgentHarness(policy, kb, tools=tools).run_episode(
            {"id": "mcp-q", "prompt": "What is 6 times 7?", "answer": "42"}
        )
        assert result.reward == 1.0
        assert result.tool_calls == 1
        assert result.trace[1]["event"] == "tool_call", repr(result.trace[1])
        assert result.trace[1]["result"]["result"] == 42
        kb.close()
    finally:
        bridge.close()


def test_memory_workspace_reads_writes_and_renames_without_touching_bank(tmp_path):
    root = tmp_path / "skills"
    (root / "general").mkdir(parents=True)
    (root / "general" / "explore.md").write_text("Explore methodically.")
    bank = SkillFileBank(root)
    workspace = bank.workspace()
    assert (
        workspace.read_file("general/explore.md")["content"] == "Explore methodically."
    )
    old_hash = workspace.read_file("general/explore.md")["sha256"]
    workspace.write_file(
        "general/explore.md", "Explore each distinct branch once.", old_hash
    )
    workspace.rename_file("general/explore.md", "task_specific/math/explore.md")
    changes = workspace.changes()
    assert len(changes) == 1
    assert changes[0]["name"] == "task_specific/math/explore.md"
    assert changes[0]["renames"][0]["from"] == "general/explore.md"
    assert (root / "general" / "explore.md").read_text() == "Explore methodically."
    workspace.close()
    bank.close()


def test_memory_path_policy_rejects_escape_symlink_and_stale_edit(tmp_path):
    root = tmp_path / "skills"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("private")
    bank = SkillFileBank(root)
    workspace = bank.workspace()
    for name in ("../outside.md", "/tmp/outside.md", "nested/../../outside.md"):
        with pytest.raises(ValueError):
            workspace.write_file(name, "escape")
    workspace.write_file("safe.md", "safe")
    with pytest.raises(ValueError, match="provide its sha256"):
        workspace.write_file("safe.md", "changed")
    link = workspace.path / "link.md"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        workspace.read_file("link.md")
    assert outside.read_text() == "private"
    workspace.close()
    bank.close()


def test_only_repeated_successful_train_memory_changes_are_promoted(tmp_path):
    root = tmp_path / "skills"
    bank = SkillFileBank(root)
    workspace = bank.workspace()
    workspace.write_file(
        "general/arithmetic.md", "Check each operation and verify units."
    )
    change = workspace.changes()
    assert bank.promote(change, "task-1", "eval", 1.0, 1.0, 2)["status"] == "rejected"
    assert (
        bank.promote(change, "task-1", "train", 0.0, 1.0, 2)["status"]
        == "pending_validation"
    )
    pending = bank.promote(change, "task-1", "train", 1.0, 1.0, 2)
    assert pending["status"] == "pending_support"
    assert not (root / "general" / "arithmetic.md").exists()
    promoted = bank.promote(change, "task-2", "train", 1.0, 1.0, 2)
    assert promoted["status"] == "promoted"
    assert (
        root / "general" / "arithmetic.md"
    ).read_text() == "Check each operation and verify units."
    duplicate = bank.promote(change, "task-3", "train", 1.0, 1.0, 2)
    assert duplicate["status"] == "promoted"
    workspace.close()
    bank.close()


def test_pair_id_cannot_fake_independent_memory_support(tmp_path):
    bank = SkillFileBank(tmp_path / "skills")
    changes = [
        {
            "name": "general/rule.md",
            "before_sha256": None,
            "content": "A reusable cross-task rule.",
        }
    ]
    first = bank.promote(changes, "orig", "train", 1.0, 1.0, 2, support_key="pair-1")
    second = bank.promote(
        changes, "counterfactual", "train", 1.0, 1.0, 2, support_key="pair-1"
    )
    assert first["status"] == second["status"] == "pending_support"
    bank.close()


def test_skillbank_retrieves_general_and_matching_task_skills(tmp_path):
    root = tmp_path / "skills"
    (root / "general").mkdir(parents=True)
    (root / "task_specific" / "spiral").mkdir(parents=True)
    (root / "task_specific" / "xor").mkdir(parents=True)
    (root / "general" / "check.md").write_text("Check spiral geometry before training.")
    (root / "task_specific" / "spiral" / "turns.md").write_text("Tune spiral turns.")
    (root / "task_specific" / "xor" / "depth.md").write_text("Increase XOR depth.")
    bank = SkillFileBank(root)
    workspace = bank.workspace()
    workspace.write_file(
        "general/new.md", "Spiral geometry requires checking staged evidence."
    )
    found = workspace.retrieve("spiral geometry staged evidence", "spiral")
    names = {item["name"] for item in found}
    assert "general/check.md" in names
    assert "task_specific/spiral/turns.md" in names
    assert "task_specific/xor/depth.md" not in names
    staged = next(item for item in found if item["name"] == "general/new.md")
    assert "staged evidence" in staged["content"]
    workspace.close()
    bank.close()


def test_failure_file_candidates_are_only_available_to_training_queries(tmp_path):
    bank = SkillFileBank(tmp_path / "skills")
    failed_change = [
        {
            "name": "general/search.md",
            "before_sha256": None,
            "content": "For web search, cite each source and cross-check the result.",
        }
    ]
    result = bank.promote(failed_change, "failed-task", "train", 0.0, 1.0, 2)
    assert result["status"] == "pending_validation"
    assert bank.pending_candidates("web search sources", "general")
    assert not bank.pending_candidates("web search sources", "general", {"failed-task"})
    bank.close()


def test_react_agent_promotes_file_memory_after_repeated_success(tmp_path):
    root = tmp_path / "skills"
    bank = SkillFileBank(root)
    proposal = {
        "name": "general/arithmetic.md",
        "content": "For arithmetic, estimate then verify each intermediate operation.",
    }
    episodes = []
    for task_id in ("task-1", "task-2"):
        policy = ScriptedPolicy(
            {
                "message": {
                    "role": "assistant",
                    "content": "Thought: save a reusable strategy\nAction: memory_write_file\nAction Input: "
                    + json.dumps(proposal),
                },
                "completion_tokens": 15,
                "finish_reason": "stop",
            },
            assistant('{"answer":"A","knowledge_updates":[]}'),
        )
        policy.protocol = "react"
        kb = KnowledgeBase(":memory:")
        tools = ToolRegistry.standard()
        tools.register_file_memory_tools()
        harness = AgentHarness(
            policy,
            kb,
            tools=tools,
            skill_file_bank=bank,
            max_tool_calls=2,
            learn_threshold=1.0,
            min_skill_support=2,
        )
        episode = harness.run_episode(
            {"id": task_id, "prompt": "Calculate 3+4.", "answer": "A"}
        )
        episodes.append(episode)
        kb.close()
    assert episodes[0].memory_promotion["status"] == "pending_support"
    assert episodes[1].memory_promotion["status"] == "promoted"
    assert (root / "general" / "arithmetic.md").read_text() == proposal["content"]

    policy = ScriptedPolicy(assistant('{"answer":"A","knowledge_updates":[]}'))
    policy.protocol = "react"
    kb = KnowledgeBase(":memory:")
    harness = AgentHarness(policy, kb, skill_file_bank=bank, max_tool_calls=0)
    result = harness.run_episode(
        {"id": "task-3", "prompt": "Calculate arithmetic carefully.", "answer": "A"}
    )
    assert result.memory_snapshot_before == result.memory_snapshot_after
    assert "verify each intermediate operation" in policy.requests[0][0][0]["content"]
    kb.close()
    bank.close()


def test_eval_file_edits_are_logged_but_never_persisted(tmp_path):
    root = tmp_path / "skills"
    bank = SkillFileBank(root)
    tools = ToolRegistry.standard()
    tools.register_file_memory_tools()
    policy = ScriptedPolicy(
        {
            "message": {
                "role": "assistant",
                "content": "Action: memory_write_file\nAction Input: "
                + json.dumps(
                    {
                        "name": "general/leak.md",
                        "content": "An eval-only memory note that must never escape the episode workspace.",
                    }
                ),
            },
            "completion_tokens": 18,
            "finish_reason": "stop",
        },
        assistant('{"answer":"A","knowledge_updates":[]}'),
    )
    policy.protocol = "react"
    kb = KnowledgeBase(":memory:")
    result = AgentHarness(
        policy, kb, tools=tools, skill_file_bank=bank, max_tool_calls=1
    ).run_episode({"id": "heldout", "prompt": "Answer A", "answer": "A"}, split="eval")
    assert result.memory_changes
    assert result.memory_promotion["status"] == "rejected"
    assert not (root / "general" / "leak.md").exists()
    kb.close()
    bank.close()
