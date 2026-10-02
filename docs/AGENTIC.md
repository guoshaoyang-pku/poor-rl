# Agentic RL harness (preview)

The first agent harness now follows the Qwen3.5-friendly **text ReAct + MCP** path:
the model emits a bounded `Action` / `Action Input`, rlforge calls the registered MCP
server, and returns an `Observation` for the next ReAct step. It does not send OpenAI
`tools` schemas on this default path. Local KB lookup and arithmetic remain built-in
tools. MCP uses the official [Python SDK](https://github.com/modelcontextprotocol/python-sdk)
and accepts the stdio `mcpServers` config shape shown by [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent).
MCP server entries launch local processes, so use only trusted configs. A legacy
`--protocol openai-tools` mode remains available for endpoints that expose native tool calls.

```bash
pip install -e '.[agent-mcp]'
```

MCP config example (`mcp.json`):

```json
{
  "mcpServers": {
    "time": {
      "command": "uvx",
      "args": ["mcp-server-time", "--local-timezone=Asia/Shanghai"]
    }
  }
}
```

Run against a Qwen3.5 OpenAI-compatible endpoint (non-thinking defaults match the
model-card text-task recipe; thinking is opt-in because 0.8B can loop):

```bash
export RLFORGE_API_KEY=EMPTY
rlforge-agentic --data data/train.jsonl --model "Qwen/Qwen3.5-0.8B" \
  --base-url http://localhost:8000/v1 --mcp-config mcp.json \
  --import-kb /path/to/kb_swarm.json --kb runs/agentic/knowledge.sqlite3 \
  --out runs/agentic/episodes.jsonl --split train --num-generations 8 \
  --max-tool-calls 4 --max-rounds 6 --max-tool-result-chars 4000 \
  --max-context-tokens 24576 --max-completion-tokens 16384
```

To turn on the persistent file SkillBank prototype, create files under `general/`,
`task_specific/<family>/`, and `common_mistakes/`, then pass `--memory-files`:

```bash
rlforge-agentic --data data/train.jsonl --model "Qwen/Qwen3.5-0.8B" \
  --base-url http://localhost:8000/v1 --mcp-config mcp.json \
  --memory-files runs/agentic/skillbank --memory-evidence-db runs/agentic/skillbank_evidence.sqlite3 \
  --kb runs/agentic/knowledge.sqlite3 --out runs/agentic/episodes.jsonl \
  --split train --num-generations 8 --min-skill-support 2
```

The model can list, read, create/edit, and rename `.md` files by filename. Each
rollout gets an isolated copy; an edit reaches the shared SkillBank only after a
successful training reward and matching content from at least two distinct task/pair
keys. Held-out splits never receive pending failure candidates and never promote edits.
The tool is confined to Markdown inside the SkillBank, enforces path/symlink/size limits,
requires the last-read hash for edits, and uses conflict checks before promotion. Changes,
rewards, provenance keys, promotion decisions, and before/after bank hashes are logged
per episode. Task-specific retrieval uses `task_specific/<family>/`; `general/` and
`common_mistakes/` are cross-task categories.

This is a **SkillRL-inspired prototype**, not a reproduction of the paper: it has
lexical top-k retrieval, per-episode file proposals, reward/support-gated promotion,
and retrieval of relevant unvalidated failed edits as candidates. It does not yet
implement an LLM skill-distillation/evolution job, embedding retrieval, category-level
validation-accuracy triggers, or automatic rewriting/pruning of existing skills. A
proper evolution controller should consume train/dev failures only and keep final
held-out evaluation sealed.

The harness keeps a provenance-aware SQLite knowledge base, applies terminal-loss
rewards, logs bounded episode traces and context estimates, and can export group-relative
terminal advantages. Evaluation episodes cannot write to the KB; learned claims require
successful training outcomes and support from distinct task/pair provenance. Context
budgets are harness-side gates, not model context extensions or measurements of actual
KV-cache allocation. FP8 KV can reduce cache memory but does not reduce the token cost of
ReAct observations. Historical ArchitectureIQ KB v4 JSON (`{"claims":[...]}`) imports
directly; its curation/leakage audit remains the experiment owner's responsibility.

This is a tested agent/tool orchestration harness, **not yet a model-training path**:
`gspo_advantage` is an auditable rollout signal, not a trainer input. It does not compute
actor token log-probabilities or update weights. GSPO training integration must preserve
the full ReAct action/observation token trajectory and align its terminal reward before
it can train the policy; do not treat the current API-generated episodes as an actual
GSPO run.
