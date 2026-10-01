from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Iterable

_WORD_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(text) if len(token) > 1}


class KnowledgeBase:
    """SQLite-backed, provenance-aware store for reusable cross-task claims."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                family TEXT NOT NULL,
                kind TEXT NOT NULL,
                source_task_id TEXT,
                source_split TEXT NOT NULL,
                source_keys TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            )"""
        )
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(claims)")}
        if "metadata" not in columns:
            self.db.execute(
                "ALTER TABLE claims ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'"
            )
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS proposal_evidence (
                text_hash TEXT NOT NULL,
                evidence_key TEXT NOT NULL,
                task_id TEXT NOT NULL,
                text TEXT NOT NULL,
                family TEXT NOT NULL,
                kind TEXT NOT NULL,
                provenance_keys TEXT NOT NULL,
                reward REAL NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (text_hash, evidence_key)
            )"""
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def import_json(self, path: str | Path) -> int:
        payload = json.loads(Path(path).read_text())
        claims = payload.get("claims") if isinstance(payload, dict) else payload
        if not isinstance(claims, list):
            raise ValueError("KB import must be a list or an object with a claims list")
        inserted = 0
        for item in claims:
            if not isinstance(item, dict):
                raise ValueError("each imported claim must be an object")
            claim_id = str(item.get("id", "")).strip()
            text = str(item.get("text", "")).strip()
            if not claim_id or not text:
                raise ValueError("each imported claim needs non-empty id and text")
            if item.get("status", "active") not in {"active", "validated", "seed"}:
                continue
            metadata = {
                key: value
                for key, value in item.items()
                if key not in {"id", "text", "family", "kind", "status"}
            }
            cur = self.db.execute(
                """INSERT OR IGNORE INTO claims
                (id,text,family,kind,source_task_id,source_split,source_keys,status,created_at,metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    claim_id,
                    text,
                    str(item.get("family", "general")),
                    str(item.get("kind", "claim")),
                    None,
                    "seed",
                    "[]",
                    "active",
                    time.time(),
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                ),
            )
            inserted += cur.rowcount
        self.db.commit()
        return inserted

    def snapshot(self) -> str:
        rows = self.db.execute(
            "SELECT id,text,family,kind,source_task_id,source_split,source_keys,status,metadata "
            "FROM claims WHERE status='active' ORDER BY id"
        ).fetchall()
        canonical = json.dumps(
            [dict(row) for row in rows], sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def retrieve(
        self,
        query: str,
        family: str = "general",
        limit: int = 8,
        exclude_task_id: str | None = None,
        exclude_keys: Iterable[str] = (),
    ) -> list[dict]:
        query_tokens = _tokens(query)
        excluded = {str(key) for key in exclude_keys if key}
        if exclude_task_id:
            excluded.add(str(exclude_task_id))
        candidates = self.db.execute(
            "SELECT * FROM claims WHERE status='active' AND source_split!='eval'"
        ).fetchall()
        ranked = []
        for row in candidates:
            provenance = set(json.loads(row["source_keys"]))
            if exclude_task_id and row["source_task_id"] == exclude_task_id:
                continue
            if excluded.intersection(provenance):
                continue
            claim_tokens = _tokens(
                row["text"] + " " + row["family"] + " " + row["kind"]
            )
            overlap = len(query_tokens & claim_tokens)
            family_match = row["family"] in {family, "general", "cross_family"}
            if row["family"] not in {family, "general", "cross_family"}:
                continue
            score = overlap / max(1, len(query_tokens)) + (
                0.25 if family_match and row["family"] == family else 0
            )
            if score > 0:
                ranked.append((score, row["id"], row))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "id": row["id"],
                "text": row["text"],
                "family": row["family"],
                "kind": row["kind"],
                "score": round(score, 6),
                "source_task_id": row["source_task_id"],
                "source_split": row["source_split"],
                "metadata": json.loads(row["metadata"]),
            }
            for score, _, row in ranked[: max(0, limit)]
        ]

    def add_successful_proposals(
        self,
        proposals: Iterable[dict],
        task_id: str,
        family: str,
        split: str,
        provenance_keys: Iterable[str],
        reward: float,
        threshold: float,
        min_support: int,
        support_key: str | None = None,
    ) -> list[str]:
        if split != "train" or reward < threshold or min_support < 1:
            return []
        inserted = []
        related_keys = sorted({str(key) for key in provenance_keys if key})
        evidence_key = support_key or (
            "|".join(related_keys) if related_keys else task_id
        )
        keys = sorted({task_id, *related_keys})
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            text = " ".join(str(proposal.get("text", "")).split())
            if not 20 <= len(text) <= 1200:
                continue
            kind = str(proposal.get("kind", "claim"))[:40]
            text_hash = hashlib.sha256(text.lower().encode()).hexdigest()
            self.db.execute(
                """INSERT OR IGNORE INTO proposal_evidence
                (text_hash,evidence_key,task_id,text,family,kind,provenance_keys,reward,created_at)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    text_hash,
                    evidence_key,
                    task_id,
                    text,
                    family or "general",
                    kind,
                    json.dumps(keys),
                    float(reward),
                    time.time(),
                ),
            )
            supports = self.db.execute(
                "SELECT COUNT(DISTINCT evidence_key) FROM proposal_evidence WHERE text_hash=?",
                (text_hash,),
            ).fetchone()[0]
            if supports < min_support:
                continue
            evidence = self.db.execute(
                "SELECT task_id,provenance_keys,family FROM proposal_evidence WHERE text_hash=? "
                "ORDER BY task_id",
                (text_hash,),
            ).fetchall()
            source_task = evidence[0]["task_id"]
            evidence_families = {row["family"] for row in evidence}
            claim_family = (
                "cross_family" if len(evidence_families) > 1 else family or "general"
            )
            all_keys = sorted(
                {key for row in evidence for key in json.loads(row["provenance_keys"])}
            )
            identifier = "K" + text_hash[:12].upper()
            cur = self.db.execute(
                """INSERT OR IGNORE INTO claims
                (id,text,family,kind,source_task_id,source_split,source_keys,status,created_at)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    identifier,
                    text,
                    claim_family,
                    kind,
                    source_task,
                    "train",
                    json.dumps(all_keys),
                    "active",
                    time.time(),
                ),
            )
            if cur.rowcount:
                inserted.append(identifier)
        self.db.commit()
        return inserted

    def __len__(self) -> int:
        return int(
            self.db.execute(
                "SELECT COUNT(*) FROM claims WHERE status='active'"
            ).fetchone()[0]
        )
