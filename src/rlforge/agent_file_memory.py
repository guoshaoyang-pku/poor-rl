from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*\.md$")
_MAX_FILE_CHARS = 12000
_MAX_FILES = 200


def _resolve_name(name: str) -> PurePosixPath:
    if not isinstance(name, str) or len(name) > 180 or not _NAME_RE.fullmatch(name):
        raise ValueError(
            "file name must be a relative .md path using letters, digits, '.', '_' or '-'"
        )
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("file path must remain inside the memory workspace")
    return relative


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MemoryWorkspace:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tempdir = tempfile.TemporaryDirectory(prefix="rlforge-memory-")
        self.path = Path(self.tempdir.name).resolve()
        self.renames: list[dict[str, str]] = []
        self._copy_seed()
        self.baseline = self._snapshot()

    def _copy_seed(self) -> None:
        files = [path for path in self.root.rglob("*.md") if path.is_file()]
        if len(files) > _MAX_FILES:
            raise ValueError(f"memory bank exceeds {_MAX_FILES} markdown files")
        for source in files:
            if source.is_symlink():
                raise ValueError("memory bank cannot contain symlinked files")
            relative = source.relative_to(self.root)
            target = self.path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    def retrieve(
        self, query: str, family: str, top_k: int = 6, max_chars: int = 4000
    ) -> list[dict[str, Any]]:
        tokens = {
            item.lower() for item in re.findall(r"[a-z0-9_]+", query) if len(item) > 1
        }
        ranked = []
        for entry in self.list_files():
            name = entry["name"]
            parts = PurePosixPath(name).parts
            if parts[0] == "task_specific" and (len(parts) < 3 or parts[1] != family):
                continue
            content = (self.path / name).read_text(encoding="utf-8")
            content_tokens = {
                item.lower()
                for item in re.findall(r"[a-z0-9_]+", f"{name} {content}")
                if len(item) > 1
            }
            overlap = len(tokens & content_tokens)
            score = overlap / max(1, len(tokens))
            if parts[0] == "general":
                score += 0.05
            if parts[0] == "common_mistakes":
                score += 0.03
            if score > 0:
                ranked.append((score, name, content))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected = []
        used = 0
        for score, name, content in ranked:
            item = {"name": name, "content": content, "score": round(score, 6)}
            size = len(json.dumps(item, ensure_ascii=False))
            if used + size > max_chars:
                continue
            selected.append(item)
            used += size
            if len(selected) >= top_k:
                break
        return selected

    def _resolve(self, name: str, allow_missing: bool = True) -> Path:
        relative = _resolve_name(name)
        target = self.path.joinpath(*relative.parts)
        parent = target.parent.resolve()
        if not parent.is_relative_to(self.path):
            raise ValueError("file path escapes the memory workspace")
        if target.is_symlink():
            raise ValueError("symlinked memory files are not allowed")
        if target.exists() and not target.resolve().is_relative_to(self.path):
            raise ValueError("file path escapes the memory workspace")
        if not allow_missing and not target.is_file():
            raise FileNotFoundError(name)
        return target

    def list_files(self) -> list[dict[str, Any]]:
        files = sorted(self.path.rglob("*.md"))
        if len(files) > _MAX_FILES:
            raise ValueError(f"workspace exceeds {_MAX_FILES} markdown files")
        return [
            {
                "name": path.relative_to(self.path).as_posix(),
                "chars": len(path.read_text(encoding="utf-8")),
            }
            for path in files
            if path.is_file() and not path.is_symlink()
        ]

    def read_file(self, name: str) -> dict[str, Any]:
        target = self._resolve(name, allow_missing=False)
        text = target.read_text(encoding="utf-8")
        return {"name": name, "content": text, "sha256": _digest(text)}

    def write_file(
        self, name: str, content: str, expected_sha256: str | None = None
    ) -> dict[str, Any]:
        if not isinstance(content, str) or len(content) > _MAX_FILE_CHARS:
            raise ValueError(
                f"file content must be text up to {_MAX_FILE_CHARS} characters"
            )
        target = self._resolve(name)
        current = target.read_text(encoding="utf-8") if target.exists() else None
        current_hash = _digest(current) if current is not None else None
        if expected_sha256 is not None and current_hash != expected_sha256:
            raise ValueError(
                "memory file changed since it was read; re-read before editing"
            )
        if current is not None and expected_sha256 is None:
            raise ValueError(
                "read the existing file and provide its sha256 before editing"
            )
        if current is None and len(self.list_files()) >= _MAX_FILES:
            raise ValueError(f"workspace is limited to {_MAX_FILES} markdown files")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temp.write_text(content, encoding="utf-8")
        os.replace(temp, target)
        return {"name": name, "chars": len(content), "sha256": _digest(content)}

    def rename_file(self, old_name: str, new_name: str) -> dict[str, str]:
        source = self._resolve(old_name, allow_missing=False)
        target = self._resolve(new_name)
        if source == target:
            raise ValueError("source and destination file names must differ")
        if target.exists():
            raise FileExistsError(new_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        source_text = source.read_text(encoding="utf-8")
        source_hash = _digest(self.baseline.get(old_name, source_text))
        source.rename(target)
        self.renames.append(
            {"from": old_name, "to": new_name, "before_sha256": source_hash}
        )
        return {"from": old_name, "to": new_name}

    def _snapshot(self) -> dict[str, str]:
        return {
            path.relative_to(self.path).as_posix(): path.read_text(encoding="utf-8")
            for path in self.path.rglob("*.md")
            if path.is_file() and not path.is_symlink()
        }

    def changes(self) -> list[dict[str, Any]]:
        current = self._snapshot()
        renamed_sources = {item["from"] for item in self.renames}
        changes = []
        for name in sorted(set(self.baseline) | set(current)):
            if name in renamed_sources:
                continue
            before = self.baseline.get(name)
            after = current.get(name)
            if before != after:
                changes.append(
                    {
                        "name": name,
                        "before_sha256": _digest(before)
                        if before is not None
                        else None,
                        "content": after,
                    }
                )
        if self.renames:
            for change in changes:
                change["renames"] = [
                    item for item in self.renames if item["to"] == change["name"]
                ]
        return changes

    def close(self) -> None:
        self.tempdir.cleanup()


class SkillFileBank:
    """Persistent Markdown skill bank with train-only, multi-task promotion."""

    def __init__(self, root: str | Path, evidence_path: str | Path | None = None):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.evidence_path = (
            Path(evidence_path).expanduser().resolve()
            if evidence_path
            else self.root.parent / f"{self.root.name}.evidence.sqlite3"
        )
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(
            self.evidence_path, timeout=30, check_same_thread=False
        )
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS proposals (
                proposal_hash TEXT NOT NULL,
                evidence_key TEXT NOT NULL,
                task_id TEXT NOT NULL,
                changes_json TEXT NOT NULL,
                reward REAL NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (proposal_hash, evidence_key)
            )"""
        )
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS promotions (
                proposal_hash TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                details TEXT NOT NULL,
                promoted_at REAL NOT NULL
            )"""
        )
        self.db.commit()
        self._lock = threading.RLock()

    def workspace(self) -> MemoryWorkspace:
        return MemoryWorkspace(self.root)

    def pending_candidates(
        self,
        query: str,
        family: str,
        exclude_keys: set[str] | tuple[str, ...] = (),
        limit: int = 3,
        max_chars: int = 2000,
        success_threshold: float = 1.0,
        min_support: int = 2,
    ) -> list[dict[str, Any]]:
        if limit < 1 or max_chars < 1:
            return []
        excluded = {str(key) for key in exclude_keys if key}
        with self._lock:
            rows = self.db.execute(
                """SELECT p.proposal_hash,p.evidence_key,p.task_id,p.changes_json,p.reward,p.created_at
                FROM proposals p LEFT JOIN promotions m USING(proposal_hash)
                WHERE p.reward < ? AND m.proposal_hash IS NULL
                AND NOT EXISTS (
                    SELECT 1 FROM proposals good
                    WHERE good.proposal_hash=p.proposal_hash AND good.reward>=?
                    GROUP BY good.proposal_hash
                    HAVING COUNT(*)>=?
                )
                ORDER BY p.created_at DESC""",
                (success_threshold, success_threshold, min_support),
            ).fetchall()
        query_tokens = {
            token.lower()
            for token in re.findall(r"[a-z0-9_]+", query)
            if len(token) > 1
        }
        ranked = []
        for row in rows:
            if row["evidence_key"] in excluded or row["task_id"] in excluded:
                continue
            changes = json.loads(row["changes_json"])
            names = [item["name"] for item in changes]
            task_categories = {
                parts[1]
                for name in names
                if (parts := PurePosixPath(name).parts)
                and parts[0] == "task_specific"
                and len(parts) > 2
            }
            if task_categories and family not in task_categories:
                continue
            candidate_text = " ".join(
                f"{item['name']} {item.get('content') or ''}" for item in changes
            )
            candidate_tokens = {
                token.lower()
                for token in re.findall(r"[a-z0-9_]+", candidate_text)
                if len(token) > 1
            }
            score = len(query_tokens & candidate_tokens) / max(1, len(query_tokens))
            if score <= 0:
                continue
            candidate = {
                "proposal_hash": row["proposal_hash"],
                "source_task_id": row["task_id"],
                "changes": changes,
                "reward": row["reward"],
                "similarity": round(score, 6),
            }
            size = len(json.dumps(candidate, ensure_ascii=False))
            if size <= max_chars:
                ranked.append((score, row["created_at"], candidate))
        ranked.sort(key=lambda item: (-item[0], -item[1]))
        return [item[2] for item in ranked[:limit]]

    def snapshot(self) -> str:
        entries = []
        for path in sorted(self.root.rglob("*.md")):
            if path.is_symlink():
                raise ValueError("memory bank cannot contain symlinked files")
            text = path.read_text(encoding="utf-8")
            entries.append((path.relative_to(self.root).as_posix(), _digest(text)))
        return _digest(json.dumps(entries, ensure_ascii=False, separators=(",", ":")))

    def promote(
        self,
        changes: list[dict[str, Any]],
        task_id: str,
        split: str,
        reward: float,
        threshold: float,
        min_support: int,
        support_key: str | None = None,
    ) -> dict[str, Any]:
        if (
            split != "train"
            or min_support < 1
            or not changes
            or not math.isfinite(reward)
        ):
            return {
                "status": "rejected",
                "reason": "split, support, score, or empty proposal",
            }
        clean = []
        rename_sources = {
            str(rename.get("from", ""))
            for item in changes
            for rename in item.get("renames", [])
        }
        for item in changes:
            name = str(item.get("name", ""))
            content = item.get("content")
            if content is not None and (
                not isinstance(content, str) or len(content) > _MAX_FILE_CHARS
            ):
                return {"status": "rejected", "reason": "invalid memory file content"}
            if content is None and name not in rename_sources:
                return {
                    "status": "rejected",
                    "reason": "deletion only allowed as part of a rename",
                }
            _resolve_name(name)
            clean.append(
                {
                    "name": name,
                    "before_sha256": item.get("before_sha256"),
                    "content": content,
                    "renames": item.get("renames", []),
                }
            )
        proposal_hash = _digest(
            json.dumps(clean, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        )
        evidence_key = support_key or task_id
        with self._lock:
            self.db.execute(
                "INSERT INTO proposals VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(proposal_hash,evidence_key) DO UPDATE SET "
                "task_id=excluded.task_id, changes_json=excluded.changes_json, "
                "reward=MAX(proposals.reward, excluded.reward)",
                (
                    proposal_hash,
                    evidence_key,
                    task_id,
                    json.dumps(clean, ensure_ascii=False),
                    float(reward),
                    time.time(),
                ),
            )
            supports = self.db.execute(
                "SELECT COUNT(*) FROM proposals WHERE proposal_hash=? AND reward>=?",
                (proposal_hash, threshold),
            ).fetchone()[0]
            if supports < min_support:
                self.db.commit()
                return {
                    "status": "pending_support"
                    if reward >= threshold
                    else "pending_validation",
                    "proposal_hash": proposal_hash,
                    "support_count": supports,
                    "required_support": min_support,
                }
            existing = self.db.execute(
                "SELECT status,details FROM promotions WHERE proposal_hash=?",
                (proposal_hash,),
            ).fetchone()
            if existing:
                return {
                    "status": existing[0],
                    "proposal_hash": proposal_hash,
                    "details": json.loads(existing[1]),
                }
            applied, conflicts = [], []
            rollback = []
            try:
                for item in clean:
                    target = self.root / item["name"]
                    if not target.resolve().is_relative_to(self.root):
                        raise ValueError("memory path escapes the bank")
                    if target.is_symlink():
                        raise ValueError("symlinked memory files are not allowed")
                    current = (
                        target.read_text(encoding="utf-8") if target.exists() else None
                    )
                    current_hash = _digest(current) if current is not None else None
                    if current_hash not in {
                        item["before_sha256"],
                        _digest(item["content"]),
                    }:
                        conflicts.append(item["name"])
                        continue
                    if item["content"] == current:
                        applied.append(item["name"])
                        continue
                    rollback.append((target, current))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
                    temp.write_text(item["content"], encoding="utf-8")
                    os.replace(temp, target)
                    applied.append(item["name"])
                    for rename in item["renames"]:
                        old_name = str(rename.get("from", ""))
                        _resolve_name(old_name)
                        old = self.root / old_name
                        if (
                            old_name == item["name"]
                            or not old.exists()
                            or old.is_symlink()
                        ):
                            continue
                        old_text = old.read_text(encoding="utf-8")
                        expected_source_hash = rename.get("before_sha256")
                        if _digest(old_text) == expected_source_hash:
                            rollback.append((old, old_text))
                            old.unlink()
                details = {"applied": applied, "conflicts": conflicts}
                status = "promoted" if applied else "conflict"
                self.db.execute(
                    "INSERT INTO promotions VALUES (?,?,?,?)",
                    (proposal_hash, status, json.dumps(details), time.time()),
                )
                self.db.commit()
                return {"status": status, "proposal_hash": proposal_hash, **details}
            except Exception:
                for target, previous in reversed(rollback):
                    if previous is None:
                        target.unlink(missing_ok=True)
                    else:
                        target.write_text(previous, encoding="utf-8")
                self.db.rollback()
                raise

    def close(self) -> None:
        self.db.close()


def _resolve_name(name: str) -> PurePosixPath:
    if not isinstance(name, str) or len(name) > 180 or not _NAME_RE.fullmatch(name):
        raise ValueError(
            "file name must be a relative .md path using letters, digits, '.', '_' or '-'"
        )
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("file path must remain inside the memory workspace")
    return relative
