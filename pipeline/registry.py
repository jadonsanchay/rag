"""SQLite registry: what is indexed, and the conversations about it.

Chroma holds vectors and FTS5 holds text, but neither can answer "is this repo
finished indexing, and did it fail?" Indexing takes minutes, so that state has to
outlive a request — hence a real table with a status column rather than a
dictionary in memory.

Conversations live here too because multi-turn retrieval needs the prior turns to
rewrite a follow-up into a standalone query.
"""

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import config

DB_PATH = config.DATA_DIR / "registry.sqlite3"

# Lifecycle. Anything not `ready` is not queryable.
QUEUED = "queued"
CLONING = "cloning"
INDEXING = "indexing"
READY = "ready"
FAILED = "failed"

SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    url           TEXT,
    variant       TEXT NOT NULL,
    collection    TEXT NOT NULL UNIQUE,
    repo_path     TEXT,
    commit_sha    TEXT,
    status        TEXT NOT NULL,
    stage         TEXT,
    error         TEXT,
    files_indexed INTEGER DEFAULT 0,
    chunks        INTEGER DEFAULT 0,
    size_bytes    INTEGER DEFAULT 0,
    languages     TEXT,
    embedding_provider TEXT,
    embedding_model    TEXT,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    indexed_at    REAL
);

CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    repo_id    TEXT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    title      TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    rewritten_query TEXT,
    trace           TEXT,
    created_at      REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, created_at);
"""


@dataclass
class Repo:
    id: str
    name: str
    url: Optional[str]
    variant: str
    collection: str
    repo_path: Optional[str]
    commit_sha: Optional[str]
    status: str
    stage: Optional[str]
    error: Optional[str]
    files_indexed: int
    chunks: int
    size_bytes: int
    languages: Dict[str, int]
    embedding_provider: Optional[str]
    embedding_model: Optional[str]
    created_at: float
    updated_at: float
    indexed_at: Optional[float]

    @property
    def ready(self) -> bool:
        return self.status == READY


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: the API reads from a threadpool and background
    # indexing writes from another thread. Writes are short and serialised by
    # SQLite's own locking; WAL keeps readers from blocking on them.
    connection = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


class Registry:
    def __init__(self) -> None:
        self.connection = _connect()
        self.connection.executescript(SCHEMA)
        self._migrate()
        self.connection.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a database was first created."""
        existing = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(repos)").fetchall()
        }
        for column in ("embedding_provider", "embedding_model"):
            if column not in existing:
                self.connection.execute(f"ALTER TABLE repos ADD COLUMN {column} TEXT")

    # --- repos -------------------------------------------------------------

    def _row_to_repo(self, row: sqlite3.Row) -> Repo:
        return Repo(
            id=row["id"],
            name=row["name"],
            url=row["url"],
            variant=row["variant"],
            collection=row["collection"],
            repo_path=row["repo_path"],
            commit_sha=row["commit_sha"],
            status=row["status"],
            stage=row["stage"],
            error=row["error"],
            files_indexed=row["files_indexed"] or 0,
            chunks=row["chunks"] or 0,
            size_bytes=row["size_bytes"] or 0,
            languages=json.loads(row["languages"]) if row["languages"] else {},
            embedding_provider=row["embedding_provider"],
            embedding_model=row["embedding_model"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            indexed_at=row["indexed_at"],
        )

    def create_repo(
        self,
        name: str,
        variant: str,
        collection: str,
        url: Optional[str] = None,
        repo_path: Optional[str] = None,
        status: str = QUEUED,
    ) -> Repo:
        now = time.time()
        repo_id = uuid.uuid4().hex[:12]
        self.connection.execute(
            "INSERT INTO repos (id, name, url, variant, collection, repo_path, "
            "status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (repo_id, name, url, variant, collection, repo_path, status, now, now),
        )
        self.connection.commit()
        return self.get_repo(repo_id)  # type: ignore[return-value]

    def upsert_indexed_repo(self, **fields: Any) -> Repo:
        """Record a repo indexed outside the API (the CLI path).

        Keyed on collection so re-running the CLI updates in place rather than
        accumulating duplicate rows.
        """
        existing = self.get_by_collection(fields["collection"])
        if existing:
            self.update_repo(existing.id, **fields)
            return self.get_repo(existing.id)  # type: ignore[return-value]

        repo = self.create_repo(
            name=fields.pop("name"),
            variant=fields.pop("variant"),
            collection=fields["collection"],
            url=fields.pop("url", None),
            repo_path=fields.pop("repo_path", None),
            status=fields.pop("status", READY),
        )
        fields.pop("collection", None)
        if fields:
            self.update_repo(repo.id, **fields)
        return self.get_repo(repo.id)  # type: ignore[return-value]

    def update_repo(self, repo_id: str, **fields: Any) -> None:
        if not fields:
            return
        if "languages" in fields and isinstance(fields["languages"], dict):
            fields["languages"] = json.dumps(fields["languages"])
        fields["updated_at"] = time.time()

        assignments = ", ".join(f"{key} = ?" for key in fields)
        self.connection.execute(
            f"UPDATE repos SET {assignments} WHERE id = ?",
            (*fields.values(), repo_id),
        )
        self.connection.commit()

    def get_repo(self, repo_id: str) -> Optional[Repo]:
        row = self.connection.execute(
            "SELECT * FROM repos WHERE id = ?", (repo_id,)
        ).fetchone()
        return self._row_to_repo(row) if row else None

    def get_by_collection(self, collection: str) -> Optional[Repo]:
        row = self.connection.execute(
            "SELECT * FROM repos WHERE collection = ?", (collection,)
        ).fetchone()
        return self._row_to_repo(row) if row else None

    def find_repo(self, name: str, variant: Optional[str] = None) -> Optional[Repo]:
        if variant:
            row = self.connection.execute(
                "SELECT * FROM repos WHERE name = ? AND variant = ?", (name, variant)
            ).fetchone()
        else:
            row = self.connection.execute(
                "SELECT * FROM repos WHERE name = ? ORDER BY updated_at DESC LIMIT 1",
                (name,),
            ).fetchone()
        return self._row_to_repo(row) if row else None

    def list_repos(self) -> List[Repo]:
        rows = self.connection.execute(
            "SELECT * FROM repos ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_repo(row) for row in rows]

    def delete_repo(self, repo_id: str) -> None:
        self.connection.execute("DELETE FROM repos WHERE id = ?", (repo_id,))
        self.connection.commit()

    # --- conversations -----------------------------------------------------

    def create_conversation(self, repo_id: str, title: Optional[str] = None) -> str:
        conversation_id = uuid.uuid4().hex[:12]
        self.connection.execute(
            "INSERT INTO conversations (id, repo_id, title, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, repo_id, title, time.time()),
        )
        self.connection.commit()
        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        row = self.connection.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        return dict(row) if row else None

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        rewritten_query: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> str:
        message_id = uuid.uuid4().hex[:12]
        self.connection.execute(
            "INSERT INTO messages (id, conversation_id, role, content, "
            "rewritten_query, trace, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                message_id,
                conversation_id,
                role,
                content,
                rewritten_query,
                json.dumps(trace) if trace is not None else None,
                time.time(),
            ),
        )
        self.connection.commit()
        return message_id

    def messages(self, conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM messages WHERE conversation_id = ? "
            "ORDER BY created_at LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        out = []
        for row in rows:
            message = dict(row)
            message["trace"] = json.loads(message["trace"]) if message["trace"] else None
            out.append(message)
        return out

    def history_turns(
        self, conversation_id: str, max_turns: int = 6
    ) -> List[Dict[str, str]]:
        """Recent turns as {role, content}, oldest first.

        Bounded deliberately: the rewriter only needs enough context to resolve a
        pronoun, and an unbounded history would grow the prompt without improving
        that.
        """
        rows = self.connection.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (conversation_id, max_turns),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def close(self) -> None:
        self.connection.close()


_registry: Optional[Registry] = None


def get_registry() -> Registry:
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry


def import_manifests() -> int:
    """Backfill the registry from manifests written before it existed."""
    from .manifests import all_manifests

    registry = get_registry()
    imported = 0
    for manifest in all_manifests():
        collection = manifest.get("collection")
        if not collection or registry.get_by_collection(collection):
            continue
        registry.upsert_indexed_repo(
            name=manifest.get("repo", collection),
            variant=manifest.get("variant") or manifest.get("strategy", ""),
            collection=collection,
            repo_path=manifest.get("repo_path"),
            commit_sha=manifest.get("commit_sha"),
            status=READY,
            files_indexed=manifest.get("files_indexed", 0),
            chunks=manifest.get("chunks", 0),
            embedding_provider=manifest.get("embedding_provider"),
            embedding_model=manifest.get("embedding_model"),
            indexed_at=time.time(),
        )
        imported += 1
    return imported
