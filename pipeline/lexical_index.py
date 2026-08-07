import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from . import config
from .ids import document_symbol_text

# BM25 column weights. Symbol matches are the strongest signal for
# "where is X defined" questions; path is a decent proxy; content is baseline.
BM25_WEIGHTS = (0.0, 2.0, 5.0, 1.0, 0.0)  # chunk_id, path, symbol, content, meta

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Question scaffolding carries no retrieval signal and only dilutes BM25.
STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "between", "but", "by",
    "called", "can", "code", "define", "defined", "definition", "did", "do", "does",
    "for", "from", "get", "give", "handle", "handled", "handles", "has", "have", "how",
    "i", "if", "implement", "implemented", "in", "into", "is", "it", "its", "me",
    "of", "on", "one", "or", "s", "same", "single", "so", "that", "the", "their",
    "them", "then", "there", "these", "this", "to", "turned", "up", "use", "used",
    "uses", "using", "was", "what", "when", "where", "which", "who", "why", "will",
    "with", "within", "work", "works", "would", "you", "your",
}


def split_identifier(token: str) -> List[str]:
    """`include_router` -> [include, router]; `APIRouter` -> [APIRouter].

    Underscores are split because FTS5's unicode61 tokenizer treats `_` as a
    separator, so that is how the content was indexed. CamelCase is NOT split,
    because `APIRouter` was indexed as the single token `apirouter`.
    """
    return [part for part in token.split("_") if part]


class LexicalIndex:
    """SQLite FTS5 index over chunks, for exact identifier lookup.

    Embeddings are weak at exact symbol names: a query for `include_router`
    is semantically almost identical to any other router method. Lexical search
    covers precisely that gap.
    """

    COLUMNS = ("chunk_id", "path", "symbol", "content", "meta")

    def __init__(self, collection_name: str, directory: Optional[Path] = None):
        self.collection_name = collection_name
        base = directory or config.LEXICAL_INDEX_DIR
        base.mkdir(parents=True, exist_ok=True)
        self.db_path = base / f"{collection_name}.sqlite3"
        self.connection = sqlite3.connect(self.db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
            "chunk_id UNINDEXED, path, symbol, content, meta UNINDEXED, "
            "tokenize='unicode61')"
        )
        self.connection.commit()

    def reset(self) -> None:
        self.connection.execute("DROP TABLE IF EXISTS chunks_fts")
        self.connection.commit()
        self._ensure_schema()

    def add_documents(self, documents: Sequence[Document], ids: Sequence[str]) -> None:
        if len(documents) != len(ids):
            raise ValueError("Number of documents must match number of ids")

        rows = [
            (
                chunk_id,
                str(doc.metadata.get("path", "")),
                document_symbol_text(doc),
                doc.page_content,
                json.dumps(doc.metadata),
            )
            for chunk_id, doc in zip(ids, documents)
        ]
        self.connection.executemany(
            "INSERT INTO chunks_fts (chunk_id, path, symbol, content, meta) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self.connection.commit()

    def count(self) -> int:
        cursor = self.connection.execute("SELECT count(*) FROM chunks_fts")
        return cursor.fetchone()[0]

    @staticmethod
    def build_query(question: str) -> str:
        """Turn a natural-language question into an FTS5 MATCH expression.

        Terms are OR'd so a single rare identifier can carry a match; BM25's IDF
        keeps common words from dominating. Multi-part identifiers also become
        phrase queries, which is how snake_case names match precisely.
        """
        terms: List[str] = []
        for raw in IDENTIFIER.findall(question):
            if raw.lower() in STOPWORDS or len(raw) < 2:
                continue

            parts = split_identifier(raw)
            if len(parts) > 1:
                # Adjacent tokens: matches how `include_router` was indexed.
                terms.append('"' + " ".join(parts) + '"')
                terms.extend(p for p in parts if p.lower() not in STOPWORDS)
            else:
                terms.append(raw)
                # Also try the de-camelised form for prose mentions.
                split_camel = CAMEL_BOUNDARY.split(raw)
                if len(split_camel) > 1:
                    terms.append('"' + " ".join(split_camel) + '"')

        unique = list(dict.fromkeys(terms))
        return " OR ".join(unique)

    def search(self, question: str, limit: int = 20) -> List[Dict[str, Any]]:
        match_query = self.build_query(question)
        if not match_query:
            return []

        weights = ", ".join(str(w) for w in BM25_WEIGHTS)
        sql = (
            "SELECT chunk_id, content, meta, bm25(chunks_fts, "
            f"{weights}) AS score "
            "FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY score LIMIT ?"
        )
        try:
            cursor = self.connection.execute(sql, (match_query, limit))
        except sqlite3.OperationalError:
            # Malformed MATCH expression: fail closed rather than break the query.
            return []

        results = []
        for chunk_id, content, meta, score in cursor.fetchall():
            results.append(
                {
                    "id": chunk_id,
                    "content": content,
                    "metadata": json.loads(meta),
                    # bm25() returns negative values, lower being better.
                    "lexical_score": -score,
                }
            )
        return results

    def close(self) -> None:
        self.connection.close()
