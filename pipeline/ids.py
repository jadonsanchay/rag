import hashlib
from typing import Any, Dict

from langchain_core.documents import Document


def make_chunk_id(metadata: Dict[str, Any], ordinal: int) -> str:
    """Stable id for a chunk, shared by the vector store and the lexical index.

    Fusion can only merge two ranked lists if both refer to chunks by the same
    identity, so ids must be deterministic rather than random per-run. Chunking
    is deterministic, so the ordinal is stable across re-indexes.
    """
    key = "|".join(
        str(metadata.get(field, ""))
        for field in ("path", "start_line", "end_line", "part")
    )
    digest = hashlib.sha1(f"{key}|{ordinal}".encode()).hexdigest()[:16]
    return f"c{ordinal:06d}_{digest}"


def chunk_ids_for(documents) -> list:
    return [make_chunk_id(doc.metadata, i) for i, doc in enumerate(documents)]


def document_symbol_text(document: Document) -> str:
    """Symbol text worth weighting heavily in lexical search."""
    metadata = document.metadata
    parts = [
        metadata.get("qualified_symbol"),
        metadata.get("symbol"),
        metadata.get("parent_symbol"),
    ]
    return " ".join(dict.fromkeys(p for p in parts if p))
