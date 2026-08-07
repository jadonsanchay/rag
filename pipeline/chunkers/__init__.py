from typing import Callable, List, Optional, Sequence

from langchain_core.documents import Document

from .. import config
from .base import Chunk, Chunker
from .python_ast import PythonASTChunker
from .text import TextChunker, default_token_counter

__all__ = [
    "Chunk",
    "Chunker",
    "PythonASTChunker",
    "TextChunker",
    "chunk_documents",
    "get_chunker",
]


def get_chunker(
    language: str,
    max_tokens: int,
    count_tokens: Optional[Callable[[str], int]] = None,
) -> Chunker:
    """Dispatch by language. Step 9 registers tree-sitter chunkers here."""
    if language == "python":
        return PythonASTChunker(max_tokens=max_tokens, count_tokens=count_tokens)
    return TextChunker(language=language, max_tokens=max_tokens, count_tokens=count_tokens)


def chunk_documents(
    documents: Sequence[Document],
    token_limit: int = 8191,
    count_tokens: Optional[Callable[[str], int]] = None,
    target_prose_tokens: int = config.TARGET_PROSE_TOKENS,
    max_code_tokens: int = config.MAX_CODE_CHUNK_TOKENS,
) -> List[Document]:
    """Chunk file-level Documents structurally, preserving their metadata.

    Code and prose get different budgets: code chunks stay whole up to
    max_code_tokens because a split function is harder to reason about, while
    prose is windowed at the smaller target size for retrieval precision.
    Both are capped by the model's real limit so nothing is truncated.
    """
    counter = count_tokens or default_token_counter
    code_budget = min(max_code_tokens, token_limit)
    prose_budget = min(target_prose_tokens, token_limit)

    chunkers: dict = {}
    chunked: List[Document] = []

    for document in documents:
        language = document.metadata.get("language", "text")
        if language not in chunkers:
            budget = code_budget if language == "python" else prose_budget
            chunkers[language] = get_chunker(language, budget, counter)

        path = document.metadata.get("path", document.metadata.get("source", "unknown"))
        base_metadata = {
            k: v for k, v in document.metadata.items() if k not in {"path", "language"}
        }

        for chunk in chunkers[language].chunk(document.page_content, path):
            chunked.append(chunk.to_document(base_metadata))

    return chunked
