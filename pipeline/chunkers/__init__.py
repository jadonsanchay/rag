from typing import Callable, List, Optional, Sequence

from langchain_core.documents import Document

from .. import config
from .base import Chunk, Chunker
from .python_ast import PythonASTChunker
from .text import TextChunker, default_token_counter
from .tree_sitter_chunker import SPECS as TREE_SITTER_SPECS
from .tree_sitter_chunker import TreeSitterChunker, supported_languages

__all__ = [
    "Chunk",
    "Chunker",
    "PythonASTChunker",
    "TextChunker",
    "TreeSitterChunker",
    "chunk_documents",
    "get_chunker",
    "structural_languages",
    "supported_languages",
]


def structural_languages() -> list:
    """Languages that get structural (declaration-aware) chunking."""
    return sorted({"python", *TREE_SITTER_SPECS})


def get_chunker(
    language: str,
    max_tokens: int,
    count_tokens: Optional[Callable[[str], int]] = None,
) -> Chunker:
    """Dispatch by language: stdlib ast for Python, tree-sitter where a grammar
    is registered, line-aware text windows otherwise."""
    if language == "python":
        return PythonASTChunker(max_tokens=max_tokens, count_tokens=count_tokens)
    if language in TREE_SITTER_SPECS:
        return TreeSitterChunker(
            language=language, max_tokens=max_tokens, count_tokens=count_tokens
        )
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
            # Code gets the larger budget so a whole function usually survives
            # intact; prose is windowed smaller for retrieval precision.
            is_code = language in config.CODE_LANGUAGES
            chunkers[language] = get_chunker(
                language, code_budget if is_code else prose_budget, counter
            )

        path = document.metadata.get("path", document.metadata.get("source", "unknown"))
        base_metadata = {
            k: v for k, v in document.metadata.items() if k not in {"path", "language"}
        }

        for chunk in chunkers[language].chunk(document.page_content, path):
            chunked.append(chunk.to_document(base_metadata))

    return chunked
