"""Retrieval-only search against an indexed repo.

Usage:
    uv run python query.py "where is APIRouter defined?"
    uv run python query.py --mode semantic "how do I upload a file?"
"""

import argparse

from pipeline import config
from pipeline.embeddings import EmbeddingManager
from pipeline.lexical_index import LexicalIndex
from pipeline.retriever import HybridRetriever
from pipeline.vector_store import VectorStore, collection_name_for_repo


def build_retriever(
    repo: str,
    variant: str,
    mode: str = config.RETRIEVAL_MODE,
    semantic_weight: float = config.SEMANTIC_WEIGHT,
    lexical_weight: float = config.LEXICAL_WEIGHT,
    rrf_k: int = config.RRF_K,
) -> HybridRetriever:
    collection = collection_name_for_repo(repo, variant)
    return HybridRetriever(
        VectorStore(collection_name=collection),
        EmbeddingManager(),
        lexical_index=LexicalIndex(collection) if mode != "semantic" else None,
        mode=mode,
        semantic_weight=semantic_weight,
        lexical_weight=lexical_weight,
        rrf_k=rrf_k,
    )


def format_location(metadata: dict) -> str:
    location = metadata.get("path", metadata.get("source", "unknown"))
    start, end = metadata.get("start_line"), metadata.get("end_line")
    return f"{location}:{start}-{end}" if start else str(location)


def main():
    parser = argparse.ArgumentParser(description="Search an indexed repo")
    parser.add_argument("question", nargs="+", help="The search query")
    parser.add_argument("--repo", default="fastapi", help="Indexed repo name")
    parser.add_argument("--variant", default="astcode-openai", help="Collection suffix")
    parser.add_argument(
        "--mode", default=config.RETRIEVAL_MODE, choices=["semantic", "lexical", "hybrid"]
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    retriever = build_retriever(args.repo, args.variant, args.mode)
    results = retriever.retrieve(" ".join(args.question), top_k=args.top_k)

    if not results:
        print("No results found.")
        return

    for result in results:
        # Retrieval trace: which path surfaced this chunk, and at what rank.
        trace = ", ".join(
            f"{name}#{result[f'{name}_rank']}"
            for name in ("semantic", "lexical")
            if result.get(f"{name}_rank")
        )
        symbol = result["metadata"].get("qualified_symbol")
        suffix = f"  [{symbol}]" if symbol else ""
        print(
            f"[{result['rank']}] score={result['score']:.4f}  "
            f"{format_location(result['metadata'])}{suffix}"
        )
        print(f"      via {trace or 'n/a'}")
        print("      " + result["content"][:200].strip().replace("\n", "\n      "))
        print("-" * 80)


if __name__ == "__main__":
    main()
