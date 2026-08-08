"""Retrieval-only search against an indexed repo.

Usage:
    uv run python query.py "where is APIRouter defined?"
    uv run python query.py --mode semantic "how do I upload a file?"
"""

import argparse
from typing import Optional

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
    max_per_file: int = config.MAX_CHUNKS_PER_FILE,
    stratify: bool = config.STRATIFY_RETRIEVAL,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> HybridRetriever:
    collection = collection_name_for_repo(repo, variant)
    # The query must be embedded by the model that built the index. Querying a
    # 384-dim MiniLM collection with a 1536-dim OpenAI vector fails outright, and
    # a same-dimension mismatch would fail silently as bad results.
    return HybridRetriever(
        VectorStore(collection_name=collection),
        EmbeddingManager(model_name=model, provider=provider or config.EMBEDDING_PROVIDER),
        lexical_index=LexicalIndex(collection) if mode != "semantic" else None,
        mode=mode,
        semantic_weight=semantic_weight,
        lexical_weight=lexical_weight,
        rrf_k=rrf_k,
        candidate_k=config.CANDIDATE_K,
        max_per_file=max_per_file,
        stratify=stratify,
        code_weight=config.CODE_WEIGHT,
        prose_weight=config.PROSE_WEIGHT,
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
        # Retrieval trace: which list surfaced this chunk, and at what rank.
        # Stratified retrieval reports per-stratum ranks; pooled reports two.
        ranks = result.get("ranks") or {
            name: result[f"{name}_rank"]
            for name in ("semantic", "lexical")
            if result.get(f"{name}_rank")
        }
        trace = ", ".join(f"{name}#{rank}" for name, rank in sorted(ranks.items()))
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
