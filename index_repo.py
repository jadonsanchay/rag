"""Index a code repository into the vector store.

Usage:
    uv run python index_repo.py repos/fastapi --exclude docs_src
"""

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from pipeline import config
from pipeline.embeddings import EmbeddingManager
from pipeline.ids import chunk_ids_for
from pipeline.lexical_index import LexicalIndex
from pipeline.repo_loader import language_stats, load_repo_documents
from pipeline.splitter import split_documents
from pipeline.vector_store import VectorStore, collection_name_for_repo

MANIFEST_DIR = config.DATA_DIR / "index_manifests"


def git_sha(repo_path: Path) -> str:
    """Record which commit was indexed, so eval labels don't silently rot."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def report_duplicates(chunks: List[Document]) -> dict:
    """Exact-duplicate chunks waste top-k slots: a query can burn its whole
    context budget on the same text retrieved several times."""
    texts = [chunk.page_content for chunk in chunks]
    unique = len(set(texts))
    duplicates = len(texts) - unique
    stats = {
        "total_chunks": len(texts),
        "unique_chunks": unique,
        "duplicate_chunks": duplicates,
        "pct_duplicate": round(100 * duplicates / len(texts), 1) if texts else 0.0,
    }
    print(f"\nDuplicate chunks: {duplicates} of {len(texts)} ({stats['pct_duplicate']}%)")
    if duplicates:
        print("  -> duplicates compete for the same top-k slots at query time")
    return stats


def report_token_budget(chunks: List[Document], embedder: EmbeddingManager) -> dict:
    """How many chunks exceed the embedding model's input limit. Chunks over
    the limit are truncated by the model, losing content silently. Checks every
    chunk, not a sample, so a single oversized outlier cannot hide."""
    limit = embedder.token_limit
    token_counts = [embedder.count_tokens(chunk.page_content) for chunk in chunks]
    over = [count for count in token_counts if count > limit]
    mean = sum(token_counts) / len(token_counts) if token_counts else 0

    stats = {
        "token_limit": limit,
        "inspected_chunks": len(token_counts),
        "chunks_over_limit": len(over),
        "pct_over_limit": round(100 * len(over) / len(token_counts), 1) if token_counts else 0.0,
        "max_tokens_seen": max(token_counts) if token_counts else 0,
        "mean_tokens": round(mean, 1),
        "total_tokens": sum(token_counts),
    }

    print(f"\nToken budget (model limit {limit}, all {len(token_counts)} chunks):")
    print(f"  over limit: {stats['chunks_over_limit']} ({stats['pct_over_limit']}%)")
    print(f"  mean {stats['mean_tokens']} tokens, largest {stats['max_tokens_seen']}")
    if stats["chunks_over_limit"]:
        print("  -> these are truncated at embed time; content past the limit is lost")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Index a code repo for RAG")
    parser.add_argument("repo_path", type=Path, help="Path to the repo to index")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Repo-relative path prefix to exclude (repeatable)",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Restrict indexing to these repo-relative prefixes (repeatable)",
    )
    parser.add_argument(
        "--strategy",
        default=config.CHUNK_STRATEGY,
        choices=["text", "ast", "ast-code"],
        help="text=baseline everywhere, ast=structural everywhere, "
        "ast-code=structural for code only (isolates the code-chunking effect)",
    )
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument(
        "--provider",
        default=config.EMBEDDING_PROVIDER,
        choices=["sentence-transformers", "openai"],
    )
    parser.add_argument("--model", default=None, help="Override the embedding model")
    parser.add_argument(
        "--variant",
        default=None,
        help="Collection suffix for this experiment (defaults to --strategy)",
    )
    parser.add_argument(
        "--cards",
        action=argparse.BooleanOptionalAction,
        default=config.INDEX_CARDS,
        help="Index structural file/package cards (--no-cards to reproduce step 4)",
    )
    args = parser.parse_args()

    repo_path = args.repo_path.resolve()

    documents, stats = load_repo_documents(repo_path, args.exclude, args.include)
    print(stats.report())
    if not documents:
        print("No indexable files found.")
        return

    print("\nLanguages:")
    for language, count in language_stats(documents).most_common(10):
        print(f"  {language}: {count}")

    # Build the embedder first: the chunkers size chunks to its token limit
    # rather than guessing, so a model swap cannot silently truncate.
    embedder = EmbeddingManager(model_name=args.model, provider=args.provider)
    print(f"\nEmbedding model: {embedder.model_name} (limit {embedder.token_limit} tokens)")

    if args.strategy == "text":
        chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)
    else:
        from pipeline.chunkers import chunk_documents

        structural_docs = documents
        chunks = []
        if args.strategy == "ast-code":
            # Hold prose chunking at the baseline so any metric change is
            # attributable to code chunking alone. "Code" is every language with a
            # structural chunker, not just Python (step 9).
            structural_docs = [
                d for d in documents
                if d.metadata.get("language") in config.CODE_LANGUAGES
            ]
            prose_docs = [
                d for d in documents
                if d.metadata.get("language") not in config.CODE_LANGUAGES
            ]
            chunks += split_documents(prose_docs, args.chunk_size, args.chunk_overlap)

        chunks += chunk_documents(
            structural_docs,
            token_limit=embedder.token_limit,
            count_tokens=embedder.count_tokens,
        )
    print(f"Split {len(documents)} files into {len(chunks)} chunks ({args.strategy})")

    if args.cards:
        from pipeline.cards import build_cards

        cards = build_cards(
            documents,
            max_tokens=config.TARGET_PROSE_TOKENS,
            count_tokens=embedder.count_tokens,
            # Non-Python languages derive their cards from chunk symbols rather
            # than a second parse.
            file_chunks=chunks,
        )
        kinds = Counter(card.metadata.get("kind") for card in cards)
        chunks += cards
        print(f"Added {len(cards)} structural cards ({dict(kinds)})")

    duplicates = report_duplicates(chunks)
    budget = report_token_budget(chunks, embedder)

    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedder.generate_embeddings(texts)

    variant = args.variant or args.strategy
    store = VectorStore.for_repo(repo_path.name, variant)
    store.reset()

    # Shared ids let rank fusion merge the two indexes.
    ids = chunk_ids_for(chunks)
    store.add_documents(chunks, embeddings, ids=ids)
    print(f"\nCollection '{store.collection_name}' now has {store.count()} chunks")

    lexical = LexicalIndex(store.collection_name)
    lexical.reset()
    lexical.add_documents(chunks, ids)
    print(f"Lexical index '{lexical.db_path.name}' now has {lexical.count()} chunks")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repo": repo_path.name,
        "repo_path": str(repo_path),
        "commit_sha": git_sha(repo_path),
        "collection": store.collection_name,
        "strategy": args.strategy,
        "variant": variant,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "embedding_provider": args.provider,
        "embedding_model": embedder.model_name,
        "token_limit": embedder.token_limit,
        "include_only": args.include,
        "excluded": args.exclude,
        "cards": args.cards,
        "files_indexed": len(documents),
        "chunks": len(chunks),
        "skipped": dict(stats.skipped),
        "token_budget": budget,
        "duplicates": duplicates,
    }
    manifest_path = MANIFEST_DIR / f"{store.collection_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path.relative_to(config.BASE_DIR)}")


if __name__ == "__main__":
    main()
