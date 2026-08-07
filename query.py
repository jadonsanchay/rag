"""Retrieval-only search against an indexed repo.

Usage:
    uv run python query.py --repo fastapi "where is APIRouter defined?"
"""

import argparse

from pipeline.embeddings import EmbeddingManager
from pipeline.retriever import RAGRetriever
from pipeline.vector_store import VectorStore


def build_retriever(repo: str, variant: str) -> RAGRetriever:
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore.for_repo(repo, variant)
    return RAGRetriever(vector_store, embedding_manager)


def main():
    parser = argparse.ArgumentParser(description="Search an indexed repo")
    parser.add_argument("question", nargs="+", help="The search query")
    parser.add_argument("--repo", default="fastapi", help="Indexed repo name")
    parser.add_argument(
        "--variant",
        default="astcode-openai",
        help="Collection suffix produced by index_repo.py",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    query = " ".join(args.question)
    retriever = build_retriever(args.repo, args.variant)
    results = retriever.retrieve(query, top_k=args.top_k)

    if not results:
        print("No results found.")
        return

    for result in results:
        meta = result["metadata"]
        location = meta.get("path", meta.get("source", "unknown"))
        lines = ""
        if meta.get("start_line"):
            lines = f":{meta['start_line']}-{meta.get('end_line', '')}"
        print(f"[{result['rank']}] score={result['score']:.4f}  {location}{lines}")
        print(result["content"][:300].strip())
        print("-" * 80)


if __name__ == "__main__":
    main()
