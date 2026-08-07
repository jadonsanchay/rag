"""Retrieve context from an indexed repo and generate an answer with OpenAI.

Usage:
    uv run python ask.py --repo fastapi "how are dependencies resolved?"
"""

import argparse

from pipeline.generator import AnswerGenerator
from pipeline.retriever import RAGRetriever
from pipeline.embeddings import EmbeddingManager
from pipeline.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Ask a question about an indexed repo")
    parser.add_argument("question", nargs="+", help="The question to answer")
    parser.add_argument("--repo", default="fastapi", help="Indexed repo name")
    parser.add_argument(
        "--variant",
        default="astcode-openai",
        help="Collection suffix produced by index_repo.py",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    query = " ".join(args.question)

    embedding_manager = EmbeddingManager()
    vector_store = VectorStore.for_repo(args.repo, args.variant)
    retriever = RAGRetriever(vector_store, embedding_manager)

    results = retriever.retrieve(query, top_k=args.top_k)
    if not results:
        print("No results found.")
        return

    generator = AnswerGenerator()
    answer = generator.generate(query, results)

    print("Answer:\n")
    print(answer)
    print("\nSources:")
    for doc in results:
        meta = doc["metadata"]
        location = meta.get("path", meta.get("source", "unknown"))
        lines = ""
        if meta.get("start_line"):
            lines = f":{meta['start_line']}-{meta.get('end_line', '')}"
        print(f"[{doc['rank']}] {location}{lines} (score={doc['score']:.4f})")


if __name__ == "__main__":
    main()
