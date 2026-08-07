"""Retrieve context from an indexed repo and generate an answer with OpenAI.

Usage:
    uv run python ask.py "how are dependencies resolved?"
"""

import argparse

from pipeline import config
from pipeline.generator import AnswerGenerator
from query import build_retriever, format_location


def main():
    parser = argparse.ArgumentParser(description="Ask a question about an indexed repo")
    parser.add_argument("question", nargs="+", help="The question to answer")
    parser.add_argument("--repo", default="fastapi", help="Indexed repo name")
    parser.add_argument("--variant", default="astcode-openai", help="Collection suffix")
    parser.add_argument(
        "--mode", default=config.RETRIEVAL_MODE, choices=["semantic", "lexical", "hybrid"]
    )
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    query = " ".join(args.question)
    retriever = build_retriever(args.repo, args.variant, args.mode)
    results = retriever.retrieve(query, top_k=args.top_k)

    if not results:
        print("No results found.")
        return

    answer = AnswerGenerator().generate(query, results)

    print("Answer:\n")
    print(answer)
    print("\nSources:")
    for doc in results:
        via = ",".join(doc.get("sources") or [])
        print(f"[{doc['rank']}] {format_location(doc['metadata'])}  (via {via})")


if __name__ == "__main__":
    main()
