"""Retrieve context from an indexed repo and generate a cited answer.

Usage:
    uv run python ask.py "how are dependencies resolved?"
"""

import argparse

from pipeline.generator import REFUSAL_TOKEN, AnswerGenerator
from pipeline.config import RETRIEVAL_MODE
from pipeline.manifests import repo_root_for
from query import build_retriever, format_location


def main():
    parser = argparse.ArgumentParser(description="Ask a question about an indexed repo")
    parser.add_argument("question", nargs="+", help="The question to answer")
    parser.add_argument("--repo", default="fastapi", help="Indexed repo name")
    parser.add_argument("--variant", default="astcode-cards", help="Collection suffix")
    parser.add_argument(
        "--mode", default=RETRIEVAL_MODE, choices=["semantic", "lexical", "hybrid"]
    )
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    query = " ".join(args.question)
    retriever = build_retriever(args.repo, args.variant, args.mode)
    results = retriever.retrieve(query, top_k=args.top_k)

    # Resolve from the manifest: the repo need not live under REPOS_DIR.
    repo_root = repo_root_for(args.repo, args.variant)
    answer = AnswerGenerator().generate(query, results, repo_root=repo_root)

    if answer.refused:
        print("Not answerable from this repository.")
        detail = answer.text[len(REFUSAL_TOKEN):].strip()
        if detail:
            print(f"  {detail}")
        return

    print(answer.text)

    print("\nSources:")
    checks = {check.index: check for check in (answer.verification.checks if answer.verification else [])}
    for index, doc in enumerate(answer.sources, start=1):
        cited = index in answer.cited_indices
        check = checks.get(index)
        if not cited:
            mark = "  "  # retrieved but not used by the answer
        elif check and check.ok:
            mark = "OK"
        else:
            mark = "!!"
        note = f"  <- {check.problem}" if check and check.problem else ""
        print(f" {mark} [{index}] {format_location(doc['metadata'])}{note}")

    if answer.verification:
        print(f"\nCitations: {answer.verification.summary()}")
        if answer.verification.fabricated_indices:
            print("  WARNING: answer cited source numbers that were never provided")


if __name__ == "__main__":
    main()
