"""Query the vector store built by ingest.py."""

import sys

from pipeline.embeddings import EmbeddingManager
from pipeline.retriever import RAGRetriever
from pipeline.vector_store import VectorStore


def main():
    query = " ".join(sys.argv[1:]) or "What is attention is all you need?"

    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    retriever = RAGRetriever(vector_store, embedding_manager)

    results = retriever.retrieve(query, top_k=5)
    if not results:
        print("No results found.")
        return

    for result in results:
        source = result["metadata"].get("source")
        print(f"[{result['rank']}] score={result['score']:.4f} source={source}")
        print(result["content"][:300])
        print("-" * 80)


if __name__ == "__main__":
    main()
