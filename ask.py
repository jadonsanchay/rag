"""Retrieve context from the vector store and generate an answer with OpenAI."""

import sys

from pipeline.embeddings import EmbeddingManager
from pipeline.generator import AnswerGenerator
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

    generator = AnswerGenerator()
    answer = generator.generate(query, results)

    print("Answer:\n")
    print(answer)
    print("\nSources:")
    for doc in results:
        print(f"[{doc['rank']}] {doc['metadata'].get('source')} (score={doc['score']:.4f})")


if __name__ == "__main__":
    main()
