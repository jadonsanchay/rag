from typing import Any, Dict, List

from .embeddings import EmbeddingManager
from .vector_store import VectorStore


class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Retrieve the most relevant document chunks for a query"""
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        retrieved_docs: List[Dict[str, Any]] = []
        documents = results.get("documents") or [[]]
        if not documents or not documents[0]:
            return retrieved_docs

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        for rank, (doc_id, document, metadata, distance) in enumerate(
            zip(ids, documents[0], metadatas, distances), start=1
        ):
            similarity_score = 1 - distance
            if similarity_score >= score_threshold:
                retrieved_docs.append(
                    {
                        "id": doc_id,
                        "content": document,
                        "metadata": metadata,
                        "score": similarity_score,
                        "distance": distance,
                        "rank": rank,
                    }
                )

        return retrieved_docs
