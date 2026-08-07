from typing import Any, Dict, List, Optional

from .embeddings import EmbeddingManager
from .fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from .lexical_index import LexicalIndex
from .vector_store import VectorStore

SEMANTIC = 0
LEXICAL = 1


class RAGRetriever:
    """Semantic-only retrieval. Kept as the Phase A baseline."""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self, query: str, top_k: int = 5, score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
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


class HybridRetriever:
    """Fuses semantic and lexical retrieval.

    Embeddings handle paraphrase ("how are deps resolved") but are weak on exact
    identifiers, because `include_router` and every other router method occupy
    almost the same semantic neighbourhood. BM25 handles identifiers precisely
    but cannot paraphrase. Rank fusion takes both without needing their scores
    to be on a comparable scale.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
        lexical_index: Optional[LexicalIndex] = None,
        mode: str = "hybrid",
        semantic_weight: float = 1.0,
        lexical_weight: float = 1.0,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_k: int = 40,
        max_per_file: Optional[int] = None,
    ):
        if mode not in {"semantic", "lexical", "hybrid"}:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        if mode in {"lexical", "hybrid"} and lexical_index is None:
            raise ValueError(f"mode={mode} requires a lexical_index")

        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.lexical_index = lexical_index
        self.mode = mode
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        # Cap chunks per file. Without it a single doc page can take every slot
        # in the context window, which wastes the generator's budget on one
        # source and hides the other files that answer the question.
        self.max_per_file = max_per_file

    def _semantic_candidates(self, query: str, limit: int) -> List[Dict[str, Any]]:
        embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[embedding.tolist()], n_results=limit
        )
        documents = results.get("documents") or [[]]
        if not documents or not documents[0]:
            return []

        return [
            {
                "id": doc_id,
                "content": content,
                "metadata": metadata,
                "semantic_score": 1 - distance,
            }
            for doc_id, content, metadata, distance in zip(
                results["ids"][0], documents[0], results["metadatas"][0], results["distances"][0]
            )
        ]

    def retrieve(
        self, query: str, top_k: int = 5, score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        limit = max(self.candidate_k, top_k)

        semantic = (
            self._semantic_candidates(query, limit)
            if self.mode in {"semantic", "hybrid"}
            else []
        )
        lexical = (
            self.lexical_index.search(query, limit)
            if self.mode in {"lexical", "hybrid"}
            else []
        )

        # Either source can supply a chunk's text and metadata.
        lookup: Dict[str, Dict[str, Any]] = {}
        for candidate in lexical + semantic:
            lookup.setdefault(candidate["id"], candidate).update(
                {k: v for k, v in candidate.items() if k.endswith("_score")}
            )

        if self.mode == "semantic":
            ordered = [(c["id"], c["semantic_score"], {SEMANTIC: i}) for i, c in enumerate(semantic, 1)]
        elif self.mode == "lexical":
            ordered = [(c["id"], c["lexical_score"], {LEXICAL: i}) for i, c in enumerate(lexical, 1)]
        else:
            ordered = reciprocal_rank_fusion(
                [[c["id"] for c in semantic], [c["id"] for c in lexical]],
                weights=[self.semantic_weight, self.lexical_weight],
                k=self.rrf_k,
            )

        retrieved: List[Dict[str, Any]] = []
        per_file: Dict[str, int] = {}

        for chunk_id, score, rank_map in ordered:
            if len(retrieved) >= top_k:
                break
            candidate = lookup.get(chunk_id)
            if candidate is None:
                continue
            if score_threshold is not None and score < score_threshold:
                continue

            path = candidate["metadata"].get("path", "")
            if self.max_per_file is not None:
                if per_file.get(path, 0) >= self.max_per_file:
                    continue
                per_file[path] = per_file.get(path, 0) + 1

            retrieved.append(
                {
                    "id": chunk_id,
                    "content": candidate["content"],
                    "metadata": candidate["metadata"],
                    "rank": len(retrieved) + 1,
                    "score": score,
                    # Retrieval trace: which path surfaced this, and at what rank.
                    "semantic_rank": rank_map.get(SEMANTIC),
                    "lexical_rank": rank_map.get(LEXICAL),
                    "semantic_score": candidate.get("semantic_score"),
                    "lexical_score": candidate.get("lexical_score"),
                    "sources": [
                        name
                        for index, name in ((SEMANTIC, "semantic"), (LEXICAL, "lexical"))
                        if index in rank_map
                    ],
                }
            )

        return retrieved
