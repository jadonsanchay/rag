from typing import Any, Dict, List, Optional, Sequence

from . import config
from .embeddings import EmbeddingManager
from .fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from .lexical_index import LexicalIndex
from .vector_store import VectorStore

SEMANTIC = 0
LEXICAL = 1

# Stratified list indices, in the order they are passed to the fusion.
STRATUM_NAMES = ("semantic:code", "semantic:prose", "lexical:code", "lexical:prose")


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
        stratify: bool = False,
        code_weight: float = 1.0,
        prose_weight: float = 1.0,
        code_languages: Optional[Sequence[str]] = None,
        min_code_results: int = 0,
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
        self.stratify = stratify
        self.code_weight = code_weight
        self.prose_weight = prose_weight
        self.code_languages = sorted(code_languages or config.CODE_LANGUAGES)
        # Softer alternative to full stratification: leave prose dominant but
        # guarantee a floor of code results, so questions whose answer is in the
        # source can never be shut out entirely by a wall of documentation.
        self.min_code_results = min_code_results

    def _semantic_candidates(
        self,
        query: str,
        limit: int,
        where: Optional[Dict[str, Any]] = None,
        embedding: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        # Accept a precomputed embedding: stratified retrieval issues two vector
        # queries for one question, and embedding it twice doubles API calls.
        if embedding is None:
            embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[embedding.tolist()], n_results=limit, where=where
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

    def _is_code(self, candidate: Dict[str, Any]) -> bool:
        return candidate["metadata"].get("language") in set(self.code_languages)

    def _retrieve_with_code_floor(
        self, query: str, top_k: int, limit: int, score_threshold: Optional[float]
    ) -> List[Dict[str, Any]]:
        """Pooled fusion, then top up with code results if too few made the cut.

        Unlike full stratification this leaves the natural ranking intact for
        prose-answered questions, and only intervenes when code is shut out.
        """
        embedding = self.embedding_manager.generate_embeddings([query])[0]
        semantic = self._semantic_candidates(query, limit, embedding=embedding)
        lexical = self.lexical_index.search(query, limit)

        lookup: Dict[str, Dict[str, Any]] = {}
        for candidate in lexical + semantic:
            entry = lookup.setdefault(candidate["id"], dict(candidate))
            entry.update({k: v for k, v in candidate.items() if k.endswith("_score")})

        fused = reciprocal_rank_fusion(
            [[c["id"] for c in semantic], [c["id"] for c in lexical]],
            weights=[self.semantic_weight, self.lexical_weight],
            k=self.rrf_k,
        )
        results = self._assemble(
            fused, lookup, top_k, score_threshold, ("semantic", "lexical")
        )

        shortfall = self.min_code_results - sum(1 for r in results if self._is_code(r))
        if shortfall <= 0:
            return results

        code_langs = self.code_languages
        code_semantic = self._semantic_candidates(
            query, limit, where={"language": {"$in": code_langs}}, embedding=embedding
        )
        code_lexical = self.lexical_index.search(query, limit, languages=code_langs)
        for candidate in code_lexical + code_semantic:
            entry = lookup.setdefault(candidate["id"], dict(candidate))
            entry.update({k: v for k, v in candidate.items() if k.endswith("_score")})

        code_fused = reciprocal_rank_fusion(
            [[c["id"] for c in code_semantic], [c["id"] for c in code_lexical]],
            weights=[self.semantic_weight, self.lexical_weight],
            k=self.rrf_k,
        )

        chosen = {r["id"] for r in results}
        promoted: List[Dict[str, Any]] = []
        for chunk_id, score, rank_map in code_fused:
            if len(promoted) >= shortfall:
                break
            if chunk_id in chosen:
                continue
            candidate = lookup.get(chunk_id)
            if candidate is None:
                continue
            promoted.append(
                {
                    "id": chunk_id,
                    "content": candidate["content"],
                    "metadata": candidate["metadata"],
                    "rank": 0,
                    "score": score,
                    "semantic_score": candidate.get("semantic_score"),
                    "lexical_score": candidate.get("lexical_score"),
                    "ranks": {"code-floor": min(rank_map.values())},
                    "sources": ["code-floor"],
                }
            )

        # Displace the lowest-ranked prose entries, keeping all code already found.
        keep = [r for r in results if self._is_code(r)]
        prose = [r for r in results if not self._is_code(r)]
        prose = prose[: max(0, top_k - len(keep) - len(promoted))]

        merged = keep + prose + promoted
        merged.sort(key=lambda r: -r["score"])
        for index, row in enumerate(merged[:top_k], start=1):
            row["rank"] = index
        return merged[:top_k]

    def _retrieve_stratified(
        self, query: str, top_k: int, limit: int, score_threshold: Optional[float]
    ) -> List[Dict[str, Any]]:
        lists = self._stratified_lists(query, limit)

        lookup: Dict[str, Dict[str, Any]] = {}
        for candidates in lists:
            for candidate in candidates:
                entry = lookup.setdefault(candidate["id"], dict(candidate))
                entry.update({k: v for k, v in candidate.items() if k.endswith("_score")})

        weights = [
            self.semantic_weight * self.code_weight,
            self.semantic_weight * self.prose_weight,
            self.lexical_weight * self.code_weight,
            self.lexical_weight * self.prose_weight,
        ]
        fused = reciprocal_rank_fusion(
            [[c["id"] for c in candidates] for candidates in lists],
            weights=weights,
            k=self.rrf_k,
        )
        return self._assemble(fused, lookup, top_k, score_threshold, STRATUM_NAMES)

    def _assemble(
        self,
        ordered: Sequence[tuple],
        lookup: Dict[str, Dict[str, Any]],
        top_k: int,
        score_threshold: Optional[float],
        list_names: Sequence[str],
    ) -> List[Dict[str, Any]]:
        """Apply the per-file cap and attach the retrieval trace."""
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
                    "semantic_score": candidate.get("semantic_score"),
                    "lexical_score": candidate.get("lexical_score"),
                    "ranks": {list_names[i]: r for i, r in rank_map.items()},
                    "sources": [list_names[i] for i in sorted(rank_map)],
                }
            )
        return retrieved

    def _stratified_lists(self, query: str, limit: int) -> tuple:
        """Four ranked lists: {semantic, lexical} x {code, prose}.

        Fetching per stratum from the stores — rather than filtering one pooled
        list — is what makes this work. Prose is 84% of chunks on this corpus, so
        a single pool of 40 candidates can contain almost no code to promote.
        """
        code_langs = self.code_languages
        per_list = max(limit // 2, 10)

        semantic_code = semantic_prose = []
        if self.mode in {"semantic", "hybrid"}:
            embedding = self.embedding_manager.generate_embeddings([query])[0]
            semantic_code = self._semantic_candidates(
                query, per_list, where={"language": {"$in": code_langs}}, embedding=embedding
            )
            semantic_prose = self._semantic_candidates(
                query, per_list, where={"language": {"$nin": code_langs}}, embedding=embedding
            )

        lexical_code = lexical_prose = []
        if self.mode in {"lexical", "hybrid"}:
            lexical_code = self.lexical_index.search(query, per_list, languages=code_langs)
            lexical_prose = self.lexical_index.search(
                query, per_list, exclude_languages=code_langs
            )

        return semantic_code, semantic_prose, lexical_code, lexical_prose

    def retrieve(
        self, query: str, top_k: int = 5, score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        limit = max(self.candidate_k, top_k)

        if self.stratify and self.mode == "hybrid":
            return self._retrieve_stratified(query, top_k, limit, score_threshold)
        if self.min_code_results and self.mode == "hybrid":
            return self._retrieve_with_code_floor(query, top_k, limit, score_threshold)

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
                    "ranks": {
                        name: rank_map[index]
                        for index, name in ((SEMANTIC, "semantic"), (LEXICAL, "lexical"))
                        if index in rank_map
                    },
                    "sources": [
                        name
                        for index, name in ((SEMANTIC, "semantic"), (LEXICAL, "lexical"))
                        if index in rank_map
                    ],
                }
            )

        return retrieved
