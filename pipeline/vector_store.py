import re
import uuid
from typing import List, Optional

import chromadb
import numpy as np
from langchain_core.documents import Document

from . import config

BATCH_SIZE = 2000


def collection_name_for_repo(repo_name: str, strategy: Optional[str] = None) -> str:
    """Derive a Chroma-safe collection name. Keeping the chunk strategy in the
    name lets baseline and AST runs coexist, which the eval harness needs to
    compare them without re-indexing every time."""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", repo_name).strip("_").lower() or "repo"
    name = f"repo_{slug}"
    if strategy:
        name = f"{name}__{strategy}"
    return name[:63]


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""

    def __init__(
        self,
        collection_name: str = config.COLLECTION_NAME,
        persist_directory: str = str(config.VECTOR_STORE_DIR),
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.persist_directory_path = config.VECTOR_STORE_DIR
        self.persist_directory_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def for_repo(cls, repo_name: str, strategy: Optional[str] = None) -> "VectorStore":
        return cls(collection_name=collection_name_for_repo(repo_name, strategy))

    def reset(self) -> None:
        """Drop and recreate the collection. Indexing runs should be
        reproducible rather than additive."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> None:
        """Add documents and their embeddings to the vector store"""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        ids: List[str] = []
        metadatas: List[dict] = []
        texts: List[str] = []
        embeddings_list: List[list] = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            metadata = {k: v for k, v in doc.metadata.items() if v is not None}
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)

            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            metadatas.append(metadata)
            texts.append(doc.page_content)
            embeddings_list.append(
                embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            )

        # Chroma rejects oversized single batches, so add in slices.
        for start in range(0, len(ids), BATCH_SIZE):
            end = start + BATCH_SIZE
            self.collection.add(
                ids=ids[start:end],
                embeddings=embeddings_list[start:end],
                metadatas=metadatas[start:end],
                documents=texts[start:end],
            )

    def count(self) -> int:
        return self.collection.count()
