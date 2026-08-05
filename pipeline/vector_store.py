import uuid
from typing import List

import chromadb
import numpy as np
from langchain_core.documents import Document

from . import config


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
            metadata={"description": "PDF document embeddings for RAG", "hnsw:space": "cosine"},
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
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)

            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            metadatas.append(metadata)
            texts.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        self.collection.add(ids=ids, embeddings=embeddings_list, metadatas=metadatas, documents=texts)

    def count(self) -> int:
        return self.collection.count()
