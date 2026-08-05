from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from . import config


class EmbeddingManager:
    """Handles document embedding using Sentence Transformers"""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not texts:
            raise ValueError("No texts provided for embedding")
        return self.model.encode(texts, show_progress_bar=True)

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
