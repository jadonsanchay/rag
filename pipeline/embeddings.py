import os
from typing import List, Optional, Protocol

import numpy as np
from dotenv import load_dotenv

from . import config

load_dotenv()


class EmbeddingProvider(Protocol):
    """Providers expose their token limit so chunkers can size chunks to the
    model instead of hardcoding a guess and silently truncating."""

    model_name: str

    def embed(self, texts: List[str]) -> np.ndarray: ...
    def count_tokens(self, text: str) -> int: ...
    @property
    def token_limit(self) -> int: ...
    @property
    def dimension(self) -> int: ...


class SentenceTransformerProvider:
    """Local embeddings. Free and offline, but small context windows."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True)

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=True))

    @property
    def token_limit(self) -> int:
        return config.MODEL_TOKEN_LIMITS.get(self.model_name, self.model.max_seq_length)

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class OpenAIEmbeddingProvider:
    """API embeddings. Long context and no local compute, at ~$0.02/1M tokens."""

    # The API caps a request at 300k tokens; stay well under it.
    BATCH_SIZE = 64
    MAX_BATCH_TOKENS = 180_000

    def __init__(self, model_name: str = config.OPENAI_EMBEDDING_MODEL):
        import tiktoken
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        self._dimension: Optional[int] = None

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))

    @property
    def token_limit(self) -> int:
        return config.MODEL_TOKEN_LIMITS.get(self.model_name, 8191)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed(["dimension probe"])[0])
        return self._dimension

    def _batches(self, texts: List[str]) -> List[List[str]]:
        """Batch by both count and token total; the API rejects either limit."""
        batches: List[List[str]] = []
        current: List[str] = []
        current_tokens = 0

        for text in texts:
            tokens = self.count_tokens(text)
            too_many = len(current) >= self.BATCH_SIZE
            too_big = current and current_tokens + tokens > self.MAX_BATCH_TOKENS
            if too_many or too_big:
                batches.append(current)
                current, current_tokens = [], 0
            current.append(text)
            current_tokens += tokens

        if current:
            batches.append(current)
        return batches

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors: List[List[float]] = []
        batches = self._batches(texts)

        for index, batch in enumerate(batches, start=1):
            # The API errors on empty strings.
            cleaned = [text if text.strip() else " " for text in batch]
            response = self.client.embeddings.create(model=self.model_name, input=cleaned)
            vectors.extend(item.embedding for item in response.data)
            print(f"  embedded batch {index}/{len(batches)}", end="\r", flush=True)

        print(" " * 40, end="\r")
        return np.array(vectors, dtype=np.float32)


def build_provider(
    provider: str = config.EMBEDDING_PROVIDER, model_name: Optional[str] = None
) -> EmbeddingProvider:
    if provider == "openai":
        return OpenAIEmbeddingProvider(model_name or config.OPENAI_EMBEDDING_MODEL)
    if provider == "sentence-transformers":
        return SentenceTransformerProvider(model_name or config.EMBEDDING_MODEL)
    raise ValueError(f"Unknown embedding provider: {provider}")


class EmbeddingManager:
    """Facade over the active provider, so call sites don't care which is in use."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        provider: str = config.EMBEDDING_PROVIDER,
    ):
        self.provider_name = provider
        self.provider = build_provider(provider, model_name)
        self.model_name = self.provider.model_name

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise ValueError("No texts provided for embedding")
        return self.provider.embed(texts)

    def get_embedding_dimension(self) -> int:
        return self.provider.dimension

    @property
    def token_limit(self) -> int:
        return self.provider.token_limit

    def count_tokens(self, text: str) -> int:
        return self.provider.count_tokens(text)
