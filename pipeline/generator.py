import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from . import config

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the provided context. "
    "If the context does not contain the answer, say you don't know."
)


class AnswerGenerator:
    """Generates an answer from retrieved context using an OpenAI chat model"""

    def __init__(self, model: str = config.OPENAI_MODEL):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")

        self.model = model
        self.client = OpenAI(api_key=api_key)

    def build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        return "\n\n".join(
            f"[{doc['rank']}] (source: {doc['metadata'].get('source', 'unknown')})\n{doc['content']}"
            for doc in retrieved_docs
        )

    def generate(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        context = self.build_context(retrieved_docs)

        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
