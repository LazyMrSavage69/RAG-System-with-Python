"""
generator.py — Answer generation via the DeepSeek API.

DeepSeek exposes an OpenAI-compatible endpoint, so we use the
`openai` SDK pointed at DeepSeek's base URL.
"""

from typing import List, Dict
from openai import OpenAI


# ── Prompt builder ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly
based on the provided context. If the context does not contain enough
information to answer the question, say so clearly — do not hallucinate."""


def build_prompt(question: str, context_chunks: List[Dict]) -> str:
    """
    Assemble a RAG prompt from retrieved chunks.

    Each chunk shows its source so the model can reference it.
    """
    context_blocks = []
    for i, chunk in enumerate(context_chunks, 1):
        context_blocks.append(
            f"[Source {i}: {chunk['source']}]\n{chunk['text']}"
        )

    context_str = "\n\n".join(context_blocks)
    return (
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


# ── Generator ─────────────────────────────────────────────────────────────────

class DeepSeekGenerator:
    """
    Calls the DeepSeek chat completion API to generate grounded answers.

    Args:
        api_key:     Your DeepSeek API key.
        model:       DeepSeek model name (default: deepseek-chat).
        temperature: Sampling temperature (0 = deterministic).
        max_tokens:  Maximum tokens in the generated answer.
    """

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.DEEPSEEK_BASE_URL,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        question: str,
        context_chunks: List[Dict],
    ) -> str:
        """
        Generate an answer grounded in the retrieved context chunks.

        Args:
            question:       The user's question.
            context_chunks: Chunks returned by VectorStore.query().

        Returns:
            The model's answer as a string.
        """
        prompt = build_prompt(question, context_chunks)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()
