"""
The contract every system under test must satisfy.

A "system under test" is a provider + model pair, not a provider alone —
ollama:nomic-embed-text and ollama:mxbai-embed-large are different systems.

Queries and documents are NOT symmetric. Several embedding models are trained
with task prefixes (nomic: "search_query: " / "search_document: ") and lose
noticeable retrieval quality without them. The contract therefore exposes
embed_queries() and embed_documents() separately rather than one embed().
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

# Takes a list of texts, returns a 2-D array of shape (len(texts), dim).
# The model name is already bound, so tasks never need to know it.
EmbedFn = Callable[[Sequence[str]], np.ndarray]


@dataclass(frozen=True)
class SystemUnderTest:
    """One measurable configuration: a provider, a model, and how it is called."""

    provider: str
    model: str
    embed: EmbedFn
    query_prefix: str = ""
    doc_prefix: str = ""
    max_tokens: int | None = None

    @property
    def id(self) -> str:
        return f"{self.provider}:{self.model}"

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed([f"{self.query_prefix}{t}" for t in texts])

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed([f"{self.doc_prefix}{t}" for t in texts])

    def describe(self) -> dict:
        """Everything that must be recorded alongside a result."""
        return {
            "id": self.id,
            "provider": self.provider,
            "model": self.model,
            "query_prefix": self.query_prefix,
            "doc_prefix": self.doc_prefix,
            "max_tokens": self.max_tokens,
        }

    def __str__(self) -> str:
        return self.id
