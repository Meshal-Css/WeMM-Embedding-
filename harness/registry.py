"""
Builds a SystemUnderTest from the existing src/providers registry.

This is an adapter: src/ stays untouched. If a provider signature ever
changes, this file is the only thing that needs updating.

MODEL_SPECS records how each model must be called. A model missing from the
table still runs, with empty prefixes and unknown limits — but its result is
then a lower bound, not a fair reading.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.providers import PROVIDERS, get_provider

from .contract import SystemUnderTest

# model name -> (query_prefix, doc_prefix, max_tokens)
MODEL_SPECS: dict[str, tuple[str, str, int | None]] = {
    "nomic-embed-text": ("search_query: ", "search_document: ", 2048),
    "mxbai-embed-large": ("Represent this sentence for searching relevant passages: ", "", 512),
    "bge-m3": ("", "", 8192),
}


def build(provider: str, model: str) -> SystemUnderTest:
    """Wrap a raw provider function into the harness contract."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}. Available: {sorted(PROVIDERS)}")

    raw_fn = get_provider(provider)
    query_prefix, doc_prefix, max_tokens = MODEL_SPECS.get(model, ("", "", None))

    def embed(texts: Sequence[str]) -> np.ndarray:
        vectors = raw_fn(list(texts), model=model)
        return np.asarray(vectors, dtype=np.float32)

    return SystemUnderTest(
        provider=provider,
        model=model,
        embed=embed,
        query_prefix=query_prefix,
        doc_prefix=doc_prefix,
        max_tokens=max_tokens,
    )


def available_providers() -> list[str]:
    return sorted(PROVIDERS)
