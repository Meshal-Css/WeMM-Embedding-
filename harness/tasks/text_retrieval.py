"""Text retrieval task.

Embed every document once, embed every query, rank documents by cosine
similarity, then score the ranking against the known answers.

The task knows nothing about which provider it is measuring - it only uses the
SystemUnderTest contract. That is what makes adding a fourth provider a
one-file change.

Timing
------
A warm-up call runs before the clock starts. Without it the first measurement
includes model loading and lazy imports, which made a GPU-resident model look
10x faster than a cold CPU one - a property of the setup, not of the model.
Timing is still only comparable across systems running on the same device;
`device_hint` records what was used so a stale comparison is at least visible.
"""

from __future__ import annotations

import platform
import time
from datetime import UTC, datetime

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..contract import SystemUnderTest
from ..datasets import Dataset
from ..metrics import (
    max_recall_at_k,
    mrr,
    ndcg_at_k,
    rank_of_first_relevant,
    recall_at_k,
    success_at_k,
)

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5)

# Rough characters-per-token for mixed Arabic/English text. Only used to warn
# when a document is likely to be silently truncated by the model.
CHARS_PER_TOKEN = 3

WARMUP_TEXT = "warm up"


def _truncation_warnings(texts: list[str], max_tokens: int | None) -> list[str]:
    if max_tokens is None:
        return []
    limit_chars = max_tokens * CHARS_PER_TOKEN
    return [
        f"text #{i} ~{len(t) // CHARS_PER_TOKEN} tokens > {max_tokens}"
        for i, t in enumerate(texts)
        if len(t) > limit_chars
    ]


def _device_hint() -> str:
    return f"{platform.system()}-{platform.machine()}"


def run(
    sut: SystemUnderTest,
    dataset: Dataset,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    reranker: str = "none",
    candidate_k: int = 10,
) -> dict:
    """Measure one system on one dataset. Returns a JSON-serialisable result."""
    if reranker not in ("none", "bge"):
        raise ValueError(f"Unknown reranker: {reranker}")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive values")
    if not dataset.corpus or not dataset.queries:
        raise ValueError("Dataset must contain documents and queries")
    ndcg_key = f"ndcg@{max(k_values)}"
    doc_texts = dataset.doc_texts
    doc_ids = dataset.doc_ids
    warnings = _truncation_warnings(doc_texts, sut.max_tokens)

    # Pay the model-loading cost before the clock starts.
    warmup_start = time.perf_counter()
    sut.embed_documents([WARMUP_TEXT])
    warmup_seconds = time.perf_counter() - warmup_start

    start = time.perf_counter()
    doc_vectors = sut.embed_documents(doc_texts)
    docs_seconds = time.perf_counter() - start

    start = time.perf_counter()
    query_vectors = sut.embed_queries(dataset.query_texts)
    queries_seconds = time.perf_counter() - start

    if doc_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError(
            f"Dimension mismatch: docs {doc_vectors.shape[1]} vs queries {query_vectors.shape[1]}"
        )

    similarities = cosine_similarity(query_vectors, doc_vectors)

    doc_lookup = dict(zip(doc_ids, doc_texts, strict=True))
    rerank_seconds = 0.0
    rerank_warmup_seconds = 0.0
    reranker_info = {"name": "none"}

    if reranker == "bge":
        from src.rerankers.bge import DEFAULT_MODEL, rerank

        start = time.perf_counter()
        rerank(WARMUP_TEXT, [("warmup", WARMUP_TEXT)])
        rerank_warmup_seconds = time.perf_counter() - start
        reranker_info = {
            "name": "bge",
            "model": DEFAULT_MODEL,
            "candidate_k": candidate_k,
            "max_length": 512,
        }

    per_query: list[dict] = []
    for index, query in enumerate(dataset.queries):
        order = np.argsort(similarities[index])[::-1]
        ranked = [doc_ids[position] for position in order]
        baseline_ranked = ranked.copy()
        rerank_scores = {}

        if reranker == "bge":
            shortlist = ranked[:candidate_k]
            candidates = [(doc_id, doc_lookup[doc_id]) for doc_id in shortlist]
            start = time.perf_counter()
            rescored = rerank(query.text, candidates)
            rerank_seconds += time.perf_counter() - start

            reranked_ids = [doc_id for doc_id, _ in rescored]
            if len(reranked_ids) != len(shortlist) or set(reranked_ids) != set(shortlist):
                raise ValueError("Reranker changed the candidate IDs")
            ranked = reranked_ids + ranked[candidate_k:]
            rerank_scores = dict(rescored)

        row: dict = {
            "id": query.id,
            "lang": query.lang,
            "text": query.text,
            "relevant": list(query.relevant),
            "top5": ranked[:5],
            "baseline_top5": baseline_ranked[:5],
            "baseline_mrr": round(mrr(baseline_ranked, query.relevant), 4),
            "rerank_scores": rerank_scores,
            "first_relevant_rank": rank_of_first_relevant(ranked, query.relevant),
            "mrr": round(mrr(ranked, query.relevant), 4),
            ndcg_key: round(ndcg_at_k(ranked, query.relevant, max(k_values)), 4),
        }
        for k in k_values:
            row[f"success@{k}"] = success_at_k(ranked, query.relevant, k)
            row[f"recall@{k}"] = round(recall_at_k(ranked, query.relevant, k), 4)
        per_query.append(row)

    def mean(key: str) -> float:
        return round(float(np.mean([row[key] for row in per_query])), 4)

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"success@{k}"] = mean(f"success@{k}")
    for k in k_values:
        metrics[f"recall@{k}"] = mean(f"recall@{k}")
    metrics["mrr"] = mean("mrr")
    metrics[ndcg_key] = mean(ndcg_key)

    # The honest yardstick for recall on THIS dataset. Dataset property, not
    # model property - identical for every system, which is exactly the point.
    ceilings = {
        f"max_recall@{k}": round(
            float(np.mean([max_recall_at_k(q.relevant, k) for q in dataset.queries])), 4
        )
        for k in k_values
    }

    return {
        "task": "text_retrieval",
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "system": sut.describe(),
        "reranker": reranker_info,
        "dataset": {
            "name": dataset.name,
            "version": dataset.version,
            "docs": len(doc_texts),
            "queries": len(dataset.queries),
            **ceilings,
        },
        "dim": int(doc_vectors.shape[1]),
        "metrics": metrics,
        "timing": {
            "device_hint": _device_hint(),
            "warmup_seconds": round(warmup_seconds, 3),
            "docs_seconds": round(docs_seconds, 3),
            "queries_seconds": round(queries_seconds, 3),
            "rerank_warmup_seconds": round(rerank_warmup_seconds, 3),
            "rerank_seconds": round(rerank_seconds, 3),
            "rerank_ms_per_query": round(rerank_seconds * 1000 / len(dataset.queries), 1),
            "ms_per_text": round(
                (docs_seconds + queries_seconds) * 1000 / (len(doc_texts) + len(dataset.queries)),
                1,
            ),
        },
        "warnings": warnings,
        "per_query": per_query,
    }
