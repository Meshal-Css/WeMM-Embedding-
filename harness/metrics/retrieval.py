"""Metrics for ranked retrieval.

Every function takes `ranked`: document ids ordered best-first, and `relevant`:
the ids that are actually correct for this query. Nothing here knows about
embeddings, models, or providers - which is why the same metrics work for a
dense retriever, BM25, or a video-frame search later on.

On reading the numbers
----------------------
`success@k` is the one to read first: it answers "did the user get an answer",
and its ceiling is always 1.0.

`recall@k` answers "how much of the answer did we get". Its ceiling is NOT 1.0
when a query has more correct documents than k - a query with 2 correct
documents can never score above 0.5 at k=1. Compare recall against
`max_recall_at_k` for the same dataset, never against 1.0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def rank_of_first_relevant(ranked: Sequence[str], relevant: Sequence[str]) -> int | None:
    """1-based position of the first correct document, or None if absent."""
    wanted = set(relevant)
    for position, doc_id in enumerate(ranked, start=1):
        if doc_id in wanted:
            return position
    return None


def success_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """1.0 if at least one correct document is in the top k, else 0.0.

    Also called hit rate. Ceiling is always 1.0, so it is directly readable:
    success@1 = 0.75 means three of four queries were answered on the first try.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    wanted = set(relevant)
    if not wanted:
        raise ValueError("relevant must not be empty")
    return 1.0 if any(doc_id in wanted for doc_id in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of the correct documents that appear in the top k.

    Ceiling is min(k, len(relevant)) / len(relevant), not 1.0 - see module docstring.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    wanted = set(relevant)
    if not wanted:
        raise ValueError("relevant must not be empty")
    found = sum(1 for doc_id in ranked[:k] if doc_id in wanted)
    return found / len(wanted)


def max_recall_at_k(relevant: Sequence[str], k: int) -> float:
    """The best recall@k this query could possibly reach. The honest yardstick."""
    if k <= 0:
        raise ValueError("k must be positive")
    total = len(set(relevant))
    if not total:
        raise ValueError("relevant must not be empty")
    return min(k, total) / total


def mrr(ranked: Sequence[str], relevant: Sequence[str]) -> float:
    """Reciprocal of the first correct rank. 1.0 if first, 0.5 if second, 0 if absent.

    Recall@k treats positions 1 and 5 as equal; MRR does not. Use both: recall
    answers "is it there", MRR answers "how far did the user have to scroll".
    """
    position = rank_of_first_relevant(ranked, relevant)
    return 0.0 if position is None else 1.0 / position


def ndcg_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    Like MRR it discounts by position, but it credits every correct document
    found, not only the first one. Already normalised against the ideal ranking,
    so its ceiling IS 1.0.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    wanted = set(relevant)
    if not wanted:
        raise ValueError("relevant must not be empty")

    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in wanted
    )
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(wanted), k) + 1))
    return dcg / ideal if ideal else 0.0
