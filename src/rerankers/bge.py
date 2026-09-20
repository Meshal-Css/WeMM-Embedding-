from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def load_model(model_name: str = DEFAULT_MODEL):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device).eval()
    return tokenizer, model, device


def rerank(
    query: str,
    candidates: list[tuple[str, str]],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 8,
) -> list[tuple[str, float]]:
    """Return (document_id, score) pairs, ordered by relevance."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if not candidates:
        return []

    ids = [doc_id for doc_id, _ in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate document IDs must be unique")

    tokenizer, model, device = load_model(model_name)
    results = []

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        pairs = [[query, text] for _, text in batch]

        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            scores = model(**inputs).logits.flatten().float().cpu()

        if not torch.isfinite(scores).all().item():
            raise ValueError("Reranker returned non-finite scores")

        results.extend(
            (doc_id, float(score))
            for (doc_id, _), score in zip(batch, scores.tolist(), strict=True)
        )

    # Python's stable sort preserves the original order when scores tie.
    return sorted(results, key=lambda item: item[1], reverse=True)
