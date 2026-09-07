"""
Wrapper لتوليد embeddings عبر sentence-transformers (Hugging Face).
مفيد للمقارنة مع موديلات Ollama المحلية.
"""
from sentence_transformers import SentenceTransformer

_model_cache: dict[str, SentenceTransformer] = {}


def embed(texts: list[str], model: str = "sentence-transformers/all-MiniLM-L6-v2") -> list[list[float]]:
    """يرجع قائمة embeddings لكل نص في texts."""
    if model not in _model_cache:
        _model_cache[model] = SentenceTransformer(model)
    vectors = _model_cache[model].encode(texts, show_progress_bar=False)
    return vectors.tolist()
