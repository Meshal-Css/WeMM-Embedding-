import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def similarity_matrix(vectors: list[list[float]]) -> np.ndarray:
    """يحسب مصفوفة التشابه (cosine) بين كل الـ embeddings المعطاة."""
    arr = np.array(vectors)
    return cosine_similarity(arr)


def most_similar(query_vector, candidate_vectors, top_k: int = 3):
    """يرجع فهارس أقرب top_k من candidate_vectors مقارنة بـ query_vector."""
    sims = cosine_similarity([query_vector], candidate_vectors)[0]
    ranked = np.argsort(sims)[::-1][:top_k]
    return [(int(i), float(sims[i])) for i in ranked]
