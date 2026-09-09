import numpy as np

from src.compare import most_similar, similarity_matrix


def test_similarity_matrix_shape():
    vectors = [[1, 0, 0], [0, 1, 0], [1, 0, 0]]
    sim = similarity_matrix(vectors)
    assert sim.shape == (3, 3)
    assert np.isclose(sim[0, 2], 1.0)  # نفس المتجه => تشابه = 1


def test_most_similar_finds_identical_vector():
    query = [1, 0, 0]
    candidates = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    result = most_similar(query, candidates, top_k=1)
    assert result[0][0] == 1  # index 1 هو الأقرب


def test_wemm_rejects_unknown_mode():
    import pytest

    from src.providers import wemm_embed

    with pytest.raises(ValueError, match="mode"):
        wemm_embed.embed(["نص"], mode="both")


def test_wemm_registered_as_provider():
    from src.providers import PROVIDERS, get_provider

    assert "wemm" in PROVIDERS
    assert get_provider("wemm") is not None
