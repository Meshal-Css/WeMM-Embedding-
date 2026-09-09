"""
Wrapper بسيط لتوليد embeddings عبر Ollama المحلي.
يتطلب: ollama يشتغل محليًا + موديل embedding مسحوب مسبقًا (مثال: nomic-embed-text).
"""

import os

import ollama

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def embed(texts: list[str], model: str = "nomic-embed-text") -> list[list[float]]:
    """يرجع قائمة embeddings لكل نص في texts."""
    client = ollama.Client(host=OLLAMA_HOST)
    vectors = []
    for text in texts:
        response = client.embeddings(model=model, prompt=text)
        vectors.append(response["embedding"])
    return vectors
