"""
تشغيل تجربة embedding كاملة من الطرفية.

مثال:
    python scripts/run_benchmark.py --provider ollama --model nomic-embed-text
    python scripts/run_benchmark.py --provider hf --model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.compare import similarity_matrix
from src.providers import get_provider
from src.utils import load_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=["ollama", "hf", "wemm"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/samples")
    args = parser.parse_args()

    texts = load_samples(args.data)
    if not texts:
        print(f"لا توجد ملفات .txt داخل {args.data}")
        return

    embed_fn = get_provider(args.provider)

    start = time.time()
    vectors = embed_fn(texts, model=args.model)
    elapsed = time.time() - start

    print(f"عدد النصوص: {len(texts)} | الموديل: {args.model} | الوقت: {elapsed:.2f}s")
    print(f"أبعاد كل embedding: {len(vectors[0])}")

    sim = similarity_matrix(vectors)
    print("\nمصفوفة التشابه (cosine):")
    print(sim.round(3))


if __name__ == "__main__":
    main()
