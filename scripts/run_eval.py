"""Run one evaluation, or print the comparison table.

Examples:
    python scripts/run_eval.py --provider ollama --model nomic-embed-text
    python scripts/run_eval.py --provider ollama --model bge-m3 --verbose
    python scripts/run_eval.py --table
    python scripts/run_eval.py --table --full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from harness import datasets, runner  # noqa: E402
from harness.registry import available_providers, build  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=available_providers())
    parser.add_argument("--model")
    parser.add_argument("--dataset", default="text_retrieval_v1")
    parser.add_argument("--task", default="text_retrieval")
    parser.add_argument("--table", action="store_true", help="print saved results and exit")
    parser.add_argument("--full", action="store_true", help="with --table, show every metric")
    parser.add_argument("--verbose", action="store_true", help="show per-query detail")
    parser.add_argument("--reranker", choices=["none", "bge"], default="none")
    parser.add_argument("--candidate-k", type=int, default=10)
    args = parser.parse_args()
    if args.candidate_k <= 0:
        parser.error("--candidate-k must be positive")

    if args.table:
        print(runner.format_table(runner.load_all(task=args.task), full=args.full))
        return 0

    if not args.provider or not args.model:
        parser.error("--provider and --model are required unless --table is used")

    dataset = datasets.load(args.dataset)
    sut = build(args.provider, args.model)

    print(
        f"Measuring {sut.id} on {dataset.name} v{dataset.version} "
        f"({len(dataset.corpus)} docs, {len(dataset.queries)} queries)"
    )
    if not sut.query_prefix and not sut.doc_prefix and sut.max_tokens is None:
        print(
            "  note: no task prefixes registered for this model - "
            "if it expects them, this score is a lower bound"
        )

    result = runner.run(
        sut,
        dataset,
        task=args.task,
        reranker=args.reranker,
        candidate_k=args.candidate_k,
    )

    for warning in result["warnings"]:
        print(f"  WARNING truncation risk: {warning}")

    print("\n  primary (ceiling 1.0):")
    for name in ("success@1", "success@3", "success@5", "mrr", "ndcg@5"):
        if name in result["metrics"]:
            print(f"    {name:<12} {result['metrics'][name]:.3f}")

    print("\n  recall (ceiling below 1.0 - this dataset has multi-answer queries):")
    for k in (1, 3, 5):
        key = f"recall@{k}"
        ceiling = result["dataset"].get(f"max_recall@{k}")
        if key in result["metrics"] and ceiling:
            value = result["metrics"][key]
            share = value / ceiling if ceiling else 0.0
            print(f"    {key:<12} {value:.3f}  of max {ceiling:.3f}  ({share:.0%})")

    timing = result["timing"]
    if args.reranker != "none":
        print(
            f"\n  reranker: {args.reranker} | candidates: {args.candidate_k}"
            f" | ms/query: {timing['rerank_ms_per_query']:.1f}"
            f" | warm-up: {timing['rerank_warmup_seconds']:.2f}s"
        )
    print(
        f"\n  {'ms/text':<12} {timing['ms_per_text']:.1f}"
        f"   (warm-up {timing['warmup_seconds']:.2f}s excluded, {timing['device_hint']})"
    )
    print(f"  {'dim':<12} {result['dim']}")

    if args.verbose:
        print("\n  per query:")
        for row in result["per_query"]:
            rank = row["first_relevant_rank"]
            mark = "hit " if rank == 1 else "MISS" if rank is None else f"#{rank}  "
            print(f"    {mark} {row['id']} [{row['lang']}] {row['text']}")
            print(f"          expected {row['relevant']} | got {row['top5']}")
            if args.reranker != "none":
                print(f"          before reranking: {row['baseline_top5']}")

    path = runner.save(result)
    print(f"\nSaved to {path.relative_to(Path.cwd())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
