"""Runs a task and persists the result.

Results are written as JSON under results/ so that a run from today can be
compared with a run from last month. A number printed to a terminal and lost is
not a measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

from .contract import SystemUnderTest
from .datasets import Dataset
from .tasks import run_text_retrieval

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

TASKS = {
    "text_retrieval": run_text_retrieval,
}

# Read these first: their ceiling is 1.0, so they mean what they look like.
PRIMARY_METRICS = ("success@1", "success@5", "mrr", "ndcg@5")


def run(
    sut: SystemUnderTest,
    dataset: Dataset,
    task: str = "text_retrieval",
    reranker: str = "none",
    candidate_k: int = 10,
) -> dict:
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}. Available: {sorted(TASKS)}")
    return TASKS[task](sut, dataset, reranker=reranker, candidate_k=candidate_k)


def save(result: dict) -> Path:
    """Write one result. The filename encodes what was measured, on what."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    system_id = result["system"]["id"].replace("/", "_").replace(":", "__")
    dataset = result["dataset"]
    filename = f"{result['task']}__{dataset['name']}_v{dataset['version']}__{system_id}.json"
    reranker = result.get("reranker", {"name": "none"})
    if reranker["name"] != "none":
        suffix = f"__rerank-{reranker['name']}__k{reranker['candidate_k']}"
        filename = f"{Path(filename).stem}{suffix}.json"
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_all(task: str | None = None, dataset_name: str | None = None) -> list[dict]:
    """Read back every saved result, optionally filtered."""
    if not RESULTS_DIR.exists():
        return []
    results = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if task and data.get("task") != task:
            continue
        if dataset_name and data.get("dataset", {}).get("name") != dataset_name:
            continue
        results.append(data)
    return results


def _render(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    top = "  ".join(h.ljust(widths[i]) for i, h in enumerate(header))
    rule = "  ".join("-" * w for w in widths)
    body = "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows)
    return f"{top}\n{rule}\n{body}"


def format_table(results: list[dict], full: bool = False) -> str:
    """A comparison table. This is the actual product of the harness."""
    if not results:
        return "No results yet."

    metric_names: list[str] = []
    for result in results:
        for name in result["metrics"]:
            if name not in metric_names:
                metric_names.append(name)
    if not full:
        metric_names = [name for name in PRIMARY_METRICS if name in metric_names]

    header = ["system", "reranker", "dim", *metric_names, "ms/text", "rerank ms/query"]
    rows = [
        [
            result["system"]["id"],
            result.get("reranker", {}).get("name", "none"),
            str(result["dim"]),
            *[f"{result['metrics'].get(name, float('nan')):.3f}" for name in metric_names],
            f"{result['timing']['ms_per_text']:.1f}",
            f"{result['timing'].get('rerank_ms_per_query', 0):.1f}",
        ]
        for result in sorted(results, key=lambda r: r["metrics"].get("success@1", 0), reverse=True)
    ]

    table = _render(header, rows)

    # Datasets differ in how many correct answers each query has, so recall has
    # a ceiling below 1.0. State it rather than letting the reader assume.
    notes = []
    datasets = {(r["dataset"]["name"], r["dataset"]["version"]): r["dataset"] for r in results}
    for (name, version), meta in datasets.items():
        ceilings = " ".join(
            f"{key}={value}" for key, value in meta.items() if key.startswith("max_recall@")
        )
        if ceilings:
            notes.append(
                f"{name} v{version}: {meta.get('docs')} docs, "
                f"{meta.get('queries')} queries | recall ceiling {ceilings}"
            )

    devices = {r["timing"].get("device_hint", "?") for r in results}
    if len(devices) > 1:
        notes.append(
            f"WARNING: results span different devices {sorted(devices)} - "
            "ms/text is not comparable across them"
        )

    if notes:
        table += "\n\n" + "\n".join(notes)
    return table
