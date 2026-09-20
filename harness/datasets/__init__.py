"""Loading and validating evaluation datasets.

A dataset is a YAML file in this directory. It pairs a corpus of documents with
queries whose correct answers are known. Without that ground truth there is no
measurement, only a similarity matrix a human has to eyeball.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DATASETS_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    lang: str = "ar"


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    relevant: tuple[str, ...]
    lang: str = "ar"
    note: str = ""


@dataclass(frozen=True)
class Dataset:
    name: str
    version: int
    corpus: tuple[Document, ...]
    queries: tuple[Query, ...]

    @property
    def doc_ids(self) -> list[str]:
        return [d.id for d in self.corpus]

    @property
    def doc_texts(self) -> list[str]:
        return [" ".join(d.text.split()) for d in self.corpus]

    @property
    def query_texts(self) -> list[str]:
        return [" ".join(q.text.split()) for q in self.queries]


def available() -> list[str]:
    return sorted(p.stem for p in DATASETS_DIR.glob("*.yaml"))


def load(name: str) -> Dataset:
    """Load a dataset by file stem, e.g. load("text_retrieval_v1")."""
    path = DATASETS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No dataset {name!r}. Available: {available()}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    corpus = tuple(
        Document(id=d["id"], text=d["text"], lang=d.get("lang", "ar")) for d in raw["corpus"]
    )

    seen: set[str] = set()
    for doc in corpus:
        if doc.id in seen:
            raise ValueError(f"Duplicate document id {doc.id!r}")
        seen.add(doc.id)

    queries = []
    for q in raw["queries"]:
        missing = set(q["relevant"]) - seen
        if missing:
            raise ValueError(f"Query {q['id']} references unknown docs: {sorted(missing)}")
        if not q["relevant"]:
            raise ValueError(f"Query {q['id']} has no relevant documents")
        queries.append(
            Query(
                id=q["id"],
                text=q["text"],
                relevant=tuple(q["relevant"]),
                lang=q.get("lang", "ar"),
                note=q.get("note", ""),
            )
        )

    return Dataset(
        name=raw["name"],
        version=int(raw.get("version", 1)),
        corpus=corpus,
        queries=tuple(queries),
    )
