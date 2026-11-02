"""Load the small SciFact benchmark into validated project data objects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Evidence:
    label: str
    sentence_ids: tuple[int, ...]


@dataclass(frozen=True)
class Claim:
    claim_id: int
    text: str
    evidence: dict[str, tuple[Evidence, ...]]

    @property
    def gold_doc_ids(self) -> set[str]:
        return set(self.evidence)

    @property
    def gold_label(self) -> str:
        labels = {item.label for entries in self.evidence.values() for item in entries}
        if not labels:
            return "NOT_ENOUGH_EVIDENCE"
        if labels == {"SUPPORT"}:
            return "SUPPORTS"
        if labels == {"CONTRADICT"}:
            return "REFUTES"
        return "MIXED"

    @property
    def gold_sentence_ids(self) -> dict[str, set[int]]:
        return {
            doc_id: {sentence_id for item in entries for sentence_id in item.sentence_ids}
            for doc_id, entries in self.evidence.items()
        }


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    abstract: tuple[str, ...]
    structured: bool

    @property
    def text(self) -> str:
        return " ".join(part.strip() for part in self.abstract if part.strip())

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.text}".strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required data file is missing: {path}. Run `make fetch` first.")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_claims(path: Path) -> list[Claim]:
    records = _read_jsonl(path)
    claims: list[Claim] = []
    seen_ids: set[int] = set()
    for record in records:
        claim_id = int(record["id"])
        if claim_id in seen_ids:
            raise ValueError(f"Duplicate claim id {claim_id} in {path}")
        seen_ids.add(claim_id)
        evidence = {
            str(doc_id): tuple(
                Evidence(
                    label=str(item["label"]).upper(),
                    sentence_ids=tuple(int(value) for value in item.get("sentences", [])),
                )
                for item in annotations
            )
            for doc_id, annotations in record.get("evidence", {}).items()
        }
        if any(
            item.label not in {"SUPPORT", "CONTRADICT"}
            for values in evidence.values()
            for item in values
        ):
            raise ValueError(f"Unexpected evidence label for claim {claim_id}")
        claims.append(Claim(claim_id, str(record["claim"]).strip(), evidence))
    return claims


def load_corpus(path: Path) -> list[Document]:
    records = _read_jsonl(path)
    documents: list[Document] = []
    seen_ids: set[str] = set()
    for record in records:
        doc_id = str(record["doc_id"])
        if doc_id in seen_ids:
            raise ValueError(f"Duplicate document id {doc_id} in {path}")
        seen_ids.add(doc_id)
        abstract = record["abstract"]
        if isinstance(abstract, str):
            abstract = [abstract]
        documents.append(
            Document(
                doc_id=doc_id,
                title=str(record["title"]).strip(),
                abstract=tuple(str(sentence).strip() for sentence in abstract),
                structured=bool(record.get("structured", False)),
            )
        )
    if not documents:
        raise ValueError(f"SciFact corpus is empty: {path}")
    return documents
