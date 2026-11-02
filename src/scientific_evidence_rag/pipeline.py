"""Application service shared by Streamlit, FastAPI and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scientific_evidence_rag.data import Claim, Document, load_corpus
from scientific_evidence_rag.generation import FlanT5Generator
from scientific_evidence_rag.paths import PROCESSED_DIR
from scientific_evidence_rag.retrieval import HybridRetriever, dense_index_cache_path


class EvidenceAssistant:
    def __init__(self, documents: list[Document], retriever: HybridRetriever) -> None:
        self.documents = documents
        self.retriever = retriever
        self.generator = FlanT5Generator()

    @classmethod
    def from_processed_data(cls) -> EvidenceAssistant:
        corpus_path = PROCESSED_DIR / "corpus.jsonl"
        documents = load_corpus(corpus_path)
        return cls(
            documents,
            HybridRetriever(documents, embedding_cache=dense_index_cache_path(corpus_path)),
        )

    def retrieve(self, claim_text: str, method: str = "hybrid", k: int = 5) -> list[dict[str, Any]]:
        return self.retriever.search(claim_text, method=method, k=k)

    def verify(self, claim_text: str, method: str = "hybrid", k: int = 5) -> dict[str, Any]:
        evidence = self.retrieve(claim_text, method=method, k=k)
        answer = self.generator.generate(claim_text, evidence)
        return {
            "claim": claim_text,
            "method": method,
            "verdict": answer.verdict,
            "answer": answer.text,
            "cited_doc_ids": list(answer.cited_doc_ids),
            "evidence": evidence,
        }


def load_eval_claims(split: str = "dev") -> list[Claim]:
    if split not in {"train", "dev"}:
        raise ValueError("Only labeled train and dev splits can be evaluated")
    from scientific_evidence_rag.data import load_claims

    return load_claims(PROCESSED_DIR / f"claims_{split}.jsonl")


def load_cached_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
