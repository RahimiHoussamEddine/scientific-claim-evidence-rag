"""Evaluate retrieval baselines and optionally one fixed local generator on dev."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scientific_evidence_rag.data import Claim, load_claims, load_corpus
from scientific_evidence_rag.generation import FlanT5Generator
from scientific_evidence_rag.metrics import (
    aggregate_retrieval,
    citation_metrics,
    classification_summary,
)
from scientific_evidence_rag.paths import PROCESSED_DIR, REPORTS_DIR
from scientific_evidence_rag.retrieval import (
    MODEL_REVISION,
    HybridRetriever,
    dense_index_cache_path,
)

METHODS = ("bm25", "dense", "hybrid")


def _retrieval_row(claim: Claim, results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "claim_id": claim.claim_id,
        "gold_doc_ids": claim.gold_doc_ids,
        "retrieved_ids": [str(result["doc_id"]) for result in results],
    }


def run_evaluation(generation: bool = False, generation_only: bool = False) -> dict[str, Any]:
    if not (PROCESSED_DIR / "claims_dev.jsonl").is_file():
        raise FileNotFoundError("SciFact data is missing. Run `make fetch` first.")
    documents = load_corpus(PROCESSED_DIR / "corpus.jsonl")
    dev_claims = load_claims(PROCESSED_DIR / "claims_dev.jsonl")
    retriever = HybridRetriever(
        documents, embedding_cache=dense_index_cache_path(PROCESSED_DIR / "corpus.jsonl")
    )
    destination = REPORTS_DIR / "validation_metrics.json"
    if generation_only:
        if not destination.is_file():
            raise FileNotFoundError(
                "Run the retrieval evaluation first to create its aggregate report."
            )
        report = json.loads(destination.read_text(encoding="utf-8"))
        if (
            report.get("dataset") != "SciFact"
            or report.get("split") != "dev"
            or report.get("corpus_documents") != len(documents)
            or report.get("claims") != len(dev_claims)
            or report.get("dense_model_revision") != MODEL_REVISION
            or not report.get("retrieval", {}).get("hybrid")
        ):
            raise ValueError(
                "The existing report does not match this SciFact dev dataset and model."
            )
    else:
        report = {
            "dataset": "SciFact",
            "split": "dev",
            "split_role": "validation; not a hidden test set",
            "corpus_documents": len(documents),
            "claims": len(dev_claims),
            "retrieval_k": 5,
            "primary_retrieval": "hybrid reciprocal-rank fusion, fixed before dev evaluation",
            "dense_model_revision": MODEL_REVISION,
            "retrieval": {},
        }
    results_by_method: dict[str, list[tuple[Claim, list[dict[str, Any]]]]] = {}
    if not generation_only:
        for method in METHODS:
            paired = [
                (claim, retriever.search(claim.text, method=method, k=5)) for claim in dev_claims
            ]
            results_by_method[method] = paired
            report["retrieval"][method] = aggregate_retrieval(
                [_retrieval_row(claim, docs) for claim, docs in paired], k=5
            )
            print(f"Retrieval evaluated: method={method}, claims={len(paired)}")

    if generation:
        generator = FlanT5Generator()
        true_labels: list[str] = []
        predicted_labels: list[str | None] = []
        citation_rows: list[dict[str, float | int]] = []
        counts: Counter[str] = Counter()
        paired = results_by_method.get("hybrid") or [
            (claim, retriever.search(claim.text, method="hybrid", k=5)) for claim in dev_claims
        ]
        for index, (claim, retrieved) in enumerate(paired, start=1):
            answer = generator.generate(claim.text, retrieved)
            true_labels.append(claim.gold_label)
            predicted_labels.append(answer.verdict)
            retrieved_ids = {str(item["doc_id"]) for item in retrieved}
            citation_rows.append(
                citation_metrics(claim.gold_doc_ids, retrieved_ids, answer.cited_doc_ids)
            )
            counts[answer.verdict or "UNPARSEABLE"] += 1
            if index % 25 == 0 or index == len(paired):
                print(f"Local FLAN-T5 evaluation: {index}/{len(paired)} dev claims", flush=True)
        citation_total = sum(int(row["citation_count"]) for row in citation_rows)
        report["generation"] = {
            "model": "google/flan-t5-small",
            "model_revision": "9471d3bc4f85c9012776f03c4c00fdfe0d789a95",
            "device": "cpu",
            "claims": len(paired),
            "verdict_metrics": classification_summary(true_labels, predicted_labels),
            "verdict_counts": dict(counts),
            "citation_count": citation_total,
            "citation_source_validity": (
                sum(
                    float(row["citation_source_validity"]) * int(row["citation_count"])
                    for row in citation_rows
                )
                / citation_total
                if citation_total
                else 0.0
            ),
            "citation_gold_precision": (
                sum(
                    float(row["citation_gold_precision"]) * int(row["citation_count"])
                    for row in citation_rows
                )
                / citation_total
                if citation_total
                else 0.0
            ),
            "citation_metric_note": (
                "Citation precision uses bracketed document IDs emitted by the model; "
                "it is not a human factuality judgment."
            ),
        }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.part")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(f"Wrote aggregate-only report: {destination.relative_to(Path.cwd())}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation", action="store_true", help="also run the pinned local FLAN-T5 model"
    )
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="reuse an existing retrieval report and evaluate generation only",
    )
    args = parser.parse_args()
    run_evaluation(
        generation=args.generation or args.generation_only, generation_only=args.generation_only
    )


if __name__ == "__main__":
    main()
