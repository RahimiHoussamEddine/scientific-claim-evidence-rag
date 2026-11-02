"""Evaluation metrics for retrieval, evidence selection and generated verdicts."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def recall_at_k(gold_ids: set[str], retrieved_ids: Sequence[str], k: int) -> float | None:
    if not gold_ids:
        return None
    return len(gold_ids.intersection(retrieved_ids[:k])) / len(gold_ids)


def precision_at_k(gold_ids: set[str], retrieved_ids: Sequence[str], k: int) -> float:
    selected = list(retrieved_ids[:k])
    if not selected:
        return 0.0
    return len(gold_ids.intersection(selected)) / len(selected)


def reciprocal_rank(gold_ids: set[str], retrieved_ids: Sequence[str]) -> float | None:
    if not gold_ids:
        return None
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(gold_ids: set[str], retrieved_ids: Sequence[str], k: int) -> float | None:
    if not gold_ids:
        return None
    gains = [1.0 if doc_id in gold_ids else 0.0 for doc_id in retrieved_ids[:k]]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal_count = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg if idcg else 0.0


def aggregate_retrieval(rows: Sequence[dict[str, object]], k: int = 5) -> dict[str, float | int]:
    evidence_rows = [row for row in rows if row["gold_doc_ids"]]
    no_evidence = len(rows) - len(evidence_rows)
    if not evidence_rows:
        raise ValueError("No claims with annotated evidence are available")
    return {
        "claims_total": len(rows),
        "claims_with_evidence": len(evidence_rows),
        "claims_without_evidence": no_evidence,
        f"evidence_recall_at_{k}": float(
            np.mean(
                [recall_at_k(row["gold_doc_ids"], row["retrieved_ids"], k) for row in evidence_rows]
            )
        ),
        f"evidence_precision_at_{k}": float(
            np.mean(
                [
                    precision_at_k(row["gold_doc_ids"], row["retrieved_ids"], k)
                    for row in evidence_rows
                ]
            )
        ),
        "evidence_mrr": float(
            np.mean(
                [
                    reciprocal_rank(row["gold_doc_ids"], row["retrieved_ids"])
                    for row in evidence_rows
                ]
            )
        ),
        f"evidence_ndcg_at_{k}": float(
            np.mean(
                [ndcg_at_k(row["gold_doc_ids"], row["retrieved_ids"], k) for row in evidence_rows]
            )
        ),
    }


def citation_metrics(
    gold_ids: set[str], retrieved_ids: set[str], cited_ids: Sequence[str]
) -> dict[str, float | int]:
    if not cited_ids:
        return {
            "citation_count": 0,
            "citation_source_validity": 0.0,
            "citation_gold_precision": 0.0,
        }
    return {
        "citation_count": len(cited_ids),
        "citation_source_validity": len(set(cited_ids).intersection(retrieved_ids))
        / len(cited_ids),
        "citation_gold_precision": len(set(cited_ids).intersection(gold_ids)) / len(cited_ids),
    }


def classification_summary(
    y_true: Sequence[str], y_pred: Sequence[str | None]
) -> dict[str, float | int | list[str]]:
    if not y_true:
        raise ValueError("No labels available for classification evaluation")
    labels = ("SUPPORTS", "REFUTES", "MIXED", "NOT_ENOUGH_EVIDENCE")
    evaluated_labels = tuple(sorted(set(y_true)))
    normalized = [value if value in labels else "UNPARSEABLE" for value in y_pred]
    return {
        "n_claims": len(y_true),
        "accuracy": float(accuracy_score(y_true, normalized)),
        "macro_f1": float(
            f1_score(y_true, normalized, labels=evaluated_labels, average="macro", zero_division=0)
        ),
        "unparseable_fraction": float(
            sum(value == "UNPARSEABLE" for value in normalized) / len(normalized)
        ),
        "evaluated_labels": list(evaluated_labels),
    }
