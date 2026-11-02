import pytest

from scientific_evidence_rag.generation import parse_citations, parse_verdict
from scientific_evidence_rag.metrics import (
    citation_metrics,
    classification_summary,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Verdict: SUPPORTS", "SUPPORTS"),
        ("Verdict: REFUTES", "REFUTES"),
        ("Verdict: MIXED", "MIXED"),
        ("Verdict: NOT ENOUGH EVIDENCE", "NOT_ENOUGH_EVIDENCE"),
        ("NOT_ENOUGH_EVIDENCE", "NOT_ENOUGH_EVIDENCE"),
        ("An abstract supports the claim, but no verdict is stated.", None),
        ("There is no reliable answer.", None),
    ],
)
def test_parse_verdict(text, expected):
    assert parse_verdict(text) == expected


def test_parse_citations_is_unique_and_preserves_order():
    assert parse_citations("See [12] and [9], then [12].") == ("12", "9")


def test_retrieval_metrics_are_exact_on_small_ranked_list():
    gold = {"a", "c"}
    ranked = ["b", "a", "c"]
    assert recall_at_k(gold, ranked, 2) == 0.5
    assert precision_at_k(gold, ranked, 2) == 0.5
    assert reciprocal_rank(gold, ranked) == 0.5
    assert ndcg_at_k(gold, ranked, 3) == pytest.approx(
        (1 / 1.5849625007 + 1 / 2.0) / (1 / 1.0 + 1 / 1.5849625007)
    )


def test_citation_metrics_separate_source_validity_from_gold_match():
    result = citation_metrics({"4"}, {"4", "5"}, ["4", "5", "99"])
    assert result["citation_source_validity"] == pytest.approx(2 / 3)
    assert result["citation_gold_precision"] == pytest.approx(1 / 3)


def test_classification_macro_f1_uses_classes_present_in_gold():
    result = classification_summary(["SUPPORTS", "NOT_ENOUGH_EVIDENCE"], ["SUPPORTS", None])
    assert result["evaluated_labels"] == ["NOT_ENOUGH_EVIDENCE", "SUPPORTS"]
    assert result["macro_f1"] == pytest.approx(0.5)
    assert result["unparseable_fraction"] == 0.5
