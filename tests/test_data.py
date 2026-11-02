import json

import pytest

from scientific_evidence_rag.data import load_claims, load_corpus


def test_load_claims_derives_supported_refuted_and_insufficient_labels(tmp_path):
    path = tmp_path / "claims.jsonl"
    records = [
        {
            "id": 1,
            "claim": "A specific claim.",
            "evidence": {"10": [{"label": "SUPPORT", "sentences": [0]}]},
        },
        {
            "id": 2,
            "claim": "Another specific claim.",
            "evidence": {"20": [{"label": "CONTRADICT", "sentences": [1]}]},
        },
        {"id": 3, "claim": "A claim without support.", "evidence": {}},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    claims = load_claims(path)

    assert [claim.gold_label for claim in claims] == ["SUPPORTS", "REFUTES", "NOT_ENOUGH_EVIDENCE"]
    assert claims[0].gold_doc_ids == {"10"}
    assert claims[0].gold_sentence_ids == {"10": {0}}


def test_load_claims_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "claims.jsonl"
    record = {"id": 1, "claim": "A claim.", "evidence": {}}
    path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate claim id"):
        load_claims(path)


def test_load_corpus_normalizes_abstract_fields(tmp_path):
    path = tmp_path / "corpus.jsonl"
    record = {
        "doc_id": 5,
        "title": "A paper",
        "abstract": ["First.", "Second."],
        "structured": True,
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    document = load_corpus(path)[0]

    assert document.full_text == "A paper. First. Second."
    assert document.structured is True
