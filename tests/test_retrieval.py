import numpy as np
import pytest

from scientific_evidence_rag.data import Document
from scientific_evidence_rag.retrieval import HybridRetriever


class FakeEncoder:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, texts, **kwargs):
        return np.asarray([self.mapping[text] for text in texts], dtype=np.float32)


def docs():
    return [
        Document("1", "Heart study", ("heart treatment improves cardiac function",), False),
        Document("2", "Weather study", ("rainfall affects regional crops",), False),
        Document("3", "Second heart paper", ("cardiac treatment and heart disease",), True),
    ]


def test_bm25_retrieves_lexically_relevant_abstracts():
    retriever = HybridRetriever(docs())

    results = retriever.search("heart treatment", method="bm25", k=2)

    assert results[0]["doc_id"] in {"1", "3"}
    assert len(results) == 2


def test_dense_retrieval_uses_supplied_vectors_without_model_download():
    documents = docs()
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)
    encoder = FakeEncoder({"heart claim": [1.0, 0.0]})
    retriever = HybridRetriever(documents, encoder=encoder, dense_vectors=vectors)

    results = retriever.search("heart claim", method="dense", k=2)

    assert [item["doc_id"] for item in results] == ["1", "3"]


def test_hybrid_retrieval_has_valid_method_and_k_guards():
    retriever = HybridRetriever(docs())

    with pytest.raises(ValueError, match="method"):
        retriever.search("claim", method="unknown")
    with pytest.raises(ValueError, match="k"):
        retriever.search("claim", k=21)
    with pytest.raises(ValueError, match="empty"):
        retriever.search("   ")
