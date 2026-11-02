"""Build a local dense index. All model/cache files stay under this repository."""

from __future__ import annotations

from scientific_evidence_rag.data import load_corpus
from scientific_evidence_rag.paths import CACHE_DIR, PROCESSED_DIR
from scientific_evidence_rag.retrieval import HybridRetriever, dense_index_cache_path


def main() -> None:
    corpus_path = PROCESSED_DIR / "corpus.jsonl"
    documents = load_corpus(corpus_path)
    cache_path = dense_index_cache_path(corpus_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    retriever = HybridRetriever(documents, embedding_cache=cache_path)
    vectors = retriever.build_dense_index(batch_size=16)
    print(f"Indexed {len(documents)} abstracts into shape {vectors.shape}.")
    print(f"Local cache: {cache_path.relative_to(CACHE_DIR.parent)}")
    print("Embeddings and model files are ignored by Git.")


if __name__ == "__main__":
    main()
