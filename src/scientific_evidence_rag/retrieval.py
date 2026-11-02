"""Sparse, dense, and reciprocal-rank-fusion retrieval over SciFact abstracts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from scientific_evidence_rag.data import Document

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def dense_index_cache_path(corpus_path: Path) -> Path:
    """Return the repository-local embedding cache for this corpus and encoder revision."""
    from scientific_evidence_rag.paths import CACHE_DIR

    digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()[:16]
    return CACHE_DIR / f"scifact-{digest}-minilm-{MODEL_REVISION[:12]}.npy"


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str], **kwargs: object) -> np.ndarray: ...


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


class HybridRetriever:
    def __init__(
        self,
        documents: Sequence[Document],
        encoder: TextEncoder | None = None,
        dense_vectors: np.ndarray | None = None,
        embedding_cache: Path | None = None,
    ) -> None:
        if not documents:
            raise ValueError("Retriever requires at least one document")
        self.documents = list(documents)
        self.encoder = encoder
        self.dense_vectors = dense_vectors
        self.embedding_cache = embedding_cache
        self.bm25 = BM25Okapi([tokenize(document.full_text) for document in self.documents])

    def _load_encoder(self) -> TextEncoder:
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError("Dense retrieval requires `make install-llm`.") from error
            from scientific_evidence_rag.paths import CACHE_DIR

            model_cache = CACHE_DIR / "sentence-transformers"
            model_cache.mkdir(parents=True, exist_ok=True)
            self.encoder = SentenceTransformer(
                MODEL_ID,
                revision=MODEL_REVISION,
                device="cpu",
                cache_folder=str(model_cache),
                trust_remote_code=False,
            )
        return self.encoder

    def build_dense_index(self, batch_size: int = 32) -> np.ndarray:
        if self.dense_vectors is not None:
            return self.dense_vectors
        if batch_size < 1 or batch_size > 64:
            raise ValueError("CPU embedding batch_size must be between 1 and 64")
        if self.embedding_cache is not None and self.embedding_cache.is_file():
            cached = np.load(self.embedding_cache, allow_pickle=False)
            if cached.shape[0] == len(self.documents):
                self.dense_vectors = cached.astype(np.float32, copy=False)
                return self.dense_vectors
        encoder = self._load_encoder()
        self.dense_vectors = np.asarray(
            encoder.encode(
                [document.full_text for document in self.documents],
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )
        if self.dense_vectors.ndim != 2 or self.dense_vectors.shape[0] != len(self.documents):
            raise ValueError("Dense encoder returned an invalid document embedding matrix")
        if self.embedding_cache is not None:
            self.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.embedding_cache.with_suffix(".npy.part")
            with temporary.open("wb") as handle:
                np.save(handle, self.dense_vectors, allow_pickle=False)
            temporary.replace(self.embedding_cache)
        return self.dense_vectors

    def _rank_sparse(self, query: str, limit: int) -> list[tuple[int, float]]:
        scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)
        order = np.argsort(-scores, kind="stable")[:limit]
        return [(int(index), float(scores[index])) for index in order]

    def _rank_dense(self, query: str, limit: int) -> list[tuple[int, float]]:
        vectors = self.build_dense_index()
        query_vector = np.asarray(
            self._load_encoder().encode(
                [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            ),
            dtype=np.float32,
        )[0]
        scores = vectors @ query_vector
        order = np.argsort(-scores, kind="stable")[:limit]
        return [(int(index), float(scores[index])) for index in order]

    def search(self, query: str, method: str = "hybrid", k: int = 5) -> list[dict[str, object]]:
        query = query.strip()
        if not query:
            raise ValueError("Claim text cannot be empty")
        if not 1 <= k <= 20:
            raise ValueError("k must be between 1 and 20")
        if method not in {"bm25", "dense", "hybrid"}:
            raise ValueError("method must be one of: bm25, dense, hybrid")

        sparse = self._rank_sparse(query, max(k, 20))
        dense = self._rank_dense(query, max(k, 20)) if method in {"dense", "hybrid"} else []
        if method == "bm25":
            ranked = sparse[:k]
            score_map = {index: {"bm25_score": score} for index, score in ranked}
        elif method == "dense":
            ranked = dense[:k]
            score_map = {index: {"dense_score": score} for index, score in ranked}
        else:
            fused: dict[int, float] = {}
            raw: dict[int, dict[str, float]] = {}
            rrf_constant = 60
            for ranking, key in ((sparse, "bm25_score"), (dense, "dense_score")):
                for rank, (index, score) in enumerate(ranking, start=1):
                    fused[index] = fused.get(index, 0.0) + 1.0 / (rrf_constant + rank)
                    raw.setdefault(index, {})[key] = score
            ordered = sorted(fused, key=lambda index: (-fused[index], index))[:k]
            ranked = [(index, fused[index]) for index in ordered]
            score_map = {index: {**raw[index], "rrf_score": score} for index, score in ranked}

        results: list[dict[str, object]] = []
        for rank, (index, _) in enumerate(ranked, start=1):
            document = self.documents[index]
            results.append(
                {
                    "rank": rank,
                    "doc_id": document.doc_id,
                    "title": document.title,
                    "abstract": list(document.abstract),
                    "structured": document.structured,
                    "scores": score_map[index],
                }
            )
        return results
