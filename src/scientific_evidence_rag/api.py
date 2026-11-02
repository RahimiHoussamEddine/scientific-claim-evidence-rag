"""FastAPI interface for local evidence search and verification."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from scientific_evidence_rag.paths import PROCESSED_DIR
from scientific_evidence_rag.pipeline import EvidenceAssistant

app = FastAPI(
    title="Scientific Claim Evidence API",
    description="Local SciFact retrieval and evidence-conditioned claim verification.",
    version="0.1.0",
)


class SearchRequest(BaseModel):
    claim: str = Field(min_length=8, max_length=2000)
    method: Literal["bm25", "dense", "hybrid"] = "hybrid"
    k: int = Field(default=5, ge=1, le=20)


@lru_cache(maxsize=1)
def assistant() -> EvidenceAssistant:
    return EvidenceAssistant.from_processed_data()


@app.get("/health")
async def health() -> dict[str, bool | str]:
    data_ready = (PROCESSED_DIR / "corpus.jsonl").is_file()
    is_cached = assistant.cache_info().currsize > 0
    return {
        "status": "ok",
        "data_ready": data_ready,
        "generation_model_loaded": (
            assistant().generator.loaded if data_ready and is_cached else False
        ),
    }


@app.post("/v1/search")
async def search(request: SearchRequest) -> dict[str, object]:
    if not (PROCESSED_DIR / "corpus.jsonl").is_file():
        raise HTTPException(status_code=503, detail="Run make fetch before using the API")
    try:
        results = assistant().retrieve(request.claim, method=request.method, k=request.k)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"claim": request.claim, "method": request.method, "results": results}


@app.post("/v1/verify")
async def verify(request: SearchRequest) -> dict[str, object]:
    if not (PROCESSED_DIR / "corpus.jsonl").is_file():
        raise HTTPException(status_code=503, detail="Run make fetch before using the API")
    try:
        return assistant().verify(request.claim, method=request.method, k=request.k)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
