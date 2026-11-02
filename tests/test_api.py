import asyncio

import httpx

from scientific_evidence_rag.api import app


def request(method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_health_reports_missing_data_without_loading_a_model(tmp_path, monkeypatch):
    from scientific_evidence_rag import api

    monkeypatch.setattr(api, "PROCESSED_DIR", tmp_path)
    response = request("GET", "/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "data_ready": False,
        "generation_model_loaded": False,
    }


def test_search_request_rejects_short_claims():
    response = request("POST", "/v1/search", json={"claim": "short"})

    assert response.status_code == 422
