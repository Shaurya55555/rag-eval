import httpx
import pytest
from fastapi.testclient import TestClient

from ragkit import store


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "client", lambda path=None: object())
    import api
    return api, TestClient(api.app)


def test_blank_question_rejected(client):
    _, c = client
    assert c.post("/ask", json={"question": ""}).status_code == 422
    assert c.post("/ask", json={"question": "   "}).status_code == 422


def test_llm_failure_becomes_502(client, monkeypatch):
    api, c = client

    def boom(*a, **k):
        raise httpx.ConnectError("provider down")

    monkeypatch.setattr(api, "answer", boom)
    r = c.post("/ask", json={"question": "what is qdrant?"})
    assert r.status_code == 502 and "unavailable" in r.json()["detail"]


def test_success_shape(client, monkeypatch):
    api, c = client
    monkeypatch.setattr(api, "answer", lambda *a, **k: {
        "answer": "x [1]", "citations": [], "contexts": [{"chunk_id": "a::0", "text": "t", "extra": 1}]})
    r = c.post("/ask", json={"question": "q"})
    assert r.status_code == 200 and r.json()["contexts"] == [{"chunk_id": "a::0", "text": "t"}]
