"""Failure paths and pipeline wiring, all offline (no network, no model downloads)."""
import json

import httpx
import pytest

from eval import run_eval
from ragkit import llm as llm_mod
from ragkit import pipeline
from ragkit.llm import LLM


class FakeResp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=self)


OK = FakeResp(200, {"choices": [{"message": {"content": "hi"}}]})


def make(monkeypatch, responses):
    calls, sleeps = [], []
    it = iter(responses)

    def post(*a, **k):
        calls.append(1)
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(llm_mod.httpx, "post", post)
    monkeypatch.setattr(llm_mod.time, "sleep", sleeps.append)
    return calls, sleeps


def test_retries_429_then_succeeds(monkeypatch):
    calls, sleeps = make(monkeypatch, [FakeResp(429, headers={"retry-after": "3"}), OK])
    assert LLM(api_key="k").complete("s", "u") == "hi"
    assert len(calls) == 2 and sleeps == [3.0]


def test_retry_after_is_capped(monkeypatch):
    _, sleeps = make(monkeypatch, [FakeResp(429, headers={"retry-after": "86400"}), OK])
    LLM(api_key="k").complete("s", "u")
    assert sleeps == [90.0]


def test_gives_up_after_six_transport_errors(monkeypatch):
    calls, _ = make(monkeypatch, [httpx.ConnectError("down")] * 6)
    with pytest.raises(httpx.ConnectError):
        LLM(api_key="k").complete("s", "u")
    assert len(calls) == 6


def test_client_error_is_not_retried(monkeypatch):
    calls, _ = make(monkeypatch, [FakeResp(401)])
    with pytest.raises(httpx.HTTPStatusError):
        LLM(api_key="k").complete("s", "u")
    assert len(calls) == 1


def test_missing_key_fails_fast(monkeypatch):
    monkeypatch.setattr(llm_mod.config, "LLM_API_KEY", "")
    with pytest.raises(RuntimeError):
        LLM(api_key="").complete("s", "u")


class StubLLM:
    def __init__(self, text):
        self.text = text

    def complete(self, system, user, temperature=0.0):
        return self.text


def fake_ctx(monkeypatch, n=3):
    ctx = [{"chunk_id": f"d.md::{i}", "doc": "d.md", "text": f"t{i}"} for i in range(n)]
    monkeypatch.setattr(pipeline, "retrieve", lambda *a, **k: ctx)


def test_citations_ignore_out_of_range_numbers(monkeypatch):
    fake_ctx(monkeypatch)
    r = pipeline.answer(None, StubLLM("A [2] and B [9] and C [1] [0]"), "q")
    assert [c["n"] for c in r["citations"]] == [1, 2]
    assert r["citations"][0]["chunk_id"] == "d.md::0"


def test_refusal_has_no_citations(monkeypatch):
    fake_ctx(monkeypatch)
    r = pipeline.answer(None, StubLLM("I don't know based on the provided documents."), "q")
    assert r["citations"] == []


def test_faithfulness_scoring_and_bad_judge_output():
    ctx = [{"text": "x"}]
    good = StubLLM('{"claims":[{"claim":"a","supported":true},{"claim":"b","supported":false}]}')
    assert run_eval.faithfulness(good, "ans", ctx) == 0.5
    assert run_eval.faithfulness(StubLLM("not json at all"), "ans", ctx) is None
    assert run_eval.faithfulness(good, "I don't know based on the provided documents.", ctx) is None
    assert run_eval.faithfulness(StubLLM('{"claims":[]}'), "ans", ctx) is None
