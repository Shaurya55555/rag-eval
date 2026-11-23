import logging

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from ragkit import store
from ragkit.llm import LLM
from ragkit.pipeline import answer

log = logging.getLogger("rag-eval")
app = FastAPI(title="rag-eval")
qc, llm = store.client(), LLM()


class Q(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    rerank: bool = True


@app.post("/ask")
def ask(q: Q):
    if not q.question.strip():
        raise HTTPException(422, "question is blank")
    try:
        r = answer(qc, llm, q.question, use_rerank=q.rerank)
    except (httpx.HTTPError, RuntimeError) as e:
        # LLM provider failed after the client's own retries, or no API key is configured
        log.error("answer failed: %s: %s", type(e).__name__, e)
        raise HTTPException(502, "the answer model is unavailable, try again later") from e
    r["contexts"] = [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in r["contexts"]]
    return r


@app.get("/")
def home():
    return FileResponse("static/index.html")
