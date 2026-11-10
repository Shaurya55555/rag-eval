from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ragkit import store
from ragkit.llm import LLM
from ragkit.pipeline import answer

app = FastAPI(title="rag-eval")
qc, llm = store.client(), LLM()


class Q(BaseModel):
    question: str
    rerank: bool = True


@app.post("/ask")
def ask(q: Q):
    r = answer(qc, llm, q.question, use_rerank=q.rerank)
    r["contexts"] = [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in r["contexts"]]
    return r


@app.get("/")
def home():
    return FileResponse("static/index.html")
