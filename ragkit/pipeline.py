import re
from . import store
from .rerank import rerank
from .llm import LLM

ANSWER_SYSTEM = (
    "Answer ONLY from the numbered context passages. Cite every claim with [n] "
    "matching the passage number. If the context does not contain the answer, "
    "reply exactly: I don't know based on the provided documents."
)


def retrieve(qc, query: str, k_fetch: int = 20, k_final: int = 5, use_rerank: bool = True) -> list[dict]:
    hits = store.search(qc, query, k_fetch)
    return rerank(query, hits, k_final) if use_rerank else hits[:k_final]


def answer(qc, llm: LLM, query: str, k_fetch=20, k_final=5, use_rerank=True) -> dict:
    ctx = retrieve(qc, query, k_fetch, k_final, use_rerank)
    block = "\n\n".join(f"[{i + 1}] ({c['chunk_id']}) {c['text']}" for i, c in enumerate(ctx))
    text = llm.complete(ANSWER_SYSTEM, f"Context:\n{block}\n\nQuestion: {query}")
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(ctx)})
    return {
        "answer": text,
        "citations": [{"n": n, "chunk_id": ctx[n - 1]["chunk_id"], "doc": ctx[n - 1]["doc"]} for n in cited],
        "contexts": ctx,
    }
