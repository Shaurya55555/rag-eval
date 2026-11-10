"""Score the pipeline on eval/questions.jsonl.

Retrieval: hit@k and MRR. A retrieved chunk is a hit if it contains
`gold_snippet` (robust to re-chunking). Generation: faithfulness = fraction of
answer claims supported by the retrieved context, via an LLM judge.

python eval/run_eval.py --no-rerank   # baseline
python eval/run_eval.py               # with reranker
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit import config, store  # noqa: E402
from ragkit.llm import LLM  # noqa: E402
from ragkit.pipeline import answer, retrieve  # noqa: E402

JUDGE = (
    "You grade faithfulness. Split the ANSWER into atomic factual claims (ignore citation markers). "
    "For each, decide if the CONTEXT supports it. Reply with JSON only: "
    '{"claims":[{"claim":"...","supported":true}]}'
)


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def first_hit_rank(ctx: list[dict], snippet: str) -> int | None:
    for i, c in enumerate(ctx, 1):
        if norm(snippet) in norm(c["text"]):
            return i
    return None


def faithfulness(llm: LLM, ans: str, ctx: list[dict]) -> float | None:
    if "don't know" in ans.lower():
        return None
    raw = llm.complete(JUDGE, "CONTEXT:\n" + "\n---\n".join(c["text"] for c in ctx) + f"\n\nANSWER:\n{ans}")
    try:
        claims = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])["claims"]
    except Exception:
        return None
    return sum(c["supported"] for c in claims) / len(claims) if claims else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval/questions.jsonl")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true", help="skip LLM answer + faithfulness")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    qc, llm, judge = store.client(), LLM(), LLM(model=config.JUDGE_MODEL)
    rows = [json.loads(line) for line in Path(a.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    ranks, faith, out = [], [], []
    for r in rows:
        if a.retrieval_only:
            ctx = retrieve(qc, r["question"], k_final=a.k, use_rerank=not a.no_rerank)
            f = None
        else:
            res = answer(qc, llm, r["question"], k_final=a.k, use_rerank=not a.no_rerank)
            ctx, f = res["contexts"], faithfulness(judge, res["answer"], res["contexts"])
            time.sleep(0.3)
        rank = first_hit_rank(ctx, r["gold_snippet"])
        ranks.append(rank)
        if f is not None:
            faith.append(f)
        out.append({"id": r["id"], "rank": rank, "faithfulness": f})

    n = len(rows)
    summary = {
        "n": n,
        "rerank": not a.no_rerank,
        "k": a.k,
        f"hit@{a.k}": round(sum(x is not None for x in ranks) / n, 3),
        "hit@1": round(sum(x == 1 for x in ranks) / n, 3),
        "mrr": round(sum(1 / x for x in ranks if x) / n, 3),
        "faithfulness": round(statistics.mean(faith), 3) if faith else None,
    }
    print(json.dumps(summary, indent=2))
    tag = "rerank" if not a.no_rerank else "baseline"
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{tag}.json").write_text(json.dumps({"summary": summary, "rows": out}, indent=2))


if __name__ == "__main__":
    main()
