# rag-eval: document Q&A with a real eval harness

Status: **scaffold working, corpus + 50-question eval set + results pending.**

Pipeline: chunk (paragraph-aware, overlap) -> bge-small embeddings -> Qdrant (embedded, no server)
-> top-20 dense fetch -> cross-encoder rerank -> top-5 -> LLM answer with `[n]` citations.

## Run
    pip install -r requirements.txt
    python ingest.py corpus/
    set LLM_API_KEY=...   (any OpenAI-compatible endpoint; see ragkit/config.py)
    uvicorn api:app       # POST /ask {"question": "..."}

## Eval
`eval/questions.jsonl`, one JSON per line: `id, question, gold_doc, gold_snippet, reference_answer`.
A retrieved chunk is a hit if it contains `gold_snippet` (independent of chunking choices).

    python eval/run_eval.py --no-rerank   # baseline  -> results/baseline.json
    python eval/run_eval.py               # reranked  -> results/rerank.json

Metrics: hit@k, hit@1, MRR (retrieval); faithfulness = supported claims / total claims (LLM judge).

## Results
| config | hit@5 | hit@1 | MRR | faithfulness |
|---|---|---|---|---|
| dense only | TBD | TBD | TBD | TBD |
| + reranker | TBD | TBD | TBD | TBD |
