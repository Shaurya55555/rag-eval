# rag-eval: document Q&A with a real eval harness

Question answering over the Qdrant and FastAPI documentation (170 markdown files, 1,696 chunks) with
dense retrieval, cross-encoder reranking and cited answers, plus an eval harness that scores retrieval
and answer faithfulness on 52 questions.

    question -> bge-small embedding -> Qdrant top-20 -> cross-encoder rerank -> top-5
             -> LLM answer with [n] citations -> citation list mapped back to chunk ids

## Design choices
- **Embedded Qdrant** (`QdrantClient(path=...)`): a real vector DB with no server or Docker; switch to a URL for deployment.
- **bge-small-en-v1.5** embeddings and an **ms-marco MiniLM cross-encoder** reranker, both via ONNX (fastembed).
- **Paragraph-aware chunking** (180 words, 40 overlap) so answers spanning a boundary stay retrievable.
- **Citations**: the prompt numbers the passages, the model must cite `[n]`, and the API returns the cited chunk ids.
  If the context lacks the answer the model is told to say so.

## Evaluation (eval/run_eval.py)
- **Eval set**: 52 questions in `eval/questions.jsonl` (id, question, gold_doc, gold_snippet, reference_answer).
  They were drafted by an LLM from sampled chunks (`eval/gen_questions.py`), each gold snippet verified to be a
  verbatim substring of its source chunk, then filtered by hand (off-topic, duplicate and ambiguous drafts removed).
- **Retrieval**: a retrieved chunk is a hit if it contains the gold snippet, so the metric does not depend on
  chunking choices. Reported: hit@5, hit@1, MRR.
- **Faithfulness**: an LLM judge (a different model from the generator) splits the answer into claims and checks
  each against the retrieved context; score = supported claims / total claims.

## Results (52 questions, k=5)
| config | hit@5 | hit@1 | MRR | faithfulness |
|---|---|---|---|---|
| dense only (baseline) | 0.923 | 0.673 | 0.752 | 0.966 |
| dense + reranker | 0.962 | 0.788 | 0.858 | not measured |

The cross-encoder lifts hit@1 by 11.5 points (0.673 to 0.788) and MRR by 0.106 (0.752 to 0.858); hit@5 rises by 3.9
points, which is 2 of 52 questions, so treat that part as small. The reranked run was retrieval-only
(`--retrieval-only`), so faithfulness was not re-measured with the reranker; the 0.966 figure is for the baseline only.
With 52 questions, one question is 1.9 points of hit@k.

## Run
    pip install -r requirements.txt
    python ingest.py corpus/
    set LLM_API_KEY=...            # any OpenAI-compatible endpoint, see ragkit/config.py
    uvicorn api:app                # open http://127.0.0.1:8000 for the demo page, or POST /ask
    python eval/run_eval.py --no-rerank
    python eval/run_eval.py
    pytest                          # 13 tests, offline (LLM calls are mocked)

## Failure handling
| situation | behavior |
|---|---|
| LLM returns 429, 500, 502 or 503 | up to 6 attempts; waits `retry-after`, capped at 90 s (a huge value means a daily quota is gone and is better surfaced than slept through), else exponential backoff |
| network error or timeout (120 s per call) | exponential backoff, re-raised after the 6th attempt |
| LLM returns 4xx other than 429 (bad key, bad request) | not retried, raised immediately |
| `LLM_API_KEY` not set | `RuntimeError` before any request is made |
| `POST /ask` and the LLM is still failing | HTTP 502 `the answer model is unavailable`; the error is logged, not leaked to the client |
| `POST /ask` with an empty or blank question | HTTP 422 |
| context lacks the answer | the model is told to reply `I don't know based on the provided documents.`; no citations are returned |
| model cites `[n]` outside the retrieved range | those numbers are dropped from the citation list |
| judge model returns non-JSON or no claims | faithfulness for that question is `None` and excluded from the mean, not counted as 0 |

Not handled: the embedded Qdrant store is single-process (a second process on the same path fails to lock it), and
there is no request-level timeout on retrieval or reranking.

## Limits
- Questions are LLM-drafted, so they skew toward passages that are easy to phrase as questions; 52 is enough to
  compare configurations, not to claim a precise absolute score.
- Faithfulness is judged by an LLM and measures support by the retrieved context, not factual truth.
- The judge model is not human-validated. Small differences (about 0.02) should be treated as noise.
- Corpus: Qdrant and FastAPI docs as of the download date; no incremental re-indexing.

Write-up: [What a cross-encoder reranker bought me](docs/reranker-gain.md)
