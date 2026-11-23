# What a cross-encoder reranker bought me on a small RAG system

I built a question-answering system over the Qdrant and FastAPI documentation: 170 markdown files, 1,696 chunks, dense retrieval
with a cross-encoder reranker, and answers with `[n]` citations. The part I want to write about is the evaluation, because the
reranker's effect only became visible once I measured retrieval separately from generation.

## The problem

My first pipeline was the standard one: embed the question with bge-small, fetch the top 5 chunks from Qdrant, give them to the LLM.
It answered most questions, but I had no idea how often the right passage was in the context, or whether it sat first or fifth.
"It looks fine on the five questions I tried" is not a measurement, and an LLM that is good at bluffing hides retrieval failures.

## Building something to measure with

I needed questions with known answers. I had an LLM draft 52 questions from sampled chunks, then checked mechanically that each
question's gold snippet is a verbatim substring of its source chunk, and then filtered by hand: off-topic, duplicate and ambiguous
drafts were removed. That is not as good as questions written by real users, and I say so in the README, because LLM-drafted
questions lean toward passages that are easy to phrase as questions.

The metric matters as much as the set. I count a retrieved chunk as a hit if it contains the gold snippet, rather than if its id
matches. That makes the score independent of chunking choices, so I can change chunk size later and compare runs. I report hit@5,
hit@1 and mean reciprocal rank (MRR), because they answer different questions: hit@5 asks whether the evidence was in the context at all,
while hit@1 and MRR ask how close to the top it was.

## What I tried

Baseline: dense retrieval, top 5. Then the change: fetch the top 20 by dense score and rerank those 20 with an ms-marco MiniLM
cross-encoder (running through ONNX), keeping the top 5. A bi-encoder compares two independently computed vectors, so it is fast but
coarse. A cross-encoder reads the question and the passage together, which is slower, so you only run it on a short candidate list.

## What I measured

| config | hit@5 | hit@1 | MRR |
|---|---|---|---|
| dense only | 0.923 | 0.673 | 0.752 |
| dense + reranker | 0.962 | 0.788 | 0.858 |

On 52 questions one question is 1.9 points. So hit@1 going from 0.673 to 0.788 is 35 questions to 41: six more questions with
the right passage in first position. MRR rose by 0.106. The hit@5 change is 48 questions to 50, which is two questions. I would not
claim that part as an improvement; it is within what chance can do on a set this size.

The pattern is the useful part. The right passage was usually already somewhere in the top 5 (0.923), and the reranker's job was
moving it up. That matters for generation, because models weight early context more, and it matters for cost: with a good order you
can pass fewer chunks.

## What I did not measure, and what went wrong

I measured faithfulness (an LLM judge that splits an answer into claims and checks each against the retrieved context) only for
the baseline: 0.966. The reranked run was retrieval-only, so I cannot say whether better ordering changed faithfulness. The README
states this next to the table instead of letting the 0.966 sit under the reranked row.

The judge is not validated against humans, so I treat differences of about 0.02 as noise. And the free-tier LLM API rate-limited
the eval hard enough that I had to add visible waits, a capped `retry-after` (a huge value means a daily quota is gone and should
be reported, not slept through), and a per-question progress line. Those are small details that decided whether the evaluation was
runnable at all.

## What I learned

1. Evaluate retrieval on its own. It is cheap, deterministic, needs no LLM, and it is where a RAG system's quality is decided.
2. Define a hit by content, not by chunk id, so your evaluation survives your own refactors.
3. Report the number of questions behind every percentage. A 3.9 point gain that is two questions is a different claim from an 11.5 point gain that is six.
4. A reranker is a ranking fix. It will not rescue a question whose evidence was never retrieved, so size the candidate list (here 20) with the baseline's recall at that depth in mind.

The code, the 52 questions and the result files are in this repository; `python eval/run_eval.py --no-rerank` and
`python eval/run_eval.py` reproduce the two rows.
