# Design decisions and likely questions: rag-eval

**Why dense retrieval then a cross-encoder rerank?**
A bi-encoder embeds query and passage separately, so it is fast and can be indexed, but it only compares two
vectors. A cross-encoder reads query and passage together and scores relevance more precisely, but it is too slow
to run over the whole corpus. So: fetch 20 candidates cheaply, rerank them, keep 5.

**Why is the gold label a snippet and not a chunk id?**
Chunk ids change whenever chunk size or overlap changes. A verbatim `gold_snippet` is judged by whether a retrieved
chunk contains it, so the same eval set stays valid while I tune chunking.

**What do hit@k, hit@1 and MRR each tell you?**
hit@5: is the answer anywhere in what the LLM sees. hit@1: is the best chunk first. MRR: average of 1/rank of the
first correct chunk, so it rewards putting the right chunk higher. The reranker should move hit@1 and MRR more than hit@5.

**How is faithfulness measured and what is wrong with it?**
A judge model splits the answer into atomic claims and checks each against the retrieved context. It measures
"supported by context", not "true". The judge is a different model from the generator to reduce self-agreement,
but it is not validated against humans, so differences of about 0.02 are noise.

**Why LLM-drafted questions, and what is the bias?**
Writing 50+ questions with exact snippets by hand is slow. Drafting from sampled chunks guarantees an answerable
gold passage, but it skews toward passages that are easy to turn into questions. I filtered off-topic,
ambiguous and duplicate drafts by hand. 52 questions compares configurations; it does not give a precise absolute score.

**Why embedded Qdrant?**
It is a real vector database (HNSW index, payloads, cosine distance) that runs in-process, so the project needs no
Docker. The client code is the same against a server, so deployment is a URL change.

**What would you do next?**
Hybrid search (BM25 plus dense, fused with RRF), chunk-size sweep using the same eval set, a small human-labelled
subset to validate the judge, and caching embeddings for incremental re-indexing.
