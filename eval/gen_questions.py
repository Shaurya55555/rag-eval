"""Draft eval questions from sampled corpus chunks. Output needs HUMAN REVIEW before use.

python eval/gen_questions.py --n 70 --out eval/questions_draft.jsonl

Each kept row has a gold_snippet verified to be a verbatim substring of its source chunk.
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit import config  # noqa: E402
from ragkit.chunk import chunk_text  # noqa: E402
from ragkit.llm import LLM  # noqa: E402

SYSTEM = (
    "You write evaluation questions for a documentation Q&A system. Given a passage, write ONE question "
    "that a developer might ask, answerable from this passage alone. The question must be self-contained: "
    "name the product/feature explicitly and never say 'this passage' or 'the text'. Also return "
    "gold_snippet: a 6-20 word span copied EXACTLY (character for character) from the passage that contains "
    "the answer, and reference_answer: one sentence. Skip passages that are only code, tables or link lists "
    'by returning {"skip": true}. Reply with JSON only: '
    '{"question": "...", "gold_snippet": "...", "reference_answer": "..."}'
)


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=70)
    ap.add_argument("--out", default="eval/questions_draft.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    random.seed(a.seed)
    chunks = []
    for f in sorted(Path("corpus").rglob("*.md")):
        text = f.read_text(encoding="utf-8")
        text = re.sub(r"^---.*?---\s*", "", text, flags=re.S)  # strip front matter
        chunks += chunk_text(f.relative_to("corpus").as_posix(), text, config.CHUNK_WORDS, config.CHUNK_OVERLAP)
    # prose-heavy chunks only, at most 2 per doc so the set spans the corpus
    chunks = [c for c in chunks if len(c.text.split()) > 80 and c.text.count("```") == 0 and c.text.count("|") < 6]
    random.shuffle(chunks)
    per_doc: dict[str, int] = {}
    llm = LLM(model=config.JUDGE_MODEL)
    kept = []
    for c in chunks:
        if len(kept) >= a.n:
            break
        if per_doc.get(c.doc, 0) >= 2:
            continue
        try:
            raw = llm.complete(SYSTEM, c.text)
            d = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        except Exception as e:  # rate limit / bad JSON: skip
            print("skip", c.id, type(e).__name__)
            time.sleep(2)
            continue
        if d.get("skip") or not all(k in d for k in ("question", "gold_snippet", "reference_answer")):
            continue
        if norm(d["gold_snippet"]) not in norm(c.text):
            continue
        per_doc[c.doc] = per_doc.get(c.doc, 0) + 1
        kept.append({"id": f"q{len(kept) + 1:03d}", "question": d["question"], "gold_doc": c.doc,
                     "gold_snippet": d["gold_snippet"], "reference_answer": d["reference_answer"]})
        print(len(kept), d["question"])
        time.sleep(0.4)
    Path(a.out).write_text("\n".join(json.dumps(k) for k in kept) + "\n", encoding="utf-8")
    print(f"wrote {len(kept)} to {a.out}")


if __name__ == "__main__":
    main()
