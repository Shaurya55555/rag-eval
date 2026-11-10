from ragkit.chunk import chunk_text
from eval.run_eval import first_hit_rank


def test_overlap_and_ids():
    text = "\n\n".join(" ".join(f"w{p}_{i}" for i in range(100)) for p in range(5))
    cs = chunk_text("d.md", text, size=120, overlap=20)
    assert len(cs) > 1 and cs[0].id == "d.md::0"
    assert all(len(c.text.split()) <= 140 for c in cs)
    tail = cs[0].text.split()[-5:]
    assert all(w in cs[1].text for w in tail)


def test_hit_rank():
    ctx = [{"text": "alpha beta"}, {"text": "The Quick  brown fox"}]
    assert first_hit_rank(ctx, "quick brown") == 2
    assert first_hit_rank(ctx, "zzz") is None
