from fastembed.rerank.cross_encoder import TextCrossEncoder
from . import config

_model = None


def rerank(query: str, hits: list[dict], top_n: int) -> list[dict]:
    global _model
    if _model is None:
        _model = TextCrossEncoder(config.RERANK_MODEL)
    scores = list(_model.rerank(query, [h["text"] for h in hits]))
    ranked = sorted(zip(hits, scores), key=lambda x: -x[1])
    return [{**h, "rerank_score": float(s)} for h, s in ranked[:top_n]]
