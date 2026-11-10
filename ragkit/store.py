from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from . import config
from .chunk import Chunk

_embedder = None


def embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(config.EMBED_MODEL)
    return _embedder


def client(path: str | None = None) -> QdrantClient:
    # Embedded Qdrant: no server/Docker. Point at a URL for a real deployment.
    return QdrantClient(path=path or config.QDRANT_PATH)


def index(chunks: list[Chunk], qc: QdrantClient, collection: str = config.COLLECTION):
    vecs = list(embedder().embed([c.text for c in chunks]))
    dim = len(vecs[0])
    if qc.collection_exists(collection):
        qc.delete_collection(collection)
    qc.create_collection(
        collection,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    qc.upsert(
        collection,
        points=[
            models.PointStruct(id=i, vector=v.tolist(), payload={"chunk_id": c.id, "doc": c.doc, "text": c.text})
            for i, (c, v) in enumerate(zip(chunks, vecs))
        ],
    )


def search(qc: QdrantClient, query: str, k: int, collection: str = config.COLLECTION) -> list[dict]:
    qv = next(iter(embedder().query_embed(query))).tolist()
    res = qc.query_points(collection, query=qv, limit=k, with_payload=True).points
    return [{**r.payload, "score": r.score} for r in res]
