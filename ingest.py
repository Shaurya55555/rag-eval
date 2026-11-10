"""python ingest.py corpus/   -> chunks every .md/.txt and indexes into Qdrant."""
import sys
from pathlib import Path
from ragkit import config, store
from ragkit.chunk import chunk_text

root = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus")
chunks = []
for f in sorted(root.rglob("*")):
    if f.suffix in {".md", ".txt"}:
        chunks += chunk_text(f.relative_to(root).as_posix(), f.read_text(encoding="utf-8"),
                             config.CHUNK_WORDS, config.CHUNK_OVERLAP)
qc = store.client()
store.index(chunks, qc)
qc.close()
print(f"indexed {len(chunks)} chunks from {root}")
