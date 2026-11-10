import re
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    doc: str
    text: str
    idx: int


def chunk_text(doc: str, text: str, size: int = 180, overlap: int = 40) -> list[Chunk]:
    """Paragraph-aware word-window chunking with overlap.

    Paragraphs are packed until `size` words; the tail `overlap` words carry
    into the next chunk so answers spanning a boundary stay retrievable.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    buf: list[str] = []

    def flush():
        if buf:
            chunks.append(Chunk(f"{doc}::{len(chunks)}", doc, " ".join(buf), len(chunks)))

    for p in paras:
        words = p.split()
        # oversize paragraph: hard-split into windows
        while len(words) > size:
            if buf:
                flush()
                buf = buf[-overlap:] if overlap else []
            take = words[:size]
            buf = buf + take
            flush()
            buf = take[-overlap:] if overlap else []
            words = words[size - overlap:] if overlap else words[size:]
        if len(buf) + len(words) > size and buf:
            flush()
            buf = buf[-overlap:] if overlap else []
        buf = buf + words
    flush()
    return chunks
