"""Split paper text into overlapping chunks carrying citation metadata."""
from __future__ import annotations

from src.schemas.paper import Chunk, Paper
from src.utils.citation import make_citation_key


def split_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Character windows with overlap, preferring to cut at a sentence end."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(". ", start + int(size * 0.6), end)
            if cut != -1:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def chunk_paper(paper: Paper, size: int = 1000, overlap: int = 150,
                citation_key: str | None = None) -> list[Chunk]:
    """Chunk available paper text and attach citation metadata."""
    key = citation_key or make_citation_key(paper)
    source_text = paper.full_text.strip() or paper.abstract.strip()
    text = "\n\n".join(part for part in (paper.title.strip(), source_text) if part)
    pieces = split_text(text, size, overlap)
    year = paper.date[:4] if paper.date[:4].isdigit() else ""
    return [
        Chunk(
            chunk_id=f"{paper.paper_id}#{i}",
            paper_id=paper.paper_id,
            text=piece,
            metadata={
                "title": paper.title,
                "authors": paper.authors,
                "year": year,
                "source": paper.source,
                "publication_status": paper.publication_status,
                "chunk_index": i,
                "citation_key": key,
            },
        )
        for i, piece in enumerate(pieces)
    ]
