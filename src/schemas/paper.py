"""Core data models shared by every module (frozen contract)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PublicationStatus = Literal["preprint", "published_unverified", "peer_reviewed", "unknown"]


class Paper(BaseModel):
    """A paper record from arXiv or PubMed.

    ``date`` may be partial ("2024" or "2024-05") when the source gives no more
    precision; we never pad missing parts.
    """

    paper_id: str  # "arxiv:2501.01234" or "pmid:12345678"
    source: Literal["arxiv", "pubmed"]
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    full_text: str = ""
    date: str = ""
    doi: str | None = None
    pdf_url: str | None = None
    journal: str | None = None
    categories: list[str] = Field(default_factory=list)
    publication_status: PublicationStatus = "unknown"


class Chunk(BaseModel):
    """A piece of paper text plus metadata used for retrieval and citation."""

    chunk_id: str  # f"{paper_id}#{chunk_index}"
    paper_id: str
    text: str
    metadata: dict = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    """A retrieval hit. Higher score means more similar."""

    chunk: Chunk
    score: float
