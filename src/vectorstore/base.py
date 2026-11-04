"""Abstract vector store interface (frozen contract)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.schemas.paper import Chunk, ScoredChunk


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> int:
        """Insert or overwrite chunks by chunk_id (idempotent). Returns chunks written."""

    @abstractmethod
    def query(self, text: str, k: int = 8, where: dict | None = None) -> list[ScoredChunk]:
        """Return up to k most similar chunks; higher score = more similar."""

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None: ...
