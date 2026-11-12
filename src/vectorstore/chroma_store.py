"""ChromaDB-backed VectorStore (local, persistent, free)."""
from __future__ import annotations

import re
from pathlib import Path

import chromadb

from src.schemas.paper import Chunk, ScoredChunk
from src.vectorstore.base import VectorStore
from src.vectorstore.embeddings import Embedder

_BATCH = 500


def _flatten(md: dict) -> dict:
    """Chroma metadata must be str/int/float/bool; join lists."""
    flat = {}
    for key, val in md.items():
        if isinstance(val, list):
            flat[key] = "; ".join(str(v) for v in val)
        elif val is None:
            flat[key] = ""
        else:
            flat[key] = val
    return flat


def _where(where: dict | None) -> dict | None:
    if not where:
        return None
    if len(where) == 1:
        return where
    return {"$and": [{k: v} for k, v in where.items()]}


class ChromaStore(VectorStore):
    def __init__(self, path: str | Path, embedder: Embedder, prefix: str = "papers"):
        self.embedder = embedder
        safe = re.sub(r"[^A-Za-z0-9._-]", "-", embedder.name)  # one collection per embedder
        self._client = chromadb.PersistentClient(path=str(path))
        self._col = self._client.get_or_create_collection(
            name=f"{prefix}_{safe}", metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks: list[Chunk]) -> int:
        written = 0
        for i in range(0, len(chunks), _BATCH):
            batch = chunks[i:i + _BATCH]
            self._col.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                embeddings=self.embedder.embed([c.text for c in batch]),
                metadatas=[_flatten({**c.metadata, "paper_id": c.paper_id}) for c in batch],
            )
            written += len(batch)
        return written

    def query(self, text: str, k: int = 8, where: dict | None = None) -> list[ScoredChunk]:
        total = self._col.count()
        if total == 0:
            return []
        res = self._col.query(query_embeddings=self.embedder.embed([text]),
                              n_results=min(k, total), where=_where(where))
        hits = []
        for cid, doc, md, dist in zip(res["ids"][0], res["documents"][0],
                                      res["metadatas"][0], res["distances"][0]):
            md = dict(md)
            paper_id = md.pop("paper_id", cid.split("#")[0])
            hits.append(ScoredChunk(chunk=Chunk(chunk_id=cid, paper_id=paper_id, text=doc, metadata=md),
                                    score=1.0 - float(dist)))  # cosine distance -> similarity
        return hits

    def count(self) -> int:
        return self._col.count()

    def delete(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self._col.delete(ids=chunk_ids)
