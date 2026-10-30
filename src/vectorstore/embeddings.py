"""Embedders: real (sentence-transformers), deterministic fake for tests, API stub."""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod


class Embedder(ABC):
    name: str

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbedder(Embedder):
    """Deterministic hashed bag-of-words vectors. Similar wording gives similar vectors.

    For tests and offline demos only; NOT a semantic model.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.name = f"fake-{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            if norm == 0:
                vec[0], norm = 1.0, 1.0
            out.append([v / norm for v in vec])
        return out


class SentenceTransformerEmbedder(Embedder):
    """Local embedding model (downloads weights on first use)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


class APIEmbedder(Embedder):
    """Placeholder for hosted embedding APIs (not implemented in Week 1)."""

    def __init__(self, *_, **__):
        raise NotImplementedError("API embeddings are not implemented yet; use SentenceTransformerEmbedder")

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError
