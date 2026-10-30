"""Normalization helpers shared by the arXiv and PubMed clients."""
from __future__ import annotations

import re

from src.schemas.paper import Paper

_RANK = {"peer_reviewed": 3, "published_unverified": 2, "preprint": 1, "unknown": 0}


class LiteratureSearchError(RuntimeError):
    """A literature API call failed."""


def clean_abstract(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


def dedupe(papers: list[Paper]) -> list[Paper]:
    """Drop duplicates by DOI, then by normalized title.

    When the same work appears twice, keep the record with the stronger publication status.
    """
    best: dict[str, Paper] = {}
    alias: dict[str, str] = {}  # secondary key -> canonical key
    for p in papers:
        keys = []
        if p.doi:
            keys.append("doi:" + p.doi.lower())
        keys.append("title:" + _title_key(p.title))
        canon = next((alias[k] for k in keys if k in alias), keys[0])
        for k in keys:
            alias[k] = canon
        if canon not in best or _RANK[p.publication_status] > _RANK[best[canon].publication_status]:
            best[canon] = p
    return list(best.values())
