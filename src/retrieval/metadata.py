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
    """Drop duplicates by DOI, then by non-empty normalized title.

    When the same work appears twice, keep the record with the stronger
    publication status. Papers without usable identifiers remain distinct.
    """
    best: dict[str, Paper] = {}
    alias: dict[str, str] = {}

    for p in papers:
        keys = []

        if p.doi and p.doi.strip():
            keys.append("doi:" + p.doi.strip().lower())

        normalized_title = _title_key(p.title)
        if normalized_title:
            keys.append("title:" + normalized_title)

        # Avoid merging unrelated papers with missing metadata.
        if not keys:
            keys.append("paper:" + p.paper_id)

        canon = next((alias[k] for k in keys if k in alias), keys[0])

        for key in keys:
            alias[key] = canon

        if (
            canon not in best
            or _RANK[p.publication_status]
            > _RANK[best[canon].publication_status]
        ):
            best[canon] = p

    return list(best.values())
