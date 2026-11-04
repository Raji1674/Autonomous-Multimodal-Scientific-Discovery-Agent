"""arXiv client (official API through the maintained `arxiv` package)."""
from __future__ import annotations

import re
from datetime import datetime, timezone

import arxiv

from src.retrieval.metadata import LiteratureSearchError, clean_abstract
from src.schemas.paper import Paper


def result_to_paper(r) -> Paper:
    """Map an arxiv.Result (or any object with the same attributes) to a Paper."""
    short = r.entry_id.split("/abs/")[-1]
    short = re.sub(r"v\d+$", "", short)  # drop version suffix
    has_pub_info = bool(r.journal_ref) or bool(r.doi)
    return Paper(
        paper_id=f"arxiv:{short}",
        source="arxiv",
        title=clean_abstract(r.title),
        authors=[a.name for a in r.authors],
        abstract=clean_abstract(r.summary),
        date=r.published.date().isoformat() if r.published else "",
        doi=r.doi or None,
        pdf_url=getattr(r, "pdf_url", None),
        journal=r.journal_ref or None,
        categories=list(r.categories or []),
        # arXiv metadata alone cannot verify peer review.
        publication_status="published_unverified" if has_pub_info else "preprint",
    )


def build_query(query: str, categories: list[str] | None) -> str:
    if not categories:
        return query
    cats = " OR ".join(f"cat:{c}" for c in categories)
    return f"({cats}) AND ({query})" if query else f"({cats})"


class ArxivClient:
    """Rate limited (3 s between requests, as arXiv asks) with automatic retries."""

    def __init__(self, client: arxiv.Client | None = None):
        self._client = client or arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)

    def search(self, query: str, max_results: int = 50, categories: list[str] | None = None,
               since: str | None = None) -> list[Paper]:
        """Newest-first search. ``since`` is an ISO date; older papers are skipped."""
        cutoff = datetime.fromisoformat(since).replace(tzinfo=timezone.utc) if since else None
        search = arxiv.Search(query=build_query(query, categories), max_results=max_results,
                              sort_by=arxiv.SortCriterion.SubmittedDate,
                              sort_order=arxiv.SortOrder.Descending)
        papers: list[Paper] = []
        try:
            for r in self._client.results(search):
                if cutoff and r.published and r.published < cutoff:
                    break  # results are newest-first
                papers.append(result_to_paper(r))
        except Exception as exc:  # noqa: BLE001
            raise LiteratureSearchError(f"arXiv search failed: {exc}") from exc
        return papers
