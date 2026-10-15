"""
data/arxiv_fetcher.py
─────────────────────────────────────────────────────────────────
Fetches scientific papers from the arXiv API.

What is arXiv?
  arXiv (pronounced "archive") is a free, open-access repository
  where scientists post their research papers. No API key is needed.

How this module works:
  1. You give it a search query (e.g. "CRISPR cancer therapy").
  2. It asks the arXiv API for matching papers.
  3. It returns a list of dictionaries, where each dict has:
       - paper_id   : unique arXiv identifier (e.g. "2401.12345")
       - title      : paper title
       - abstract   : paper summary
       - authors    : list of author names
       - published  : publication date
       - url        : link to the arXiv page
       - pdf_url    : direct link to the PDF
       - source     : always "arxiv" (useful later when mixing sources)

Usage example:
  from data.arxiv_fetcher import fetch_arxiv_papers

  papers = fetch_arxiv_papers(query="quantum computing", max_results=5)
  for p in papers:
      print(p["title"])
      print(p["abstract"][:200])   # first 200 characters
      print(p["pdf_url"])
      print("---")
"""

import arxiv                          # official arXiv Python client
from typing import List, Dict, Any
import logging

# Set up a logger so we can see what is happening in the console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_arxiv_papers(
    query: str,
    max_results: int = 10,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
) -> List[Dict[str, Any]]:
    """
    Search arXiv for papers matching a query and return structured metadata.

    Parameters
    ----------
    query : str
        The search query. You can use natural language or arXiv field codes.
        Examples:
          "CRISPR gene editing"
          "ti:transformer AND abs:attention"   (title contains 'transformer',
                                                abstract contains 'attention')

    max_results : int
        How many papers to fetch. Default is 10. arXiv allows up to 300 per
        request but be polite — start small.

    sort_by : arxiv.SortCriterion
        How to rank results. Options:
          arxiv.SortCriterion.Relevance       ← default (best match first)
          arxiv.SortCriterion.SubmittedDate   (newest first)
          arxiv.SortCriterion.LastUpdatedDate (recently updated first)

    Returns
    -------
    List[Dict[str, Any]]
        A list of paper dictionaries. Each dict contains:
        {
            "paper_id":  str,   # e.g. "2401.12345"
            "title":     str,
            "abstract":  str,
            "authors":   List[str],
            "published": str,   # ISO date string, e.g. "2024-01-15"
            "url":       str,   # arXiv page
            "pdf_url":   str,   # direct PDF link
            "source":    str,   # always "arxiv"
        }
    """

    logger.info(f"[arXiv] Searching for: '{query}' (max {max_results} results)")

    # ── Step 1: Build the search request ──────────────────────────
    # arxiv.Search() defines WHAT to search for.
    # It does NOT hit the network yet — just stores your parameters.
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
    )

    # ── Step 2: Run the search and collect results ─────────────────
    # arxiv.Client() handles rate limiting automatically (3 sec between calls).
    # client.results(search) hits the arXiv API and returns paper objects.
    client = arxiv.Client()

    papers: List[Dict[str, Any]] = []

    try:
        for result in client.results(search):
            # Each `result` is an arxiv.Result object.
            # We convert it to a plain Python dict for easy use elsewhere.
            paper = {
                # short_id() gives "2401.12345" instead of the full URL
                "paper_id":  result.get_short_id(),
                "title":     result.title.strip(),
                # The abstract often has extra newlines — clean them up
                "abstract":  result.summary.strip().replace("\n", " "),
                # result.authors is a list of Author objects; we extract names
                "authors":   [author.name for author in result.authors],
                # published is a datetime object; convert to ISO date string
                "published": result.published.strftime("%Y-%m-%d"),
                # entry_id is the arXiv page URL
                "url":       result.entry_id,
                # pdf_url is the direct PDF download link
                "pdf_url":   result.pdf_url,
                # Tag so downstream code knows where this came from
                "source":    "arxiv",
            }
            papers.append(paper)
            logger.info(f"  ✓ Fetched: {paper['title'][:70]}...")

    except Exception as e:
        # If anything goes wrong (network error, API timeout, etc.)
        # we log it and return whatever we managed to get so far.
        logger.error(f"[arXiv] Error during fetch: {e}")

    logger.info(f"[arXiv] Done. Fetched {len(papers)} papers.")
    return papers


# ─────────────────────────────────────────────────────────────────
# Quick test — run this file directly to see it in action:
#   python data/arxiv_fetcher.py
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = fetch_arxiv_papers(
        query="large language models scientific discovery",
        max_results=3,
    )

    print(f"\n{'='*60}")
    print(f"Fetched {len(results)} papers from arXiv")
    print(f"{'='*60}\n")

    for i, paper in enumerate(results, start=1):
        print(f"[{i}] {paper['title']}")
        print(f"     Authors  : {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
        print(f"     Published: {paper['published']}")
        print(f"     ID       : {paper['paper_id']}")
        print(f"     URL      : {paper['url']}")
        print(f"     Abstract : {paper['abstract'][:150]}...")
        print()
