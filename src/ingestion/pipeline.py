"""Ingestion: arXiv + PubMed -> dedupe -> chunk/embed -> vector store."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from src.retrieval.arxiv_client import ArxivClient
from src.retrieval.metadata import LiteratureSearchError, dedupe
from src.retrieval.pubmed_client import PubMedClient
from src.schemas.paper import Paper
from src.schemas.state import FailureState
from src.utils.logging import get_logger, log_event
from src.vectorstore.base import VectorStore
from src.vectorstore.indexing import index_papers

log = get_logger("ingestion")


@dataclass
class IngestionReport:
    fetched: dict[str, int] = field(default_factory=dict)
    after_dedupe: int = 0
    chunks_written: int = 0
    failures: list[dict] = field(default_factory=list)


def run_ingestion(query: str, store: VectorStore, arxiv_client=None, pubmed_client=None,
                  max_per_source: int = 50, categories: list[str] | None = None,
                  pdf_extractor=None, max_pdf_papers: int = 3) -> IngestionReport:
    """One source failing never stops the other; failures are recorded, not raised."""
    report = IngestionReport()
    papers: list[Paper] = []
    sources = {"arxiv": lambda: (arxiv_client or ArxivClient()).search(query, max_per_source, categories),
               "pubmed": lambda: (pubmed_client or PubMedClient()).search(query, max_per_source)}
    for name, fetch in sources.items():
        try:
            got = fetch()
            report.fetched[name] = len(got)
            papers.extend(got)
        except LiteratureSearchError as exc:
            report.fetched[name] = 0
            report.failures.append({"source": name, "failure_state": FailureState.LITERATURE_SEARCH_FAILED.value,
                                    "error": str(exc)})
            log_event(log, "ingestion", f"fetch_{name}", "error", error=str(exc))
    unique = dedupe(papers)
    report.after_dedupe = len(unique)
    idx = index_papers(unique, store, pdf_extractor=pdf_extractor,
                       max_pdf_papers=max_pdf_papers)
    report.chunks_written = idx.chunks_written
    report.failures.extend(idx.failures)
    log_event(log, "ingestion", "run", "ok", fetched=report.fetched, unique=report.after_dedupe,
              chunks=report.chunks_written, failures=len(report.failures))
    return report


def main() -> None:  # pragma: no cover - needs network and model download
    from src.utils.config import load_config
    from src.vectorstore.chroma_store import ChromaStore
    from src.vectorstore.embeddings import SentenceTransformerEmbedder

    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--max", type=int, default=50)
    args = ap.parse_args()
    cfg = load_config()
    store = ChromaStore(cfg.retrieval.chroma_path, SentenceTransformerEmbedder(cfg.retrieval.embedding_model))
    print(run_ingestion(args.query, store, max_per_source=args.max))
    print("chunks in store:", store.count())


if __name__ == "__main__":
    main()
