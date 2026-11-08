"""Index papers into a vector store; one bad paper never stops the batch."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.schemas.paper import Paper
from src.schemas.state import FailureState
from src.utils.citation import make_citation_key
from src.utils.logging import get_logger, log_event
from src.vectorstore.base import VectorStore
from src.vectorstore.chunking import chunk_paper

log = get_logger("indexing")


@dataclass
class IndexReport:
    papers_in: int = 0
    chunks_written: int = 0
    failures: list[dict] = field(default_factory=list)


def index_papers(papers: list[Paper], store: VectorStore, chunk_size: int = 1000,
                 chunk_overlap: int = 150, pdf_extractor=None,
                 max_pdf_papers: int = 3) -> IndexReport:
    """Chunk, embed (inside the store) and upsert. Citation keys are unique within the batch."""
    report = IndexReport(papers_in=len(papers))
    used: set[str] = set()
    pdfs_processed = 0
    for paper in papers:
        try:
            if (pdf_extractor is not None and paper.pdf_url and not paper.full_text.strip()
                    and pdfs_processed < max_pdf_papers):
                pdfs_processed += 1
                try:
                    paper = paper.model_copy(update={"full_text": pdf_extractor.extract(paper.pdf_url)})
                except Exception as exc:  # Keep indexing the abstract when a PDF is unavailable.
                    report.failures.append({
                        "paper_id": paper.paper_id,
                        "failure_state": FailureState.PAPER_PARSE_FAILED.value,
                        "error": str(exc),
                    })
            if not paper.abstract.strip():
                if not paper.full_text.strip():
                    continue  # metadata-only records are skipped
            key = make_citation_key(paper, used)
            used.add(key)
            chunks = chunk_paper(paper, chunk_size, chunk_overlap, citation_key=key)
            report.chunks_written += store.upsert(chunks)
        except Exception as exc:  # noqa: BLE001 - isolate per-paper failures
            report.failures.append({"paper_id": paper.paper_id,
                                    "failure_state": FailureState.EMBEDDING_FAILED.value,
                                    "error": str(exc)})
            log_event(log, "indexing", "index_paper", "error", error=str(exc), paper_id=paper.paper_id)
    return report
