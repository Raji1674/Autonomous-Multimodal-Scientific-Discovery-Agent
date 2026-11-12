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
                 chunk_overlap: int = 150) -> IndexReport:
    """Chunk, embed (inside the store) and upsert. Citation keys are unique within the batch."""
    report = IndexReport(papers_in=len(papers))
    used: set[str] = set()
    for paper in papers:
        try:
            if not paper.abstract.strip():
                continue  # nothing to index; metadata-only records are skipped
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
