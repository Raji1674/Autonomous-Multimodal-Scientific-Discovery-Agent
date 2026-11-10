"""Index papers and optional figure descriptions into a vector store."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.schemas.paper import Chunk, Paper
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
    figures_extracted: int = 0
    figures_described: int = 0


def index_papers(
    papers: list[Paper],
    store: VectorStore,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    pdf_extractor=None,
    max_pdf_papers: int = 3,
    figure_extractor=None,
    vision_client=None,
    extract_figures: bool = False,
    vision_enabled: bool = False,
    max_figures_per_pdf: int = 10,
) -> IndexReport:
    report = IndexReport(papers_in=len(papers))
    used: set[str] = set()
    pdfs_processed = 0

    for paper in papers:
        try:
            pdf_bytes = None

            if (
                pdf_extractor is not None
                and paper.pdf_url
                and (not paper.full_text.strip() or extract_figures)
                and pdfs_processed < max_pdf_papers
            ):
                pdfs_processed += 1

                try:
                    if hasattr(pdf_extractor, "extract_with_content"):
                        extracted_text, pdf_bytes = (
                            pdf_extractor.extract_with_content(paper.pdf_url)
                        )
                    else:
                        extracted_text = pdf_extractor.extract(paper.pdf_url)

                    paper = paper.model_copy(
                        update={"full_text": extracted_text}
                    )

                except Exception as exc:
                    report.failures.append({
                        "paper_id": paper.paper_id,
                        "failure_state": FailureState.PAPER_PARSE_FAILED.value,
                        "error": str(exc),
                    })

            if not paper.abstract.strip() and not paper.full_text.strip():
                continue

            key = make_citation_key(paper, used)
            used.add(key)

            chunks = chunk_paper(
                paper,
                chunk_size,
                chunk_overlap,
                citation_key=key,
            )

            if chunks:
                report.chunks_written += store.upsert(chunks)

            if (
                extract_figures
                and pdf_bytes is not None
                and figure_extractor is not None
            ):
                try:
                    figures = figure_extractor.extract(pdf_bytes)
                    figures = figures[:max_figures_per_pdf]
                    report.figures_extracted += len(figures)
                except Exception as exc:
                    report.failures.append({
                        "paper_id": paper.paper_id,
                        "failure_state": FailureState.PAPER_PARSE_FAILED.value,
                        "error": f"Figure extraction failed: {exc}",
                    })
                    continue

                if not vision_enabled or vision_client is None:
                    continue

                figure_chunks = []

                for figure in figures:
                    try:
                        description = vision_client.describe_image(
                            system=(
                                "You are a scientific figure analysis assistant. "
                                "Describe only visible information. Do not invent "
                                "results, labels, or conclusions."
                            ),
                            user=(
                                "Describe this scientific figure for literature "
                                "search. Include visible chart type, axes, labels, "
                                "trends, legends, and important observations. "
                                "Mention uncertainty when text is unreadable."
                            ),
                            image_bytes=figure.image_bytes,
                            mime_type=figure.mime_type,
                        )

                        if not description or not description.strip():
                            raise ValueError("Vision model returned an empty description")

                        figure_chunks.append(
                            Chunk(
                                chunk_id=(
                                    f"{paper.paper_id}#figure-{figure.figure_index}"
                                ),
                                paper_id=paper.paper_id,
                                text=(
                                    f"Figure {figure.figure_index}, "
                                    f"page {figure.page_number}: {description}"
                                ),
                                metadata={
                                    "title": paper.title,
                                    "authors": paper.authors,
                                    "year": paper.date[:4] if paper.date else "",
                                    "source": paper.source,
                                    "publication_status": paper.publication_status,
                                    "citation_key": key,
                                    "content_type": "figure",
                                    "figure_index": figure.figure_index,
                                    "page_number": figure.page_number,
                                },
                            )
                        )

                        report.figures_described += 1

                    except Exception as exc:
                        report.failures.append({
                            "paper_id": paper.paper_id,
                            "failure_state": FailureState.PAPER_PARSE_FAILED.value,
                            "error": (
                                f"Figure {figure.figure_index} description failed: {exc}"
                            ),
                        })

                if figure_chunks:
                    report.chunks_written += store.upsert(figure_chunks)

        except Exception as exc:
            report.failures.append({
                "paper_id": paper.paper_id,
                "failure_state": FailureState.EMBEDDING_FAILED.value,
                "error": str(exc),
            })
            log.exception(
                "Paper indexing failed: %s",
                paper.paper_id,
            )

    return report
