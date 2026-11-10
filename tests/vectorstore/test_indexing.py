
import pytest

from src.schemas.paper import Paper
from src.schemas.state import FailureState
from src.vectorstore.indexing import index_papers


class FakeStore:
    def __init__(self):
        self.chunks = []

    def upsert(self, chunks):
        self.chunks.extend(chunks)
        return len(chunks)


def make_paper(paper_id, abstract="A scientific abstract.", full_text=""):
    return Paper(
        paper_id=paper_id,
        source="arxiv",
        title=f"Research Paper {paper_id}",
        abstract=abstract,
        full_text=full_text,
        date="2025-01-15",
        publication_status="preprint",
    )


def test_index_papers_indexes_abstract_when_pdf_is_unavailable():
    store = FakeStore()
    paper = make_paper("paper1")

    report = index_papers([paper], store)

    assert report.papers_in == 1
    assert report.chunks_written > 0
    assert len(store.chunks) > 0
    assert store.chunks[0].paper_id == "paper1"


def test_index_papers_uses_extracted_pdf_text():
    store = FakeStore()

    class FakePdfExtractor:
        def extract(self, url):
            return "Extracted full scientific paper text."

    paper = make_paper("paper2", full_text="")
    paper = paper.model_copy(update={"pdf_url": "https://arxiv.org/pdf/2501.00001"})

    report = index_papers(
        [paper],
        store,
        pdf_extractor=FakePdfExtractor(),
    )

    assert report.chunks_written > 0
    assert any(
        "Extracted full scientific paper text." in chunk.text
        for chunk in store.chunks
    )


def test_index_papers_falls_back_to_abstract_when_pdf_extraction_fails():
    store = FakeStore()

    class FailingPdfExtractor:
        def extract(self, url):
            raise RuntimeError("PDF download failed")

    paper = make_paper("paper3", abstract="Fallback abstract text.")
    paper = paper.model_copy(update={"pdf_url": "https://arxiv.org/pdf/2501.00002"})

    report = index_papers(
        [paper],
        store,
        pdf_extractor=FailingPdfExtractor(),
    )

    assert report.chunks_written > 0
    assert any(
        failure["failure_state"] == FailureState.PAPER_PARSE_FAILED.value
        for failure in report.failures
    )
    assert any("Fallback abstract text." in chunk.text for chunk in store.chunks)


def test_index_papers_continues_after_embedding_failure():
    class PartiallyFailingStore(FakeStore):
        def upsert(self, chunks):
            if chunks[0].paper_id == "paper4":
                raise RuntimeError("Embedding failed")
            return super().upsert(chunks)

    store = PartiallyFailingStore()

    papers = [
        make_paper("paper4"),
        make_paper("paper5"),
    ]

    report = index_papers(papers, store)

    assert report.papers_in == 2
    assert report.chunks_written > 0
    assert any(
        failure["failure_state"] == FailureState.EMBEDDING_FAILED.value
        for failure in report.failures
    )
    assert any(chunk.paper_id == "paper5" for chunk in store.chunks)


def test_index_papers_skips_papers_without_usable_text():
    store = FakeStore()
    paper = make_paper("paper6", abstract="", full_text="")

    report = index_papers([paper], store)

    assert report.papers_in == 1
    assert report.chunks_written == 0
    assert store.chunks == []

def test_index_papers_indexes_figure_description():
    store = FakeStore()

    class FakePdfExtractor:
        def extract_with_content(self, url):
            return "Extracted paper text.", b"fake-pdf-bytes"

    class FakeFigureExtractor:
        def extract(self, pdf_bytes):
            assert pdf_bytes == b"fake-pdf-bytes"

            from src.retrieval.figures import ExtractedFigure

            return [
                ExtractedFigure(
                    figure_index=1,
                    page_number=2,
                    image_bytes=b"fake-image-bytes",
                    mime_type="image/png",
                )
            ]

    class FakeVisionClient:
        def describe_image(self, system, user, image_bytes, mime_type):
            assert image_bytes == b"fake-image-bytes"
            assert mime_type == "image/png"
            return "Figure 1 shows a positive correlation."

    paper = make_paper("paper7")
    paper = paper.model_copy(
        update={"pdf_url": "https://arxiv.org/pdf/2501.00007"}
    )

    report = index_papers(
        [paper],
        store,
        pdf_extractor=FakePdfExtractor(),
        figure_extractor=FakeFigureExtractor(),
        vision_client=FakeVisionClient(),
        extract_figures=True,
        vision_enabled=True,
        max_figures_per_pdf=10,
    )

    figure_chunks = [
        chunk for chunk in store.chunks
        if chunk.metadata.get("content_type") == "figure"
    ]

    assert len(figure_chunks) == 1
    assert "positive correlation" in figure_chunks[0].text
    assert figure_chunks[0].metadata["page_number"] == 2
    assert report.figures_extracted == 1
    assert report.figures_described == 1

def test_index_papers_extracts_figures_when_full_text_exists():
    store = FakeStore()

    class FakePdfExtractor:
        def extract_with_content(self, url):
            return "Existing PDF text.", b"fake-pdf-bytes"

    class FakeFigureExtractor:
        def extract(self, pdf_bytes):
            assert pdf_bytes == b"fake-pdf-bytes"

            from src.retrieval.figures import ExtractedFigure

            return [
                ExtractedFigure(
                    figure_index=1,
                    page_number=2,
                    image_bytes=b"fake-image-bytes",
                    mime_type="image/png",
                )
            ]

    class FakeVisionClient:
        def describe_image(self, system, user, image_bytes, mime_type):
            return "Figure shows a positive correlation."

    paper = make_paper(
        "paper8",
        full_text="This paper already has extracted text.",
    )
    paper = paper.model_copy(
        update={"pdf_url": "https://arxiv.org/pdf/2501.00008"}
    )

    report = index_papers(
        [paper],
        store,
        pdf_extractor=FakePdfExtractor(),
        figure_extractor=FakeFigureExtractor(),
        vision_client=FakeVisionClient(),
        extract_figures=True,
        vision_enabled=True,
    )

    assert report.figures_extracted == 1
    assert report.figures_described == 1
    assert any(
        chunk.metadata.get("content_type") == "figure"
        for chunk in store.chunks
    )
