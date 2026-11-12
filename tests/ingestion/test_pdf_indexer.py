
from src.ingestion.pdf_indexer import index_pdf_content


class FakeStore:
    def __init__(self):
        self.chunks = []

    def upsert(self, chunks):
        self.chunks.extend(chunks)
        return len(chunks)


def test_pdf_chunks_preserve_page_and_citation_metadata():
    store = FakeStore()

    pdf_result = {
        "pdf_name": "research_paper.pdf",
        "pages": [
            {
                "page_number": 1,
                "text": "Introduction and research background."
            },
            {
                "page_number": 2,
                "text": "Methodology and experimental results."
            }
        ]
    }

    result = index_pdf_content(pdf_result, store)

    assert result["success"] is True
    assert result["chunks_created"] == 2
    assert result["chunks_indexed"] == 2

    assert store.chunks[0].metadata["page_number"] == 1
    assert store.chunks[1].metadata["page_number"] == 2

    assert store.chunks[0].metadata["citation_key"] == "pdf:research_paper"
    assert store.chunks[1].metadata["citation_key"] == "pdf:research_paper"

    assert store.chunks[0].metadata["source"] == "uploaded_pdf"


def test_pdf_chunk_ids_are_unique_across_pages():
    store = FakeStore()

    pdf_result = {
        "pdf_name": "sample.pdf",
        "pages": [
            {"page_number": 1, "text": "Page one content."},
            {"page_number": 2, "text": "Page two content."}
        ]
    }

    index_pdf_content(pdf_result, store)

    chunk_ids = [chunk.chunk_id for chunk in store.chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_empty_pdf_pages_are_skipped():
    store = FakeStore()

    pdf_result = {
        "pdf_name": "empty.pdf",
        "pages": [
            {"page_number": 1, "text": ""},
            {"page_number": 2, "text": "   "}
        ]
    }

    result = index_pdf_content(pdf_result, store)

    assert result["chunks_created"] == 0
    assert result["chunks_indexed"] == 0
    assert result["success"] is True
    assert store.chunks == []
