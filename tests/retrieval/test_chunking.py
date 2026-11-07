
import pytest

from src.schemas.paper import Paper
from src.vectorstore.chunking import split_text, chunk_paper


def make_paper():
    return Paper(
        paper_id="arxiv:2501.01234",
        source="arxiv",
        title="Machine Learning Research",
        authors=["Author One"],
        abstract="This is a research abstract. " * 100,
        date="2025-01-15",
        publication_status="preprint",
    )


def test_split_short_text():
    assert split_text("Short text") == ["Short text"]


def test_split_empty_text():
    assert split_text("   ") == []


def test_split_text_into_multiple_chunks():
    result = split_text("A" * 250, size=100, overlap=20)

    assert len(result) > 1
    assert all(len(chunk) <= 100 for chunk in result)


@pytest.mark.parametrize(
    "size, overlap",
    [(0, 0), (-1, 0), (100, -1), (100, 100), (50, 60)],
)
def test_invalid_chunk_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_text("Some research text", size=size, overlap=overlap)


def test_chunk_paper_metadata():
    paper = make_paper()

    chunks = chunk_paper(paper, size=100, overlap=20)

    assert len(chunks) > 1
    assert chunks[0].paper_id == paper.paper_id
    assert chunks[0].metadata["title"] == paper.title
    assert chunks[0].metadata["source"] == "arxiv"
    assert chunks[0].metadata["year"] == "2025"
    assert chunks[0].metadata["citation_key"]


def test_chunk_paper_preserves_publication_status():
    paper = make_paper()

    chunks = chunk_paper(paper, size=100, overlap=20)

    assert all(
        chunk.metadata["publication_status"] == "preprint"
        for chunk in chunks
    )


def test_chunk_ids_are_unique():
    chunks = chunk_paper(make_paper(), size=100, overlap=20)

    ids = [chunk.chunk_id for chunk in chunks]

    assert len(ids) == len(set(ids))


def test_chunk_paper_with_partial_date():
    paper = make_paper()
    paper.date = "2025"

    chunks = chunk_paper(paper)

    assert chunks[0].metadata["year"] == "2025"
