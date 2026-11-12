
import pytest

from src.schemas.paper import Chunk
from src.vectorstore.chroma_store import ChromaStore
from src.vectorstore.embeddings import FakeEmbedder


@pytest.fixture
def store(tmp_path):
    return ChromaStore(
        path=tmp_path / "chroma_test",
        embedder=FakeEmbedder(dim=64),
    )


def make_chunk(
    chunk_id="paper1#0",
    paper_id="paper1",
    text="Cardiovascular disease prediction using machine learning",
    metadata=None,
):
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        text=text,
        metadata=metadata or {},
    )


def test_initial_store_is_empty(store):
    assert store.count() == 0


def test_upsert_single_chunk(store):
    result = store.upsert([make_chunk()])

    assert result == 1
    assert store.count() == 1


def test_upsert_multiple_chunks(store):
    chunks = [
        make_chunk("paper1#0", "paper1"),
        make_chunk("paper2#0", "paper2"),
    ]

    result = store.upsert(chunks)

    assert result == 2
    assert store.count() == 2


def test_upsert_empty_list(store):
    assert store.upsert([]) == 0
    assert store.count() == 0


def test_query_returns_relevant_chunks(store):
    store.upsert([
        make_chunk(
            "paper1#0",
            text="Cardiovascular disease prediction using machine learning",
        ),
        make_chunk(
            "paper2#0",
            text="Natural language processing and text classification",
        ),
    ])

    results = store.query("cardiovascular disease prediction", k=2)

    assert len(results) > 0
    assert results[0].chunk.text
    assert isinstance(results[0].score, float)


def test_query_empty_store(store):
    assert store.query("machine learning") == []


def test_query_respects_k(store):
    store.upsert([
        make_chunk("paper1#0"),
        make_chunk("paper2#0", "paper2", "Deep learning research"),
        make_chunk("paper3#0", "paper3", "Data science research"),
    ])

    results = store.query("research", k=2)

    assert len(results) <= 2


def test_query_with_metadata_filter(store):
    store.upsert([
        make_chunk(
            "paper1#0",
            metadata={"source": "arxiv"},
        ),
        make_chunk(
            "paper2#0",
            "paper2",
            metadata={"source": "pubmed"},
        ),
    ])

    results = store.query(
        "machine learning",
        where={"source": "arxiv"},
    )

    assert all(
        result.chunk.metadata["source"] == "arxiv"
        for result in results
    )


def test_list_metadata_is_flattened(store):
    store.upsert([
        make_chunk(
            metadata={"authors": ["Author A", "Author B"]}
        )
    ])

    results = store.query("cardiovascular")

    assert results[0].chunk.metadata["authors"] == "Author A; Author B"


def test_count_after_delete(store):
    store.upsert([
        make_chunk("paper1#0"),
        make_chunk("paper2#0", "paper2"),
    ])

    store.delete(["paper1#0"])

    assert store.count() == 1


def test_delete_empty_list(store):
    store.upsert([make_chunk()])

    store.delete([])

    assert store.count() == 1


def test_persistent_storage(tmp_path):
    path = tmp_path / "persistent_chroma"
    embedder = FakeEmbedder(dim=64)

    first_store = ChromaStore(path, embedder)
    first_store.upsert([make_chunk()])

    second_store = ChromaStore(path, embedder)

    assert second_store.count() == 1


def test_pdf_citation_metadata_survives_storage_and_retrieval(store):
    pdf_chunk = make_chunk(
        chunk_id="pdf:sample#page3#0",
        paper_id="pdf:sample",
        text="Verified research evidence from the uploaded PDF.",
        metadata={
            "title": "Sample Research PDF",
            "source": "uploaded_pdf",
            "page_number": 3,
            "citation_key": "pdf:sample",
            "publication_status": "peer_reviewed",
        },
    )

    store.upsert([pdf_chunk])

    results = store.query(
        "verified research evidence",
        k=1,
    )

    assert len(results) == 1

    metadata = results[0].chunk.metadata

    assert results[0].chunk.paper_id == "pdf:sample"
    assert metadata["page_number"] == 3
    assert metadata["citation_key"] == "pdf:sample"
    assert metadata["source"] == "uploaded_pdf"
    assert metadata["publication_status"] == "peer_reviewed"
