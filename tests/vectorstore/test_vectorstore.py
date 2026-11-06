
from src.schemas.paper import Chunk
from src.vectorstore.chroma_store import ChromaStore
from src.vectorstore.embeddings import FakeEmbedder


def make_chunk(chunk_id, paper_id, text, source="arxiv"):
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        text=text,
        metadata={
            "title": paper_id,
            "source": source,
            "publication_status": "preprint",
        },
    )


def test_chroma_store_upserts_and_retrieves_chunks(tmp_path):
    store = ChromaStore(tmp_path, FakeEmbedder())

    chunks = [
        make_chunk(
            "paper1#0",
            "paper1",
            "Machine learning predicts student academic performance.",
        ),
        make_chunk(
            "paper2#0",
            "paper2",
            "Neural networks classify handwritten digits.",
        ),
    ]

    written = store.upsert(chunks)

    assert written == 2
    assert store.count() == 2

    results = store.query("machine learning student performance", k=2)

    assert len(results) == 2
    assert results[0].chunk.paper_id == "paper1"
    assert results[0].chunk.text == chunks[0].text


def test_chroma_store_preserves_metadata(tmp_path):
    store = ChromaStore(tmp_path, FakeEmbedder())

    chunk = make_chunk(
        "paper3#0",
        "paper3",
        "Student performance prediction using educational data.",
    )

    store.upsert([chunk])

    results = store.query("student performance prediction", k=1)

    assert len(results) == 1
    assert results[0].chunk.metadata["title"] == "paper3"
    assert results[0].chunk.metadata["source"] == "arxiv"
    assert results[0].chunk.metadata["publication_status"] == "preprint"


def test_chroma_store_filters_by_metadata(tmp_path):
    store = ChromaStore(tmp_path, FakeEmbedder())

    chunks = [
        make_chunk("paper4#0", "paper4", "Student performance prediction.", "arxiv"),
        make_chunk("paper5#0", "paper5", "Student performance prediction.", "pubmed"),
    ]

    store.upsert(chunks)

    results = store.query(
        "student performance",
        k=2,
        where={"source": "pubmed"},
    )

    assert len(results) == 1
    assert results[0].chunk.paper_id == "paper5"


def test_chroma_store_deletes_chunks(tmp_path):
    store = ChromaStore(tmp_path, FakeEmbedder())

    chunk = make_chunk(
        "paper6#0",
        "paper6",
        "Educational data and student performance.",
    )

    store.upsert([chunk])

    assert store.count() == 1

    store.delete(["paper6#0"])

    assert store.count() == 0
    assert store.query("student performance") == []


def test_chroma_store_returns_empty_when_no_chunks_exist(tmp_path):
    store = ChromaStore(tmp_path, FakeEmbedder())

    assert store.count() == 0
    assert store.query("student performance") == []


def test_chroma_store_persists_data_after_reopening(tmp_path):
    first_store = ChromaStore(tmp_path, FakeEmbedder())

    chunk = make_chunk(
        "paper7#0",
        "paper7",
        "Machine learning models predict student performance.",
    )

    first_store.upsert([chunk])

    second_store = ChromaStore(tmp_path, FakeEmbedder())

    assert second_store.count() == 1

    results = second_store.query("machine learning student performance", k=1)

    assert len(results) == 1
    assert results[0].chunk.paper_id == "paper7"
