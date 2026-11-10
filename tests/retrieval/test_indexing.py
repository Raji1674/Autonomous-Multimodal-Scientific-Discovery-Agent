
from src.schemas.paper import Paper
from src.schemas.paper import Chunk
from src.vectorstore.indexing import index_papers


class FakeVectorStore:
    def __init__(self, fail=False):
        self.chunks = []
        self.fail = fail

    def upsert(self, chunks):
        if self.fail:
            raise RuntimeError("Vector store unavailable")
        self.chunks.extend(chunks)
        return len(chunks)


def make_paper(paper_id, abstract="This is a research abstract. " * 20):
    return Paper(
        paper_id=paper_id,
        source="arxiv",
        title=f"Research paper {paper_id}",
        authors=["Author A"],
        abstract=abstract,
        date="2025-01-10",
        publication_status="preprint",
    )


def test_index_single_paper():
    store = FakeVectorStore()
    report = index_papers([make_paper("paper1")], store)

    assert report.papers_in == 1
    assert report.chunks_written > 0
    assert len(store.chunks) > 0


def test_index_multiple_papers():
    store = FakeVectorStore()
    papers = [make_paper("paper1"), make_paper("paper2")]

    report = index_papers(papers, store)

    assert report.papers_in == 2
    assert report.chunks_written > 0


def test_skip_empty_abstract():
    store = FakeVectorStore()
    report = index_papers([make_paper("paper1", "")], store)

    assert report.papers_in == 1
    assert report.chunks_written == 0
    assert len(store.chunks) == 0


def test_continue_after_store_failure():
    store = FakeVectorStore(fail=True)
    report = index_papers([make_paper("paper1")], store)

    assert report.papers_in == 1
    assert report.chunks_written == 0
    assert len(report.failures) == 1


def test_failure_does_not_stop_next_paper():
    class PartialFailureStore:
        def __init__(self):
            self.calls = 0

        def upsert(self, chunks):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("First paper failed")
            return len(chunks)

    store = PartialFailureStore()
    report = index_papers(
        [make_paper("paper1"), make_paper("paper2")],
        store,
    )

    assert report.papers_in == 2
    assert len(report.failures) == 1
    assert report.chunks_written > 0


def test_chunk_metadata_contains_citation_key():
    store = FakeVectorStore()
    index_papers([make_paper("paper1")], store)

    assert store.chunks
    assert store.chunks[0].metadata["citation_key"]


def test_chunk_metadata_contains_source_and_year():
    store = FakeVectorStore()
    index_papers([make_paper("paper1")], store)

    metadata = store.chunks[0].metadata

    assert metadata["source"] == "arxiv"
    assert metadata["year"] == "2025"


def test_unique_chunk_ids():
    store = FakeVectorStore()
    index_papers(
        [make_paper("paper1"), make_paper("paper2")],
        store,
    )

    ids = [chunk.chunk_id for chunk in store.chunks]

    assert len(ids) == len(set(ids))
