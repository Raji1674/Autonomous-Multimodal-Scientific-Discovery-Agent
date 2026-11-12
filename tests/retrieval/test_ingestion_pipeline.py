
from src.ingestion.pipeline import run_ingestion
from src.retrieval.metadata import LiteratureSearchError
from src.schemas.paper import Paper


def make_paper(paper_id, title="Research Paper"):
    return Paper(
        paper_id=paper_id,
        source="arxiv",
        title=title,
        authors=["Author A"],
        abstract="This is a research abstract. " * 20,
        date="2025-01-10",
        publication_status="preprint",
    )


class FakeClient:
    def __init__(self, papers=None, error=None):
        self.papers = papers or []
        self.error = error
        self.calls = []

    def search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.papers


class FakeStore:
    def __init__(self):
        self.chunks = []

    def upsert(self, chunks):
        self.chunks.extend(chunks)
        return len(chunks)


def test_ingestion_from_both_sources():
    arxiv = FakeClient([make_paper("a1")])
    pubmed = FakeClient([
        make_paper("p1", "Another Research Paper")
    ])
    store = FakeStore()

    report = run_ingestion(
        "machine learning",
        store,
        arxiv_client=arxiv,
        pubmed_client=pubmed,
    )

    assert report.fetched == {"arxiv": 1, "pubmed": 1}
    assert report.after_dedupe == 2
    assert report.chunks_written > 0
    assert len(report.failures) == 0


def test_arxiv_failure_pubmed_success():
    arxiv = FakeClient(error=LiteratureSearchError("Arxiv unavailable"))
    pubmed = FakeClient([make_paper("p1")])
    store = FakeStore()

    report = run_ingestion(
        "AI research",
        store,
        arxiv_client=arxiv,
        pubmed_client=pubmed,
    )

    assert report.fetched["arxiv"] == 0
    assert report.fetched["pubmed"] == 1
    assert report.after_dedupe == 1
    assert len(report.failures) == 1


def test_pubmed_failure_arxiv_success():
    arxiv = FakeClient([make_paper("a1")])
    pubmed = FakeClient(error=LiteratureSearchError("PubMed unavailable"))
    store = FakeStore()

    report = run_ingestion(
        "AI research",
        store,
        arxiv_client=arxiv,
        pubmed_client=pubmed,
    )

    assert report.fetched["arxiv"] == 1
    assert report.fetched["pubmed"] == 0
    assert report.after_dedupe == 1
    assert len(report.failures) == 1


def test_both_sources_fail():
    arxiv = FakeClient(error=LiteratureSearchError("Arxiv unavailable"))
    pubmed = FakeClient(error=LiteratureSearchError("PubMed unavailable"))
    store = FakeStore()

    report = run_ingestion(
        "AI research",
        store,
        arxiv_client=arxiv,
        pubmed_client=pubmed,
    )

    assert report.after_dedupe == 0
    assert report.chunks_written == 0
    assert len(report.failures) == 2


def test_duplicate_papers_are_removed():
    arxiv = FakeClient([make_paper("a1", "Same Research")])
    pubmed = FakeClient([make_paper("p1", "Same Research")])
    store = FakeStore()

    report = run_ingestion(
        "research",
        store,
        arxiv_client=arxiv,
        pubmed_client=pubmed,
    )

    assert report.after_dedupe == 1


def test_empty_search_results():
    arxiv = FakeClient()
    pubmed = FakeClient()
    store = FakeStore()

    report = run_ingestion(
        "unknown topic",
        store,
        arxiv_client=arxiv,
        pubmed_client=pubmed,
    )

    assert report.after_dedupe == 0
    assert report.chunks_written == 0
    assert report.failures == []


def test_max_per_source_is_passed_to_clients():
    arxiv = FakeClient()
    pubmed = FakeClient()
    store = FakeStore()

    run_ingestion(
        "research",
        store,
        arxiv_client=arxiv,
        pubmed_client=pubmed,
        max_per_source=10,
    )

    assert arxiv.calls[0][0][1] == 10
    assert pubmed.calls[0][0][1] == 10
