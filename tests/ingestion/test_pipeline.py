from src.retrieval.metadata import LiteratureSearchError
from src.ingestion.pipeline import run_ingestion
from src.schemas.paper import Paper


class FakeStore:
    def __init__(self):
        self.chunks = []

    def upsert(self, chunks):
        self.chunks.extend(chunks)
        return len(chunks)


class FakeArxivClient:
    def search(self, query, max_results, categories=None):
        return [
            Paper(
                paper_id="arxiv:2501.00001",
                source="arxiv",
                title="Machine Learning for Student Performance",
                authors=["Author A"],
                abstract="Machine learning can help predict student performance.",
                date="2025",
                publication_status="preprint",
            )
        ]


class FakePubMedClient:
    def search(self, query, max_results):
        return [
            Paper(
                paper_id="pmid:12345678",
                source="pubmed",
                title="Educational Data Mining",
                authors=["Author B"],
                abstract="Educational data mining analyzes student learning data.",
                date="2024",
                publication_status="published_unverified",
            )
        ]


def test_ingestion_fetches_deduplicates_and_indexes_papers():
    store = FakeStore()

    report = run_ingestion(
        query="Predict student performance",
        store=store,
        arxiv_client=FakeArxivClient(),
        pubmed_client=FakePubMedClient(),
        max_per_source=5,
    )

    assert report.fetched == {"arxiv": 1, "pubmed": 1}
    assert report.after_dedupe == 2
    assert report.chunks_written > 0
    assert len(store.chunks) == report.chunks_written
    assert report.failures == []

class FailingArxivClient:
    def search(self, query, max_results, categories=None):
        raise LiteratureSearchError("Simulated arXiv failure")


def test_ingestion_continues_when_one_source_fails():
    store = FakeStore()

    report = run_ingestion(
        query="Predict student performance",
        store=store,
        arxiv_client=FailingArxivClient(),
        pubmed_client=FakePubMedClient(),
        max_per_source=5,
    )

    assert report.fetched["arxiv"] == 0
    assert report.fetched["pubmed"] == 1
    assert report.after_dedupe == 1
    assert report.chunks_written > 0
    assert len(report.failures) == 1
    assert report.failures[0]["source"] == "arxiv"

class DuplicateArxivClient:
    def search(self, query, max_results, categories=None):
        return [
            Paper(
                paper_id="arxiv:2501.00001",
                source="arxiv",
                title="Machine Learning for Student Performance",
                abstract="A study of student performance.",
                doi="10.1234/example",
                publication_status="preprint",
            )
        ]


class DuplicatePubMedClient:
    def search(self, query, max_results):
        return [
            Paper(
                paper_id="pmid:12345678",
                source="pubmed",
                title="MACHINE LEARNING FOR STUDENT PERFORMANCE!",
                abstract="A published study of student performance.",
                doi="10.1234/example",
                publication_status="peer_reviewed",
            )
        ]


def test_ingestion_deduplicates_papers_and_keeps_stronger_status():
    store = FakeStore()

    report = run_ingestion(
        query="Predict student performance",
        store=store,
        arxiv_client=DuplicateArxivClient(),
        pubmed_client=DuplicatePubMedClient(),
    )

    assert report.fetched == {"arxiv": 1, "pubmed": 1}
    assert report.after_dedupe == 1
    assert report.chunks_written > 0
    assert store.chunks[0].metadata["publication_status"] == "peer_reviewed"
