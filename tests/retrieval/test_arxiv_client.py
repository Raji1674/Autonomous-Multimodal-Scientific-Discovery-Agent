
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.retrieval.arxiv_client import (
    ArxivClient,
    build_query,
    result_to_paper,
)
from src.retrieval.metadata import LiteratureSearchError


def make_arxiv_result(
    entry_id="https://arxiv.org/abs/2501.01234v2",
    title="  Machine Learning for Students  ",
    summary="  Predicting academic performance.  ",
    published=None,
    doi=None,
    journal_ref=None,
):
    return SimpleNamespace(
        entry_id=entry_id,
        title=title,
        summary=summary,
        authors=[SimpleNamespace(name="Author One")],
        published=published or datetime(2025, 1, 15, tzinfo=timezone.utc),
        doi=doi,
        journal_ref=journal_ref,
        pdf_url="https://arxiv.org/pdf/2501.01234",
        categories=["cs.LG", "stat.ML"],
    )


def test_build_query_without_categories():
    assert build_query("machine learning", None) == "machine learning"


def test_build_query_with_categories():
    query = build_query("machine learning", ["cs.LG", "stat.ML"])

    assert query == "(cat:cs.LG OR cat:stat.ML) AND (machine learning)"


def test_result_to_paper_maps_arxiv_metadata():
    result = make_arxiv_result()

    paper = result_to_paper(result)

    assert paper.paper_id == "arxiv:2501.01234"
    assert paper.source == "arxiv"
    assert paper.title == "Machine Learning for Students"
    assert paper.abstract == "Predicting academic performance."
    assert paper.authors == ["Author One"]
    assert paper.date == "2025-01-15"
    assert paper.pdf_url == "https://arxiv.org/pdf/2501.01234"
    assert paper.categories == ["cs.LG", "stat.ML"]
    assert paper.publication_status == "preprint"


def test_result_to_paper_marks_publication_metadata_as_unverified():
    result = make_arxiv_result(
        doi="10.1234/example",
        journal_ref="Example Journal",
    )

    paper = result_to_paper(result)

    assert paper.doi == "10.1234/example"
    assert paper.journal == "Example Journal"
    assert paper.publication_status == "published_unverified"


def test_arxiv_search_returns_papers_newest_first():
    newest = make_arxiv_result(
        entry_id="https://arxiv.org/abs/2502.00001",
        published=datetime(2025, 2, 1, tzinfo=timezone.utc),
    )
    older = make_arxiv_result(
        entry_id="https://arxiv.org/abs/2501.00001",
        published=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    class FakeClient:
        def results(self, search):
            self.search_request = search
            return iter([newest, older])

    fake_client = FakeClient()
    client = ArxivClient(client=fake_client)

    papers = client.search("student performance", max_results=10)

    assert len(papers) == 2
    assert papers[0].paper_id == "arxiv:2502.00001"
    assert papers[1].paper_id == "arxiv:2501.00001"
    assert fake_client.search_request.max_results == 10


def test_arxiv_search_filters_papers_before_since_date():
    newest = make_arxiv_result(
        entry_id="https://arxiv.org/abs/2502.00001",
        published=datetime(2025, 2, 1, tzinfo=timezone.utc),
    )
    older = make_arxiv_result(
        entry_id="https://arxiv.org/abs/2501.00001",
        published=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )

    class FakeClient:
        def results(self, search):
            return iter([newest, older])

    client = ArxivClient(client=FakeClient())

    papers = client.search("student performance", since="2025-01-15")

    assert len(papers) == 1
    assert papers[0].paper_id == "arxiv:2502.00001"


def test_arxiv_search_wraps_api_errors():
    class FailingClient:
        def results(self, search):
            raise RuntimeError("API unavailable")

    client = ArxivClient(client=FailingClient())

    with pytest.raises(LiteratureSearchError, match="arXiv search failed"):
        client.search("student performance")
