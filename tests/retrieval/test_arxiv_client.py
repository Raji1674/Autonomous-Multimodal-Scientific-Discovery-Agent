
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import arxiv

from src.retrieval.arxiv_client import (
    ArxivClient,
    build_query,
    result_to_paper,
)
from src.retrieval.metadata import LiteratureSearchError


def make_arxiv_result(**overrides):
    data = {
        "entry_id": "https://arxiv.org/abs/2501.01234v2",
        "title": "  Machine Learning Research  ",
        "summary": "  Research   abstract. ",
        "authors": [SimpleNamespace(name="Author One")],
        "published": datetime(2025, 1, 10, tzinfo=timezone.utc),
        "journal_ref": None,
        "doi": None,
        "pdf_url": "https://arxiv.org/pdf/2501.01234",
        "categories": ["cs.AI"],
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_result_to_paper():
    paper = result_to_paper(make_arxiv_result())

    assert paper.paper_id == "arxiv:2501.01234"
    assert paper.source == "arxiv"
    assert paper.title == "Machine Learning Research"
    assert paper.abstract == "Research abstract."
    assert paper.authors == ["Author One"]
    assert paper.date == "2025-01-10"
    assert paper.publication_status == "preprint"


def test_result_to_paper_with_doi():
    paper = result_to_paper(
        make_arxiv_result(doi="10.1000/example")
    )

    assert paper.doi == "10.1000/example"
    assert paper.publication_status == "published_unverified"


def test_result_to_paper_with_journal_reference():
    paper = result_to_paper(
        make_arxiv_result(journal_ref="Example Journal")
    )

    assert paper.journal == "Example Journal"
    assert paper.publication_status == "published_unverified"


def test_build_query_without_categories():
    assert build_query("machine learning", None) == "machine learning"


def test_build_query_with_categories():
    result = build_query(
        "machine learning",
        ["cs.AI", "cs.LG"],
    )

    assert result == "(cat:cs.AI OR cat:cs.LG) AND (machine learning)"


def test_build_query_categories_without_keyword():
    assert build_query("", ["cs.AI"]) == "(cat:cs.AI)"


def test_search_returns_papers():
    class MockClient:
        def results(self, search):
            yield make_arxiv_result()

    client = ArxivClient(client=MockClient())
    papers = client.search("machine learning", max_results=5)

    assert len(papers) == 1
    assert papers[0].paper_id == "arxiv:2501.01234"


def test_search_applies_date_filter():
    class MockClient:
        def results(self, search):
            yield make_arxiv_result(
                published=datetime(2025, 2, 1, tzinfo=timezone.utc)
            )
            yield make_arxiv_result(
                entry_id="https://arxiv.org/abs/2401.00001v1",
                published=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )

    client = ArxivClient(client=MockClient())
    papers = client.search("AI", since="2025-01-01")

    assert len(papers) == 1
    assert papers[0].date == "2025-02-01"


def test_search_wraps_api_errors():
    class MockClient:
        def results(self, search):
            raise RuntimeError("API unavailable")
            yield

    client = ArxivClient(client=MockClient())

    with pytest.raises(LiteratureSearchError):
        client.search("AI")


def test_search_empty_result():
    class MockClient:
        def results(self, search):
            return iter([])

    client = ArxivClient(client=MockClient())

    assert client.search("unknown topic") == []
