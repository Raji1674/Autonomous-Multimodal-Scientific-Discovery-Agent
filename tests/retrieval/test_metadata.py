import pytest

from src.retrieval.metadata import (
    clean_abstract,
    dedupe,
    LiteratureSearchError,
)
from src.schemas.paper import Paper


def make_paper(
    paper_id,
    title="Machine Learning in Healthcare",
    doi=None,
    status="preprint",
    source="arxiv",
):
    return Paper(
        paper_id=paper_id,
        source=source,
        title=title,
        authors=["Author A"],
        abstract="Research abstract",
        doi=doi,
        publication_status=status,
    )


def test_clean_abstract_removes_extra_spaces():
    assert clean_abstract("  Machine   Learning \n Research ") == (
        "Machine Learning Research"
    )


def test_clean_abstract_handles_none():
    assert clean_abstract(None) == ""


def test_dedupe_by_same_doi():
    papers = [
        make_paper("paper1", doi="10.1234/ABC"),
        make_paper("paper2", doi="10.1234/abc"),
    ]

    result = dedupe(papers)

    assert len(result) == 1


def test_dedupe_by_normalized_title():
    papers = [
        make_paper("paper1", title="Machine Learning in Healthcare"),
        make_paper("paper2", title="Machine-Learning in Healthcare!"),
    ]

    assert len(dedupe(papers)) == 1


def test_dedupe_prefers_peer_reviewed():
    papers = [
        make_paper("paper1", status="preprint"),
        make_paper("paper2", status="peer_reviewed"),
    ]

    result = dedupe(papers)

    assert len(result) == 1
    assert result[0].publication_status == "peer_reviewed"


def test_dedupe_prefers_published_unverified_over_preprint():
    papers = [
        make_paper("paper1", status="preprint"),
        make_paper("paper2", status="published_unverified"),
    ]

    result = dedupe(papers)

    assert len(result) == 1
    assert result[0].publication_status == "published_unverified"


def test_dedupe_keeps_different_papers():
    papers = [
        make_paper("paper1", title="Heart Disease Prediction"),
        make_paper("paper2", title="Cancer Detection"),
    ]

    assert len(dedupe(papers)) == 2


def test_dedupe_empty_list():
    assert dedupe([]) == []


def test_literature_search_error():
    with pytest.raises(LiteratureSearchError):
        raise LiteratureSearchError("API unavailable")