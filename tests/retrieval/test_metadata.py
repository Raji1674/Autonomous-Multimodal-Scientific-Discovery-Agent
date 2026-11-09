
from src.schemas.paper import Paper
from src.retrieval.metadata import clean_abstract, dedupe


def make_paper(paper_id, title, doi=None, status="unknown"):
    return Paper(
        paper_id=paper_id,
        source="arxiv",
        title=title,
        doi=doi,
        publication_status=status,
    )


def test_clean_abstract_normalizes_whitespace():
    assert clean_abstract("  Machine   learning\n predicts   outcomes. ") == (
        "Machine learning predicts outcomes."
    )


def test_dedupe_removes_duplicate_titles():
    papers = [
        make_paper("paper1", "Machine Learning in Education"),
        make_paper("paper2", "Machine-Learning in Education"),
    ]

    result = dedupe(papers)

    assert len(result) == 1


def test_dedupe_matches_doi_case_insensitively():
    papers = [
        make_paper("paper1", "Research A", "10.1234/ABC"),
        make_paper("paper2", "Different Title", "10.1234/abc"),
    ]

    result = dedupe(papers)

    assert len(result) == 1


def test_dedupe_keeps_stronger_publication_status():
    preprint = make_paper(
        "arxiv:123",
        "Research Study",
        status="preprint",
    )
    published = make_paper(
        "pmid:456",
        "Research Study",
        status="peer_reviewed",
    )

    result = dedupe([preprint, published])

    assert len(result) == 1
    assert result[0].paper_id == "pmid:456"
    assert result[0].publication_status == "peer_reviewed"


def test_dedupe_matches_doi_and_title_aliases():
    papers = [
        make_paper("paper1", "Research Study", "10.1234/example"),
        make_paper("paper2", "Research Study"),
        make_paper("paper3", "Another Title", "10.1234/example"),
    ]

    result = dedupe(papers)

    assert len(result) == 1


def test_dedupe_preserves_distinct_papers():
    papers = [
        make_paper("paper1", "Research in Education"),
        make_paper("paper2", "Research in Healthcare"),
    ]

    assert len(dedupe(papers)) == 2


def test_dedupe_does_not_merge_papers_with_empty_titles_and_different_dois():
    papers = [
        make_paper("paper1", "", "10.1234/first"),
        make_paper("paper2", "", "10.1234/second"),
    ]

    result = dedupe(papers)

    assert len(result) == 2
