
from src.retrieval.pubmed_client import (
    PubMedClient,
    parse_esearch_ids,
    parse_pubmed_xml,
)
from src.retrieval.metadata import LiteratureSearchError


ARTICLE_XML = b"""
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <Journal>
          <Title>Journal of Educational Research</Title>
          <JournalIssue>
            <PubDate>
              <Year>2025</Year>
              <Month>Jan</Month>
              <Day>15</Day>
            </PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Machine Learning in Education</ArticleTitle>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>John</ForeName>
          </Author>
          <Author>
            <CollectiveName>Research Group</CollectiveName>
          </Author>
        </AuthorList>
        <Abstract>
          <AbstractText Label="BACKGROUND">Academic prediction.</AbstractText>
          <AbstractText Label="RESULTS">Improved accuracy.</AbstractText>
        </Abstract>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1234/example</ArticleId>
        <ArticleId IdType="pmc">PMC123456</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_parse_esearch_ids():
    xml = b"<eSearchResult><IdList><Id>123</Id><Id>456</Id></IdList></eSearchResult>"

    assert parse_esearch_ids(xml) == ["123", "456"]


def test_parse_pubmed_xml_extracts_article_metadata():
    papers = parse_pubmed_xml(ARTICLE_XML)

    assert len(papers) == 1

    paper = papers[0]
    assert paper.paper_id == "pmid:12345678"
    assert paper.source == "pubmed"
    assert paper.title == "Machine Learning in Education"
    assert paper.authors == ["John Smith", "Research Group"]
    assert "BACKGROUND: Academic prediction." in paper.abstract
    assert "RESULTS: Improved accuracy." in paper.abstract
    assert paper.date == "2025-01-15"
    assert paper.doi == "10.1234/example"
    assert paper.pdf_url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC123456/pdf/"
    assert paper.journal == "Journal of Educational Research"
    assert paper.publication_status == "peer_reviewed"


def test_parse_pubmed_xml_identifies_preprints():
    xml = b"""
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>789</PMID>
          <Article>
            <ArticleTitle>Preprint Study</ArticleTitle>
            <PublicationTypeList>
              <PublicationType>Preprint</PublicationType>
            </PublicationTypeList>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """

    papers = parse_pubmed_xml(xml)

    assert len(papers) == 1
    assert papers[0].publication_status == "preprint"


def test_parse_pubmed_xml_skips_articles_without_pmid():
    xml = b"""
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <Article>
            <ArticleTitle>Article Without PMID</ArticleTitle>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """

    assert parse_pubmed_xml(xml) == []


def test_parse_pubmed_xml_supports_partial_publication_date():
    xml = b"""
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>111</PMID>
          <Article>
            <Journal>
              <JournalIssue>
                <PubDate>
                  <Year>2024</Year>
                  <Month>Mar</Month>
                </PubDate>
              </JournalIssue>
            </Journal>
            <ArticleTitle>Partial Date Study</ArticleTitle>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """

    papers = parse_pubmed_xml(xml)

    assert papers[0].date == "2024-03"


def test_pubmed_search_fetches_ids_and_articles():
    calls = []
    delays = []

    def fake_fetcher(endpoint, **kwargs):
        calls.append((endpoint, kwargs))

        if endpoint == "esearch":
            return b"<eSearchResult><IdList><Id>123</Id></IdList></eSearchResult>"

        if endpoint == "efetch":
            return ARTICLE_XML

        raise ValueError("Unexpected endpoint")

    client = PubMedClient(
        fetcher=fake_fetcher,
        sleep=lambda seconds: delays.append(seconds),
    )

    papers = client.search("machine learning", max_results=10)

    assert len(papers) == 1
    assert papers[0].paper_id == "pmid:12345678"
    assert calls[0][0] == "esearch"
    assert calls[0][1]["term"] == "machine learning"
    assert calls[1][0] == "efetch"
    assert calls[1][1]["id"] == "123"
    assert len(delays) == 1


def test_pubmed_search_wraps_api_errors():
    def failing_fetcher(endpoint, **kwargs):
        raise RuntimeError("NCBI unavailable")

    client = PubMedClient(fetcher=failing_fetcher, sleep=lambda _: None)

    import pytest

    with pytest.raises(LiteratureSearchError, match="PubMed search failed"):
        client.search("machine learning")
