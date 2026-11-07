
import pytest

from src.retrieval.pubmed_client import (
    PubMedClient,
    parse_pubmed_xml,
    parse_esearch_ids,
    _parse_date,
)
from src.retrieval.metadata import LiteratureSearchError


XML = b"""
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>Cardiovascular Disease Prediction</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Research background.</AbstractText>
          <AbstractText>Research findings.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>John</ForeName>
          </Author>
        </AuthorList>
        <Journal>
          <Title>Medical Journal</Title>
          <JournalIssue>
            <PubDate>
              <Year>2025</Year>
              <Month>Jan</Month>
              <Day>5</Day>
            </PubDate>
          </JournalIssue>
        </Journal>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1000/example</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_parse_pubmed_xml():
    papers = parse_pubmed_xml(XML)

    assert len(papers) == 1
    assert papers[0].paper_id == "pmid:12345678"
    assert papers[0].source == "pubmed"
    assert papers[0].title == "Cardiovascular Disease Prediction"
    assert papers[0].authors == ["John Smith"]
    assert papers[0].doi == "10.1000/example"
    assert papers[0].date == "2025-01-05"
    assert papers[0].publication_status == "peer_reviewed"


def test_parse_pubmed_preprint_status():
    xml = XML.replace(
        b"Journal Article",
        b"Preprint",
    )

    papers = parse_pubmed_xml(xml)

    assert papers[0].publication_status == "preprint"


def test_parse_pubmed_empty_xml():
    papers = parse_pubmed_xml(b"<PubmedArticleSet />")

    assert papers == []


def test_parse_pubmed_invalid_xml():
    with pytest.raises(Exception):
        parse_pubmed_xml(b"<invalid")


def test_parse_esearch_ids():
    xml = b"<IdList><Id>123</Id><Id>456</Id></IdList>"

    assert parse_esearch_ids(xml) == ["123", "456"]


def test_parse_esearch_empty_ids():
    assert parse_esearch_ids(b"<IdList />") == []


def test_parse_date_full():
    from xml.etree import ElementTree as ET

    article = ET.fromstring(
        """
        <Article>
            <ArticleDate>
                <Year>2025</Year>
                <Month>Feb</Month>
                <Day>9</Day>
            </ArticleDate>
        </Article>
        """
    )

    assert _parse_date(article) == "2025-02-09"

def test_parse_date_year_only():
    from xml.etree import ElementTree as ET

    article = ET.fromstring(
        """
        <Article>
            <ArticleDate>
                <Year>2024</Year>
            </ArticleDate>
        </Article>
        """
    )

    assert _parse_date(article) == "2024"


def test_pubmed_search_success():
    calls = []

    def fake_fetch(endpoint, **kwargs):
        calls.append(endpoint)

        if endpoint == "esearch":
            return b"<IdList><Id>123</Id></IdList>"

        return XML

    client = PubMedClient(fetcher=fake_fetch, sleep=lambda _: None)
    papers = client.search("heart disease", max_results=10)

    assert len(papers) == 1
    assert calls == ["esearch", "efetch"]


def test_pubmed_search_failure():
    def fake_fetch(endpoint, **kwargs):
        raise RuntimeError("NCBI unavailable")

    client = PubMedClient(fetcher=fake_fetch, sleep=lambda _: None)

    with pytest.raises(LiteratureSearchError):
        client.search("heart disease")
