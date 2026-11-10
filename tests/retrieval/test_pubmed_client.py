
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



from xml.etree import ElementTree as ET

from src.retrieval.pubmed_client import (
    _parse_date,
    parse_pubmed_xml,
    PubMedClient,
)
from src.retrieval.metadata import LiteratureSearchError


def test_parse_journal_publication_date():
    article = ET.fromstring("""
    <Article>
        <Journal>
            <JournalIssue>
                <PubDate>
                    <Year>2023</Year>
                    <Month>Mar</Month>
                </PubDate>
            </JournalIssue>
        </Journal>
    </Article>
    """)

    assert _parse_date(article) == "2023-03"


def test_parse_year_and_numeric_month():
    article = ET.fromstring("""
    <Article>
        <ArticleDate>
            <Year>2024</Year>
            <Month>2</Month>
        </ArticleDate>
    </Article>
    """)

    assert _parse_date(article) == "2024-02"


def test_parse_medline_date():
    article = ET.fromstring("""
    <Article>
        <ArticleDate>
            <MedlineDate>2020 Jan-Feb</MedlineDate>
        </ArticleDate>
    </Article>
    """)

    assert _parse_date(article) == "2020"


def test_parse_missing_date():
    article = ET.fromstring("<Article><Title>Research</Title></Article>")

    assert _parse_date(article) == ""


def test_skip_article_without_pmid():
    xml = """
    <PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <Article>
                    <ArticleTitle>Missing ID</ArticleTitle>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    </PubmedArticleSet>
    """

    assert parse_pubmed_xml(xml) == []


def test_parse_collective_author():
    xml = """
    <PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345</PMID>
                <Article>
                    <ArticleTitle>Research</ArticleTitle>
                    <AuthorList>
                        <Author>
                            <CollectiveName>Research Group</CollectiveName>
                        </Author>
                    </AuthorList>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    </PubmedArticleSet>
    """

    papers = parse_pubmed_xml(xml)

    assert papers[0].authors == ["Research Group"]


def test_parse_doi():
    xml = """
    <PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345</PMID>
                <Article>
                    <ArticleTitle>Research</ArticleTitle>
                </Article>
            </MedlineCitation>
            <PubmedData>
                <ArticleIdList>
                    <ArticleId IdType="doi">10.1234/example</ArticleId>
                </ArticleIdList>
            </PubmedData>
        </PubmedArticle>
    </PubmedArticleSet>
    """

    papers = parse_pubmed_xml(xml)

    assert papers[0].doi == "10.1234/example"


def test_pubmed_search_empty_ids():
    calls = []

    def fake_fetch(endpoint, **kwargs):
        calls.append(endpoint)
        return b"<eSearchResult><IdList></IdList></eSearchResult>"

    client = PubMedClient(fetcher=fake_fetch, sleep=lambda _: None)

    assert client.search("unknown topic") == []
    assert calls == ["esearch"]


def test_pubmed_search_multiple_batches():
    calls = []
    sleep_calls = []

    def fake_fetch(endpoint, **kwargs):
        calls.append((endpoint, kwargs))

        if endpoint == "esearch":
            return b"""
            <eSearchResult>
                <IdList>
                    <Id>1</Id><Id>2</Id><Id>3</Id>
                </IdList>
            </eSearchResult>
            """

        return b"<PubmedArticleSet></PubmedArticleSet>"

    client = PubMedClient(
        fetcher=fake_fetch,
        sleep=lambda delay: sleep_calls.append(delay),
    )

    result = client.search("research", max_results=3, batch=2)

    assert result == []
    assert len(calls) == 3
    assert len(sleep_calls) == 2


def test_pubmed_search_invalid_batch():
    client = PubMedClient(
        fetcher=lambda *args, **kwargs: (
            b"<eSearchResult><IdList><Id>1</Id></IdList></eSearchResult>"
        ),
        sleep=lambda _: None,
    )

    with pytest.raises(LiteratureSearchError):
        client.search("research", batch=0)
