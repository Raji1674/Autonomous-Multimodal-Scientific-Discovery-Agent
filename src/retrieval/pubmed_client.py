"""PubMed client (official NCBI E-utilities through Biopython's Entrez)."""
from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable

from src.retrieval.metadata import LiteratureSearchError, clean_abstract
from src.schemas.paper import Paper

_MONTHS = {m: f"{i:02d}" for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _text(el: ET.Element | None) -> str:
    return clean_abstract("".join(el.itertext())) if el is not None else ""


def _parse_date(article: ET.Element) -> str:
    """Return YYYY, YYYY-MM or YYYY-MM-DD; never pads missing parts."""
    node = article.find("ArticleDate")
    if node is None:
        node = article.find("Journal/JournalIssue/PubDate")
    if node is None:
        return ""
    year = _text(node.find("Year"))
    if not year:  # e.g. <MedlineDate>2020 Jan-Feb</MedlineDate>
        md = _text(node.find("MedlineDate"))
        return md[:4] if md[:4].isdigit() else ""
    month = _text(node.find("Month"))
    month = _MONTHS.get(month[:3].title(), month.zfill(2) if month.isdigit() else "")
    day = _text(node.find("Day"))
    if month and day.isdigit():
        return f"{year}-{month}-{int(day):02d}"
    return f"{year}-{month}" if month else year


def parse_pubmed_xml(xml_bytes: bytes | str) -> list[Paper]:
    """Parse an efetch PubmedArticleSet into Paper objects."""
    root = ET.fromstring(xml_bytes)
    papers = []
    for art in root.iter("PubmedArticle"):
        cit = art.find("MedlineCitation")
        article = cit.find("Article") if cit is not None else None
        pmid = _text(cit.find("PMID")) if cit is not None else ""
        if article is None or not pmid:
            continue
        parts = []
        for ab in article.findall("Abstract/AbstractText"):
            label = ab.get("Label")
            parts.append(f"{label}: {_text(ab)}" if label else _text(ab))
        authors = []
        for au in article.findall("AuthorList/Author"):
            last, fore = _text(au.find("LastName")), _text(au.find("ForeName"))
            name = f"{fore} {last}".strip() or _text(au.find("CollectiveName"))
            if name:
                authors.append(name)
        doi = None
        for aid in art.findall("PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = _text(aid)
        types = [_text(t) for t in article.findall("PublicationTypeList/PublicationType")]
        if "Preprint" in types:
            status = "preprint"
        else:
            status = "unknown"
        papers.append(Paper(
            paper_id=f"pmid:{pmid}", source="pubmed", title=_text(article.find("ArticleTitle")),
            authors=authors, abstract=clean_abstract(" ".join(parts)), date=_parse_date(article),
            doi=doi, pdf_url=None, journal=_text(article.find("Journal/Title")) or None,
            categories=types, publication_status=status))
    return papers


def parse_esearch_ids(xml_bytes: bytes | str) -> list[str]:
    return [_text(i) for i in ET.fromstring(xml_bytes).iter("Id")]


class PubMedClient:
    """esearch then efetch. Fetcher is injectable so tests need no network."""

    def __init__(self, fetcher: Callable[..., bytes] | None = None, sleep: Callable[[float], None] = time.sleep):
        self._api_key = os.environ.get("NCBI_API_KEY") or None
        self._delay = 0.11 if self._api_key else 0.34  # 10 req/s with key, 3 req/s without
        self._sleep = sleep
        self._fetch = fetcher or self._entrez_fetch

    def _entrez_fetch(self, endpoint: str, **kwargs) -> bytes:
        from Bio import Entrez  # lazy import

        Entrez.email = os.environ.get("NCBI_EMAIL", "")
        Entrez.tool = "autosci"
        if self._api_key:
            Entrez.api_key = self._api_key
        handle = getattr(Entrez, endpoint)(**kwargs)
        try:
            data = handle.read()
            return data if isinstance(data, bytes) else data.encode()
        finally:
            handle.close()

    def search(self, term: str, max_results: int = 50, batch: int = 50) -> list[Paper]:
        try:
            ids_xml = self._fetch("esearch", db="pubmed", term=term, retmax=max_results, sort="date")
            ids = parse_esearch_ids(ids_xml)
            papers: list[Paper] = []
            for i in range(0, len(ids), batch):
                self._sleep(self._delay)
                xml = self._fetch("efetch", db="pubmed", id=",".join(ids[i:i + batch]), retmode="xml")
                papers.extend(parse_pubmed_xml(xml))
            return papers
        except Exception as exc:  # noqa: BLE001
            raise LiteratureSearchError(f"PubMed search failed: {exc}") from exc
