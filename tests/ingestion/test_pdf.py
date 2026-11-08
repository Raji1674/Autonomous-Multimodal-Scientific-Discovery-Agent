import pymupdf
import pytest
from unittest.mock import Mock, patch

from src.retrieval.pdf import PDFProcessingError, PdfTextExtractor


def make_pdf(*page_texts):
    """Create a small PDF in memory for testing."""
    document = pymupdf.open()

    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)

    content = document.tobytes()
    document.close()
    return content


def make_response(
    *,
    content=b"%PDF-test",
    status_code=200,
    headers=None,
    is_redirect=False,
):
    response = Mock()
    response.is_redirect = is_redirect
    response.headers = headers or {}
    response.status_code = status_code
    response.iter_content.return_value = [content]
    response.raise_for_status.return_value = None
    return response


@pytest.mark.parametrize(
    "url",
    [
        "http://arxiv.org/pdf/1234",
        "https://example.com/paper.pdf",
        "https://user:password@arxiv.org/pdf/1234",
        "https://arxiv.org:444/pdf/1234",
    ],
)
def test_rejects_unsafe_urls(url):
    with pytest.raises(PDFProcessingError):
        PdfTextExtractor._validate_url(url)


def test_rejects_redirect_to_unapproved_host():
    extractor = PdfTextExtractor()
    response = make_response(
        headers={"Location": "https://example.com/file.pdf"},
        is_redirect=True,
    )

    with patch("src.retrieval.pdf.requests.get", return_value=response):
        with pytest.raises(PDFProcessingError):
            extractor.download("https://arxiv.org/pdf/1234")

    response.close.assert_called_once()


def test_rejects_redirect_without_location():
    extractor = PdfTextExtractor()
    response = make_response(is_redirect=True)

    with patch("src.retrieval.pdf.requests.get", return_value=response):
        with pytest.raises(PDFProcessingError, match="without a location"):
            extractor.download("https://arxiv.org/pdf/1234")


def test_follows_redirect_to_approved_host():
    extractor = PdfTextExtractor()

    redirect = make_response(
        headers={"Location": "https://export.arxiv.org/pdf/1234"},
        is_redirect=True,
    )
    final_response = make_response(content=b"%PDF-valid")

    with patch(
        "src.retrieval.pdf.requests.get",
        side_effect=[redirect, final_response],
    ):
        result = extractor.download("https://arxiv.org/pdf/1234")

    assert result == b"%PDF-valid"
    redirect.close.assert_called_once()
    final_response.close.assert_called_once()


def test_rejects_pdf_exceeding_content_length_limit():
    extractor = PdfTextExtractor(max_bytes=10)
    response = make_response(
        headers={"Content-Length": "100"},
    )

    with patch("src.retrieval.pdf.requests.get", return_value=response):
        with pytest.raises(PDFProcessingError, match="size limit"):
            extractor.download("https://arxiv.org/pdf/1234")

    response.close.assert_called_once()


def test_rejects_non_pdf_response():
    extractor = PdfTextExtractor()
    response = make_response(content=b"This is not a PDF")

    with patch("src.retrieval.pdf.requests.get", return_value=response):
        with pytest.raises(PDFProcessingError, match="did not return a PDF"):
            extractor.download("https://arxiv.org/pdf/1234")


def test_extracts_text_from_pdf():
    extractor = PdfTextExtractor()
    pdf_content = make_pdf("Scientific research test")

    with patch.object(extractor, "download", return_value=pdf_content):
        result = extractor.extract("https://arxiv.org/pdf/1234")

    assert "Scientific research test" in result


def test_extract_respects_max_pages():
    extractor = PdfTextExtractor(max_pages=1)
    pdf_content = make_pdf("First page text", "Second page text")

    with patch.object(extractor, "download", return_value=pdf_content):
        result = extractor.extract("https://arxiv.org/pdf/1234")

    assert "First page text" in result
    assert "Second page text" not in result
