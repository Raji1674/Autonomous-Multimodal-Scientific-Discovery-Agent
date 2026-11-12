
import pytest
import pymupdf

from src.ingestion.pdf_processor import extract_pdf_content


def create_test_pdf(path):
    """Create a small PDF with two pages for testing."""
    doc = pymupdf.open()

    page1 = doc.new_page()
    page1.insert_text(
        (72, 72),
        "Research background and introduction."
    )

    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "Methodology and experimental results."
    )

    doc.save(path)
    doc.close()


def test_extract_pdf_content_pagewise(tmp_path):
    pdf_path = tmp_path / "research.pdf"
    create_test_pdf(pdf_path)

    result = extract_pdf_content(
        str(pdf_path),
        str(tmp_path / "images")
    )

    assert result["pdf_name"] == "research.pdf"
    assert result["total_pages"] == 2
    assert len(result["pages"]) == 2

    assert result["pages"][0]["page_number"] == 1
    assert result["pages"][1]["page_number"] == 2

    assert "Research background" in result["pages"][0]["text"]
    assert "Methodology" in result["pages"][1]["text"]


def test_extract_pdf_content_word_count(tmp_path):
    pdf_path = tmp_path / "words.pdf"
    create_test_pdf(pdf_path)

    result = extract_pdf_content(
        str(pdf_path),
        str(tmp_path / "images")
    )

    expected_words = sum(
        len(page["text"].split())
        for page in result["pages"]
    )

    assert result["total_words"] == expected_words

    for page in result["pages"]:
        assert page["word_count"] == len(page["text"].split())


def test_extract_missing_pdf_raises_error(tmp_path):
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_pdf_content(str(missing_pdf))


def test_extract_pdf_without_images(tmp_path):
    pdf_path = tmp_path / "text_only.pdf"
    create_test_pdf(pdf_path)

    result = extract_pdf_content(
        str(pdf_path),
        str(tmp_path / "images")
    )

    assert result["images"] == []
