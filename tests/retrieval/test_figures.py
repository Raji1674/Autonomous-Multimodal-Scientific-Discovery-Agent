
import pymupdf

from src.retrieval.figures import PdfFigureExtractor


def make_pdf_with_image():
    """Create an in-memory PDF containing an embedded PNG image."""
    document = pymupdf.open()
    page = document.new_page()

    pixmap = pymupdf.Pixmap(
        pymupdf.csRGB,
        pymupdf.IRect(0, 0, 20, 20),
        0,
    )
    pixmap.clear_with(255)
    image_bytes = pixmap.tobytes("png")

    page.insert_image(
        pymupdf.Rect(50, 50, 250, 250),
        stream=image_bytes,
    )

    content = document.tobytes()
    document.close()
    return content


def make_pdf_without_images():
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Scientific paper without images")

    content = document.tobytes()
    document.close()
    return content


def test_extracts_embedded_pdf_image():
    extractor = PdfFigureExtractor()
    pdf_bytes = make_pdf_with_image()

    figures = extractor.extract(pdf_bytes)

    assert len(figures) == 1
    assert figures[0].page_number == 1
    assert figures[0].figure_index == 1
    assert figures[0].mime_type == "image/png"
    assert figures[0].image_bytes.startswith(b"\x89PNG")


def test_returns_empty_list_when_pdf_has_no_images():
    extractor = PdfFigureExtractor()

    figures = extractor.extract(make_pdf_without_images())

    assert figures == []


def test_respects_max_pages():
    document = pymupdf.open()

    first_page = document.new_page()
    second_page = document.new_page()

    pixmap = pymupdf.Pixmap(
        pymupdf.csRGB,
        pymupdf.IRect(0, 0, 20, 20),
        0,
    )
    pixmap.clear_with(255)
    image_bytes = pixmap.tobytes("png")

    second_page.insert_image(
        pymupdf.Rect(50, 50, 250, 250),
        stream=image_bytes,
    )

    pdf_bytes = document.tobytes()
    document.close()

    extractor = PdfFigureExtractor(max_pages=1)

    figures = extractor.extract(pdf_bytes)

    assert figures == []


def test_respects_max_figures():
    document = pymupdf.open()
    page = document.new_page()

    first_pixmap = pymupdf.Pixmap(
        pymupdf.csRGB,
        pymupdf.IRect(0, 0, 20, 20),
        0,
    )
    first_pixmap.clear_with(255)

    second_pixmap = pymupdf.Pixmap(
        pymupdf.csRGB,
        pymupdf.IRect(0, 0, 20, 20),
        0,
    )
    second_pixmap.clear_with(0)

    page.insert_image(
        pymupdf.Rect(20, 20, 100, 100),
        stream=first_pixmap.tobytes("png"),
    )
    page.insert_image(
        pymupdf.Rect(120, 20, 200, 100),
        stream=second_pixmap.tobytes("png"),
    )

    pdf_bytes = document.tobytes()
    document.close()

    extractor = PdfFigureExtractor(max_figures=1)

    figures = extractor.extract(pdf_bytes)

    assert len(figures) == 1

def test_rejects_empty_pdf_data():
    extractor = PdfFigureExtractor()

    import pytest

    with pytest.raises(ValueError, match="cannot be empty"):
        extractor.extract(b"")
