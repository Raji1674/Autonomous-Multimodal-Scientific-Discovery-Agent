
"""Extract embedded images from scientific PDF documents."""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf


@dataclass(frozen=True)
class ExtractedFigure:
    """An image extracted from a PDF page."""

    figure_index: int
    page_number: int
    image_bytes: bytes
    mime_type: str


class PdfFigureExtractor:
    """Extract unique embedded images from PDF documents."""

    def __init__(
        self,
        *,
        max_pages: int = 300,
        max_figures: int = 20,
        max_image_bytes: int = 10 * 1024 * 1024,
    ):
        if max_pages < 1:
            raise ValueError("max_pages must be at least 1")
        if max_figures < 1:
            raise ValueError("max_figures must be at least 1")
        if max_image_bytes < 1:
            raise ValueError("max_image_bytes must be at least 1")

        self.max_pages = max_pages
        self.max_figures = max_figures
        self.max_image_bytes = max_image_bytes

    def extract(self, pdf_bytes: bytes) -> list[ExtractedFigure]:
        """Extract unique embedded images from PDF bytes."""
        if not pdf_bytes:
            raise ValueError("PDF data cannot be empty")

        figures: list[ExtractedFigure] = []
        seen_xrefs: set[int] = set()

        try:
            document = pymupdf.open(stream=pdf_bytes, filetype="pdf")

            try:
                for page_index in range(min(len(document), self.max_pages)):
                    page = document[page_index]

                    for image_info in page.get_images(full=True):
                        xref = image_info[0]

                        if xref in seen_xrefs:
                            continue

                        seen_xrefs.add(xref)

                        extracted = document.extract_image(xref)
                        image_bytes = extracted["image"]

                        if not image_bytes:
                            continue

                        if len(image_bytes) > self.max_image_bytes:
                            continue

                        extension = extracted.get("ext", "").lower()
                        mime_type = {
                            "png": "image/png",
                            "jpg": "image/jpeg",
                            "jpeg": "image/jpeg",
                            "webp": "image/webp",
                            "gif": "image/gif",
                        }.get(extension)

                        if mime_type is None:
                            continue

                        figures.append(
                            ExtractedFigure(
                                figure_index=len(figures) + 1,
                                page_number=page_index + 1,
                                image_bytes=image_bytes,
                                mime_type=mime_type,
                            )
                        )

                        if len(figures) >= self.max_figures:
                            return figures

            finally:
                document.close()

        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(
                f"Could not extract figures from PDF: {exc}"
            ) from exc

        return figures
