"""Bounded PDF download and text extraction for known scholarly repositories."""
from __future__ import annotations

from urllib.parse import urljoin, urlparse

import pymupdf
import requests


class PDFProcessingError(RuntimeError):
    """Raised when a scholarly PDF cannot be safely downloaded or parsed."""


class PdfTextExtractor:
    """Download and extract text from arXiv/PMC PDFs with strict size limits."""

    ALLOWED_HOSTS = {"arxiv.org", "export.arxiv.org", "pmc.ncbi.nlm.nih.gov", "europepmc.org"}

    def __init__(self, *, timeout: tuple[float, float] = (5.0, 30.0),
                 max_bytes: int = 20_000_000, max_pages: int = 300):
        if max_bytes <= 0 or max_pages <= 0:
            raise ValueError("max_bytes and max_pages must be positive")
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_pages = max_pages

    @classmethod
    def _validate_url(cls, url: str) -> None:
        parsed = urlparse(url)
        if (parsed.scheme != "https" or parsed.hostname not in cls.ALLOWED_HOSTS
                or parsed.username or parsed.password or parsed.port not in (None, 443)):
            raise PDFProcessingError("PDF URL must use HTTPS on an approved scholarly host")

    def download(self, url: str) -> bytes:
        """Download a PDF without following redirects to unapproved hosts."""
        current = url
        for _ in range(4):
            self._validate_url(current)
            try:
                response = requests.get(
                    current, timeout=self.timeout, stream=True, allow_redirects=False,
                    headers={"User-Agent": "AutonomousScientificDiscoveryAgent/1.0 (research use)"},
                )
            except requests.RequestException as exc:
                raise PDFProcessingError(f"PDF download failed: {exc}") from exc
            try:
                if response.is_redirect:
                    location = response.headers.get("Location")
                    if not location:
                        raise PDFProcessingError("PDF server returned a redirect without a location")
                    current = urljoin(current, location)
                    continue
                response.raise_for_status()
                declared_size = response.headers.get("Content-Length")
                if declared_size and int(declared_size) > self.max_bytes:
                    raise PDFProcessingError("PDF exceeds the configured download size limit")
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > self.max_bytes:
                        raise PDFProcessingError("PDF exceeds the configured download size limit")
                    chunks.append(chunk)
                content = b"".join(chunks)
                if not content.startswith(b"%PDF-"):
                    raise PDFProcessingError("The scholarly URL did not return a PDF file")
                return content
            except requests.RequestException as exc:
                raise PDFProcessingError(f"PDF download failed: {exc}") from exc
            finally:
                response.close()
        raise PDFProcessingError("PDF server redirected too many times")

    def extract(self, url: str) -> str:
        """Extract page text, stopping at the configured page limit."""
        content = self.download(url)
        try:
            with pymupdf.open(stream=content, filetype="pdf") as document:
                pages = [page.get_text("text") for page in document[:self.max_pages]]
        except Exception as exc:  # noqa: BLE001 - normalize parser errors at this boundary
            raise PDFProcessingError(f"Could not parse downloaded PDF: {exc}") from exc
        text = "\n\n".join(page.strip() for page in pages if page.strip())
        if not text:
            raise PDFProcessingError("PDF contains no extractable text")
        return text
