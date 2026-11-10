
"""Analyse PDF figures and charts using a vision-capable LLM."""
from __future__ import annotations

import json
from pathlib import Path

from src.llm.providers import OpenAICompatClient
from src.utils.config import VisionModelConfig, get_secret
from src.llm.providers import PROVIDERS


class PDFVisionAnalyzer:
    def __init__(self, config: VisionModelConfig):
        if config.provider not in PROVIDERS:
            raise ValueError(f"Unsupported provider: {config.provider}")

        url, key_name = PROVIDERS[config.provider]

        self.client = OpenAICompatClient(
            url=url,
            api_key=get_secret(key_name),
            model=config.model,
            temperature=config.temperature,
        )

    def analyze_image(self, image_path: str, page_number: int) -> dict:
        """Extract scientific information from one PDF figure."""

        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        prompt = f"""
Analyze this figure extracted from a scientific research paper.

PDF page number: {page_number}

Identify:
1. Figure type (chart, graph, diagram, table, photograph, or other).
2. Title or caption, if visible.
3. Axes, labels, units, legends and categories, if applicable.
4. Main trends, comparisons and patterns.
5. Important numerical values only when clearly readable.
6. Scientific interpretation supported by the figure.
7. Limitations or unreadable information.

Do not invent values or conclusions.
If information is unclear, explicitly say so.

Return valid JSON with these keys:
figure_type, title, description, axes, key_observations,
numerical_values, scientific_interpretation, limitations.
"""

        response = self.client.complete_with_images(
            system=(
                "You are a scientific figure analysis assistant. "
                "Report only information visible in the supplied image."
            ),
            user=prompt,
            image_paths=[str(path)],
            json_mode=True,
        )

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {
                "figure_type": "unknown",
                "title": "",
                "description": response,
                "axes": [],
                "key_observations": [],
                "numerical_values": [],
                "scientific_interpretation": "",
                "limitations": ["Model response was not valid JSON."],
            }

        result["page_number"] = page_number
        result["image_path"] = str(path)

        return result

    def analyze_pdf_images(self, images: list[dict]) -> list[dict]:
        """Analyse all extracted images from a PDF."""

        findings = []

        for image in images:
            try:
                finding = self.analyze_image(
                    image_path=image["image_path"],
                    page_number=image["page_number"],
                )
                findings.append(finding)

            except Exception as exc:
                findings.append({
                    "page_number": image.get("page_number"),
                    "image_path": image.get("image_path"),
                    "error": str(exc),
                })

        return findings
