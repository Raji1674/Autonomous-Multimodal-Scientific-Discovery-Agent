
import json

import pytest

from src.agents.pdf_vision_analyzer import PDFVisionAnalyzer
from src.utils.config import VisionModelConfig


@pytest.fixture
def analyzer():
    instance = PDFVisionAnalyzer.__new__(PDFVisionAnalyzer)
    instance.client = None
    return instance


@pytest.fixture
def fake_image(tmp_path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"fake-image-content")
    return str(image)


def test_analyze_image_returns_structured_result(
    analyzer, fake_image, monkeypatch
):
    expected = {
        "figure_type": "chart",
        "title": "Accuracy comparison",
        "description": "A comparison chart",
        "axes": ["Model", "Accuracy"],
        "key_observations": ["Model A performs better"],
        "numerical_values": [95],
        "scientific_interpretation": "Model A is more accurate",
        "limitations": [],
    }

    def fake_complete(**kwargs):
        assert kwargs["image_paths"] == [fake_image]
        assert kwargs["json_mode"] is True
        return json.dumps(expected)

    monkeypatch.setattr(
        analyzer,
        "client",
        type("FakeClient", (), {
            "complete_with_images": staticmethod(fake_complete)
        })(),
    )

    result = analyzer.analyze_image(fake_image, 3)

    assert result["figure_type"] == "chart"
    assert result["page_number"] == 3
    assert result["image_path"] == fake_image


def test_invalid_json_returns_fallback(
    analyzer, fake_image, monkeypatch
):
    analyzer.client = type(
        "FakeClient",
        (),
        {
            "complete_with_images": staticmethod(
                lambda **kwargs: "Not valid JSON"
            )
        },
    )()

    result = analyzer.analyze_image(fake_image, 2)

    assert result["figure_type"] == "unknown"
    assert result["description"] == "Not valid JSON"
    assert "Model response was not valid JSON." in result["limitations"]


def test_missing_image_raises_error(analyzer, tmp_path):
    missing = str(tmp_path / "missing.png")

    with pytest.raises(FileNotFoundError):
        analyzer.analyze_image(missing, 1)


def test_empty_image_list_returns_empty_findings(analyzer):
    assert analyzer.analyze_pdf_images([]) == []


def test_multiple_images_are_analyzed(analyzer, fake_image, monkeypatch):
    calls = []

    def fake_analyze(image_path, page_number):
        calls.append(page_number)
        return {
            "figure_type": "chart",
            "page_number": page_number,
            "image_path": image_path,
        }

    monkeypatch.setattr(analyzer, "analyze_image", fake_analyze)

    images = [
        {"image_path": fake_image, "page_number": 1},
        {"image_path": fake_image, "page_number": 2},
    ]

    results = analyzer.analyze_pdf_images(images)

    assert len(results) == 2
    assert calls == [1, 2]


def test_one_image_failure_does_not_stop_remaining_images(
    analyzer, fake_image, monkeypatch
):
    def fake_analyze(image_path, page_number):
        if page_number == 1:
            raise RuntimeError("Vision API failed")

        return {
            "figure_type": "chart",
            "page_number": page_number,
            "image_path": image_path,
        }

    monkeypatch.setattr(analyzer, "analyze_image", fake_analyze)

    images = [
        {"image_path": fake_image, "page_number": 1},
        {"image_path": fake_image, "page_number": 2},
    ]

    results = analyzer.analyze_pdf_images(images)

    assert len(results) == 2
    assert "error" in results[0]
    assert results[1]["page_number"] == 2
