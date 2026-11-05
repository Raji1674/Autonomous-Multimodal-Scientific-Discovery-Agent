
"""Dataset discovery node for the autonomous research workflow."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

import requests

from src.experiments.dataset_search import search_datasets
from src.experiments.dataset_validator import validate_csv_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "downloaded"
API_URL = "https://huggingface.co/api/datasets"

MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024  # 100 MB


def _safe_dataset_filename(dataset_id: str, csv_name: str) -> str:
    """Build a safe local filename from Hugging Face metadata."""
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "__", dataset_id)
    safe_csv = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(csv_name).name)
    return f"{safe_id}__{safe_csv}"


def _download_limited(response, destination: Path) -> None:
    """Download a response without allowing files larger than the limit."""
    headers = getattr(response, "headers", {}) or {}
    content_length = headers.get("Content-Length", "")
    if content_length:
        try:
            if int(content_length) > MAX_DOWNLOAD_BYTES:
                raise ValueError("Dataset exceeds the 100 MB download limit")
        except ValueError as exc:
            if "100 MB" in str(exc):
                raise
            raise ValueError("Invalid Content-Length header") from exc

    total = 0

    try:
        with destination.open("wb") as output:
            iterator = (
                response.iter_content(chunk_size=1024 * 1024)
                if callable(getattr(response, "iter_content", None))
                else [response.content]
            )

            for chunk in iterator:
                if not chunk:
                    continue

                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Dataset exceeds the 100 MB download limit")

                output.write(chunk)

    except Exception:
        destination.unlink(missing_ok=True)
        raise


def dataset_discovery_node(state: dict) -> dict:
    """Discover, download, and validate a dataset for a research hypothesis."""

    research_question = state.get("research_question", "").strip()
    hypothesis = state.get("selected_hypothesis") or {}

    if not research_question:
        return {
            "failure_state": "DATASET_DISCOVERY_FAILED",
            "error": "Research question is missing",
        }

    if isinstance(hypothesis, dict):
        keyword = str(
            hypothesis.get("statement")
            or hypothesis.get("hypothesis")
            or research_question
        )
    else:
        keyword = str(hypothesis) or research_question

    target_column = state.get("target_column")
    candidates = search_datasets(keyword, limit=10)

    if not candidates:
        return {
            "failure_state": "DATASET_DISCOVERY_FAILED",
            "error": "No Hugging Face datasets were found",
            "dataset_candidates": [],
        }

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    errors = []

    for candidate in candidates:
        dataset_id = candidate.get("name")

        if not isinstance(dataset_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", dataset_id
        ):
            continue

        try:
            encoded_id = quote(dataset_id, safe="/")
            metadata_url = f"{API_URL}/{encoded_id}/tree/main"

            response = requests.get(metadata_url, timeout=30)
            response.raise_for_status()
            files = response.json()

            if not isinstance(files, list):
                errors.append(f"{dataset_id}: Invalid repository file listing")
                continue

            csv_files = [
                item["path"]
                for item in files
                if isinstance(item, dict)
                and isinstance(item.get("path"), str)
                and item["path"].lower().endswith(".csv")
                and Path(item["path"]).name == item["path"]
            ]

            for csv_name in csv_files:
                download_url = (
                    f"https://huggingface.co/datasets/{encoded_id}"
                    f"/resolve/main/{quote(csv_name)}"
                )

                destination = DOWNLOAD_DIR / _safe_dataset_filename(
                    dataset_id, csv_name
                )

                try:
                    response = requests.get(
                        download_url,
                        timeout=60,
                        stream=True,
                    )
                    response.raise_for_status()

                    _download_limited(response, destination)

                    validation = validate_csv_dataset(
                        destination,
                        target_column,
                    )

                    if validation["valid"]:
                        metadata = {
                            "dataset_id": dataset_id,
                            "dataset_url": candidate.get("url"),
                            "source_file": csv_name,
                            "downloads": candidate.get("downloads", 0),
                        }

                        return {
                            "dataset_candidates": candidates,
                            "selected_dataset": candidate,
                            "dataset_path": str(destination),
                            "dataset_validation": validation,
                            "dataset_metadata": metadata,
                            "failure_state": None,
                            "error": None,
                        }

                    errors.append(
                        f"{dataset_id}/{csv_name}: {validation['issues']}"
                    )
                    destination.unlink(missing_ok=True)

                except (requests.RequestException, OSError, ValueError) as exc:
                    destination.unlink(missing_ok=True)
                    errors.append(f"{dataset_id}/{csv_name}: {exc}")

        except (requests.RequestException, OSError, ValueError) as exc:
            errors.append(f"{dataset_id}: {exc}")

    return {
        "dataset_candidates": candidates,
        "failure_state": "DATASET_DISCOVERY_FAILED",
        "error": "; ".join(errors) or "No valid CSV dataset was found",
    }
