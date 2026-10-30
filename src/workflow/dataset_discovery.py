
"""Dataset discovery node for the autonomous research workflow."""

from pathlib import Path

import requests

from src.experiments.dataset_search import search_datasets
from src.experiments.dataset_validator import validate_csv_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "downloaded"
API_URL = "https://huggingface.co/api/datasets"


def dataset_discovery_node(state: dict) -> dict:
    """Discover, download, and validate a dataset for a research hypothesis."""

    research_question = state.get("research_question", "").strip()
    hypothesis = state.get("selected_hypothesis") or {}

    if not research_question:
        return {
            "failure_state": "DATASET_DISCOVERY_FAILED",
            "error": "Research question is missing",
        }

    # Use hypothesis content when available, otherwise use the question.
    if isinstance(hypothesis, dict):
        keyword = str(
            hypothesis.get("statement")
            or hypothesis.get("hypothesis")
            or research_question
        )
    else:
        keyword = str(hypothesis) or research_question

    target_column = state.get("target_column")

    if not target_column:
        return {
            "failure_state": "DATASET_DISCOVERY_FAILED",
            "error": (
                "Target column is not specified. "
                "Automatic target selection is not enabled."
            ),
        }

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

        if not dataset_id:
            continue

        try:
            metadata_url = f"{API_URL}/{dataset_id}/tree/main"
            response = requests.get(metadata_url, timeout=30)
            response.raise_for_status()
            files = response.json()

            csv_files = [
                item["path"]
                for item in files
                if isinstance(item, dict)
                and isinstance(item.get("path"), str)
                and item["path"].lower().endswith(".csv")
            ]

            for csv_name in csv_files:
                # Restrict downloads to CSV files in the repository root.
                if Path(csv_name).name != csv_name:
                    continue

                download_url = (
                    f"https://huggingface.co/datasets/{dataset_id}"
                    f"/resolve/main/{csv_name}"
                )

                response = requests.get(download_url, timeout=60)
                response.raise_for_status()

                safe_dataset_id = dataset_id.replace("/", "__")
                destination = DOWNLOAD_DIR / (
                    f"{safe_dataset_id}__{Path(csv_name).name}"
                )

                destination.write_bytes(response.content)

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

        except (requests.RequestException, OSError, ValueError) as exc:
            errors.append(f"{dataset_id}: {exc}")

    return {
        "dataset_candidates": candidates,
        "failure_state": "DATASET_DISCOVERY_FAILED",
        "error": "; ".join(errors) or "No valid CSV dataset was found",
    }
