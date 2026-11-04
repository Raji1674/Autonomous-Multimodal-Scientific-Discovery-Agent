
from pathlib import Path

import requests

from src.experiments.dataset_search import search_datasets
from src.experiments.csv_trainer import train_csv_dataset

API_URL = "https://huggingface.co/api/datasets"


def download_and_train(
    keyword: str,
    target_column: str,
    output_dir: str | Path = "data/downloaded",
    limit: int = 10,
) -> dict:
    """Search Hugging Face, find a CSV, validate it, and train a classifier."""

    datasets = search_datasets(keyword, limit=limit)

    if not datasets:
        return {
            "success": False,
            "stage": "search",
            "issues": ["No datasets found or dataset search failed"],
        }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    errors = []

    for dataset in datasets:
        dataset_id = dataset["name"]
        csv_path = None

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

            if not csv_files:
                errors.append(f"{dataset_id}: no CSV file found at repository root")
                continue

            # Try CSVs until one validates and trains successfully.
            for csv_name in csv_files:
                url = (
                    f"https://huggingface.co/datasets/{dataset_id}"
                    f"/resolve/main/{csv_name}"
                )
                response = requests.get(url, timeout=60)
                response.raise_for_status()

                safe_name = (
                    dataset_id.replace("/", "__")
                    + "__"
                    + Path(csv_name).name
                )
                csv_path = output_path / safe_name
                csv_path.write_bytes(response.content)

                result = train_csv_dataset(csv_path, target_column)

                if result.get("success"):
                    result["dataset_id"] = dataset_id
                    result["dataset_url"] = dataset["url"]
                    result["source_file"] = csv_name
                    return result

                errors.append(
                    f"{dataset_id}/{csv_name}: "
                    f"{result.get('issues', ['training failed'])}"
                )

        except (requests.RequestException, OSError, ValueError) as exc:
            errors.append(f"{dataset_id}: {exc}")

        finally:
            if csv_path is not None:
                csv_path.unlink(missing_ok=True)

    return {
        "success": False,
        "stage": "download_or_training",
        "issues": errors or ["No usable CSV datasets found"],
    }
