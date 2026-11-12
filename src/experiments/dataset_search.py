
"""Search public datasets on Hugging Face."""
from __future__ import annotations

import requests

API_URL = "https://huggingface.co/api/datasets"


def search_datasets(keyword: str, limit: int = 10) -> list[dict]:
    """Search public Hugging Face datasets with basic metadata validation."""
    keyword = str(keyword).strip()

    if not keyword:
        return []

    limit = max(1, min(int(limit), 50))

    try:
        response = requests.get(
            API_URL,
            params={
                "search": keyword,
                "limit": limit,
                "sort": "downloads",
                "direction": -1,
            },
            timeout=20,
        )
        response.raise_for_status()
        datasets = response.json()

        if not isinstance(datasets, list):
            return []

        results = []

        for dataset in datasets:
            if not isinstance(dataset, dict):
                continue

            dataset_id = dataset.get("id")

            if not isinstance(dataset_id, str) or not dataset_id.strip():
                continue

            results.append({
                "name": dataset_id,
                "downloads": dataset.get("downloads", 0),
                "url": f"https://huggingface.co/datasets/{dataset_id}",
            })

        return results

    except (requests.RequestException, ValueError, TypeError) as exc:
        print(f"Dataset search failed: {exc}")
        return []
