import requests


def search_datasets(keyword: str, limit: int = 10):
    """Search public datasets on Hugging Face."""
    url = "https://huggingface.co/api/datasets"

    try:
        response = requests.get(
            url,
            params={"search": keyword, "limit": limit},
            timeout=20,
        )
        response.raise_for_status()
        datasets = response.json()

        return [
            {
                "name": dataset.get("id", "Unknown"),
                "downloads": dataset.get("downloads", 0),
                "url": (
                    "https://huggingface.co/datasets/"
                    + dataset["id"]
                ),
            }
            for dataset in datasets
        ]

    except requests.RequestException as exc:
        print(f"Dataset search failed: {exc}")
        return []
