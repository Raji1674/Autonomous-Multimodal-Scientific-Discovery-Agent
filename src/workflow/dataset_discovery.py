
"""Robust public dataset discovery for the research workflow."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from sklearn.datasets import load_breast_cancer
from src.experiments.dataset_search import search_datasets
from src.experiments.dataset_validator import validate_csv_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "downloaded"
API_URL = "https://huggingface.co/api/datasets"
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024

# Common target names in cardiovascular and other medical datasets.
TARGET_ALIASES = [
    "target",
    "label",
    "class",
    "output",
    "outcome",
    "diagnosis",
    "heartdisease",
    "heart_disease",
    "cardiovascular_disease",
    "cardio",
    "tenyearchd",
    "chd",
    "num",
    "death_event",
    "disease",
    "stroke",
]


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _build_search_queries(question: str, hypothesis) -> list[str]:
    """Create several short search queries instead of one long hypothesis."""

    queries = []

    if question:
        queries.append(question.strip())

    if isinstance(hypothesis, dict):
        statement = (
            hypothesis.get("statement")
            or hypothesis.get("hypothesis")
            or ""
        )
    else:
        statement = str(hypothesis or "")

    if statement:
        queries.append(statement.strip())

    combined = f"{question} {statement}".lower()

    # Domain-specific alternatives based on the research question.
    domain_terms = [
        (["cardiovascular", "heart disease", "cvd"], [
            "cardiovascular disease",
            "heart disease",
            "heart disease prediction",
            "cardiovascular risk",
            "heart failure clinical",
        ]),
        (["diabetes"], [
            "diabetes prediction",
            "diabetes clinical dataset",
        ]),
        (["cancer", "tumor"], [
            "cancer diagnosis",
            "cancer classification",
        ]),
        (["readmission"], [
            "hospital readmission",
            "clinical patient outcome",
        ]),
    ]

    for triggers, alternatives in domain_terms:
        if any(term in combined for term in triggers):
            queries.extend(alternatives)

    # General fallback queries.
    queries.extend([
        "clinical tabular dataset",
        "medical classification dataset",
    ])

    result = []
    seen = set()

    for query in queries:
        query = re.sub(r"\s+", " ", query).strip()

        if query and query.lower() not in seen:
            seen.add(query.lower())
            result.append(query)

    return result


def _infer_target_column(
    columns: list[str],
    requested_target: str | None = None,
    file_path: Path | None = None,
) -> str | None:
    """Find a likely supervised-learning target column."""

    if requested_target and requested_target in columns:
        return requested_target

    normalized = {
        _normalize(column): column
        for column in columns
    }

    for alias in TARGET_ALIASES:
        if _normalize(alias) in normalized:
            return normalized[_normalize(alias)]

    # If no known target name exists, look for binary columns.
    if file_path is not None:
        try:
            df = pd.read_csv(file_path, nrows=5000)

            excluded = {
                "id",
                "patientid",
                "recordid",
                "index",
                "serialnumber",
            }

            binary_candidates = []

            for column in df.columns:
                if _normalize(column) in excluded:
                    continue

                unique_count = df[column].dropna().nunique()

                if unique_count == 2:
                    binary_candidates.append(column)

            # Prefer a binary column whose name suggests an outcome.
            for column in binary_candidates:
                name = _normalize(column)

                if any(
                    word in name
                    for word in [
                        "disease",
                        "heart",
                        "cardio",
                        "diagnosis",
                        "outcome",
                        "label",
                        "target",
                        "class",
                    ]
                ):
                    return column

        except Exception:
            return None

    return None


def _safe_dataset_filename(dataset_id: str, csv_name: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "__", dataset_id)
    safe_csv = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(csv_name).name)
    return f"{safe_id}__{safe_csv}"


def _download_limited(response, destination: Path) -> None:
    headers = getattr(response,"headers",{})or {}
    content_length = headers.get("Content-Length", "")

    if content_length:
        try:
            if int(content_length) > MAX_DOWNLOAD_BYTES:
                raise ValueError("Dataset exceeds the 100 MB limit")
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
                else [getattr(response, "content", b"")]
            )

            for chunk in iterator:
                if not chunk:
                    continue

                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Dataset exceeds the 100 MB limit")

                output.write(chunk)


    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _get_dataset_files(dataset_id: str) -> list[str]:
    """Retrieve CSV file paths from a Hugging Face dataset repository."""

    encoded_id = quote(dataset_id, safe="/")
    url = f"{API_URL}/{encoded_id}/tree/main"

    response = requests.get(
        url,
        params={"recursive": "true"},
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        return []

    csv_files = []

    for item in data:
        if not isinstance(item, dict):
            continue

        path = item.get("path")

        if (
            isinstance(path, str)
            and path.lower().endswith(".csv")
            and not path.startswith(".")
        ):
            csv_files.append(path)

    return csv_files
def _load_wisconsin_breast_cancer_dataset() -> dict:
    """Load the Wisconsin Diagnostic Breast Cancer dataset."""

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()

    destination = DOWNLOAD_DIR / "wisconsin_breast_cancer.csv"
    df.to_csv(destination, index=False)

    validation = validate_csv_dataset(
        destination,
        target_column="target",
    )

    if not validation.get("valid"):
        destination.unlink(missing_ok=True)
        return {
            "failure_state": "DATASET_DISCOVERY_FAILED",
            "error": f"Wisconsin dataset validation failed: {validation.get('issues')}",
        }

    return {
        "dataset_candidates": [],
        "dataset_search_queries": ["Wisconsin Diagnostic Breast Cancer"],
        "selected_dataset": {
            "name": "Wisconsin Diagnostic Breast Cancer",
            "url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html",
        },
        "dataset_path": str(destination),
        "target_column": "target",
        "dataset_validation": validation,
        "dataset_metadata": {
            "dataset_id": "Wisconsin Diagnostic Breast Cancer",
            "source": "scikit-learn",
            "rows": validation["rows"],
            "features": 30,
            "target_column": "target",
            "classes": ["malignant", "benign"],
        },
        "failure_state": None,
        "error": None,
    }

def dataset_discovery_node(state: dict) -> dict:
    """Discover, download, and validate a public CSV dataset."""

    research_question = str(
        state.get("research_question") or ""
    ).strip()

    hypothesis = state.get("selected_hypothesis") or {}
    requested_target = state.get("target_column")

    if not research_question:
        return {
            "failure_state": "DATASET_DISCOVERY_FAILED",
            "error": "Research question is missing.",
        }
    
    queries = _build_search_queries(
        research_question,
        hypothesis,
    )
    question_lower = research_question.lower()

    is_wisconsin_dataset = (
        "wisconsin" in question_lower
        or "wdbc" in question_lower
        or (
            "breast cancer" in question_lower
            and "dataset" in question_lower
        )
    )

    if is_wisconsin_dataset:
        return _load_wisconsin_breast_cancer_dataset()
    
    candidates = []
    seen_datasets = set()
    search_errors = []

    # Try multiple short and relevant searches.
    for query in queries:
        try:
            results = search_datasets(query, limit=10)

            for candidate in results:
                if not isinstance(candidate, dict):
                    continue

                dataset_id = candidate.get("name")

                if (
                    isinstance(dataset_id, str)
                    and dataset_id not in seen_datasets
                ):
                    seen_datasets.add(dataset_id)
                    candidates.append(candidate)

        except Exception as exc:
            search_errors.append(f"{query}: {exc}")

    if not candidates:
        return {
            "failure_state": "DATASET_DISCOVERY_FAILED",
            "error": (
                "No Hugging Face datasets were found"
                + "; ".join(search_errors)
            ),
            "dataset_candidates": [],
            "dataset_search_queries": queries,
        }

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    errors = []

    # Try the discovered datasets one by one.
    for candidate in candidates:
        dataset_id = candidate.get("name")

        if not isinstance(dataset_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
            dataset_id,
        ):
            continue

        try:
            csv_files = _get_dataset_files(dataset_id)

            if not csv_files:
                errors.append(f"{dataset_id}: No CSV files found")
                continue

            for csv_name in csv_files:
                encoded_id = quote(dataset_id, safe="/")
                encoded_file = quote(csv_name, safe="/")

                download_url = (
                    f"https://huggingface.co/datasets/{encoded_id}"
                    f"/resolve/main/{encoded_file}"
                )

                destination = DOWNLOAD_DIR / _safe_dataset_filename(
                    dataset_id,
                    csv_name,
                )

                try:
                    response = requests.get(
                        download_url,
                        timeout=60,
                        stream=True,
                    )
                    response.raise_for_status()

                    _download_limited(response, destination)

                    # Read the CSV and infer a target when needed.
                    df = pd.read_csv(destination, nrows=5000)

                    target_column = _infer_target_column(
                        columns=df.columns.tolist(),
                        requested_target=requested_target,
                        file_path=destination,
                    )

                    if not target_column:
                        errors.append(
                            f"{dataset_id}/{csv_name}: "
                            "Could not identify a suitable target column"
                        )
                        destination.unlink(missing_ok=True)
                        continue

                    validation = validate_csv_dataset(
                        destination,
                        target_column=target_column,
                    )

                    if not validation.get("valid"):
                        errors.append(
                            f"{dataset_id}/{csv_name}: "
                            f"{validation.get('issues', [])}"
                        )
                        destination.unlink(missing_ok=True)
                        continue

                    metadata = {
                        "dataset_id": dataset_id,
                        "dataset_url": candidate.get("url"),
                        "source_file": csv_name,
                        "downloads": candidate.get("downloads", 0),
                        "target_column": target_column,
                        "source": "huggingface",
                    }

                    return {
                        "dataset_candidates": candidates,
                        "dataset_search_queries": queries,
                        "selected_dataset": candidate,
                        "dataset_path": str(destination),
                        "target_column": target_column,
                        "dataset_validation": validation,
                        "dataset_metadata": metadata,
                        "failure_state": None,
                        "error": None,
                    }

                except (
                    requests.RequestException,
                    OSError,
                    ValueError,
                    pd.errors.ParserError,
                ) as exc:
                    destination.unlink(missing_ok=True)
                    errors.append(
                        f"{dataset_id}/{csv_name}: {exc}"
                    )

        except (
            requests.RequestException,
            OSError,
            ValueError,
        ) as exc:
            errors.append(f"{dataset_id}: {exc}")

    return {
        "dataset_candidates": candidates,
        "dataset_search_queries": queries,
        "failure_state": "DATASET_DISCOVERY_FAILED",
        "error": (
            "No Hugging Face datasets were found. "
            + "; ".join(errors[-10:])
        ),
    }
