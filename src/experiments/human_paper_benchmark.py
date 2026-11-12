
"""External human-authored research benchmark.

This module performs a contextual comparison only when the
agent experiment and published reference are compatible.
"""

from statistics import mean


REFERENCE_PAPERS = [
    {
        "paper_id": "peerj-cs-2951",
        "title": (
            "Advanced deep learning and transfer learning "
            "approaches for breast cancer classification"
        ),
        "year": 2025,
        "venue": "PeerJ Computer Science",
        "dataset": "Wisconsin Breast Cancer Dataset",
        "dataset_rows": 554,
        "model": "Random Forest",
        "published_accuracy": 0.965,
        "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12453660/",
    }
]


def _normalize(value) -> str:
    """Normalize metadata for compatibility checks."""
    return "".join(
        char.lower()
        for char in str(value or "")
        if char.isalnum()
    )


def _get_model(state: dict, results: dict) -> str | None:
    """Identify the actual model used by the experiment."""

    protocol = results.get("experiment_protocol") or {}
    plan = state.get("experiment_plan") or {}

    model = (
        protocol.get("algorithm")
        or results.get("algorithm")
        or plan.get("model")
    )

    if not model:
        return None

    normalized = _normalize(model)

    aliases = {
        "randomforest": "randomforest",
        "randomforestclassifier": "randomforest",
        "rf": "randomforest",
        "logisticregression": "logisticregression",
        "logisticregressionclassifier": "logisticregression",
        "logreg": "logisticregression",
        "svm": "svm",
        "supportvectormachine": "svm",
        "xgboost": "xgboost",
    }

    return aliases.get(normalized, normalized)


def _get_dataset_names(state: dict, results: dict) -> set[str]:
    """Collect available dataset identifiers."""

    values = []

    experiment = results.get("experiment") or {}
    plan = state.get("experiment_plan") or {}
    selected = state.get("selected_dataset") or {}
    metadata = state.get("dataset_metadata") or {}

    values.extend([
        results.get("dataset"),
        experiment.get("dataset"),
        plan.get("dataset"),
        selected.get("name"),
        selected.get("filename"),
        selected.get("dataset_name"),
        metadata.get("name"),
        metadata.get("filename"),
        metadata.get("dataset_name"),
    ])

    return {
        _normalize(value)
        for value in values
        if value
    }



def _is_reference_dataset(dataset_names: set[str]) -> bool:
    """Check whether available identifiers match the reference dataset."""

    aliases = {
        "wisconsinbreastcancerdataset",
        "wisconsinbreastcancer",
        "wisconsindiagnosticbreastcancer",
        "breastcancer",
        "breastcancerdataset",
        "wdbc",
    }

    for name in dataset_names:
        normalized_name = name.removesuffix("csv").removesuffix("data")

        if normalized_name in aliases:
            return True

    return False


    return any(
        name in aliases
        for name in dataset_names
    )


def evaluate_human_paper_benchmark(state: dict) -> dict:
    """Compare compatible agent results with published research."""

    results = state.get("experiment_results") or {}

    if not isinstance(results, dict):
        results = {}

    # Support both general experiment and CSV runner result structures.
    experiment = results.get("experiment") or {}

    scores = results.get("baseline")

    if scores is None:
        scores = experiment.get("baseline")

    reference = REFERENCE_PAPERS[0]

    if not isinstance(scores, (list, tuple)) or not scores:
        return {
            "status": "not_evaluated",
            "reason": "Baseline accuracy scores are unavailable.",
            "references": REFERENCE_PAPERS,
        }

    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= 1
        for value in scores
    ):
        return {
            "status": "not_evaluated",
            "reason": "Baseline scores are not valid accuracy values.",
            "references": REFERENCE_PAPERS,
        }

    model = _get_model(state, results)

    if model is None:
        return {
            "status": "not_evaluated",
            "reason": "The experiment algorithm could not be identified.",
            "references": REFERENCE_PAPERS,
        }

    if model != _normalize("Random Forest"):
        return {
            "status": "not_comparable",
            "reason": (
                "The agent experiment uses a different algorithm "
                "from the published Random Forest reference."
            ),
            "agent_model": model,
            "published_model": reference["model"],
            "references": REFERENCE_PAPERS,
        }

    dataset_names = _get_dataset_names(state, results)

    if not dataset_names:
        return {
            "status": "not_evaluated",
            "reason": "Dataset identity is unavailable.",
            "references": REFERENCE_PAPERS,
        }

    if not _is_reference_dataset(dataset_names):
        return {
            "status": "not_comparable",
            "reason": (
                "The agent dataset could not be confirmed as "
                "the published Wisconsin Breast Cancer Dataset."
            ),
            "agent_dataset_identifiers": sorted(dataset_names),
            "published_dataset": reference["dataset"],
            "references": REFERENCE_PAPERS,
        }

    agent_accuracy = mean(float(value) for value in scores)

    validation = state.get("dataset_validation") or {}
    current_rows = validation.get("rows")

    return {
        "status": "contextual_comparison",
        "agent_model": "RandomForestClassifier",
        "agent_baseline_accuracy": agent_accuracy,
        "published_model": reference["model"],
        "published_accuracy": reference["published_accuracy"],
        "difference_percentage_points": round(
            (agent_accuracy - reference["published_accuracy"]) * 100,
            4,
        ),
        "agent_dataset_rows": current_rows,
        "published_dataset_rows": reference["dataset_rows"],
        "same_dataset_size": (
            current_rows == reference["dataset_rows"]
            if current_rows is not None
            else None
        ),
        "reference_paper": reference,
        "limitations": [
            "The published paper reports 554 cleaned instances.",
            "Dataset sample counts may differ.",
            "Experimental splits and validation protocols may differ.",
            "This is a contextual comparison, not an independent reproduction.",
            "Statistical significance between the two studies was not tested.",
            "The published and agent results may use different preprocessing.",
        ],
    }
