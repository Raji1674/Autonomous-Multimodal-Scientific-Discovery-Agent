"""External human-authored research benchmark."""
from statistics import mean


REFERENCE_PAPERS = [
    {
        "paper_id": "peerj-cs-2951",
        "title": "Advanced deep learning and transfer learning approaches for breast cancer classification",
        "year": 2025,
        "venue": "PeerJ Computer Science",
        "dataset": "Wisconsin Breast Cancer Dataset",
        "dataset_rows": 554,
        "model": "Random Forest",
        "published_accuracy": 0.965,
        "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12453660/",
    }
]


def evaluate_human_paper_benchmark(state: dict) -> dict:
    """Compare the completed experiment with published human research."""
    results = state.get("experiment_results") or {}
    scores = results.get("baseline")

    if not isinstance(scores, (list, tuple)) or not scores:
        return {
            "status": "not_evaluated",
            "reason": "Baseline accuracy scores are unavailable.",
            "references": REFERENCE_PAPERS,
        }

    agent_accuracy = mean(float(value) for value in scores)
    reference = REFERENCE_PAPERS[0]
    published_accuracy = reference["published_accuracy"]

    validation = state.get("dataset_validation") or {}
    current_rows = validation.get("rows")

    return {
        "status": "contextual_comparison",
        "agent_model": "RandomForestClassifier",
        "agent_baseline_accuracy": agent_accuracy,
        "published_model": reference["model"],
        "published_accuracy": published_accuracy,
        "difference_percentage_points": round(
            (agent_accuracy - published_accuracy) * 100, 4
        ),
        "agent_dataset_rows": current_rows,
        "published_dataset_rows": reference["dataset_rows"],
        "same_dataset_size": current_rows == reference["dataset_rows"],
        "reference_paper": reference,
        "limitations": [
            "Dataset sample counts differ.",
            "Experimental splits and validation protocols may differ.",
            "This is a contextual comparison, not an independent reproduction.",
            "Statistical significance between the two studies was not tested.",
            "Overlap with the existing RAG corpus has not yet been verified.",
        ],
    }
