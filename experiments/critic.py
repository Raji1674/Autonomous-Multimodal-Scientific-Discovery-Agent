
import json
from pathlib import Path


def review_experiment(results):
    issues = []
    warnings = []

    # 1. Check whether required metrics exist
    required_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]

    for metric in required_metrics:
        if metric not in results:
            issues.append(f"Missing metric: {metric}")
            continue

        value = results[metric]

        # 2. Check metric values
        if not isinstance(value, (int, float)):
            issues.append(f"{metric} must be numeric.")
        elif not 0 <= value <= 1:
            issues.append(
                f"{metric} must be between 0 and 1."
            )

    # 3. Check model and dataset information
    if not results.get("model"):
        warnings.append("Model name is missing.")

    if not results.get("dataset"):
        warnings.append("Dataset name is missing.")

    # 4. Check confusion matrix
    matrix = results.get("confusion_matrix")

    if matrix is None:
        warnings.append("Confusion matrix is missing.")
    else:
        try:
            if len(matrix) != 2 or any(
                len(row) != 2 for row in matrix
            ):
                issues.append(
                    "Expected a 2x2 confusion matrix "
                    "for binary classification."
                )
            elif any(
                not isinstance(value, int) or value < 0
                for row in matrix
                for value in row
            ):
                issues.append(
                    "Confusion matrix counts must be "
                    "non-negative integers."
                )
        except TypeError:
            issues.append("Confusion matrix format is invalid.")

    # 5. Give a review status
    if issues:
        status = "NEEDS_REVIEW"
    elif warnings:
        status = "REVIEW_WITH_WARNINGS"
    else:
        status = "BASIC_CHECKS_PASSED"

    return {
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "note": (
            "These checks verify result format and basic "
            "consistency. They do not prove scientific validity."
        ),
    }


if __name__ == "__main__":
    metrics_path = Path("outputs/metrics.json")
    results_path = Path("outputs/experiment_results.json")

    # Prefer metrics.json; use experiment results as fallback
    if metrics_path.exists():
        input_path = metrics_path
    elif results_path.exists():
        input_path = results_path
    else:
        raise FileNotFoundError(
            "No experiment results found. Run trainer.py "
            "or metrics.py first."
        )

    with input_path.open("r", encoding="utf-8") as file:
        results = json.load(file)

    report = review_experiment(results)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "critic_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("Critic review completed!")
    print(json.dumps(report, indent=4))
    print(f"\nReport saved to: {report_path}")
