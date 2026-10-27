
import pandas as pd

from src.experiments.csv_evaluation import evaluate_csv_models


def test_csv_evaluation_returns_paired_scores(tmp_path):
    csv_path = tmp_path / "sample.csv"

    pd.DataFrame({
        "feature1": list(range(1, 61)),
        "feature2": [i % 7 for i in range(60)],
        "target": [0] * 30 + [1] * 30,
    }).to_csv(csv_path, index=False)

    result = evaluate_csv_models(
        csv_path,
        "target",
        baseline_params={"n_estimators": 10},
        treatment_params={"n_estimators": 20},
        n_seeds=10,
    )

    assert result["success"] is True
    assert result["result_label"] == "executed"
    assert len(result["baseline"]) == 10
    assert len(result["treatment"]) == 10
    assert result["analysis"]["n"] == 10


def test_csv_evaluation_rejects_same_parameters(tmp_path):
    csv_path = tmp_path / "sample.csv"

    pd.DataFrame({
        "feature": list(range(20)),
        "target": [0] * 10 + [1] * 10,
    }).to_csv(csv_path, index=False)

    result = evaluate_csv_models(
        csv_path,
        "target",
        baseline_params={"n_estimators": 10},
        treatment_params={"n_estimators": 10},
        n_seeds=5,
    )

    assert result["success"] is False
    assert result["stage"] == "configuration"
