
import pandas as pd

from src.experiments.csv_evaluation import evaluate_csv_models
from src.experiments.dataset_validator import validate_csv_dataset


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


def test_missing_feature_values_are_allowed(tmp_path):
    file_path = tmp_path / "data.csv"

    pd.DataFrame({
        "feature": [1, None, 3, 4],
        "target": [0, 1, 1, 0],
    }).to_csv(file_path, index=False)

    result = validate_csv_dataset(file_path, "target")

    assert result["valid"] is True
    assert result["missing_values"] == 1
    assert result["missing_feature_values"] == 1
    assert result["missing_target_values"] == 0


def test_missing_target_values_are_allowed(tmp_path):
    file_path = tmp_path / "data.csv"

    pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "target": [0, None, 1, 0],
    }).to_csv(file_path, index=False)

    result = validate_csv_dataset(file_path, "target")

    assert result["valid"] is True
    assert result["missing_target_values"] == 1
    assert result["target_classes"] == 2


def test_all_null_feature_is_rejected(tmp_path):
    file_path = tmp_path / "data.csv"

    pd.DataFrame({
        "feature": [None, None, None],
        "target": [0, 1, 0],
    }).to_csv(file_path, index=False)

    result = validate_csv_dataset(file_path, "target")

    assert result["valid"] is False
    assert any(
        "only missing values" in issue
        for issue in result["issues"]
    )

def test_csv_evaluation_handles_missing_feature_values(tmp_path):
    csv_path = tmp_path / "missing_features.csv"

    pd.DataFrame({
        "feature": [1, None, 3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "target": [0] * 10 + [1] * 10,
    }).to_csv(csv_path, index=False)

    result = evaluate_csv_models(
        csv_path,
        "target",
        baseline_params={"n_estimators": 10},
        treatment_params={"n_estimators": 20},
        n_seeds=5,
    )

    assert result["success"] is True
    assert len(result["baseline"]) == 5


def test_csv_evaluation_handles_missing_target_values(tmp_path):
    csv_path = tmp_path / "missing_targets.csv"

    pd.DataFrame({
        "feature": list(range(21)),
        "target": [0] * 10 + [1] * 10 + [None],
    }).to_csv(csv_path, index=False)

    result = evaluate_csv_models(
        csv_path,
        "target",
        baseline_params={"n_estimators": 10},
        treatment_params={"n_estimators": 20},
        n_seeds=5,
    )

    assert result["success"] is True
    assert result["n_train"] + result["n_test"] == 20