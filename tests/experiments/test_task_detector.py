
import pandas as pd

from src.experiments.task_detector import detect_task


def test_detect_classification(tmp_path):
    path = tmp_path / "classification.csv"
    pd.DataFrame({
        "age": range(1, 31),
        "target": [0, 1] * 15,
    }).to_csv(path, index=False)

    result = detect_task(path, "target")

    assert result["success"] is True
    assert result["task"] == "classification"


def test_detect_regression(tmp_path):
    path = tmp_path / "regression.csv"
    pd.DataFrame({
        "feature": range(1, 31),
        "target": [x * 2.5 for x in range(1, 31)],
    }).to_csv(path, index=False)

    result = detect_task(path, "target")

    assert result["success"] is True
    assert result["task"] == "regression"


def test_detect_clustering_without_target(tmp_path):
    path = tmp_path / "clustering.csv"
    pd.DataFrame({
        "feature1": range(1, 31),
        "feature2": range(31, 61),
    }).to_csv(path, index=False)

    result = detect_task(path)

    assert result["success"] is True
    assert result["task"] == "clustering"
    assert result["requires_confirmation"] is True


def test_task_override(tmp_path):
    result = detect_task(
        "unused.csv",
        task_override="regression",
    )

    assert result["success"] is True
    assert result["task"] == "regression"


def test_invalid_target(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({
        "feature": [1, 2, 3],
        "target": [0, 1, 0],
    }).to_csv(path, index=False)

    result = detect_task(path, "wrong_column")

    assert result["success"] is False
