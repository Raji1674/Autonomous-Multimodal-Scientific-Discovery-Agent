
import pandas as pd

from src.experiments.csv_trainer import train_csv_dataset


def test_csv_trainer_returns_metrics(tmp_path):
    csv_path = tmp_path / "sample.csv"

    pd.DataFrame({
        "feature1": list(range(1, 41)),
        "feature2": list(range(41, 81)),
        "target": [0] * 20 + [1] * 20,
    }).to_csv(csv_path, index=False)

    result = train_csv_dataset(csv_path, "target")

    assert result["success"] is True
    assert result["model"] == "RandomForestClassifier"
    assert 0 <= result["accuracy"] <= 1
    assert 0 <= result["f1_macro"] <= 1
    assert result["train_rows"] + result["test_rows"] == 40


def test_csv_trainer_rejects_missing_file(tmp_path):
    result = train_csv_dataset(tmp_path / "missing.csv", "target")

    assert result["success"] is False
    assert result["stage"] == "validation"
