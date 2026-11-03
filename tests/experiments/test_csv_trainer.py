
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

def test_train_all_models_returns_comparison(tmp_path):
    from src.experiments.csv_trainer import train_all_models

    csv_path = tmp_path / "sample.csv"

    pd.DataFrame({
        "feature1": list(range(1, 101)),
        "feature2": list(range(101, 201)),
        "target": [0] * 50 + [1] * 50,
    }).to_csv(csv_path, index=False)

    result = train_all_models(csv_path, "target")

    assert result["success"] is True
    assert len(result["models"]) == 4
    assert result["best_model"] in [
        "LogisticRegression",
        "DecisionTree",
        "RandomForest",
        "SVM",
    ]

    for model in result["models"]:
        assert 0 <= model["accuracy"] <= 1
        assert 0 <= model["f1_macro"] <= 1
