
import pandas as pd

from src.experiments.regression_trainer import train_regression_models


def test_regression_trainer_returns_model_comparison(tmp_path):
    csv_path = tmp_path / "regression.csv"

    pd.DataFrame({
        "feature1": list(range(1, 101)),
        "feature2": [x * 2 for x in range(1, 101)],
        "target": [x * 3 + 5 for x in range(1, 101)],
    }).to_csv(csv_path, index=False)

    result = train_regression_models(csv_path, "target")

    assert result["success"] is True
    assert result["task"] == "regression"
    assert len(result["models"]) == 7
    assert result["best_model"] in [
        "LinearRegression",
        "Ridge",
        "Lasso",
        "DecisionTree",
        "RandomForest",
        "GradientBoosting",
        "SVR",
    ]

    for model in result["models"]:
        assert model["mae"] >= 0
        assert model["rmse"] >= 0
        assert isinstance(model["r2"], float)


def test_regression_trainer_rejects_missing_target(tmp_path):
    csv_path = tmp_path / "regression.csv"

    pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "target": [10, 20, 30, 40, 50],
    }).to_csv(csv_path, index=False)

    result = train_regression_models(csv_path, "wrong_target")

    assert result["success"] is False
    assert result["stage"] == "validation"
