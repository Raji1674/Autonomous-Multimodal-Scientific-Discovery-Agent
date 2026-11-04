
import pandas as pd

from src.experiments.csv_critic import evaluate_csv_with_critic


def test_csv_critic_returns_statistical_verdict(tmp_path):
    rows = []

    # A small, reproducible classification dataset.
    for i in range(60):
        rows.append({
            "feature_a": float(i % 10),
            "feature_b": float(i // 10),
            "target": "A" if i < 30 else "B",
        })

    csv_path = tmp_path / "critic_dataset.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = evaluate_csv_with_critic(
        file_path=str(csv_path),
        target_column="target",
        baseline_params={"n_estimators": 10},
        treatment_params={"n_estimators": 20},
        n_seeds=10,
    )

    assert result["success"] is True
    assert result["critic"]["verdict"] in {
        "accept", "reject", "inconclusive"
    }
    assert result["critic"]["analysis"]["n"] == 10
    assert "statistical_quality" in result["critic"]
