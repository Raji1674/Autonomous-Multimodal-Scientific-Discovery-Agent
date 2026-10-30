
import json

import pandas as pd

from src.experiments import sandbox_csv_runner as runner


def create_dataset(path):
    path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "feature1": list(range(1, 21)),
        "feature2": [i % 5 for i in range(20)],
        "target": [0] * 10 + [1] * 10,
    }).to_csv(path, index=False)


def test_sandbox_csv_runner_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    dataset = tmp_path / "data" / "downloaded" / "sample.csv"
    create_dataset(dataset)

    experiment = {
        "result_label": "executed",
        "baseline": [0.8] * 10,
        "treatment": [0.9] * 10,
        "n_seeds": 10,
        "n_train": 16,
        "n_test": 4,
    }

    monkeypatch.setattr(
        runner,
        "execute_python",
        lambda code, dataset_path: {
            "result_label": "executed",
            "output": json.dumps(experiment),
            "exit_code": 0,
            "output_truncated": False,
        },
    )

    class FakeCritic:
        def evaluate_scores(self, result, **kwargs):
            return {"verdict": "accept"}

    monkeypatch.setattr(runner, "StatisticalCritic", FakeCritic)

    result = runner.run_sandbox_csv_experiment(
        dataset,
        "target",
        baseline_params={"n_estimators": 10},
        treatment_params={"n_estimators": 20},
        n_seeds=10,
    )

    assert result["success"] is True
    assert result["experiment"]["result_label"] == "executed"
    assert result["critic"]["verdict"] == "accept"


def test_sandbox_csv_runner_rejects_invalid_target(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    dataset = tmp_path / "data" / "downloaded" / "sample.csv"
    create_dataset(dataset)

    result = runner.run_sandbox_csv_experiment(
        dataset,
        "unknown_target",
    )

    assert result["success"] is False
    assert result["stage"] == "validation"


def test_sandbox_csv_runner_rejects_dataset_outside_downloaded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    dataset = tmp_path / "outside.csv"
    create_dataset(dataset)

    try:
        runner.run_sandbox_csv_experiment(dataset, "target")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "data/downloaded" in str(exc)


def test_sandbox_csv_runner_rejects_same_parameters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    dataset = tmp_path / "data" / "downloaded" / "sample.csv"
    create_dataset(dataset)

    result = runner.run_sandbox_csv_experiment(
        dataset,
        "target",
        baseline_params={"n_estimators": 10},
        treatment_params={"n_estimators": 10},
    )

    assert result["success"] is False
    assert result["stage"] == "configuration"


def test_sandbox_csv_runner_rejects_invalid_seed_count(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    dataset = tmp_path / "data" / "downloaded" / "sample.csv"
    create_dataset(dataset)

    try:
        runner.run_sandbox_csv_experiment(
            dataset,
            "target",
            n_seeds=1,
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "n_seeds" in str(exc)
