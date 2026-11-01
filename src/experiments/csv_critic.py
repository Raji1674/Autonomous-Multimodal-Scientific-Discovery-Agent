
from src.agents.statistical_critic import StatisticalCritic
from src.experiments.csv_evaluation import evaluate_csv_models


def evaluate_csv_with_critic(
    file_path: str,
    target_column: str,
    baseline_params: dict | None = None,
    treatment_params: dict | None = None,
    n_seeds: int = 20,
    test_size: float = 0.2,
    expected_direction: str = "higher",
    n_tests: int = 1,
) -> dict:
    """Evaluate CSV models and obtain a Statistical Critic verdict."""

    if expected_direction not in {"higher", "lower"}:
        raise ValueError("expected_direction must be 'higher' or 'lower'")

    experiment = evaluate_csv_models(
        file_path=file_path,
        target_column=target_column,
        baseline_params=baseline_params,
        treatment_params=treatment_params,
        n_seeds=n_seeds,
        test_size=test_size,
    )

    if not experiment.get("success"):
        return {
            "success": False,
            "stage": experiment.get("stage", "evaluation"),
            "issues": experiment.get("issues", []),
        }

    critic = StatisticalCritic()
    verdict = critic.evaluate_scores(
    result={
        "result_label": "executed",
        "baseline": experiment["baseline"],
        "treatment": experiment["treatment"],
        "n_train": experiment["n_train"],
        "n_test": experiment["n_test"],
    },
    expected_direction=expected_direction,
    n_tests=n_tests,
)

    return {
        "success": True,
        "dataset": experiment["dataset"],
        "target_column": target_column,
        "experiment": experiment,
        "critic": verdict,
        }
