
"""Reliability tests for the actual StatisticalCritic implementation."""

import numpy as np
import pytest

from src.agents.statistical_critic import StatisticalCritic
from src.experiments.protocol import ResultError


def make_result(baseline, treatment):
    """Create a valid executed experiment result."""
    return {
        "experiment_name": "critic_reliability_test",
        "result_label": "executed",
        "baseline": list(baseline),
        "treatment": list(treatment),
        "n_train": 100,
        "n_test": 50,
        "seeds": list(range(len(baseline))),
    }


@pytest.fixture
def critic():
    return StatisticalCritic(
        alpha=0.05,
        min_seeds=10,
        min_abs_diff=0.005,
        min_effect=0.5,
    )


def test_accepts_significant_improvement(critic):
    rng = np.random.default_rng(42)
    baseline = rng.uniform(0.65, 0.75, 20)
    treatment = baseline + rng.normal(0.10, 0.015, 20)

    result = make_result(baseline, treatment)
    evaluation = critic.evaluate_scores(result, "higher")

    assert evaluation["verdict"] == "accept"
    assert evaluation["analysis"]["mean_diff"] > 0
    assert evaluation["analysis"]["p_adjusted"] < 0.05


def test_rejects_significant_opposite_direction(critic):
    rng = np.random.default_rng(42)
    baseline = rng.uniform(0.80, 0.90, 20)
    treatment = baseline + rng.normal(-0.10, 0.015, 20)

    result = make_result(baseline, treatment)
    evaluation = critic.evaluate_scores(result, "higher")

    assert evaluation["verdict"] == "reject"
    assert evaluation["analysis"]["mean_diff"] < 0
    assert evaluation["analysis"]["p_adjusted"] < 0.05


def test_inconclusive_when_no_mean_improvement(critic):
    baseline = np.linspace(0.70, 0.85, 20)
    treatment = baseline.copy()

    result = make_result(baseline, treatment)
    evaluation = critic.evaluate_scores(result, "higher")

    assert evaluation["verdict"] == "inconclusive"


def test_inconclusive_when_too_few_seeds(critic):
    baseline = np.linspace(0.70, 0.80, 5)
    treatment = baseline + 0.10

    result = make_result(baseline, treatment)
    evaluation = critic.evaluate_scores(result, "higher")

    assert evaluation["verdict"] == "inconclusive"
    assert any(
        issue["severity"] == "critical"
        for issue in evaluation["issues"]
    )


def test_rejects_unexecuted_results(critic):
    result = make_result(
        [0.70] * 20,
        [0.80] * 20,
    )
    result["result_label"] = "simulated"

    with pytest.raises(ResultError):
        critic.evaluate_scores(result, "higher")


def test_rejects_invalid_direction(critic):
    result = make_result(
        [0.70] * 20,
        [0.80] * 20,
    )

    with pytest.raises(ValueError):
        critic.evaluate_scores(result, "invalid")


def test_multiple_comparison_correction(critic):
    rng = np.random.default_rng(42)
    baseline = rng.uniform(0.65, 0.75, 20)
    treatment = baseline + rng.normal(0.10, 0.015, 20)

    result = make_result(baseline, treatment)

    evaluation = critic.evaluate_scores(
        result,
        expected_direction="higher",
        n_tests=5,
    )

    analysis = evaluation["analysis"]

    assert analysis["p_adjusted"] == pytest.approx(
        min(1.0, analysis["p"] * 5)
    )


def test_quality_score_is_bounded(critic):
    rng = np.random.default_rng(42)
    baseline = rng.uniform(0.70, 0.80, 20)
    treatment = baseline + rng.normal(0.10, 0.015, 20)

    result = make_result(baseline, treatment)
    evaluation = critic.evaluate_scores(result, "higher")

    assert 0 <= evaluation["statistical_quality"] <= 10


def test_lower_direction_accepts_significant_decrease(critic):
    rng = np.random.default_rng(42)
    baseline = rng.uniform(0.70, 0.80, 20)
    treatment = baseline + rng.normal(-0.10, 0.015, 20)

    result = make_result(baseline, treatment)
    evaluation = critic.evaluate_scores(result, "lower")

    assert evaluation["verdict"] == "accept"
    assert evaluation["analysis"]["mean_diff"] < 0
