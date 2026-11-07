import numpy as np
import pytest

from src.agents.statistical_critic import StatisticalCritic, StatisticalCriticAgent, check_reproducibility
from src.experiments.protocol import ExperimentPlan, ResultError
from src.experiments.templates import run_plan
from src.llm.client import FakeLLM

PLAN = dict(research_question="q", hypothesis="h", null_hypothesis="n", dataset="iris", model="logreg",
            independent_variable="C", baseline_value=1.0, treatment_value=0.01, n_seeds=20)


def synthetic(diffs, n_train=105, n_test=45, label="executed"):
    """Hand-built SYNTHETIC scores for unit-testing the critic (not experiment results)."""
    base = np.full(len(diffs), 0.9) + np.tile([0.004, -0.004], len(diffs))[: len(diffs)]
    return {"result_label": label, "baseline": base.tolist(), "treatment": (base + np.asarray(diffs)).tolist(),
            "n_train": n_train, "n_test": n_test}


def test_clear_improvement_is_accepted():
    diffs = 0.02 + np.tile([0.002, -0.002, 0.004, -0.004], 5)  # mean +0.02, small spread, 20 runs
    ev = StatisticalCritic().evaluate(PLAN, synthetic(diffs))
    assert ev["verdict"] == "accept" and ev["analysis"]["p_adjusted"] < 0.05


def test_zero_mean_difference_is_rejected():
    diffs = np.tile([0.003, -0.003], 10)  # mean exactly 0
    ev = StatisticalCritic().evaluate(PLAN, synthetic(diffs))
    assert ev["verdict"] == "reject" and ev["analysis"]["p"] == pytest.approx(1.0)


def test_significant_but_opposite_direction_is_rejected():
    diffs = -0.02 + np.tile([0.002, -0.002, 0.004, -0.004], 5)
    ev = StatisticalCritic().evaluate(PLAN, synthetic(diffs))
    assert ev["verdict"] == "reject" and any("OPPOSITE" in r for r in ev["reasons"])


def test_too_few_runs_is_inconclusive():
    ev = StatisticalCritic().evaluate(PLAN, synthetic(np.full(6, 0.05) + np.tile([0.001, -0.001], 3)))
    assert ev["verdict"] == "inconclusive"
    assert any(i["severity"] == "critical" for i in ev["issues"])


def test_bonferroni_can_flip_a_borderline_result():
    diffs = 0.004 + np.tile([0.003, -0.003, 0.006, -0.006], 5)
    single = StatisticalCritic(min_abs_diff=0.001, min_effect=0.1).evaluate(PLAN, synthetic(diffs), n_tests=1)
    many = StatisticalCritic(min_abs_diff=0.001, min_effect=0.1).evaluate(PLAN, synthetic(diffs), n_tests=50)
    assert many["analysis"]["p_adjusted"] >= single["analysis"]["p_adjusted"]
    assert many["verdict"] == "reject"


@pytest.mark.parametrize("label", ["not_executed", "simulated"])
def test_critic_refuses_unexecuted_results(label):
    with pytest.raises(ResultError):
        StatisticalCritic().evaluate(PLAN, synthetic(np.zeros(10), label=label))


def test_agent_sets_failure_state_for_not_executed_stub_output():
    agent = StatisticalCriticAgent()
    out = agent({"experiment_plan": PLAN, "experiment_results": {"result_label": "not_executed"}})
    assert out["failure_state"] == "STATISTICAL_ANALYSIS_FAILED"


def test_agent_end_to_end_on_real_template_run_with_llm_explanation():
    plan = ExperimentPlan(**{**PLAN, "n_seeds": 10})
    result = run_plan(plan)
    llm = FakeLLM(["Scripted explanation."])
    out = StatisticalCriticAgent(llm=llm, check_repro=True)(
        {"experiment_plan": plan.model_dump(), "experiment_results": result})
    assert out["verdict"] in {"accept", "reject", "inconclusive"}
    assert out["critic_feedback"]["interpretation"] == "Scripted explanation."
    assert out["critic_feedback"]["reproducibility"]["reproduced"] is True
    assert "p" in out["statistical_results"]
    assert "never recompute" in llm.calls[0]["system"].lower()  # LLM is told not to compute numbers


def test_reproducibility_check_detects_tampered_scores():
    plan = ExperimentPlan(**{**PLAN, "n_seeds": 5})
    result = run_plan(plan)
    assert check_reproducibility(plan.model_dump(), result)["reproduced"] is True
    result["treatment"][0] += 0.05
    assert check_reproducibility(plan.model_dump(), result)["reproduced"] is False

def test_lower_is_better_metric_accepts_improvement():
    diffs = -0.5 + np.tile([0.02, -0.02, 0.04, -0.04], 5)

    result = synthetic(diffs)
    plan = {
        **PLAN,
        "task_type": "regression",
        "model": "random_forest_regressor",
        "metric": "mae",
        "expected_direction": "lower",
        "independent_variable": "n_estimators",
        "baseline_value": 10,
        "treatment_value": 20,
    }

    ev = StatisticalCritic(
        min_abs_diff=0.1,
        min_effect=0.1,
    ).evaluate(plan, result)

    assert ev["verdict"] == "accept"


def test_regression_overfit_gap_uses_lower_is_better_direction():
    result = synthetic(np.full(20, -0.2))
    result["treatment_train"] = [1.0] * 20
    result["treatment"] = [2.0] * 20

    critic = StatisticalCritic(overfit_gap=0.15)
    ev = critic.evaluate_scores(
        result,
        expected_direction="lower",
    )

    assert any(
        "possible overfitting" in issue["message"]
        for issue in ev["issues"]
    )


def test_regression_does_not_trigger_accuracy_ceiling_warning():
    diffs = -0.2 + np.tile(
        [0.02, -0.02, 0.04, -0.04],
        5,
    )

    result = synthetic(diffs)
    result["baseline"] = [0.999] * 20
    result["treatment"] = [0.999] * 20

    ev = StatisticalCritic().evaluate_scores(
        result,
        expected_direction="lower",
    )

    assert not any(
        "ceiling effect" in issue["message"]
        for issue in ev["issues"]
    )
