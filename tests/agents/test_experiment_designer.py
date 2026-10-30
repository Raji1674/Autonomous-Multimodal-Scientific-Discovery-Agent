import json

from src.agents.experiment_designer import ExperimentDesigner
from src.llm.client import FakeLLM

STATE = {"research_question": "Does regularization help?",
         "selected_hypothesis": {"hypothesis": "Stronger L2 lowers variance.", "experiment": "vary C",
                                 "success_metric": "accuracy", "falsification_condition": "no change"}}
GOOD = {"null_hypothesis": "no change", "dataset": "iris", "model": "logreg", "metric": "accuracy",
        "independent_variable": "C", "baseline_value": 1.0, "treatment_value": 0.1,
        "expected_direction": "higher", "fixed_params": {}, "falsification_criteria": "no difference"}
BAD = {**GOOD, "model": "knn"}  # C is not allowed for knn


def test_valid_plan_accepted_and_seeds_forced():
    out = ExperimentDesigner(FakeLLM([json.dumps({**GOOD, "n_seeds": 2})]))(STATE)
    plan = out["experiment_plan"]
    assert plan["n_seeds"] == 20 and plan["hypothesis"] == "Stronger L2 lowers variance."


def test_invalid_plan_is_retried_with_error_feedback():
    llm = FakeLLM([json.dumps(BAD), json.dumps(GOOD)])
    out = ExperimentDesigner(llm)(STATE)
    assert "experiment_plan" in out
    assert "invalid" in llm.calls[1]["user"].lower() and "not allowed" in llm.calls[1]["user"]


def test_gives_up_after_max_retries():
    out = ExperimentDesigner(FakeLLM([json.dumps(BAD)] * 3), max_retries=2)(STATE)
    assert out["failure_state"] == "CODE_GENERATION_FAILED"
