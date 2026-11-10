
import pytest

from src.experiments.protocol import (
    ExperimentPlan,
    ResultError,
    parse_result,
    validate_result,
)


def make_plan(**overrides):
    data = {
        "research_question": "Does increasing C improve accuracy?",
        "hypothesis": "Increasing C improves accuracy.",
        "null_hypothesis": "C has no effect.",
        "dataset": "breast_cancer",
        "model": "logreg",
        "independent_variable": "C",
        "baseline_value": 0.1,
        "treatment_value": 1.0,
        "expected_direction": "higher",
    }
    data.update(overrides)
    return data


def valid_result():
    return {
        "result_label": "executed",
        "baseline": [0.82, 0.84, 0.81],
        "treatment": [0.86, 0.87, 0.85],
    }


def test_valid_experiment_plan():
    plan = ExperimentPlan(**make_plan())
    assert plan.model == "logreg"


def test_reject_unsupported_parameter():
    with pytest.raises(ValueError):
        ExperimentPlan(**make_plan(independent_variable="learning_rate"))


def test_reject_same_baseline_and_treatment():
    with pytest.raises(ValueError):
        ExperimentPlan(**make_plan(baseline_value=1.0))


def test_reject_invalid_dataset():
    with pytest.raises(ValueError):
        ExperimentPlan(**make_plan(dataset="unknown_dataset"))


def test_reject_invalid_seed_count():
    with pytest.raises(ValueError):
        ExperimentPlan(**make_plan(n_seeds=1))


def test_valid_result():
    result = valid_result()
    assert validate_result(result) == result


def test_reject_non_dictionary_result():
    with pytest.raises(ResultError):
        validate_result([])


def test_reject_missing_baseline():
    result = valid_result()
    result.pop("baseline")

    with pytest.raises(ResultError):
        validate_result(result)


def test_reject_unequal_paired_seed_counts():
    result = valid_result()
    result["treatment"] = [0.86, 0.87]

    with pytest.raises(ResultError):
        validate_result(result)


def test_reject_non_finite_scores():
    result = valid_result()
    result["baseline"][0] = float("nan")

    with pytest.raises(ResultError):
        validate_result(result)


def test_parse_valid_json_output():
    output = """
    Training started...
    Experiment completed.
    {"result_label":"executed","baseline":[0.8,0.82],"treatment":[0.85,0.86]}
    """

    result = parse_result(output)

    assert result["result_label"] == "executed"


def test_reject_empty_output():
    with pytest.raises(ResultError):
        parse_result("")


def test_reject_invalid_json_output():
    with pytest.raises(ResultError):
        parse_result("This is not JSON")
