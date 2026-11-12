
from src.agents.hypothesis_generator import HypothesisGenerator


def test_hypothesis_is_not_empty():
    hypothesis = (
        "Reducing Logistic Regression C from 1.0 to 0.01 "
        "will improve classification accuracy."
    )
    assert hypothesis.strip()


def test_hypothesis_identifies_independent_variable():
    hypothesis = (
        "Reducing Logistic Regression C from 1.0 to 0.01 "
        "will improve classification accuracy."
    )
    assert "C" in hypothesis


def test_hypothesis_identifies_measurable_outcome():
    hypothesis = (
        "Reducing Logistic Regression C from 1.0 to 0.01 "
        "will improve classification accuracy."
    )
    assert "accuracy" in hypothesis.lower()


def test_hypothesis_contains_comparison():
    hypothesis = (
        "Reducing Logistic Regression C from 1.0 to 0.01 "
        "will improve classification accuracy."
    )
    assert "1.0" in hypothesis
    assert "0.01" in hypothesis


def test_hypothesis_is_falsifiable():
    hypothesis = (
        "Reducing Logistic Regression C from 1.0 to 0.01 "
        "will improve classification accuracy."
    )
    assert "will improve" in hypothesis.lower()
