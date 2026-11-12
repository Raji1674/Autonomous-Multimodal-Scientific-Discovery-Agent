
import pytest

from src.experiments.human_paper_benchmark import (
    REFERENCE_PAPERS,
    _get_dataset_names,
    _get_model,
    _is_reference_dataset,
    evaluate_human_paper_benchmark,
)


def normalize_for_test(value):
    return "".join(
        char.lower()
        for char in str(value or "")
        if char.isalnum()
    )


def make_state(
    scores=None,
    model="Random Forest",
    dataset="Wisconsin Breast Cancer Dataset",
    rows=554,
):
    results = {}

    if scores is not None:
        results["baseline"] = scores
        results["experiment_protocol"] = {
            "algorithm": model,
        }

    return {
        "experiment_results": results,
        "experiment_plan": {"model": model},
        "selected_dataset": {"name": dataset},
        "dataset_validation": {"rows": rows},
    }


# -----------------------------
# Missing and invalid score tests
# -----------------------------

def test_missing_baseline_scores_returns_not_evaluated():
    result = evaluate_human_paper_benchmark(
        make_state(scores=None)
    )

    assert result["status"] == "not_evaluated"
    assert "unavailable" in result["reason"].lower()


@pytest.mark.parametrize(
    "scores",
    [
        [],
        [True, 0.9],
        [0.8, False],
        [-0.1, 0.9],
        [0.8, 1.1],
        ["0.9", 0.8],
        [None, 0.9],
    ],
)
def test_invalid_baseline_scores_are_rejected(scores):
    result = evaluate_human_paper_benchmark(
        make_state(scores=scores)
    )

    assert result["status"] == "not_evaluated"

    if scores:
        assert "valid accuracy" in result["reason"].lower()
    else:
        assert "unavailable" in result["reason"].lower()


# -----------------------------
# Model identification tests
# -----------------------------

@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("Random Forest", "randomforest"),
        ("RandomForestClassifier", "randomforest"),
        ("RF", "randomforest"),
        ("Logistic Regression", "logisticregression"),
        ("LogisticRegressionClassifier", "logisticregression"),
        ("LogReg", "logisticregression"),
        ("Support Vector Machine", "svm"),
        ("SVM", "svm"),
        ("XGBoost", "xgboost"),
    ],
)
def test_model_alias_normalization(model, expected):
    state = {"experiment_plan": {"model": model}}

    assert _get_model(state, {}) == expected


def test_protocol_algorithm_has_priority_over_plan_model():
    state = {
        "experiment_plan": {"model": "Logistic Regression"}
    }

    results = {
        "experiment_protocol": {
            "algorithm": "Random Forest"
        }
    }

    assert _get_model(state, results) == "randomforest"


def test_missing_model_returns_none():
    assert _get_model({}, {}) is None


def test_different_model_returns_not_comparable():
    result = evaluate_human_paper_benchmark(
        make_state(
            scores=[0.90, 0.92, 0.91],
            model="Logistic Regression",
        )
    )

    assert result["status"] == "not_comparable"
    assert result["published_model"] == "Random Forest"


# -----------------------------
# Dataset identification tests
# -----------------------------

@pytest.mark.parametrize(
    "dataset",
    [
        "Wisconsin Breast Cancer Dataset",
        "Wisconsin Breast Cancer",
        "Wisconsin Diagnostic Breast Cancer",
        "Breast Cancer",
        "Breast Cancer Dataset",
        "WDBC",
        "breast_cancer.csv",
    ],
)
def test_reference_dataset_aliases(dataset):
    names = {normalize_for_test(dataset)}

    assert _is_reference_dataset(names)


def test_unrelated_dataset_is_not_reference_dataset():
    assert not _is_reference_dataset({"iris"})
    assert not _is_reference_dataset({"diabetes"})


def test_dataset_identifiers_are_collected_from_state():
    state = {
        "selected_dataset": {
            "name": "Wisconsin Breast Cancer Dataset"
        },
        "dataset_metadata": {
            "filename": "breast_cancer.csv"
        },
    }

    names = _get_dataset_names(state, {})

    assert "wisconsinbreastcancerdataset" in names
    assert "breastcancercsv" in names


def test_missing_dataset_identity_returns_not_evaluated():
    state = {
        "experiment_results": {
            "baseline": [0.90, 0.91, 0.92],
            "experiment_protocol": {
                "algorithm": "Random Forest"
            },
        }
    }

    result = evaluate_human_paper_benchmark(state)

    assert result["status"] == "not_evaluated"
    assert "identity" in result["reason"].lower()


def test_unrelated_dataset_returns_not_comparable():
    result = evaluate_human_paper_benchmark(
        make_state(
            scores=[0.90, 0.92, 0.91],
            dataset="Iris",
        )
    )

    assert result["status"] == "not_comparable"
    assert result["published_dataset"] == (
        "Wisconsin Breast Cancer Dataset"
    )


# -----------------------------
# Compatible comparison tests
# -----------------------------

def test_compatible_experiment_returns_contextual_comparison():
    scores = [0.90, 0.92, 0.94]

    result = evaluate_human_paper_benchmark(
        make_state(scores=scores)
    )

    assert result["status"] == "contextual_comparison"
    assert result["agent_model"] == "RandomForestClassifier"
    assert result["agent_baseline_accuracy"] == pytest.approx(
        sum(scores) / len(scores)
    )
    assert result["published_accuracy"] == 0.965


def test_comparison_difference_is_in_percentage_points():
    result = evaluate_human_paper_benchmark(
        make_state(scores=[0.90, 0.92, 0.94])
    )

    expected = (0.92 - 0.965) * 100

    assert result["difference_percentage_points"] == pytest.approx(
        expected
    )


def test_same_dataset_size_is_true_when_rows_match():
    result = evaluate_human_paper_benchmark(
        make_state(scores=[0.90, 0.92], rows=554)
    )

    assert result["same_dataset_size"] is True


def test_different_dataset_size_is_false():
    result = evaluate_human_paper_benchmark(
        make_state(scores=[0.90, 0.92], rows=569)
    )

    assert result["same_dataset_size"] is False


def test_unknown_dataset_size_returns_none():
    state = make_state(scores=[0.90, 0.92])
    state["dataset_validation"] = {}

    result = evaluate_human_paper_benchmark(state)

    assert result["same_dataset_size"] is None


def test_comparison_includes_methodological_limitations():
    result = evaluate_human_paper_benchmark(
        make_state(scores=[0.90, 0.92])
    )

    assert result["status"] == "contextual_comparison"
    assert len(result["limitations"]) >= 5

    assert any(
        "reproduction" in item.lower()
        for item in result["limitations"]
    )


def test_reference_paper_metadata_is_available():
    reference = REFERENCE_PAPERS[0]

    assert reference["model"] == "Random Forest"
    assert reference["published_accuracy"] == 0.965
    assert reference["dataset"] == (
        "Wisconsin Breast Cancer Dataset"
    )
    assert reference["source"].startswith("https://")
