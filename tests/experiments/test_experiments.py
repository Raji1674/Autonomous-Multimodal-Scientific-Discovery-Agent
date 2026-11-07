import json
import subprocess
import sys

import pytest

from src.experiments.datasets import load_dataset
from src.experiments.protocol import ExperimentPlan, ResultError, parse_result, validate_result
from src.experiments.templates import render_script, run_plan


def make_plan(**over) -> ExperimentPlan:
    base = dict(research_question="q", hypothesis="h", null_hypothesis="n", dataset="iris", model="logreg",
                independent_variable="C", baseline_value=1.0, treatment_value=0.01, n_seeds=5)
    base.update(over)
    return ExperimentPlan(**base)


def test_plan_rejects_disallowed_variable():
    with pytest.raises(ValueError):
        make_plan(model="knn", independent_variable="C")


def test_plan_rejects_same_values_and_overlapping_fixed_params():
    with pytest.raises(ValueError):
        make_plan(treatment_value=1.0)
    with pytest.raises(ValueError):
        make_plan(fixed_params={"C": 2.0})


def test_dataset_loader():
    X, y, meta = load_dataset("iris")
    assert X.shape == (150, 4) and meta["n_classes"] == 3
    with pytest.raises(ValueError):
        load_dataset("nope")


def test_run_plan_shapes_label_and_split_sizes():
    r = run_plan(make_plan())
    assert r["result_label"] == "executed"
    assert len(r["baseline"]) == len(r["treatment"]) == 5
    assert r["n_train"] + r["n_test"] == 150
    assert all(0.0 <= v <= 1.0 for v in r["baseline"] + r["treatment"])


def test_run_plan_is_deterministic():
    a, b = run_plan(make_plan()), run_plan(make_plan())
    assert a["baseline"] == b["baseline"] and a["treatment"] == b["treatment"]


def test_rendered_script_matches_in_process_result():
    """The sandbox script and the in-process path run identical code (here run as a local subprocess)."""
    plan = make_plan(model="knn", independent_variable="n_neighbors", baseline_value=3, treatment_value=15)
    out = subprocess.run([sys.executable, "-I", "-c", render_script(plan)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    from_script = parse_result(out.stdout)
    in_proc = run_plan(plan)
    assert from_script["baseline"] == in_proc["baseline"]
    assert from_script["treatment"] == in_proc["treatment"]


def test_parse_result_uses_last_line_and_rejects_bad_output():
    good = {"result_label": "executed", "baseline": [0.9, 0.8], "treatment": [0.95, 0.85]}
    assert parse_result("warning: something\n" + json.dumps(good))["treatment"] == [0.95, 0.85]
    for bad in ["", "not json", json.dumps({**good, "result_label": "made_up"}),
                json.dumps({**good, "treatment": [0.9]}), json.dumps({**good, "baseline": [0.9, "x"]}),
                json.dumps({**good, "baseline": [0.9, 1e999]})]:
        with pytest.raises(ResultError):
            parse_result(bad)


def test_validate_result_rejects_non_dict():
    with pytest.raises(ResultError):
        validate_result([1, 2, 3])

from src.experiments.dataset_validator import validate_csv_dataset


def test_validate_csv_dataset_accepts_valid_csv(tmp_path):
    csv_file = tmp_path / "valid.csv"
    csv_file.write_text(
        "feature_a,feature_b,target\n"
        "1,2,0\n"
        "3,4,1\n"
        "5,6,0\n",
        encoding="utf-8",
    )

    result = validate_csv_dataset(csv_file, "target")

    assert result["valid"] is True
    assert result["rows"] == 3
    assert result["columns"] == 3
    assert result["target_classes"] == 2
    assert result["issues"] == []


def test_validate_csv_dataset_rejects_missing_file(tmp_path):
    result = validate_csv_dataset(tmp_path / "missing.csv", "target")

    assert result["valid"] is False
    assert "CSV file not found" in result["issues"]


def make_csv_dataset(path):
    """Create a reproducible dataset with numeric and categorical features."""
    rows = ["study_hours,attendance,school_type,passed"]

    for i in range(120):
        study_hours = (i % 10) + 1
        attendance = 60 + (i % 40)
        school_type = "public" if i % 2 == 0 else "private"
        passed = 1 if study_hours >= 5 else 0

        rows.append(
            f"{study_hours},{attendance},{school_type},{passed}"
        )

    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def test_run_plan_with_csv_supports_categorical_features(tmp_path):
    csv_file = make_csv_dataset(tmp_path / "students.csv")

    plan = make_plan(
        dataset="uploaded_csv",
        task_type="classification",
        model="random_forest",
        independent_variable="n_estimators",
        baseline_value=10,
        treatment_value=30,
    )

    result = run_plan(
        plan,
        dataset_path=str(csv_file),
        target_column="passed",
    )

    assert result["result_label"] == "executed"
    assert result["dataset_source"] == "uploaded_csv"
    assert result["dataset_name"] == "uploaded_csv"
    assert result["target_column"] == "passed"
    assert result["n_samples"] == 120
    assert result["n_features"] == 3
    assert len(result["baseline"]) == 5
    assert len(result["treatment"]) == 5


def test_run_plan_with_csv_rejects_missing_target(tmp_path):
    csv_file = make_csv_dataset(tmp_path / "students.csv")

    plan = make_plan(
        model="random_forest",
        independent_variable="n_estimators",
        baseline_value=10,
        treatment_value=30,
    )

    with pytest.raises(ValueError, match="target column"):
        run_plan(
            plan,
            dataset_path=str(csv_file),
            target_column=None,
        )


def test_rendered_csv_script_matches_in_process_result(tmp_path):
    csv_file = make_csv_dataset(tmp_path / "students.csv")

    plan = make_plan(
        dataset="uploaded_csv",
        task_type="classification",
        model="random_forest",
        independent_variable="n_estimators",
        baseline_value=10,
        treatment_value=30,
    )

    script = render_script(
        plan,
        dataset_path=str(csv_file),
        target_column="passed",
    )

    out = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert out.returncode == 0, out.stderr

    from_script = parse_result(out.stdout)
    in_process = run_plan(
        plan,
        dataset_path=str(csv_file),
        target_column="passed",
    )

    assert from_script["baseline"] == in_process["baseline"]
    assert from_script["treatment"] == in_process["treatment"]
    assert from_script["dataset_source"] == "uploaded_csv"

def test_run_plan_with_csv_supports_regression_and_excluded_columns(tmp_path):
    import pandas as pd

    csv_file = tmp_path / "students.csv"
    df = pd.DataFrame({
        "studytime": [1, 2, 3, 4, 5] * 20,
        "absences": [0, 2, 1, 3, 4] * 20,
        "G1": [10, 12, 14, 16, 18] * 20,
        "G2": [11, 13, 15, 17, 19] * 20,
        "G3": [9, 11, 13, 15, 17] * 20,
    })
    df.to_csv(csv_file, index=False)

    plan = make_plan(
        dataset="uploaded_csv",
        task_type="regression",
        model="random_forest_regressor",
        metric="mae",
        independent_variable="n_estimators",
        baseline_value=10,
        treatment_value=30,
        expected_direction="lower",
        excluded_columns=["G1", "G2"],
    )

    result = run_plan(
        plan,
        dataset_path=str(csv_file),
        target_column="G3",
    )

    assert result["result_label"] == "executed"
    assert result["dataset_source"] == "uploaded_csv"
    assert result["task_type"] == "regression"
    assert result["target_column"] == "G3"
    assert "G1" not in result["feature_columns"]
    assert "G2" not in result["feature_columns"]
    assert "G3" not in result["feature_columns"]
    assert result["n_features"] == 2
    assert len(result["baseline"]) == 5
    assert len(result["treatment"]) == 5
