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