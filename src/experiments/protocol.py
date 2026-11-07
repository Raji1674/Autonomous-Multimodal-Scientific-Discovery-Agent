
"""Experiment plan + result contracts: the JSON protocol between agents and the sandbox."""
from __future__ import annotations

import json
import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# Hyperparameters an experiment may vary, per model.
ALLOWED_PARAMS: dict[str, set[str]] = {
    "logreg": {"C"},
    "svm": {"C"},
    "knn": {"n_neighbors"},
    "tree": {"max_depth"},
    "random_forest": {"n_estimators", "max_depth"},
    "linear_regression": {"fit_intercept"},
    "decision_tree_regressor": {"max_depth"},
    "random_forest_regressor": {"n_estimators", "max_depth"},
    "ridge": {"alpha"},
}

CLASSIFICATION_MODELS = {
    "logreg", "svm", "knn", "tree", "random_forest"
}

REGRESSION_MODELS = {
    "linear_regression",
    "decision_tree_regressor",
    "random_forest_regressor",
    "ridge",
}

CLASSIFICATION_METRICS = {"accuracy", "f1_macro"}
REGRESSION_METRICS = {"mae", "rmse", "r2"}

RESULT_LABELS = {"executed", "simulated", "not_executed"}


class ExperimentPlan(BaseModel):
    """Validated, machine-runnable experiment design."""

    research_question: str
    hypothesis: str
    null_hypothesis: str

    dataset: Literal[
        "iris", "wine", "breast_cancer", "digits", "uploaded_csv"
    ] = "iris"

    task_type: Literal["classification", "regression"] = "classification"

    model: Literal[
        "logreg", "svm", "knn", "tree", "random_forest",
        "linear_regression", "decision_tree_regressor",
        "random_forest_regressor", "ridge",
    ]

    metric: Literal["accuracy", "f1_macro", "mae", "rmse", "r2"] = "accuracy"

    independent_variable: str
    baseline_value: float | int | bool | None
    treatment_value: float | int | bool | None

    expected_direction: Literal["higher", "lower"] = "higher"
    n_seeds: int = Field(20, ge=2, le=100)
    test_size: float = Field(0.3, ge=0.1, le=0.5)

    fixed_params: dict[str, float | int | bool | None] = Field(default_factory=dict)
    excluded_columns: list[str] = Field(default_factory=list)
    falsification_criteria: str = ""

    @model_validator(mode="after")
    def _check(self) -> "ExperimentPlan":
        allowed = ALLOWED_PARAMS[self.model]

        if self.independent_variable not in allowed:
            raise ValueError(
                f"{self.independent_variable!r} is not allowed for "
                f"{self.model}; use {sorted(allowed)}"
            )

        bad = set(self.fixed_params) - allowed
        if bad:
            raise ValueError(
                f"fixed_params not allowed for {self.model}: {sorted(bad)}"
            )

        if self.independent_variable in self.fixed_params:
            raise ValueError(
                "independent_variable must not also appear in fixed_params"
            )

        if self.baseline_value == self.treatment_value:
            raise ValueError("baseline_value and treatment_value must differ")

        if self.task_type == "classification":
            if self.model not in CLASSIFICATION_MODELS:
                raise ValueError("Classification requires a classification model")
            if self.metric not in CLASSIFICATION_METRICS:
                raise ValueError("Classification requires accuracy or f1_macro")

        if self.task_type == "regression":
            if self.model not in REGRESSION_MODELS:
                raise ValueError("Regression requires a regression model")
            if self.metric not in REGRESSION_METRICS:
                raise ValueError("Regression requires mae, rmse, or r2")

        if self.dataset == "uploaded_csv" and self.dataset == "iris":
            raise ValueError("Invalid dataset configuration")

        if len(self.excluded_columns) != len(set(self.excluded_columns)):
            raise ValueError("excluded_columns must not contain duplicates")

        if any(not column.strip() for column in self.excluded_columns):
            raise ValueError("excluded_columns cannot contain empty names")

        return self


class ResultError(ValueError):
    """The experiment output does not follow the result protocol."""


def validate_result(data: dict) -> dict:
    """Check result structure, not scientific quality."""
    if not isinstance(data, dict):
        raise ResultError("result must be a JSON object")

    if data.get("result_label") not in RESULT_LABELS:
        raise ResultError(
            f"result_label must be one of {sorted(RESULT_LABELS)}"
        )

    for key in ("baseline", "treatment"):
        vals = data.get(key)

        if not isinstance(vals, list) or not vals:
            raise ResultError(f"{key!r} must be a non-empty list of scores")

        if not all(
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(v)
            for v in vals
        ):
            raise ResultError(
                f"{key!r} contains non-numeric or non-finite values"
            )

    if len(data["baseline"]) != len(data["treatment"]):
        raise ResultError(
            "baseline and treatment must have the same length (paired seeds)"
        )

    if len(data["baseline"]) < 2:
        raise ResultError("need at least 2 seeds")

    return data


def parse_result(stdout: str) -> dict:
    """Parse the LAST non-empty stdout line as JSON."""
    lines = [ln for ln in stdout.strip().splitlines() if ln.strip()]

    if not lines:
        raise ResultError("experiment produced no output")

    try:
        data = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ResultError(f"last output line is not JSON: {exc}") from exc

    return validate_result(data)
