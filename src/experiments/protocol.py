"""Experiment plan + result contracts: the JSON protocol between agents and the sandbox."""
from __future__ import annotations

import json
import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Which hyperparameters an experiment may vary, per model (keeps plans testable and safe).
ALLOWED_PARAMS: dict[str, set[str]] = {
    "logreg": {"C"},
    "svm": {"C"},
    "knn": {"n_neighbors"},
    "tree": {"max_depth"},
    "random_forest": {"n_estimators", "max_depth"},
}
RESULT_LABELS = {"executed", "simulated", "not_executed"}


class ExperimentPlan(BaseModel):
    """The 12-field design the Experiment Designer produces (subset that is machine-runnable)."""

    research_question: str
    hypothesis: str
    null_hypothesis: str
    dataset: Literal["iris", "wine", "breast_cancer", "digits"]
    model: Literal["logreg", "svm", "knn", "tree", "random_forest"]
    metric: Literal["accuracy", "f1_macro"] = "accuracy"
    independent_variable: str
    baseline_value: float | int | None
    treatment_value: float | int | None
    expected_direction: Literal["higher", "lower"] = "higher"
    n_seeds: int = Field(20, ge=2, le=100)
    test_size: float = Field(0.3, ge=0.1, le=0.5)
    fixed_params: dict[str, float | int | None] = Field(default_factory=dict)
    falsification_criteria: str = ""

    @model_validator(mode="after")
    def _check(self) -> "ExperimentPlan":
        allowed = ALLOWED_PARAMS[self.model]
        if self.independent_variable not in allowed:
            raise ValueError(f"{self.independent_variable!r} is not allowed for {self.model}; use {sorted(allowed)}")
        bad = set(self.fixed_params) - allowed
        if bad:
            raise ValueError(f"fixed_params not allowed for {self.model}: {sorted(bad)}")
        if self.independent_variable in self.fixed_params:
            raise ValueError("independent_variable must not also appear in fixed_params")
        if self.baseline_value == self.treatment_value:
            raise ValueError("baseline_value and treatment_value must differ")
        return self


class ResultError(ValueError):
    """The experiment output does not follow the result protocol."""


def validate_result(data: dict) -> dict:
    """Check structure only (not scientific quality). Raises ResultError on problems."""
    if not isinstance(data, dict):
        raise ResultError("result must be a JSON object")
    if data.get("result_label") not in RESULT_LABELS:
        raise ResultError(f"result_label must be one of {sorted(RESULT_LABELS)}")
    for key in ("baseline", "treatment"):
        vals = data.get(key)
        if not isinstance(vals, list) or not vals:
            raise ResultError(f"{key!r} must be a non-empty list of scores")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vals):
            raise ResultError(f"{key!r} contains non-numeric or non-finite values")
    if len(data["baseline"]) != len(data["treatment"]):
        raise ResultError("baseline and treatment must have the same length (paired seeds)")
    if len(data["baseline"]) < 2:
        raise ResultError("need at least 2 seeds")
    return data


def parse_result(stdout: str) -> dict:
    """Parse the LAST non-empty stdout line as JSON (earlier lines may be library chatter)."""
    lines = [ln for ln in stdout.strip().splitlines() if ln.strip()]
    if not lines:
        raise ResultError("experiment produced no output")
    try:
        data = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ResultError(f"last output line is not JSON: {exc}") from exc
    return validate_result(data)
