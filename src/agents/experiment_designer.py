
"""Experiment Designer: creates validated classification or regression plans."""
from __future__ import annotations

import json

from pydantic import ValidationError

from src.agents.base import BaseAgent
from src.experiments.protocol import ALLOWED_PARAMS, ExperimentPlan
from src.llm.client import LLMClient, complete_json
from src.schemas.state import FailureState, ResearchState


CLASSIFICATION_SYSTEM = (
    "Design a reproducible scikit-learn classification experiment. "
    "Reply with ONLY JSON containing: null_hypothesis, dataset, task_type, "
    "model, metric, independent_variable, baseline_value, treatment_value, "
    "expected_direction, fixed_params, excluded_columns, falsification_criteria. "
    "Use dataset iris, wine, breast_cancer, digits, or uploaded_csv. "
    "Use task_type classification, a classification model, and metric accuracy or f1_macro. "
    "For uploaded_csv, use dataset uploaded_csv. "
    "Use only these canonical model identifiers: logreg, svm, knn, tree, random_forest. "
)

REGRESSION_SYSTEM = (
    "Design a reproducible scikit-learn regression experiment. "
    "Reply with ONLY JSON containing: null_hypothesis, dataset, task_type, "
    "model, metric, independent_variable, baseline_value, treatment_value, "
    "expected_direction, fixed_params, excluded_columns, falsification_criteria. "
    "Use dataset uploaded_csv and task_type regression for uploaded CSV data. "
    "Allowed regression models: linear_regression, decision_tree_regressor, "
    "random_forest_regressor, ridge. "
    "Allowed metrics: mae, rmse, r2. "
    "For mae and rmse, expected_direction must be lower when improvement means "
    "smaller prediction error. For r2, expected_direction must be higher. "
)


def normalize_model_name(model: str) -> str:
    """Convert common sklearn class names to project model identifiers."""

    aliases = {
        "LogisticRegression": "logreg",
        "SVC": "svm",
        "KNeighborsClassifier": "knn",
        "DecisionTreeClassifier": "tree",
        "RandomForestClassifier": "random_forest",
        "LinearRegression": "linear_regression",
        "DecisionTreeRegressor": "decision_tree_regressor",
        "RandomForestRegressor": "random_forest_regressor",
        "Ridge": "ridge",
    }

    return aliases.get(model.strip(), model.strip())


class ExperimentDesigner(BaseAgent):
    name = "experiment_designer"
    failure_on_error = FailureState.CODE_GENERATION_FAILED

    def __init__(
        self,
        llm: LLMClient,
        max_retries: int = 2,
        n_seeds: int = 20,
    ):
        super().__init__()
        self.llm = llm
        self.max_retries = max_retries
        self.n_seeds = n_seeds

    def run(self, state: ResearchState) -> dict:
        h = state["selected_hypothesis"]

        dataset_path = state.get("dataset_path")
        target_column = state.get("target_column")
        task_type = state.get("dataset_task_type") or state.get("task_type")

        if dataset_path and not target_column:
            return {
                "failure_state": self.failure_on_error.value,
                "error": "An uploaded CSV requires a target column.",
            }

        if dataset_path and task_type not in {"classification", "regression"}:
            return {
                "failure_state": self.failure_on_error.value,
                "error": "Uploaded CSV task type must be classification or regression.",
            }

        is_uploaded_csv = bool(dataset_path)
        is_regression = task_type == "regression"

        if is_uploaded_csv:
            system = REGRESSION_SYSTEM if is_regression else CLASSIFICATION_SYSTEM
            dataset_instructions = (
                f"\nDataset: uploaded_csv"
                f"\nTarget column: {target_column}"
                f"\nTask type: {task_type}"
                "\nUse only the uploaded CSV. Do not choose a built-in dataset."
            )

            available_columns = state.get("dataset_columns")
            if available_columns:
                dataset_instructions += (
                    "\nAvailable CSV columns: "
                    + json.dumps(available_columns)
                )
        else:
            system = CLASSIFICATION_SYSTEM
            dataset_instructions = (
                "\nNo CSV was uploaded. Choose a suitable built-in "
                "classification dataset."
            )

        base = (
            f"Research question: {state['research_question']}\n"
            f"Hypothesis: {h['hypothesis']}\n"
            f"Proposed experiment: {h.get('experiment', '')}\n"
            f"Success metric: {h.get('success_metric', '')}\n"
            f"Falsification: {h.get('falsification_condition', '')}\n"
            f"{dataset_instructions}"
        )

        requested_exclusions = state.get("excluded_columns")
        if requested_exclusions:
            base += (
                "\nColumns that MUST be excluded from predictors: "
                + json.dumps(requested_exclusions)
                + "\nDo not include the target column among excluded columns."
            )

        if is_uploaded_csv:
            base += (
                "\nThe plan dataset must be uploaded_csv."
                "\nThe plan task_type must match the supplied task type."
            )

        if is_regression:
            base += (
                "\nThe target is numerical. Use a regression model and "
                "a regression metric. Do not use accuracy or f1_macro."
            )

        critic_feedback = state.get("critic_feedback")
        previous_plan = state.get("experiment_plan")

        if critic_feedback and previous_plan:
            base += (
                "\n\nREVISION REQUIRED"
                "\nKeep the original scientific hypothesis unchanged."
                "\nImprove the experiment design using this critic feedback:"
                f"\n{json.dumps(critic_feedback, default=str)}"
                "\nPrevious experiment plan:"
                f"\n{json.dumps(previous_plan, default=str)}"
                "\nChange the experiment design only where scientifically justified."
                "\nDo not weaken the statistical acceptance criteria."
            )

        prompt = base
        last = ""

        for _ in range(self.max_retries + 1):
            data = complete_json(self.llm, system, prompt)

            try:
                if isinstance(data.get("model"), str):
                    data["model"] = normalize_model_name(data["model"])

                if is_uploaded_csv:
                    data["dataset"] = "uploaded_csv"
                    data["task_type"] = task_type

                    if requested_exclusions is not None:
                        data["excluded_columns"] = requested_exclusions

                plan = ExperimentPlan(
                    **{
                        **data,
                        "research_question": state["research_question"],
                        "hypothesis": h["hypothesis"],
                        "n_seeds": self.n_seeds,
                    }
                )

                return {"experiment_plan": plan.model_dump()}

            except (ValidationError, TypeError, ValueError) as exc:
                last = str(exc)
                prompt = (
                    base
                    + f"\n\nYour previous plan was invalid:\n{last}"
                    + "\nFix it and reply with ONLY JSON."
                )

        return {
            "failure_state": self.failure_on_error.value,
            "error": f"no valid experiment plan: {last[:300]}",
        }