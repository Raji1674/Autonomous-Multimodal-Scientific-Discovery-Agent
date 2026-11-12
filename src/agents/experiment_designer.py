
"""Experiment Designer: converts a selected hypothesis into a validated,
machine-runnable experiment plan.

The LLM proposes a plan, but deterministic checks enforce consistency
between the research question, hypothesis, and selected model.
"""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from src.agents.base import BaseAgent
from src.experiments.protocol import ALLOWED_PARAMS, ExperimentPlan
from src.llm.client import LLMClient, complete_json
from src.schemas.state import FailureState, ResearchState


SYSTEM = (
    "You design small, reproducible scikit-learn experiments that test ONE "
    "hypothesis by changing ONE hyperparameter. "
    "Reply with ONLY JSON with keys: null_hypothesis, dataset "
    "(iris|wine|breast_cancer|digits), model "
    "(logreg|svm|knn|tree|random_forest), metric "
    "(accuracy|f1_macro), independent_variable, baseline_value, "
    "treatment_value, expected_direction (higher|lower), "
    "fixed_params (object, may be empty), falsification_criteria. "
    f"Allowed independent_variable per model: "
    f"{json.dumps({k: sorted(v) for k, v in ALLOWED_PARAMS.items()})}. "
    "The model MUST match the algorithm explicitly named in the research "
    "question or selected hypothesis. Never substitute another algorithm. "
    "For Logistic Regression regularization experiments, use model='logreg' "
    "and independent_variable='C'. "
    "Baseline and treatment values must differ. "
    "The experiment must test the selected hypothesis, not a different "
    "model or unrelated hyperparameter."
)


# Explicit model names and commonly used aliases.
MODEL_TERMS = {
    "logreg": (
        r"logistic\s+regression",
        r"logreg",
    ),
    "random_forest": (
        r"random\s+forest",
        r"randomforest",
    ),
    "svm": (
        r"support\s+vector\s+machine",
        r"\bsvm\b",
        r"\bsvc\b",
    ),
    "knn": (
        r"k[\s-]*nearest\s+neighbou?r",
        r"\bknn\b",
    ),
    "tree": (
        r"decision\s+tree",
        r"decisiontree",
    ),
}


def infer_expected_model(research_question: str, hypothesis: str) -> str | None:
    """Return the explicitly named model when exactly one model is identified."""

    text = f"{research_question} {hypothesis}".lower()

    matches = []

    for model, patterns in MODEL_TERMS.items():
        if any(re.search(pattern, text) for pattern in patterns):
            matches.append(model)

    # If multiple models are explicitly discussed, don't guess.
    if len(matches) == 1:
        return matches[0]

    return None


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
        hypothesis_data = state["selected_hypothesis"]

        hypothesis = hypothesis_data["hypothesis"]
        research_question = state["research_question"]

        expected_model = infer_expected_model(
            research_question,
            hypothesis,
        )

        base = (
            f"Research question: {research_question}\n"
            f"Selected hypothesis: {hypothesis}\n"
            f"Proposed experiment: "
            f"{hypothesis_data.get('experiment', '')}\n"
            f"Success metric: "
            f"{hypothesis_data.get('success_metric', '')}\n"
            f"Falsification: "
            f"{hypothesis_data.get('falsification_condition', '')}\n"
        )

        if expected_model:
            base += (
                f"\nMANDATORY MODEL CONSTRAINT: "
                f"model must be '{expected_model}'. "
                f"Do not choose another model.\n"
            )

        prompt = base
        last_error = ""

        for attempt in range(self.max_retries + 1):
            try:
                data = complete_json(
                    self.llm,
                    SYSTEM,
                    prompt,
                )

                if not isinstance(data, dict):
                    raise ValueError(
                        "LLM response must be a JSON object."
                    )

                proposed_model = data.get("model")

                # Reject a valid but scientifically unrelated model.
                if (
                    expected_model is not None
                    and proposed_model != expected_model
                ):
                    raise ValueError(
                        f"Model mismatch: the research question and "
                        f"hypothesis require '{expected_model}', but "
                        f"the LLM proposed '{proposed_model}'. "
                        f"Regenerate the plan using '{expected_model}'."
                    )

                plan = ExperimentPlan(
                    **{
                        **data,
                        "research_question": research_question,
                        "hypothesis": hypothesis,
                        "n_seeds": self.n_seeds,
                    }
                )

                return {
                    "experiment_plan": plan.model_dump(),
                }

            except (
                ValidationError,
                TypeError,
                ValueError,
            ) as exc:
                last_error = str(exc)

                prompt = (
                    base
                    + "\n\nYour previous plan was invalid:\n"
                    + last_error
                    + "\n\nCorrect the problem. "
                    + "Return ONLY valid JSON."
                )

        return {
            "failure_state": self.failure_on_error.value,
            "error": (
                f"No valid experiment plan after "
                f"{self.max_retries + 1} attempts: "
                f"{last_error[:500]}"
            ),
        }
