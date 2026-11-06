
"""Experiment Designer: turns the selected hypothesis into a validated, machine-runnable plan.

The LLM proposes design choices; pydantic validation (ExperimentPlan) is the gatekeeper.
Validation errors are fed back to the LLM. Critic feedback is used for scientific revisions.
"""
from __future__ import annotations

import json

from pydantic import ValidationError

from src.agents.base import BaseAgent
from src.experiments.protocol import ALLOWED_PARAMS, ExperimentPlan
from src.llm.client import LLMClient, complete_json
from src.schemas.state import FailureState, ResearchState

SYSTEM = (
    "You design small, reproducible scikit-learn experiments that test ONE hypothesis by changing ONE "
    "hyperparameter. Reply with ONLY JSON with keys: null_hypothesis, dataset (iris|wine|breast_cancer|digits), "
    "model (logreg|svm|knn|tree|random_forest), metric (accuracy|f1_macro), independent_variable, "
    "baseline_value, treatment_value, expected_direction (higher|lower), fixed_params (object, may be empty), "
    f"falsification_criteria. Allowed independent_variable per model: {json.dumps({k: sorted(v) for k, v in ALLOWED_PARAMS.items()})}. "
    "Baseline and treatment values must differ."
)


class ExperimentDesigner(BaseAgent):
    name = "experiment_designer"
    failure_on_error = FailureState.CODE_GENERATION_FAILED

    def __init__(self, llm: LLMClient, max_retries: int = 2, n_seeds: int = 20):
        super().__init__()
        self.llm, self.max_retries, self.n_seeds = llm, max_retries, n_seeds

    def run(self, state: ResearchState) -> dict:
        h = state["selected_hypothesis"]

        base = (
            f"Hypothesis: {h['hypothesis']}\n"
            f"Proposed experiment: {h.get('experiment', '')}\n"
            f"Success metric: {h.get('success_metric', '')}\n"
            f"Falsification: {h.get('falsification_condition', '')}"
        )

        # Add critic feedback only when this is a scientific revision.
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

        prompt, last = base, ""

        for _ in range(self.max_retries + 1):
            data = complete_json(self.llm, SYSTEM, prompt)

            try:
                plan = ExperimentPlan(
                    **{
                        **data,
                        "research_question": state["research_question"],
                        "hypothesis": h["hypothesis"],
                        "n_seeds": self.n_seeds,
                    }
                )
                return {"experiment_plan": plan.model_dump()}

            except (ValidationError, TypeError) as exc:
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
