
"""CSV Experiment Agent: runs discovered datasets in Docker and stores critic results."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.experiments.sandbox_csv_runner import run_sandbox_csv_experiment
from src.schemas.state import FailureState, ResearchState


class CSVExperimentAgent(BaseAgent):
    name = "csv_experiment_agent"
    failure_on_error = FailureState.EXECUTION_FAILED

    def run(self, state: ResearchState) -> dict:
        dataset_path = state.get("dataset_path")
        target_column = state.get("target_column")
        hypothesis = state.get("selected_hypothesis") or {}

        if not dataset_path:
            return {
                "failure_state": FailureState.EXECUTION_FAILED.value,
                "error": "Dataset path is missing.",
            }

        if not target_column:
            return {
                "failure_state": FailureState.EXECUTION_FAILED.value,
                "error": "Target column is missing.",
            }

        if not isinstance(hypothesis, dict):
            return {
                "failure_state": FailureState.EXECUTION_FAILED.value,
                "error": "Selected hypothesis must be a dictionary.",
            }

        result = run_sandbox_csv_experiment(
            file_path=dataset_path,
            target_column=target_column,
            n_seeds=20,
            test_size=0.2,
            expected_direction="higher",
            n_tests=1,
        )

        if not result.get("success"):
            return {
                "failure_state": FailureState.EXECUTION_FAILED.value,
                "error": (
                    f"CSV experiment failed at {result.get('stage')}: "
                    f"{result.get('issues', [])}"
                ),
                "experiment_results": {
                    "result_label": "execution_failed"
                },
            }

        critic = result["critic"]

        return {
            "experiment_results": result["experiment"],
            "statistical_results": critic["analysis"],
            "critic_feedback": {
                "issues": critic["issues"],
                "reasons": critic["reasons"],
                "statistical_quality": critic["statistical_quality"],
                "verdict": critic["verdict"],
            },
            "verdict": critic["verdict"],
            "dataset_metadata": {
                **(state.get("dataset_metadata") or {}),
                "selected_file": result["dataset"],
                "target_column": result["target_column"],
            },
        }
