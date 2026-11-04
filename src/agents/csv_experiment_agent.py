
"""CSV Experiment Agent: detects task and selects the appropriate ML trainer."""

from __future__ import annotations

from pathlib import Path

from src.agents.base import BaseAgent
from src.experiments.task_detector import detect_task
from src.experiments.csv_trainer import train_all_models
from src.experiments.regression_trainer import train_regression_models
from src.experiments.clustering_trainer import train_clustering_models
from src.experiments.sandbox_csv_runner import run_sandbox_csv_experiment
from src.schemas.state import FailureState, ResearchState


class CSVExperimentAgent(BaseAgent):
    name = "csv_experiment_agent"
    failure_on_error = FailureState.EXECUTION_FAILED

    def run(self, state: ResearchState) -> dict:
        dataset_path = state.get("dataset_path")
        target_column = state.get("target_column")
        task_override = state.get("task_override")
        hypothesis = state.get("selected_hypothesis") or {}

        if not dataset_path or not Path(dataset_path).is_file():
            return self._failure("Dataset file is missing or invalid.")

        if not isinstance(hypothesis, dict):
            return self._failure("Selected hypothesis must be a dictionary.")

        detection = detect_task(
            file_path=dataset_path,
            target_column=target_column,
            task_override=task_override,
        )

        if not detection.get("success"):
            return self._failure(detection.get("error", "Task detection failed."))

        task = detection["task"]

        # A missing target means unsupervised clustering.
        if task != "clustering" and not target_column:
            return self._failure("Target column is required for supervised learning.")

        if task == "classification":
            result = self._run_classification(
                dataset_path, target_column, state
            )
        elif task == "regression":
            result = train_regression_models(
                file_path=dataset_path,
                target_column=target_column,
            )
        else:
            result = train_clustering_models(file_path=dataset_path)

        if not result.get("success"):
            return self._failure(
                f"{task.title()} experiment failed: "
                f"{result.get('issues', result.get('error', 'Unknown error'))}",
                task=task,
                experiment_results=result,
            )

        return {
            "detected_task": task,
            "task_detection": detection,
            "experiment_results": result,
            "dataset_metadata": {
                **(state.get("dataset_metadata") or {}),
                "selected_file": Path(dataset_path).name,
                "target_column": target_column,
                "task": task,
            },
            "verdict": "completed",
        }

    @staticmethod
    def _failure(
        message: str,
        task: str | None = None,
        experiment_results: dict | None = None,
    ) -> dict:
        update = {
            "failure_state": FailureState.EXECUTION_FAILED.value,
            "error": message,
        }
        if task:
            update["detected_task"] = task
        if experiment_results is not None:
            update["experiment_results"] = experiment_results
        return update

    @staticmethod
    def _run_classification(
        dataset_path: str,
        target_column: str,
        state: ResearchState,
    ) -> dict:
        # Preserve the existing Docker-isolated paired experiment and critic.
        return run_sandbox_csv_experiment(
            file_path=dataset_path,
            target_column=target_column,
            n_seeds=20,
            test_size=0.2,
            expected_direction="higher",
            n_tests=1,
        )
