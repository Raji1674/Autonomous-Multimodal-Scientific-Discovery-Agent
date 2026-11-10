
"""CSV Experiment Agent with reward-based debugging and automatic retries."""

from __future__ import annotations

from pathlib import Path

from src.agents.base import BaseAgent
from src.experiments.task_detector import detect_task
from src.experiments.regression_trainer import train_regression_models
from src.experiments.clustering_trainer import train_clustering_models
from src.experiments.sandbox_csv_runner import run_sandbox_csv_experiment
from src.schemas.state import FailureState, ResearchState


class CSVExperimentAgent(BaseAgent):
    name = "csv_experiment_agent"
    failure_on_error = FailureState.EXECUTION_FAILED

    MAX_ATTEMPTS = 3

    def __init__(self, debugger=None, reward_policy=None):
        super().__init__()

        self.debugger = debugger

        # Reuse the debugger's policy so selected strategies
        # receive rewards in the same policy instance.
        self.reward_policy = (
            reward_policy
            if reward_policy is not None
            else getattr(debugger, "policy", None)
        )

    def run(self, state: ResearchState) -> dict:
        """Detect the task, execute the experiment and return results."""

        dataset_path = state.get("dataset_path")
        target_column = state.get("target_column")
        task_override = state.get("task_override")
        hypothesis = state.get("selected_hypothesis") or {}

        if not dataset_path or not Path(dataset_path).is_file():
            return self._failure("Dataset file is missing or invalid.")

        if not isinstance(hypothesis, dict):
            return self._failure(
                "Selected hypothesis must be a dictionary."
            )

        detection = detect_task(
            file_path=dataset_path,
            target_column=target_column,
            task_override=task_override,
        )

        if not isinstance(detection, dict):
            return self._failure(
                "Task detector returned an invalid result: "
                f"{type(detection).__name__}"
            )

        if not detection.get("success"):
            return self._failure(
                detection.get("error", "Task detection failed.")
            )

        task = detection.get("task")

        if task not in {"classification", "regression", "clustering"}:
            return self._failure(
                f"Unsupported or missing detected task: {task}"
            )

        if task != "clustering" and not target_column:
            return self._failure(
                "Target column is required for supervised learning."
            )

        try:
            if task == "classification":
                result = self._run_classification(
                    dataset_path,
                    target_column,
                    state,
                )

            elif task == "regression":
                result = train_regression_models(
                    file_path=dataset_path,
                    target_column=target_column,
                )

            else:
                result = train_clustering_models(
                    file_path=dataset_path
                )

        except Exception as exc:
            return self._failure(
                f"{task.title()} experiment raised "
                f"{type(exc).__name__}: {exc}",
                task=task,
            )

        if not isinstance(result, dict):
            return self._failure(
                f"{task.title()} experiment returned an invalid result: "
                f"{type(result).__name__}",
                task=task,
            )

        debug_fields = {
            key: result[key]
            for key in (
                "debug_attempts",
                "debug_reward_history",
                "debug_policy_stats",
                "execution_history",
            )
            if key in result
        }
        clean_result = {
                key: value
                for key, value in result.items()
                if key not in debug_fields
}
        if not result.get("success"):
            update = self._failure(
                f"{task.title()} experiment failed: "
                f"{result.get('issues', result.get('error', 'Unknown error'))}",
                task=task,
                experiment_results=clean_result,
            )
            update.update(debug_fields)
            return update

        return {
            "detected_task": task,
            "task_detection": detection,
            "experiment_results": clean_result,
            "dataset_metadata": {
                **(state.get("dataset_metadata") or {}),
                "selected_file": Path(dataset_path).name,
                "target_column": target_column,
                "task": task,
            },
            "verdict": "completed",
            **debug_fields,
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

    def _record_strategy_reward(
        self,
        strategy: str | None,
        reward: float,
        attempt: int,
        reason: str,
        reward_history: list,
    ) -> list:
        """Update the selected debugging strategy after its retry."""

        if not strategy or self.reward_policy is None:
            return reward_history

        self.reward_policy.update(strategy, reward)

        reward_history.append({
            "attempt": attempt,
            "strategy": strategy,
            "reward": reward,
            "reason": reason,
        })

        return reward_history

    def _run_classification(
        self,
        dataset_path: str,
        target_column: str,
        state: ResearchState,
    ) -> dict:
        """Run paired Random Forest experiments with automatic repair."""

        script_override = None
        pending_strategy = None

        reward_history = list(
            state.get("debug_reward_history") or []
        )

        execution_history = list(
            state.get("execution_history") or []
        )

        last_result = None

        for attempt in range(1, self.MAX_ATTEMPTS + 1):

            result = run_sandbox_csv_experiment(
                file_path=dataset_path,
                target_column=target_column,
                n_seeds=20,
                test_size=0.2,
                expected_direction="higher",
                n_tests=1,
                script_override=script_override,
            )

            last_result = result

            # Successful execution and statistical evaluation.
            if result.get("success"):

                self._record_strategy_reward(
                    pending_strategy,
                    1.0,
                    attempt,
                    "execution_success",
                    reward_history,
                )

                execution_history.append({
                    "attempt": attempt,
                    "status": "success",
                })

                return {
                    **result,
                    "debug_attempts": attempt,
                    "debug_reward_history": reward_history,
                    "debug_policy_stats": (
                        self.reward_policy.get_stats()
                        if self.reward_policy is not None
                        else []
                    ),
                    "execution_history": execution_history,
                }

            # Only Docker execution failures are repairable.
            # Validation, parsing and critic failures must not
            # be silently repaired or bypassed.
            if result.get("stage") != "sandbox":
                return {
                    **result,
                    "debug_attempts": attempt,
                    "debug_reward_history": reward_history,
                    "debug_policy_stats": (
                        self.reward_policy.get_stats()
                        if self.reward_policy is not None
                        else []
                    ),
                    "execution_history": execution_history,
                }

            error_message = str(
                result.get("debug_feedback")
                or result.get("issues", ["Unknown sandbox error"])[0]
            )

            execution_history.append({
                "attempt": attempt,
                "status": "failed",
                "error": error_message,
            })

            # The previous debugging strategy receives a negative
            # reward if its repaired script also failed.
            self._record_strategy_reward(
                pending_strategy,
                -1.0,
                attempt,
                "execution_failure",
                reward_history,
            )

            pending_strategy = None

            # No more retries.
            if attempt >= self.MAX_ATTEMPTS:
                break

            if self.debugger is None:
                break

            generated_code = result.get("generated_code")

            if not isinstance(generated_code, dict):
                break

            # Send the failed script and sandbox error to the debugger.
            debug_state = {
                **state,
                "generated_code": generated_code,
                "debug_feedback": error_message,
                "error": error_message,
            }

            repair = self.debugger(debug_state)

            if not isinstance(repair, dict):
                break

            if repair.get("failure_state"):
                return {
                    "success": False,
                    "stage": "debugger",
                    "issues": [
                        repair.get("error", "Code repair failed.")
                    ],
                    "debug_attempts": attempt,
                    "debug_reward_history": reward_history,
                    "debug_policy_stats": (
                        self.reward_policy.get_stats()
                        if self.reward_policy is not None
                        else []
                    ),
                    "execution_history": execution_history,
                }

            repaired_code = (
                repair.get("generated_code") or {}
            ).get("code")

            strategy = repair.get("debug_policy_strategy")

            if (
                not isinstance(repaired_code, str)
                or not repaired_code.strip()
                or not strategy
            ):
                return {
                    "success": False,
                    "stage": "debugger",
                    "issues": [
                        "Debugger did not return valid repaired code "
                        "and a debugging strategy."
                    ],
                    "debug_attempts": attempt,
                    "debug_reward_history": reward_history,
                    "execution_history": execution_history,
                }

            script_override = repaired_code
            pending_strategy = strategy

        return {
            "success": False,
            "stage": "sandbox",
            "issues": (
                last_result.get("issues", [])
                if isinstance(last_result, dict)
                else ["Maximum execution attempts reached."]
            ),
            "debug_attempts": self.MAX_ATTEMPTS,
            "debug_reward_history": reward_history,
            "debug_policy_stats": (
                self.reward_policy.get_stats()
                if self.reward_policy is not None
                else []
            ),
            "execution_history": execution_history,
        }
