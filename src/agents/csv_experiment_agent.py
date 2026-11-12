
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
        self.reward_policy = (
            reward_policy
            if reward_policy is not None
            else getattr(debugger, "policy", None)
        )

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

        try:
            detection = detect_task(
                file_path=dataset_path,
                target_column=target_column,
                task_override=task_override,
                research_question=state.get("research_question"),
            )
        except Exception as exc:
            detection = {
                "success": False,
                "error": str(exc),
            }

        if not isinstance(detection, dict):
            detection = {
                "success": False,
                "error": "Task detector returned an invalid result.",
            }

        # An explicit, supported task override takes precedence when
        # automatic detection cannot classify a small or unusual dataset.
        if (
            not detection.get("success")
            and task_override in {
                "classification",
                "regression",
                "clustering",
            }
        ):
            detection = {
                "success": True,
                "task": task_override,
                "target_column": target_column,
                "reason": "Explicit task override",
            }

        if not detection.get("success"):
            return self._failure(
                detection.get("error", "Task detection failed.")
            )

        task = detection.get("task")
        target_column = detection.get("target_column", target_column)

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
                f"{task.title()} experiment returned an invalid result.",
                task=task,
            )

        debug_keys = (
            "debug_attempts",
            "debug_reward_history",
            "debug_policy_stats",
            "execution_history",
        )

        debug_fields = {
            key: result[key]
            for key in debug_keys
            if key in result
        }

        clean_result = {
            key: value
            for key, value in result.items()
            if key not in debug_keys
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

    @staticmethod
    def _get_model_configuration(state: dict):
        """Choose algorithms and parameters from the research wording."""

        hypothesis = state.get("selected_hypothesis") or {}

        if isinstance(hypothesis, dict):
            hypothesis_text = " ".join(
                str(hypothesis.get(key, ""))
                for key in (
                    "hypothesis",
                    "statement",
                    "experiment",
                    "success_metric",
                    "falsification_condition",
                )
            )
        else:
            hypothesis_text = str(hypothesis)

        comparison_text = (
            hypothesis_text + " " +
            str(state.get("research_question", ""))
        ).lower()

        # Logistic Regression regularization experiment.
        if any(term in comparison_text for term in (
            "logistic regression",
            "logisticregression",
            "logreg",
        )):
            return (
                "logistic_regression",
                "logistic_regression",
                {"C": 1.0, "max_iter": 2000},
                {"C": 0.1, "max_iter": 2000},
            )

        # Direct comparison between Random Forest and XGBoost.
        has_rf = (
            "random forest" in comparison_text
            or "randomforest" in comparison_text
        )
        has_xgb = (
            "xgboost" in comparison_text
            or "xgb" in comparison_text
        )

        if has_rf and has_xgb:
            return (
                "random_forest",
                "xgboost",
                {"n_estimators": 100},
                {"n_estimators": 200},
            )

        if has_xgb:
            return (
                "xgboost",
                "xgboost",
                {"n_estimators": 100},
                {"n_estimators": 200},
            )

        if has_rf:
            return (
                "random_forest",
                "random_forest",
                {"n_estimators": 100},
                {"n_estimators": 200},
            )

        # Preserve the project's default classification experiment
        # when no algorithm is specified.
        return (
            "random_forest",
            "random_forest",
            {"n_estimators": 100},
            {"n_estimators": 200},
        )

    def _run_classification(
        self,
        dataset_path: str,
        target_column: str,
        state: ResearchState,
    ) -> dict:
        """Run paired model experiments with automatic repair."""

        (
            baseline_algorithm,
            treatment_algorithm,
            baseline_params,
            treatment_params,
        ) = self._get_model_configuration(state)

        script_override = None
        pending_strategy = None

        reward_history = list(
            state.get("debug_reward_history") or []
        )
        execution_history = list(
            state.get("execution_history") or []
        )

        last_result = None
        attempts_made = 0

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            attempts_made = attempt

            result = run_sandbox_csv_experiment(
                file_path=dataset_path,
                target_column=target_column,
                n_seeds=20,
                test_size=0.2,
                expected_direction="higher",
                n_tests=1,
                script_override=script_override,
                baseline_algorithm=baseline_algorithm,
                treatment_algorithm=treatment_algorithm,
                baseline_params=baseline_params,
                treatment_params=treatment_params,
            )

            last_result = result

            if not isinstance(result, dict):
                return {
                    "success": False,
                    "stage": "result_validation",
                    "issues": [
                        "Sandbox runner returned an invalid result."
                    ],
                    "debug_attempts": attempt,
                    "debug_reward_history": reward_history,
                    "execution_history": execution_history,
                }

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

            # Only sandbox failures can be repaired by the debugger.
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

            issues = result.get("issues") or ["Unknown sandbox error"]
            error_message = str(
                result.get("debug_feedback") or issues[0]
            )

            execution_history.append({
                "attempt": attempt,
                "status": "failed",
                "error": error_message,
            })

            self._record_strategy_reward(
                pending_strategy,
                -1.0,
                attempt,
                "execution_failure",
                reward_history,
            )
            pending_strategy = None

            if attempt >= self.MAX_ATTEMPTS or self.debugger is None:
                break

            generated_code = result.get("generated_code")

            if not isinstance(generated_code, dict):
                break

            debug_state = {
                **state,
                "generated_code": generated_code,
                "debug_feedback": error_message,
                "error": error_message,
            }

            try:
                repair = self.debugger(debug_state)
            except Exception as exc:
                error_message = (
                    f"Debugger raised {type(exc).__name__}: {exc}"
                )
                break

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

            if (
                not isinstance(repaired_code, str)
                or not repaired_code.strip()
            ):
                break

            script_override = repaired_code
            pending_strategy = repair.get("debug_policy_strategy")

        final_issues = (
            last_result.get("issues", [])
            if isinstance(last_result, dict)
            else ["Maximum execution attempts reached."]
        )

        return {
            "success": False,
            "stage": "sandbox",
            "issues": final_issues,
            "debug_attempts": attempts_made,
            "debug_reward_history": reward_history,
            "debug_policy_stats": (
                self.reward_policy.get_stats()
                if self.reward_policy is not None
                else []
            ),
            "execution_history": execution_history,
        }
