
"""Code Executor: runs validated experiment plans or supplied code in Docker."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.experiments.protocol import ExperimentPlan, parse_result
from src.experiments.templates import render_script
from src.sandbox.executor import execute_python, SandboxExecutionError
from src.schemas.state import FailureState, ResearchState


class CodeExecutor(BaseAgent):
    name = "code_executor"
    failure_on_error = FailureState.EXECUTION_FAILED

    def run(self, state: ResearchState) -> dict:
        experiment_name = "unnamed_experiment"
        dataset_path = state.get("dataset_path")
        target_column = state.get("target_column")
        plan_data = state.get("experiment_plan")

        # Uploaded CSV experiments must always use a validated plan.
        # Never silently fall back to generated code or a built-in dataset.
        if dataset_path and plan_data is None:
            return {
                "failure_state": FailureState.CODE_GENERATION_FAILED.value,
                "error": (
                    "Uploaded CSV experiment requires a validated "
                    "experiment plan. Refusing to run a different dataset."
                ),
            }

        if bool(dataset_path) != bool(target_column):
            return {
                "failure_state": FailureState.CODE_GENERATION_FAILED.value,
                "error": "Dataset path and target column must be provided together.",
            }

        if plan_data is not None:
            try:
                plan = ExperimentPlan(**plan_data)

                if dataset_path:
                    if plan.dataset != "uploaded_csv":
                        raise ValueError(
                            "Uploaded CSV requires dataset='uploaded_csv'."
                        )

                    expected_task = (
                        state.get("dataset_task_type")
                        or state.get("task_type")
                    )

                    if expected_task and plan.task_type != expected_task:
                        raise ValueError(
                            "Experiment plan task type does not match "
                            "the uploaded dataset."
                        )

                    code = render_script(
                        plan,
                        dataset_path="/dataset.csv",
                        target_column=target_column,
                    )
                else:
                    code = render_script(plan)

                experiment_name = "validated_experiment"

            except (ValueError, TypeError) as exc:
                return {
                    "failure_state": FailureState.CODE_GENERATION_FAILED.value,
                    "error": f"Invalid experiment plan: {exc}",
                }

        else:
            # Backward-compatible path for non-CSV generated code.
            generated = state.get("generated_code")

            if not isinstance(generated, dict):
                return {
                    "failure_state": FailureState.CODE_GENERATION_FAILED.value,
                    "error": "Missing experiment_plan or generated_code.",
                }

            code = generated.get("code")
            experiment_name = generated.get(
                "experiment_name", experiment_name
            )

            if not isinstance(code, str) or not code.strip():
                return {
                    "failure_state": FailureState.CODE_GENERATION_FAILED.value,
                    "error": "Generated code must be a non-empty string.",
                }

        try:
            if dataset_path:
                result = execute_python(
                    code,
                    dataset_path=dataset_path,
                )
            else:
                result = execute_python(code)

            if result.get("result_label") != "executed":
                raise SandboxExecutionError(
                    "Sandbox did not complete execution successfully."
                )

            structured_result = parse_result(result.get("output", ""))

            return {
                "experiment_results": {
                    "experiment_name": experiment_name,
                    **structured_result,
                }
            }

        except ValueError as exc:
            return {
                "failure_state": FailureState.SANDBOX_REJECTED.value,
                "error": str(exc),
            }

        except SandboxExecutionError as exc:
            return {
                "failure_state": FailureState.EXECUTION_FAILED.value,
                "error": str(exc),
                "experiment_results": {
                    "result_label": "execution_failed"
                },
            }
