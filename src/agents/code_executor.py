
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

        # Preferred path: generate code from a validated experiment plan.
        plan_data = state.get("experiment_plan")

        if plan_data is not None:
            try:
                plan = ExperimentPlan(**plan_data)
                code = render_script(plan)
                experiment_name = "validated_experiment"

            except (ValueError, TypeError) as exc:
                return {
                    "failure_state": FailureState.CODE_GENERATION_FAILED.value,
                    "error": f"Invalid experiment plan: {exc}",
                }

        else:
            # Backward-compatible path: execute supplied generated code.
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
