"""Execute validated experiment plans with reward-based feedback."""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.experiments.debug_policy import DebugPolicy
from src.experiments.protocol import ExperimentPlan, ResultError, parse_result
from src.experiments.templates import render_script
from src.sandbox.executor import execute_python, SandboxExecutionError
from src.schemas.state import FailureState, ResearchState


class CodeExecutor(BaseAgent):
    name = "code_executor"
    failure_on_error = FailureState.EXECUTION_FAILED
    MAX_ATTEMPTS = 3

    def __init__(self):
        super().__init__()
        self.policy = DebugPolicy()

    def _record_reward(self, state, attempt, reward, reason):
        strategy = state.get("debug_policy_strategy")
        history = state.get("debug_reward_history", [])

        if strategy:
            self.policy.update(strategy, reward)

            history = history + [{
                "attempt": attempt,
                "strategy": strategy,
                "reward": reward,
                "reason": reason,
            }]

        return history

    def run(self, state: ResearchState) -> dict:
        attempts = state.get("debug_attempts", 0)

        if attempts >= self.MAX_ATTEMPTS:
            return {
                "failure_state": FailureState.EXECUTION_FAILED.value,
                "error": "Maximum execution attempts reached.",
            }

        generated = state.get("generated_code")
        plan_data = state.get("experiment_plan")

        if isinstance(generated, dict) and generated.get("code"):
            code = generated["code"]
            experiment_name = generated.get(
                "experiment_name", "validated_experiment"
            )
        elif plan_data is not None:
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
            return {
                "failure_state": FailureState.CODE_GENERATION_FAILED.value,
                "error": "Missing experiment_plan or generated_code.",
            }

        attempt = attempts + 1

        try:
            result = execute_python(code)

            if result.get("result_label") != "executed":
                raise SandboxExecutionError(
                    "Sandbox did not complete execution successfully."
                )

            structured = parse_result(result.get("output", ""))

            reward_history = self._record_reward(
                state, attempt, 1.0, "execution_success"
            )

            return {
                "debug_attempts": attempt,
                "debug_feedback": "",
                "debug_reward_history": reward_history,
                "debug_policy_stats": self.policy.get_stats(),
                "execution_history": state.get("execution_history", []) + [{
                    "attempt": attempt,
                    "status": "success",
                }],
                "experiment_results": {
                    "experiment_name": experiment_name,
                    **structured,
                },
                "failure_state": None,
                "error": None,
            }

        except ResultError as exc:
            return {
                "debug_attempts": attempt,
                "failure_state": FailureState.RESULT_VALIDATION_FAILED.value,
                "error": str(exc),
                "experiment_results": {"result_label": "invalid_result"},
            }

        except ValueError as exc:
            return {
                "debug_attempts": attempt,
                "failure_state": FailureState.SANDBOX_REJECTED.value,
                "error": str(exc),
            }

        except SandboxExecutionError as exc:
            reward_history = self._record_reward(
                state, attempt, -1.0, "execution_failure"
            )

            history = state.get("execution_history", []) + [{
                "attempt": attempt,
                "status": "failed",
                "error": str(exc),
            }]

            return {
                "debug_attempts": attempt,
                "generated_code": {
                    "code": code,
                    "experiment_name": experiment_name,
                },
                "debug_feedback": str(exc),
                "debug_reward_history": reward_history,
                "debug_policy_stats": self.policy.get_stats(),
                "execution_history": history,
                "failure_state": FailureState.EXECUTION_FAILED.value,
                "error": str(exc),
                "experiment_results": {
                    "result_label": "execution_failed"
                },
            }