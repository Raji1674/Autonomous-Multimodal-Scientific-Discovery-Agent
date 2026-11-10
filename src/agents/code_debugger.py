
"""LLM-based repair agent with reward-based debugging strategy selection."""
from __future__ import annotations

import json

from src.agents.base import BaseAgent
from src.experiments.debug_policy import DebugPolicy
from src.llm.client import LLMClient, LLMError
from src.schemas.state import FailureState, ResearchState


SYSTEM = """
You are a Python debugging assistant for scientific machine-learning experiments.

Fix the supplied Python code based on the sandbox execution error.

Rules:
- Return ONLY a JSON object with one key: corrected_code.
- corrected_code must contain the complete corrected Python script.
- Preserve the original experiment's scientific objective and methodology.
- Do not remove validation, security checks, or statistical calculations.
- Do not add network access, filesystem access outside the supplied dataset,
  subprocess execution, shell commands, or credential access.
- Do not fabricate experiment results.
- Make the smallest necessary correction.
"""


STRATEGY_GUIDANCE = {
    "minimal_fix": "Make only the smallest change required to fix the error.",
    "type_error_fix": "Inspect variable types and correct type mismatches.",
    "dependency_fix": "Inspect imports and dependency usage. Do not install packages or access networks.",
    "data_validation_fix": "Inspect dataset columns, values, shapes and validation assumptions.",
}


class CodeDebugger(BaseAgent):
    name = "code_debugger"
    failure_on_error = FailureState.CODE_GENERATION_FAILED

    def __init__(self, llm: LLMClient):
        super().__init__()
        self.llm = llm
        self.policy = DebugPolicy()

    def run(self, state: ResearchState) -> dict:
        generated = state.get("generated_code") or {}
        code = generated.get("code")
        error = state.get("debug_feedback") or state.get("error", "")

        if not isinstance(code, str) or not code.strip():
            return {
                "failure_state": FailureState.CODE_GENERATION_FAILED.value,
                "error": "No generated Python code available for debugging.",
            }

        if not isinstance(error, str) or not error.strip():
            return {
                "failure_state": FailureState.CODE_GENERATION_FAILED.value,
                "error": "No execution error available for debugging.",
            }

        try:
            strategy = self.policy.select_strategy()
            guidance = STRATEGY_GUIDANCE[strategy]

            prompt = (
                "Fix this failed Python experiment.\n\n"
                f"Selected debugging strategy: {strategy}\n"
                f"Strategy guidance: {guidance}\n\n"
                f"Execution error:\n{error[:5000]}\n\n"
                f"Original Python code:\n{code[:20000]}\n\n"
                "Return the complete corrected script in JSON."
            )

            raw = self.llm.complete(SYSTEM, prompt, json_mode=True)
            data = json.loads(raw)

            corrected_code = data.get("corrected_code")

            if not isinstance(corrected_code, str) or not corrected_code.strip():
                raise ValueError("LLM did not return valid corrected_code.")

            if corrected_code == code:
                raise ValueError("Debugger returned unchanged code.")

            return {
                "generated_code": {
                    **generated,
                    "code": corrected_code,
                },
                "debug_feedback": "",
                "debug_policy_strategy": strategy,
                "debug_policy_stats": self.policy.get_stats(),
                "failure_state": None,
                "error": None,
            }

        except (json.JSONDecodeError, ValueError, LLMError) as exc:
            return {
                "failure_state": FailureState.CODE_GENERATION_FAILED.value,
                "error": f"Code debugging failed: {exc}",
            }
