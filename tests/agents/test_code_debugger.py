
import json

from src.agents.code_debugger import CodeDebugger
from src.agents.code_executor import CodeExecutor
from src.schemas.state import FailureState
from src.sandbox.executor import SandboxExecutionError


class MockLLM:
    def complete(self, system, user, *, json_mode=False):
        return json.dumps({
            "corrected_code": "print('corrected experiment')"
        })


def test_debugger_corrects_failed_code():
    state = {
        "generated_code": {
            "code": "print(undefined_variable)",
            "experiment_name": "test_experiment",
        },
        "debug_feedback": "NameError: undefined_variable",
        "debug_attempts": 1,
        "failure_state": FailureState.EXECUTION_FAILED.value,
        "error": "NameError",
    }

    result = CodeDebugger(MockLLM()).run(state)

    assert result["generated_code"]["code"] == "print('corrected experiment')"
    assert result["failure_state"] is None
    assert result["error"] is None
    assert result["debug_feedback"] == ""


def test_debugger_and_executor_retry_integration(monkeypatch):
    execution_count = {"value": 0}

    def mock_execute(code):
        execution_count["value"] += 1

        if execution_count["value"] == 1:
            raise SandboxExecutionError("NameError: undefined_variable")

        return {
            "result_label": "executed",
            "exit_code": 0,
            "output": (
                '{"result_label":"executed",'
                '"baseline":[0.82,0.84,0.81],'
                '"treatment":[0.86,0.87,0.85]}'
            ),
            "output_truncated": False,
        }

    monkeypatch.setattr(
        "src.agents.code_executor.execute_python",
        mock_execute,
    )

    state = {
        "generated_code": {
            "code": "print(undefined_variable)",
            "experiment_name": "retry_test",
        },
        "debug_attempts": 0,
        "execution_history": [],
    }

    executor = CodeExecutor()
    debugger = CodeDebugger(MockLLM())

    # First execution fails.
    first_result = executor.run(state)
    state.update(first_result)

    assert state["failure_state"] == FailureState.EXECUTION_FAILED.value
    assert state["debug_attempts"] == 1

    # Debugger corrects the failed code.
    debug_result = debugger.run(state)
    state.update(debug_result)

    assert state["failure_state"] is None
    assert state["error"] is None

    # Second execution succeeds.
    second_result = executor.run(state)

    assert second_result["failure_state"] is None
    assert second_result["debug_attempts"] == 2
    assert second_result["experiment_results"]["result_label"] == "executed"
    assert execution_count["value"] == 2


def test_executor_stops_after_max_attempts(monkeypatch):
    def should_not_execute(code):
        raise AssertionError("Executor should not run again")

    monkeypatch.setattr(
        "src.agents.code_executor.execute_python",
        should_not_execute,
    )

    result = CodeExecutor().run({
        "generated_code": {"code": "print('test')"},
        "debug_attempts": 3,
    })

    assert result["failure_state"] == FailureState.EXECUTION_FAILED.value
    assert result["error"] == "Maximum execution attempts reached."
