import pytest

from src.agents.code_executor import CodeExecutor
from src.schemas.state import FailureState


def test_successful_code_execution(monkeypatch):
    def mock_execute(code):
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
            "experiment_name": "test_experiment",
            "code": "print('Hello from sandbox')",
        }
    }

    result = CodeExecutor().run(state)

    assert result["experiment_results"]["result_label"] == "executed"
    assert result["experiment_results"]["experiment_name"] == "test_experiment"


def test_missing_generated_code():
    result = CodeExecutor().run({})

    assert result["failure_state"] == FailureState.CODE_GENERATION_FAILED.value


def test_invalid_generated_code():
    state = {"generated_code": {"code": ""}}

    result = CodeExecutor().run(state)

    assert result["failure_state"] == FailureState.CODE_GENERATION_FAILED.value


def test_sandbox_rejection(monkeypatch):
    def mock_execute(code):
        raise ValueError("Code rejected")

    monkeypatch.setattr(
        "src.agents.code_executor.execute_python",
        mock_execute,
    )

    state = {"generated_code": {"code": "print('test')"}}

    result = CodeExecutor().run(state)

    assert result["failure_state"] == FailureState.SANDBOX_REJECTED.value


def test_execution_failure(monkeypatch):
    from src.sandbox.executor import SandboxExecutionError

    def mock_execute(code):
        raise SandboxExecutionError("Docker failed")

    monkeypatch.setattr(
        "src.agents.code_executor.execute_python",
        mock_execute,
    )

    state = {"generated_code": {"code": "print('test')"}}

    result = CodeExecutor().run(state)

    assert result["failure_state"] == FailureState.EXECUTION_FAILED.value
    assert result["experiment_results"]["result_label"] == "execution_failed"


def test_experiment_plan_execution(monkeypatch):
    from src.experiments.protocol import ExperimentPlan

    captured = {}

    def mock_render_script(plan):
        captured["plan"] = plan
        return "print('experiment executed')"

    def mock_execute(code):
        captured["code"] = code
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
        "src.agents.code_executor.render_script",
        mock_render_script,
    )
    monkeypatch.setattr(
        "src.agents.code_executor.execute_python",
        mock_execute,
    )

    plan = ExperimentPlan(
        research_question="Does increasing C improve classification accuracy?",
        hypothesis="Increasing C improves accuracy.",
        null_hypothesis="Changing C does not improve accuracy.",
        dataset="breast_cancer",
        model="logreg",
        metric="accuracy",
        independent_variable="C",
        baseline_value=0.1,
        treatment_value=1.0,
        expected_direction="higher",
        n_seeds=20,
        test_size=0.3,
        fixed_params={},
        falsification_criteria="Treatment accuracy does not improve.",
    )

    result = CodeExecutor().run({
        "experiment_plan": plan.model_dump()
    })

    assert captured["plan"].independent_variable == "C"
    assert captured["code"] == "print('experiment executed')"
    assert result["experiment_results"]["result_label"] == "executed"
    assert result["experiment_results"]["baseline"] == [0.82, 0.84, 0.81]
    assert result["experiment_results"]["treatment"] == [0.86, 0.87, 0.85]
