
from src.workflow.graph import build_graph
from src.schemas.state import FailureState


def test_graph_retries_failed_execution_then_calls_critic():
    calls = {"executor": 0, "debugger": 0, "critic": 0}

    def reviewer(state):
        return {"literature": [{"paper_id": "test"}]}

    def hypothesis(state):
        return {"hypotheses": [{"hypothesis": "test"}]}

    def executor(state):
        calls["executor"] += 1

        if calls["executor"] == 1:
            return {
                "debug_attempts": 1,
                "failure_state": FailureState.EXECUTION_FAILED.value,
                "error": "NameError",
            }

        return {
            "debug_attempts": 2,
            "failure_state": None,
            "error": None,
            "experiment_results": {"result_label": "executed"},
        }

    def debugger(state):
        calls["debugger"] += 1
        return {
            "failure_state": None,
            "error": None,
            "debug_feedback": "",
        }

    def critic(state):
        calls["critic"] += 1
        return {"verdict": "accept"}

    app = build_graph(
        reviewer=reviewer,
        hypothesis=hypothesis,
        executor=executor,
        critic=critic,
        debugger=debugger,
    )

    result = app.invoke({
        "research_question": "Test research question"
    })

    assert calls["executor"] == 2
    assert calls["debugger"] == 1
    assert calls["critic"] == 1
    assert result["verdict"] == "accept"


def test_graph_stops_after_maximum_execution_attempts():
    calls = {"executor": 0, "debugger": 0, "critic": 0}

    def reviewer(state):
        return {}

    def hypothesis(state):
        return {}

    def executor(state):
        calls["executor"] += 1
        return {
            "debug_attempts": 3,
            "failure_state": FailureState.EXECUTION_FAILED.value,
            "error": "Execution failed",
        }

    def debugger(state):
        calls["debugger"] += 1
        return {}

    def critic(state):
        calls["critic"] += 1
        return {}

    app = build_graph(
        reviewer=reviewer,
        hypothesis=hypothesis,
        executor=executor,
        critic=critic,
        debugger=debugger,
    )

    result = app.invoke({
        "research_question": "Test research question"
    })

    assert calls["executor"] == 1
    assert calls["debugger"] == 0
    assert calls["critic"] == 0
    assert result["failure_state"] == FailureState.EXECUTION_FAILED.value

def test_graph_calls_synthesis_after_critic():
    calls = {"critic": 0, "synthesis": 0}

    def reviewer(state):
        return {}

    def hypothesis(state):
        return {}

    def executor(state):
        return {"experiment_results": {"result_label": "executed"}}

    def critic(state):
        calls["critic"] += 1
        return {"verdict": "accept"}

    def synthesis(state):
        calls["synthesis"] += 1
        return {"final_report": "Test research report"}

    app = build_graph(
        reviewer=reviewer,
        hypothesis=hypothesis,
        executor=executor,
        critic=critic,
        synthesis=synthesis,
    )

    result = app.invoke({"research_question": "Test question"})

    assert calls["critic"] == 1
    assert calls["synthesis"] == 1
    assert result["final_report"] == "Test research report"

def test_csv_success_calls_synthesis():
    calls = []

    def reviewer(state):
        return {}

    def hypothesis(state):
        return {}

    def discovery(state):
        return {}

    def csv_experiment(state):
        return {"experiment_results": {"success": True}}

    def synthesis(state):
        calls.append("synthesis")
        return {"final_report": "Research report"}

    app = build_graph(
        reviewer=reviewer,
        hypothesis=hypothesis,
        executor=lambda state: {},
        critic=lambda state: {},
        dataset_discovery=discovery,
        csv_experiment=csv_experiment,
        synthesis=synthesis,
    )

    app.invoke({})

    assert calls == ["synthesis"]


def test_csv_failure_skips_synthesis():
    calls = []

    def csv_experiment(state):
        return {
            "failure_state": FailureState.EXECUTION_FAILED.value,
            "error": "Experiment failed",
        }

    def synthesis(state):
        calls.append("synthesis")
        return {"final_report": "Research report"}

    app = build_graph(
        reviewer=lambda state: {},
        hypothesis=lambda state: {},
        executor=lambda state: {},
        critic=lambda state: {},
        dataset_discovery=lambda state: {},
        csv_experiment=csv_experiment,
        synthesis=synthesis,
    )

    app.invoke({})

    assert calls == []

def test_graph_skips_synthesis_when_critic_fails():
    calls = {"synthesis": 0}

    def synthesis(state):
        calls["synthesis"] += 1
        return {"final_report": "Should not be generated"}

    app = build_graph(
        reviewer=lambda state: {},
        hypothesis=lambda state: {},
        executor=lambda state: {
            "experiment_results": {"result_label": "executed"}
        },
        critic=lambda state: {
            "failure_state": FailureState.STATISTICAL_ANALYSIS_FAILED.value,
            "error": "Statistical analysis failed",
        },
        synthesis=synthesis,
    )

    result = app.invoke({"research_question": "Test question"})

    assert calls["synthesis"] == 0
    assert result["failure_state"] == (
        FailureState.STATISTICAL_ANALYSIS_FAILED.value
    )