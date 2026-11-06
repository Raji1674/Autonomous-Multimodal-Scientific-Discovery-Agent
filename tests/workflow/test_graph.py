import pytest

from src.workflow.graph import build_graph, new_config


def test_graph_executes_agents_in_correct_order():
    execution_order = []

    def reviewer(state):
        execution_order.append("reviewer")
        return {"literature_summary": "Literature reviewed"}

    def hypothesis(state):
        execution_order.append("hypothesis")
        return {"selected_hypothesis": {"statement": "Test hypothesis"}}

    def designer(state):
        execution_order.append("designer")
        return {"experiment_plan": {"model": "RandomForest"}}

    def executor(state):
        execution_order.append("executor")
        return {"experiment_results": {"accuracy": 0.90}}

    def critic(state):
        execution_order.append("critic")
        return {"verdict": "APPROVED"}

    def synthesis(state):
        execution_order.append("synthesis")
        return {"final_report": "Research completed"}

    graph = build_graph(
        reviewer=reviewer,
        hypothesis=hypothesis,
        designer=designer,
        executor=executor,
        critic=critic,
        synthesis=synthesis,
    )

    result = graph.invoke({
        "research_question": "Predict student performance"
    })

    assert execution_order == [
        "reviewer",
        "hypothesis",
        "designer",
        "executor",
        "critic",
        "synthesis",
    ]
    assert result["final_report"] == "Research completed"


def test_graph_stops_when_reviewer_fails():
    execution_order = []

    def reviewer(state):
        execution_order.append("reviewer")
        return {
            "failure_state": "LITERATURE_SEARCH_FAILED",
            "error": "Literature search failed",
        }

    def hypothesis(state):
        execution_order.append("hypothesis")
        return {}

    def executor(state):
        execution_order.append("executor")
        return {}

    def critic(state):
        execution_order.append("critic")
        return {}

    graph = build_graph(
        reviewer=reviewer,
        hypothesis=hypothesis,
        executor=executor,
        critic=critic,
    )

    result = graph.invoke({
        "research_question": "Predict student performance"
    })

    assert execution_order == ["reviewer"]
    assert result["failure_state"] == "LITERATURE_SEARCH_FAILED"


def test_graph_supports_optional_designer_and_synthesis():
    execution_order = []

    def reviewer(state):
        execution_order.append("reviewer")
        return {}

    def hypothesis(state):
        execution_order.append("hypothesis")
        return {}

    def executor(state):
        execution_order.append("executor")
        return {}

    def critic(state):
        execution_order.append("critic")
        return {"verdict": "APPROVED"}

    graph = build_graph(
        reviewer=reviewer,
        hypothesis=hypothesis,
        executor=executor,
        critic=critic,
    )

    graph.invoke({
        "research_question": "Predict student performance"
    })

    assert execution_order == [
        "reviewer",
        "hypothesis",
        "executor",
        "critic",
    ]


def test_new_config_generates_unique_thread_ids():
    config_one = new_config()
    config_two = new_config()

    assert "configurable" in config_one
    assert "thread_id" in config_one["configurable"]

    assert "configurable" in config_two
    assert "thread_id" in config_two["configurable"]

    assert (
        config_one["configurable"]["thread_id"]
        != config_two["configurable"]["thread_id"]
    )
    
def test_graph_saves_checkpoint_to_sqlite(tmp_path):
    checkpoint_path = str(tmp_path / "workflow.db")

    def reviewer(state):
        return {"literature_summary": "Saved literature"}

    def hypothesis(state):
        return {"selected_hypothesis": {"statement": "Test hypothesis"}}

    def executor(state):
        return {"experiment_results": {"accuracy": 0.90}}

    def critic(state):
        return {"verdict": "APPROVED"}

    graph = build_graph(
        reviewer=reviewer,
        hypothesis=hypothesis,
        executor=executor,
        critic=critic,
        checkpoint_path=checkpoint_path,
    )

    config = new_config()

    graph.invoke(
        {"research_question": "Predict student performance"},
        config=config,
    )

    assert (tmp_path / "workflow.db").exists()

    checkpoint = graph.get_state(config)

    assert checkpoint.values["research_question"] == (
        "Predict student performance"
    )
    assert checkpoint.values["verdict"] == "APPROVED"

def test_graph_interrupts_after_reviewer(tmp_path):
    execution_order = []

    def reviewer(state):
        execution_order.append("reviewer")
        return {"literature_summary": "Literature reviewed"}

    def hypothesis(state):
        execution_order.append("hypothesis")
        return {}

    def executor(state):
        execution_order.append("executor")
        return {}

    def critic(state):
        execution_order.append("critic")
        return {}

    graph = build_graph(
        reviewer=reviewer,
        hypothesis=hypothesis,
        executor=executor,
        critic=critic,
        interrupt_after=["reviewer"],
        checkpoint_path=str(tmp_path / "interrupt.db"),
    )

    config = new_config()

    result = graph.invoke(
        {"research_question": "Predict student performance"},
        config=config,
    )

    assert execution_order == ["reviewer"]
    assert result["literature_summary"] == "Literature reviewed"

    checkpoint = graph.get_state(config)

    assert checkpoint.next == ("hypothesis",)

def test_graph_resumes_from_checkpoint_without_rerunning_reviewer(tmp_path):
    execution_order = []

    def reviewer(state):
        execution_order.append("reviewer")
        return {"literature_summary": "Literature reviewed"}

    def hypothesis(state):
        execution_order.append("hypothesis")
        return {"selected_hypothesis": {"statement": "Test hypothesis"}}

    def executor(state):
        execution_order.append("executor")
        return {"experiment_results": {"accuracy": 0.90}}

    def critic(state):
        execution_order.append("critic")
        return {"verdict": "APPROVED"}

    graph = build_graph(
        reviewer=reviewer,
        hypothesis=hypothesis,
        executor=executor,
        critic=critic,
        checkpoint_path=str(tmp_path / "resume.db"),
        interrupt_after=["reviewer"],
    )

    config = new_config()

    initial_result = graph.invoke(
        {"research_question": "Predict student performance"},
        config=config,
    )

    assert execution_order == ["reviewer"]

    resumed_graph = build_graph(
        reviewer=reviewer,
        hypothesis=hypothesis,
        executor=executor,
        critic=critic,
        checkpoint_path=str(tmp_path / "resume.db"),
    )

    final_result = resumed_graph.invoke(None, config=config)

    assert execution_order == [
        "reviewer",
        "hypothesis",
        "executor",
        "critic",
    ]
    assert final_result["verdict"] == "APPROVED"

def test_graph_stops_when_hypothesis_fails():
    execution_order = []

    def reviewer(state):
        execution_order.append("reviewer")
        return {}

    def hypothesis(state):
        execution_order.append("hypothesis")
        return {
            "failure_state": "HYPOTHESIS_FAILED",
            "error": "Hypothesis generation failed",
        }

    def executor(state):
        execution_order.append("executor")
        return {}

    def critic(state):
        execution_order.append("critic")
        return {}

    graph = build_graph(
        reviewer=reviewer,
        hypothesis=hypothesis,
        executor=executor,
        critic=critic,
    )

    result = graph.invoke({
        "research_question": "Predict student performance"
    })

    assert execution_order == ["reviewer", "hypothesis"]
    assert result["failure_state"] == "HYPOTHESIS_FAILED"