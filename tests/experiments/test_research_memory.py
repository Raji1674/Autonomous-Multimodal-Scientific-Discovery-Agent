
from src.experiments.research_memory import ResearchMemory


def test_save_and_retrieve_research(tmp_path):
    memory = ResearchMemory(str(tmp_path / "memory.sqlite"))

    state = {
        "research_question": "Predict cardiovascular disease using machine learning",
        "selected_hypothesis": {
            "hypothesis": "XGBoost can predict cardiovascular disease accurately"
        },
        "experiment_results": {"success": True, "accuracy": 0.91},
        "statistical_results": {"p_value": 0.02},
        "verdict": "SUPPORTED",
        "selected_dataset": {"name": "heart_disease.csv"},
        "target_column": "cardio",
    }

    record_id = memory.save(state, "thread-001")

    assert record_id is not None
    assert memory.count() == 1

    results = memory.retrieve(
        "cardiovascular disease prediction machine learning"
    )

    assert len(results) == 1
    assert "cardiovascular" in results[0]["research_question"]
    assert results[0]["verdict"] == "SUPPORTED"


def test_memory_persists_across_instances(tmp_path):
    db_path = str(tmp_path / "memory.sqlite")

    first = ResearchMemory(db_path)

    state = {
        "research_question": "Predict diabetes using machine learning",
        "selected_hypothesis": {
            "hypothesis": "Machine learning predicts diabetes"
        },
        "experiment_results": {"success": True},
    }

    first.save(state, "thread-002")

    second = ResearchMemory(db_path)

    assert second.count() == 1

    results = second.retrieve("diabetes prediction")
    assert len(results) == 1


def test_failed_research_is_not_saved(tmp_path):
    memory = ResearchMemory(str(tmp_path / "memory.sqlite"))

    state = {
        "research_question": "Test failed research",
        "failure_state": "EXECUTION_FAILED",
        "error": "Execution failed",
    }

    assert memory.save(state, "thread-003") is None
    assert memory.count() == 0


def test_same_thread_does_not_create_duplicate(tmp_path):
    memory = ResearchMemory(str(tmp_path / "memory.sqlite"))

    state = {
        "research_question": "Predict heart disease",
        "selected_hypothesis": {"hypothesis": "Test hypothesis"},
        "experiment_results": {"success": True},
    }

    first_id = memory.save(state, "thread-004")
    second_id = memory.save(state, "thread-004")

    assert first_id == second_id
    assert memory.count() == 1


def test_unrelated_question_returns_no_results(tmp_path):
    memory = ResearchMemory(str(tmp_path / "memory.sqlite"))

    state = {
        "research_question": "Predict cardiovascular disease",
        "selected_hypothesis": {
            "hypothesis": "Test cardiovascular hypothesis"
        },
        "experiment_results": {"success": True},
    }

    memory.save(state, "thread-005")

    results = memory.retrieve("satellite image segmentation")
    assert results == []


def test_retrieval_matches_previous_hypothesis(tmp_path):
    memory = ResearchMemory(str(tmp_path / "memory.sqlite"))

    state = {
        "research_question": "Predict cardiovascular disease",
        "selected_hypothesis": {
            "hypothesis": (
                "Reducing Logistic Regression regularization "
                "improves classification accuracy"
            )
        },
        "experiment_results": {"success": True},
    }

    memory.save(state, "thread-hypothesis-001")

    results = memory.retrieve(
        "Logistic Regression classification accuracy"
    )

    assert len(results) == 1
    assert "Logistic Regression" in results[0]["hypothesis"]
