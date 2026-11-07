from types import SimpleNamespace

import pytest

from src.workflow import service as service_module
from src.workflow.service import ResearchService


class FakeGraph:
    def __init__(self, result=None, error=None):
        self.result = result or {}
        self.error = error
        self.received_state = None
        self.received_config = None

    def invoke(self, state, config):
        self.received_state = state
        self.received_config = config

        if self.error:
            raise self.error

        return self.result


@pytest.fixture
def research_service():
    """Create a service without initializing production dependencies."""
    service = ResearchService.__new__(ResearchService)
    service.cfg = SimpleNamespace(
        retrieval=SimpleNamespace(
            download_pdfs=False,
            max_pdf_papers=0,
        )
    )
    service.store = object()
    service.graph = FakeGraph()
    return service

@pytest.fixture
def successful_ingestion(monkeypatch):
    """Prevent real arXiv and PubMed requests."""
    report = SimpleNamespace(
        fetched={"arxiv": 5, "pubmed": 3},
        after_dedupe=7,
        chunks_written=14,
        failures=[],
    )

    monkeypatch.setattr(
        service_module,
        "run_ingestion",
        lambda *args, **kwargs: report,
    )

    return report


def test_run_rejects_empty_question(research_service):
    result = research_service.run("   ")

    assert result["ok"] is False
    assert result["error"] == "Please enter a research question."


def test_run_rejects_question_over_500_characters(research_service):
    result = research_service.run("a" * 501)

    assert result["ok"] is False
    assert result["error"] == "Research questions must be 500 characters or fewer."


def test_run_rejects_dataset_without_target_column(research_service):
    result = research_service.run(
        "Predict student performance",
        dataset_path="students.csv",
    )

    assert result["ok"] is False
    assert "must be provided together" in result["error"]


def test_run_rejects_target_column_without_dataset(research_service):
    result = research_service.run(
        "Predict student performance",
        target_column="passed",
    )

    assert result["ok"] is False
    assert "must be provided together" in result["error"]


def test_run_rejects_invalid_dataset(research_service, tmp_path):
    result = research_service.run(
        "Predict student performance",
        dataset_path=str(tmp_path / "missing.csv"),
        target_column="passed",
    )

    assert result["ok"] is False
    assert result["error"] == "Uploaded dataset validation failed."
    assert result["dataset_issues"]


def test_run_stops_when_ingestion_writes_no_chunks(
    research_service,
    monkeypatch,
):
    report = SimpleNamespace(
        fetched={"arxiv": 0, "pubmed": 0},
        after_dedupe=0,
        chunks_written=0,
        failures=[],
    )

    monkeypatch.setattr(
        service_module,
        "run_ingestion",
        lambda *args, **kwargs: report,
    )

    result = research_service.run("Predict student performance")

    assert result["ok"] is False
    assert "No new literature" in result["error"]
    assert result["ingestion"]["chunks_written"] == 0
    assert research_service.graph.received_state is None


def test_run_returns_successful_workflow_result(
    research_service,
    successful_ingestion,
    monkeypatch,
):
    research_service.graph = FakeGraph(
        result={
            "final_report": "Research completed successfully.",
            "citations": {"paper_1": {"title": "Sample paper"}},
            "verdict": "APPROVED",
            "experiment_results": {"accuracy": 0.92},
        }
    )

    monkeypatch.setattr(
        service_module,
        "new_config",
        lambda: {"configurable": {"thread_id": "test-run"}},
    )

    result = research_service.run("Predict student performance")

    assert result["ok"] is True
    assert result["report"] == "Research completed successfully."
    assert result["verdict"] == "APPROVED"
    assert result["citations"]["paper_1"]["title"] == "Sample paper"
    assert result["results"]["accuracy"] == 0.92

    assert research_service.graph.received_state["research_question"] == (
        "Predict student performance"
    )
    assert "run_id" in research_service.graph.received_state
    assert research_service.graph.received_config == {
        "configurable": {"thread_id": "test-run"}
    }


def test_run_returns_failure_when_workflow_sets_failure_state(
    research_service,
    successful_ingestion,
):
    research_service.graph = FakeGraph(
        result={
            "failure_state": "HYPOTHESIS_FAILED",
            "error": "Could not generate a valid hypothesis.",
        }
    )

    result = research_service.run("Predict student performance")

    assert result["ok"] is False
    assert result["failure_state"] == "HYPOTHESIS_FAILED"
    assert result["error"] == "Could not generate a valid hypothesis."


def test_run_handles_unexpected_workflow_exception(
    research_service,
    successful_ingestion,
):
    research_service.graph = FakeGraph(
        error=RuntimeError("Internal test exception")
    )

    result = research_service.run("Predict student performance")

    assert result["ok"] is False
    assert result["error"] == (
        "The research workflow encountered an unexpected error."
    )
    assert "ingestion" in result