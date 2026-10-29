from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    """Shared state passed between scientific research agents."""

    research_question: str

    # Literature Agent
    literature_findings: list[dict[str, Any]]
    literature_summary: str

    # Hypothesis Agent
    hypothesis: str
    experiment_plan: dict[str, Any]

    # Code / Experiment Agent
    experiment_results: dict[str, Any]

    # Critic Agent
    critique: str
    approved: bool
    revision_count: int

    # Synthesis Agent
    final_report: str

    # Workflow diagnostics
    errors: list[str] 