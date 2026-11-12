"""Workflow state and failure enum.

State values are plain dicts/strings (not pydantic objects) so the LangGraph
SQLite checkpointer can serialize them reliably.
"""
from __future__ import annotations

from enum import Enum
from typing import TypedDict


class FailureState(str, Enum):
    LITERATURE_SEARCH_FAILED = "LITERATURE_SEARCH_FAILED"
    PAPER_PARSE_FAILED = "PAPER_PARSE_FAILED"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
    HYPOTHESIS_FAILED = "HYPOTHESIS_FAILED"
    CODE_GENERATION_FAILED = "CODE_GENERATION_FAILED"
    SANDBOX_REJECTED = "SANDBOX_REJECTED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    RESULT_VALIDATION_FAILED = "RESULT_VALIDATION_FAILED"
    STATISTICAL_ANALYSIS_FAILED = "STATISTICAL_ANALYSIS_FAILED"
    SYNTHESIS_FAILED = "SYNTHESIS_FAILED"
    DATASET_DISCOVERY_FAILED = "DATASET_DISCOVERY_FAILED"


class ResearchState(TypedDict, total=False):
    run_id: str
    research_question: str
    literature: list[dict]  # ScoredChunk.model_dump() items
    literature_summary: str
    citations: dict[str, dict]  # citation_key -> paper info
    hypotheses: list[dict]
    selected_hypothesis: dict
    generated_code: dict
    experiment_plan: dict
    experiment_results: dict
    statistical_results: dict
    critic_feedback: dict
    verdict: str
    failure_state: str | None
    error: str | None
    costs: dict
    target_column: str
    dataset_candidates: list[dict]
    selected_dataset: dict
    dataset_path: str
    dataset_validation: dict
    dataset_metadata: dict
    task_override: str
    detected_task: str
    task_detection: dict
    debug_attempts: int
    execution_history: list[dict]
    debug_reward_history: list[dict]
    debug_feedback: str
    debug_policy_strategy: str
    debug_policy_stats: list[dict]
    workflow_thread_id: str
    final_report: str
    human_paper_benchmark: dict
    revision_count: int
    revision_history: list[dict]
    latex_abstract: str
    latex_abstract_path: str
        # Long-term research memory
    memory_context: list[dict]
    memory_retrieval: dict
    memory_saved: bool
    memory_record_id: str