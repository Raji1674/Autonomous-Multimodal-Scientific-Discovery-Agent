
"""LangGraph workflow with bounded, critic-authorized experiment revisions."""
from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.schemas.state import ResearchState

Node = Callable[[ResearchState], dict]


def _route(next_node: str) -> Callable[[ResearchState], str]:
    def route(state: ResearchState) -> str:
        return "end" if state.get("failure_state") else next_node

    return route


def _critic_route(
    max_revisions: int,
    synthesis_available: bool,
) -> Callable[[ResearchState], str]:
    """Revise only for an explicitly authorized methodological correction."""

    def route(state: ResearchState) -> str:
        feedback = state.get("critic_feedback")
        revision_count = state.get("revision_count", 0)

        authorized = (
            isinstance(feedback, dict)
            and feedback.get("revision_required") is True
            and isinstance(feedback.get("revision_reason"), str)
            and bool(feedback["revision_reason"].strip())
        )

        valid_count = (
            isinstance(revision_count, int)
            and not isinstance(revision_count, bool)
            and revision_count >= 0
        )

        if (
            authorized
            and valid_count
            and revision_count < max_revisions
        ):
            return "prepare_revision"

        return "synthesis" if synthesis_available else "end"

    return route


def _prepare_revision(state: ResearchState) -> dict:
    """Record the previous plan and critic feedback before redesign."""

    feedback = state.get("critic_feedback", {})
    previous_plan = state.get("experiment_plan")
    revision_count = state.get("revision_count", 0)

    history = list(state.get("revision_history", []))
    history.append(
        {
            "revision": revision_count + 1,
            "reason": feedback.get("revision_reason", ""),
            "critic_feedback": feedback,
            "previous_plan": previous_plan,
        }
    )

    return {
        "revision_count": revision_count + 1,
        "revision_history": history,
    }


def build_graph(
    reviewer: Node,
    hypothesis: Node,
    executor: Node,
    critic: Node,
    checkpoint_path: str | None = None,
    interrupt_after: list[str] | None = None,
    *,
    designer: Node | None = None,
    synthesis: Node | None = None,
    max_revisions: int = 2,
):
    """Build the research workflow with a bounded revision loop."""

    if (
        isinstance(max_revisions, bool)
        or not isinstance(max_revisions, int)
        or max_revisions < 0
    ):
        raise ValueError("max_revisions must be a non-negative integer")

    graph = StateGraph(ResearchState)

    graph.add_node("reviewer", reviewer)
    graph.add_node("hypothesis", hypothesis)
    graph.add_node("executor", executor)
    graph.add_node("critic", critic)

    if designer is not None:
        graph.add_node("designer", designer)
        graph.add_node("prepare_revision", _prepare_revision)

    if synthesis is not None:
        graph.add_node("synthesis", synthesis)

    graph.add_edge(START, "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        _route("hypothesis"),
        {"hypothesis": "hypothesis", "end": END},
    )

    next_after_hypothesis = "designer" if designer is not None else "executor"

    graph.add_conditional_edges(
        "hypothesis",
        _route(next_after_hypothesis),
        {
            next_after_hypothesis: next_after_hypothesis,
            "end": END,
        },
    )

    if designer is not None:
        graph.add_conditional_edges(
            "designer",
            _route("executor"),
            {"executor": "executor", "end": END},
        )

    graph.add_conditional_edges(
        "executor",
        _route("critic"),
        {"critic": "critic", "end": END},
    )

    if designer is not None:
        graph.add_conditional_edges(
            "critic",
            _critic_route(max_revisions, synthesis is not None),
            {
                "prepare_revision": "prepare_revision",
                "synthesis": "synthesis" if synthesis is not None else END,
                "end": END,
            },
        )
        graph.add_edge("prepare_revision", "designer")
    elif synthesis is not None:
        graph.add_conditional_edges(
            "critic",
            _route("synthesis"),
            {"synthesis": "synthesis", "end": END},
        )
    else:
        graph.add_edge("critic", END)

    if synthesis is not None:
        graph.add_edge("synthesis", END)

    saver = None

    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            checkpoint_path,
            check_same_thread=False,
        )
        saver = SqliteSaver(connection)

    return graph.compile(
        checkpointer=saver,
        interrupt_after=interrupt_after,
    )


def new_config(thread_id: str | None = None):
    return {
        "configurable": {
            "thread_id": thread_id or f"run-{uuid.uuid4().hex[:8]}"
        }
    }
