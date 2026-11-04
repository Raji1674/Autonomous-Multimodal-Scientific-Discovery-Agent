
"""LangGraph workflow: reviewer -> hypothesis -> designer -> executor -> critic."""
from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.schemas.state import ResearchState

Node = Callable[[ResearchState], dict]


def _route(next_node: str) -> Callable[[ResearchState], str]:
    def route(state: ResearchState) -> str:
        return "end" if state.get("failure_state") else next_node

    return route


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
):
    """Build the research workflow.

    The designer is optional to preserve compatibility with existing callers.
    """
    graph = StateGraph(ResearchState)

    graph.add_node("reviewer", reviewer)
    graph.add_node("hypothesis", hypothesis)
    graph.add_node("executor", executor)
    graph.add_node("critic", critic)

    if designer is not None:
        graph.add_node("designer", designer)

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

    if synthesis is not None:
        graph.add_conditional_edges(
            "critic",
            _route("synthesis"),
            {"synthesis": "synthesis", "end": END},
        )
        graph.add_edge("synthesis", END)
    else:
        graph.add_edge("critic", END)

    saver = None

    if checkpoint_path:
        connection = sqlite3.connect(
            checkpoint_path,
            check_same_thread=False,
        )
        saver = SqliteSaver(connection)

    return graph.compile(
        checkpointer=saver,
        interrupt_after=interrupt_after,
    )


def new_config():
    return {
        "configurable": {
            "thread_id": f"run-{uuid.uuid4().hex[:8]}"
        }
    }
