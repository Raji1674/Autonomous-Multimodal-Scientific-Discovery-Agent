
"""Literature Reviewer: retrieve evidence, then summarize using ONLY retrieved text."""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.schemas.state import FailureState, ResearchState
from src.vectorstore.base import VectorStore

SYSTEM = (
    "You are a careful literature reviewer. Use ONLY the provided sources. "
    "Cite every claim as [citation_key]. Papers marked 'preprint' are NOT peer reviewed; say so. "
    "If the sources do not support something, say that instead of guessing."
)

PEER_REVIEWED_SYSTEM = (
    "You are a careful literature reviewer. Use ONLY the provided peer-reviewed sources. "
    "Cite every claim as [citation_key]. Do not introduce claims unsupported by the sources. "
    "If the sources do not support something, say that instead of guessing."
)


class LiteratureReviewer(BaseAgent):
    name = "literature_reviewer"
    failure_on_error = FailureState.RETRIEVAL_FAILED

    def __init__(
        self,
        store: VectorStore,
        llm: LLMClient,
        top_k: int = 8,
        peer_reviewed_only: bool = False,
    ):
        super().__init__()
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.peer_reviewed_only = peer_reviewed_only

    def run(self, state: ResearchState) -> dict:
        question = state["research_question"]

        if self.peer_reviewed_only:
            hits = self.store.query(
                question,
                k=self.top_k,
                where={"publication_status": "peer_reviewed"},
            )
            system_prompt = PEER_REVIEWED_SYSTEM
        else:
            hits = self.store.query(question, k=self.top_k)
            system_prompt = SYSTEM

        if not hits:
            return {
                "failure_state": FailureState.RETRIEVAL_FAILED.value,
                "error": "no evidence retrieved for the question",
            }

        citations: dict[str, dict] = {}
        blocks = []

        for h in hits:
            md = h.chunk.metadata
            key = md["citation_key"]

            if key in citations and citations[key]["paper_id"] != h.chunk.paper_id:
                key = f"{key}_{h.chunk.paper_id.replace(':', '')}"

            citations.setdefault(
                key,
                {
                    "paper_id": h.chunk.paper_id,
                    "title": md.get("title", ""),
                    "authors": md.get("authors", ""),
                    "year": md.get("year", ""),
                    "source": md.get("source", ""),
                    "publication_status": md.get("publication_status", "unknown"),
                },
            )

            blocks.append(
                f"[{key}] ({md.get('publication_status', 'unknown')}) "
                f"{md.get('title', '')}\n{h.chunk.text}"
            )

        summary = self.llm.complete(
            system_prompt,
            f"Research question: {question}\n\nSources:\n\n" + "\n\n".join(blocks),
        )

        return {
            "literature": [h.model_dump() for h in hits],
            "citations": citations,
            "literature_summary": summary,
        }
