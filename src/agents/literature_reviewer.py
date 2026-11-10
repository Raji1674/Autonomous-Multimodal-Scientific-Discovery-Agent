"""Literature Reviewer: retrieve evidence, then summarize using ONLY retrieved text."""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.schemas.state import FailureState, ResearchState
from src.vectorstore.base import VectorStore

SYSTEM = (
    "You are a careful literature reviewer. Use ONLY the provided sources. "
    "Cite every claim as [citation_key]. "
    "For uploaded PDFs, include the page number in citations, e.g. [pdf:paper, p. 3]. "
    "Papers marked 'preprint' are NOT peer reviewed; say so. "
    "If the sources do not support something, say that instead of guessing."
)


class LiteratureReviewer(BaseAgent):
    name = "literature_reviewer"
    failure_on_error = FailureState.RETRIEVAL_FAILED

    def __init__(self, store: VectorStore, llm: LLMClient, top_k: int = 8):
        super().__init__()
        self.store, self.llm, self.top_k = store, llm, top_k

    def run(self, state: ResearchState) -> dict:
        question = state["research_question"]
        hits = self.store.query(question, k=self.top_k)

        if not hits:
            return {
                "failure_state": FailureState.RETRIEVAL_FAILED.value,
                "error": "no evidence retrieved for the question"
            }

        citations: dict[str, dict] = {}
        blocks = []

        for h in hits:
            md = h.chunk.metadata
            key = md.get("citation_key", h.chunk.paper_id)
            paper_id = h.chunk.paper_id

            if key in citations and citations[key]["paper_id"] != paper_id:
                key = f"{key}_{paper_id.replace(':', '')}"

            citations.setdefault(key, {
                "paper_id": paper_id,
                "title": md.get("title", ""),
                "authors": md.get("authors", ""),
                "year": md.get("year", ""),
                "source": md.get("source", ""),
                "publication_status": md.get("publication_status", "unknown"),
                "pages": []
            })

            page = md.get("page_number")
            if page is not None and page not in citations[key]["pages"]:
                citations[key]["pages"].append(page)

            page_label = f", page {page}" if page is not None else ""

            blocks.append(
                f"[{key}{page_label}] "
                f"({md.get('publication_status', 'unknown')}) "
                f"{md.get('title', '')}\n{h.chunk.text}"
            )

        summary = self.llm.complete(
            SYSTEM,
            f"Research question: {question}\n\n"
            "Use the supplied evidence only. Preserve page references for uploaded PDFs.\n\n"
            "Sources:\n\n" + "\n\n".join(blocks)
        )

        return {
            "literature": [h.model_dump() for h in hits],
            "citations": citations,
            "literature_summary": summary
        }
