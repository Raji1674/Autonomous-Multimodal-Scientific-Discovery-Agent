
"""Literature Reviewer: retrieve and summarize only peer-reviewed evidence."""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.schemas.state import FailureState, ResearchState
from src.vectorstore.base import VectorStore


SYSTEM = (
    "You are a careful scientific literature reviewer. "
    "Use ONLY the supplied peer-reviewed sources. "
    "Cite every factual claim using [citation_key]. "
    "For uploaded PDFs, include page numbers in citations. "
    "Never use preprints, unpublished papers, or unverified sources "
    "as scientific evidence. "
    "Do not invent references, results, or conclusions. "
    "If the evidence is insufficient, explicitly say so."
)


class LiteratureReviewer(BaseAgent):
    name = "literature_reviewer"
    failure_on_error = FailureState.RETRIEVAL_FAILED

    def __init__(
        self,
        store: VectorStore,
        llm: LLMClient,
        top_k: int = 8,
    ):
        super().__init__()

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        self.store = store
        self.llm = llm
        self.top_k = top_k

    def run(self, state: ResearchState) -> dict:
        question = state["research_question"]

        # Retrieve extra candidates because some may be unverified.
        candidate_k = max(self.top_k * 4, self.top_k)
        hits = self.store.query(question, k=candidate_k)

        # Only explicitly classified peer-reviewed records are eligible.
        eligible_hits = [
            hit
            for hit in hits
            if str(
                hit.chunk.metadata.get(
                    "publication_status", "unknown"
                )
            ).strip().lower() == "peer_reviewed"
        ]

        filter_report = {
            "retrieved_candidates": len(hits),
            "eligible_evidence": len(eligible_hits),
            "excluded_unverified": len(hits) - len(eligible_hits),
            "policy": "peer_reviewed_only",
        }

        if not eligible_hits:
            return {
                "failure_state": FailureState.RETRIEVAL_FAILED.value,
                "error": (
                    "No explicitly peer-reviewed evidence was found. "
                    "Preprints, published-unverified records and "
                    "unknown publication statuses were excluded."
                ),
                "literature": [],
                "citations": {},
                "literature_summary": "",
                "literature_filter": filter_report,
            }

        # Limit the final evidence to the configured top_k.
        eligible_hits = eligible_hits[:self.top_k]

        citations: dict[str, dict] = {}
        blocks: list[str] = []

        for hit in eligible_hits:
            metadata = hit.chunk.metadata

            key = metadata.get(
                "citation_key",
                hit.chunk.paper_id,
            )
            paper_id = hit.chunk.paper_id

            if (
                key in citations
                and citations[key]["paper_id"] != paper_id
            ):
                key = f"{key}_{paper_id.replace(':', '')}"

            citations.setdefault(
                key,
                {
                    "paper_id": paper_id,
                    "title": metadata.get("title", ""),
                    "authors": metadata.get("authors", ""),
                    "year": metadata.get("year", ""),
                    "source": metadata.get("source", ""),
                    "publication_status": metadata.get(
                        "publication_status", "unknown"
                    ),
                    "pages": [],
                },
            )

            page = metadata.get("page_number")

            if (
                page is not None
                and page not in citations[key]["pages"]
            ):
                citations[key]["pages"].append(page)

            page_label = (
                f", page {page}" if page is not None else ""
            )

            blocks.append(
                f"[{key}{page_label}] "
                f"(peer_reviewed) "
                f"{metadata.get('title', '')}\n"
                f"{hit.chunk.text}"
            )

        summary = self.llm.complete(
            SYSTEM,
            (
                f"Research question: {question}\n\n"
                "Use only the peer-reviewed evidence supplied below. "
                "Preserve page references for uploaded PDFs. "
                "Cite every factual claim.\n\n"
                "Sources:\n\n"
                + "\n\n".join(blocks)
            ),
        )

        return {
            "literature": [
                hit.model_dump() for hit in eligible_hits
            ],
            "citations": citations,
            "literature_summary": summary,
            "literature_filter": filter_report,
        }
