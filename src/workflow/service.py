
"""Production research service: ingestion -> agents -> research report."""
from __future__ import annotations

import uuid

from src.agents.literature_reviewer import LiteratureReviewer
from src.agents.hypothesis_generator import HypothesisGenerator
from src.agents.experiment_designer import ExperimentDesigner
from src.agents.code_executor import CodeExecutor
from src.agents.statistical_critic import (
    StatisticalCritic,
    StatisticalCriticAgent,
)
from src.agents.synthesis import SynthesisAgent

from src.ingestion.pipeline import run_ingestion
from src.llm.providers import make_client
from src.utils.config import load_config
from src.vectorstore.embeddings import SentenceTransformerEmbedder
from src.vectorstore.chroma_store import ChromaStore
from src.workflow.graph import build_graph, new_config


class ResearchService:
    """Initializes production components once and runs research questions."""

    def __init__(self):
        self.cfg = load_config()

        # Production LLM and persistent vector store.
        self.llm = make_client(self.cfg.models)
        embedder = SentenceTransformerEmbedder(
            self.cfg.retrieval.embedding_model
        )
        self.store = ChromaStore(
            self.cfg.retrieval.chroma_path,
            embedder,
        )

        # Production agents.
        reviewer = LiteratureReviewer(
            self.store,
            self.llm,
            top_k=self.cfg.retrieval.top_k,
        )
        hypothesis = HypothesisGenerator(
            self.llm,
            max_retries=self.cfg.agents.max_hypothesis_retries,
        )
        designer = ExperimentDesigner(
            self.llm,
            max_retries=self.cfg.agents.max_hypothesis_retries,
        )
        executor = CodeExecutor()
        critic = StatisticalCriticAgent(
            critic=StatisticalCritic(),
            llm=self.llm,
        )
        synthesis = SynthesisAgent(self.llm)

        self.graph = build_graph(
            reviewer=reviewer,
            hypothesis=hypothesis,
            designer=designer,
            executor=executor,
            critic=critic,
            synthesis=synthesis,
        )

    def run(self, question: str) -> dict:
        """Ingest real literature and execute the production workflow."""

        question = question.strip()

        if not question:
            return {
                "ok": False,
                "error": "Please enter a research question.",
            }

        if len(question) > 500:
            return {
                "ok": False,
                "error": "Research questions must be 500 characters or fewer.",
            }

        # Fetch and index real literature from arXiv and PubMed.
        ingestion = run_ingestion(
            question,
            self.store,
            max_per_source=10,
        )

        ingestion_details = {
            "fetched": ingestion.fetched,
            "after_dedupe": ingestion.after_dedupe,
            "chunks_written": ingestion.chunks_written,
            "failures": ingestion.failures,
        }

        if ingestion.chunks_written <= 0:
            return {
                "ok": False,
                "error": "No new literature could be indexed for this question.",
                "ingestion": ingestion_details,
            }

        initial_state = {
            "run_id": str(uuid.uuid4()),
            "research_question": question,
        }

        try:
            result = self.graph.invoke(
                initial_state,
                config=new_config(),
            )
        except Exception:
            return {
                "ok": False,
                "error": "The research workflow encountered an unexpected error.",
                "ingestion": ingestion_details,
            }

        failed = bool(result.get("failure_state"))

        return {
            "ok": not failed,
            "error": result.get("error") if failed else None,
            "failure_state": result.get("failure_state"),
            "report": result.get("final_report", ""),
            "citations": result.get("citations", {}),
            "verdict": result.get("verdict"),
            "results": result.get("experiment_results"),
            "ingestion": ingestion_details,
        }
