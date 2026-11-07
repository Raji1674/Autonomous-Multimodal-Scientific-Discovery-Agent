
"""Production research service: ingestion -> agents -> research report."""
from __future__ import annotations

import re

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


def find_excluded_columns(question: str, column_names: list[str]) -> list[str]:
    """Find dataset columns explicitly excluded in the research question."""

    pattern = (
        r"\b(?:exclude|excluding|omit|omitting|remove|removing|drop|dropping)"
        r"\s+(.+?)"
        r"(?=\s+from\s+(?:the\s+)?(?:predictors|features|input\s+variables)"
        r"|\s+as\s+(?:predictors|features)|[.;\n]|$)"
    )

    matches = re.findall(pattern, question, flags=re.IGNORECASE)
    requested_text = " ".join(matches)

    excluded = []

    for column in column_names:
        column_pattern = rf"(?<!\w){re.escape(column)}(?!\w)"

        if re.search(column_pattern, requested_text, flags=re.IGNORECASE):
            excluded.append(column)

    return excluded


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
            max_revisions=self.cfg.agents.max_revisions,
            checkpoint_path=self.cfg.agents.checkpoint_path,
        )

    def run(
        self,
        question: str,
        dataset_path: str | None = None,
        target_column: str | None = None,
    ) -> dict:
        """Ingest literature and execute the production research workflow."""

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

        if bool(dataset_path) != bool(target_column):
            return {
                "ok": False,
                "error": "Dataset path and target column must be provided together.",
            }

        validation = None

        if dataset_path:
            from src.experiments.dataset_validator import validate_csv_dataset

            validation = validate_csv_dataset(
                dataset_path,
                target_column,
            )

            if not validation.get("valid"):
                return {
                    "ok": False,
                    "error": "Uploaded dataset validation failed.",
                    "dataset_issues": validation.get("issues", []),
                }

        # Fetch and index real literature from arXiv and PubMed.
        ingestion = run_ingestion(
            question,
            self.store,
            max_per_source=10,
            pdf_extractor=self._pdf_extractor(),
            max_pdf_papers=self.cfg.retrieval.max_pdf_papers,
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

        run_config = new_config()
        run_id = run_config["configurable"]["thread_id"]

        initial_state = {
            "run_id": run_id,
            "workflow_thread_id": run_id,
            "research_question": question,
        }

        if dataset_path and validation:
            column_names = validation.get("column_names", [])
            excluded_columns = find_excluded_columns(
                question,
                column_names,
            )

            initial_state.update({
                "dataset_path": dataset_path,
                "target_column": target_column,
                "dataset_task_type": validation.get("task_type"),
                "dataset_columns": column_names,
                "excluded_columns": excluded_columns,
            })

        try:
            result = self.graph.invoke(
                initial_state,
                config=run_config,
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
            "run_id": run_id,
            "error": result.get("error") if failed else None,
            "failure_state": result.get("failure_state"),
            "report": result.get("final_report", ""),
            "citations": result.get("citations", {}),
            "verdict": result.get("verdict"),
            "results": result.get("experiment_results"),
            "ingestion": ingestion_details,
        }

    def _pdf_extractor(self):
        if not self.cfg.retrieval.download_pdfs:
            return None

        from src.retrieval.pdf import PdfTextExtractor

        return PdfTextExtractor()
