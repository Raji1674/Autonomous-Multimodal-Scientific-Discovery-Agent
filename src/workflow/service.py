
"""Application service for scientific research workflows."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from src.experiments.dataset_validator import validate_csv_dataset
from src.agents.code_debugger import CodeDebugger
from src.agents.code_executor import CodeExecutor
from src.agents.csv_experiment_agent import CSVExperimentAgent
from src.agents.experiment_designer import ExperimentDesigner
from src.agents.hypothesis_generator import HypothesisGenerator
from src.agents.literature_reviewer import LiteratureReviewer
from src.agents.statistical_critic import StatisticalCriticAgent
from src.agents.synthesis import SynthesisAgent
from src.ingestion.pipeline import run_ingestion
from src.llm.providers import make_client
from src.schemas.state import ResearchState
from src.utils.config import load_config
from src.vectorstore.chroma_store import ChromaStore
from src.vectorstore.embeddings import SentenceTransformerEmbedder
from src.workflow.dataset_discovery import dataset_discovery_node
from src.workflow.graph import build_graph, new_config


class ResearchService:
    """Initialize and execute research workflows."""

    def __init__(
        self,
        config_dir: str = "configs",
        checkpoint_path: str = "data/checkpoints/research.sqlite",
        llm=None,
        store=None,
        enable_ingestion: bool = True,
    ):
        self.config = load_config(config_dir)

        self.llm = (
            llm if llm is not None
            else make_client(self.config.models)
        )

        self.store = (
            store if store is not None
            else ChromaStore(
                self.config.retrieval.chroma_path,
                SentenceTransformerEmbedder(
                    self.config.retrieval.embedding_model
                ),
            )
        )

        self.enable_ingestion = enable_ingestion
        self.checkpoint_path = checkpoint_path

        self.reviewer = LiteratureReviewer(
            self.store,
            self.llm,
            top_k=self.config.retrieval.top_k,
        )

        self.hypothesis = HypothesisGenerator(
            self.llm,
            max_retries=self.config.agents.max_hypothesis_retries,
        )

        self.designer = ExperimentDesigner(self.llm)
        self.executor = CodeExecutor()
        self.debugger = CodeDebugger(self.llm)
        self.critic = StatisticalCriticAgent(llm=self.llm)

        self.csv_experiment = CSVExperimentAgent(
    debugger=self.debugger,
    reward_policy=self.executor.policy,)
        self.synthesis = SynthesisAgent(self.llm)

    def _discover_or_use_dataset(self, state: ResearchState) -> dict:
        """Validate a supplied CSV or discover a public dataset."""

        dataset_path = state.get("dataset_path")
        target_column = state.get("target_column")

        if not dataset_path:
            return dataset_discovery_node(state)

        source_path = Path(dataset_path).expanduser().resolve()

        if (
            not source_path.is_file()
            or source_path.suffix.lower() != ".csv"
        ):
            return {
                "failure_state": "DATASET_DISCOVERY_FAILED",
                "error": "The supplied dataset must be an existing CSV file.",
            }

        validation = validate_csv_dataset(
            source_path,
            target_column=target_column,
        )

        if not validation["valid"]:
            return {
                "failure_state": "DATASET_DISCOVERY_FAILED",
                "error": "; ".join(validation["issues"]),
                "dataset_validation": validation,
            }

        allowed_root = Path("data/downloaded").resolve()
        allowed_root.mkdir(parents=True, exist_ok=True)

        if source_path.is_relative_to(allowed_root):
            staged_path = source_path
        else:
            staged_path = (
                allowed_root / f"{uuid.uuid4().hex}.csv"
            )
            shutil.copyfile(source_path, staged_path)

        return {
            "dataset_path": str(staged_path),
            "selected_dataset": {
                "name": source_path.name,
                "source": "user_supplied",
            },
            "dataset_validation": validation,
            "failure_state": None,
            "error": None,
        }

    def _build_app(self, experiment_mode: str):
        """Build the research workflow graph."""

        common = {
            "reviewer": self.reviewer,
            "hypothesis": self.hypothesis,
            "executor": self.executor,
            "critic": self.critic,
            "debugger": self.debugger,
            "synthesis": self.synthesis,
            "checkpoint_path": self.checkpoint_path,
        }

        if experiment_mode == "csv":
            return build_graph(
                **common,
                dataset_discovery=self._discover_or_use_dataset,
                csv_experiment=self.csv_experiment,
            )

        if experiment_mode == "protocol":
            return build_graph(
                **common,
                designer=self.designer,
            )

        raise ValueError(
            "experiment_mode must be 'csv' or 'protocol'"
        )

    def run(
        self,
        research_question: str,
        *,
        dataset_path: str | None = None,
        target_column: str | None = None,
        task_override: str | None = None,
        experiment_mode: str = "csv",
        ingest: bool | None = None,
        thread_id: str | None = None,
        resume: bool = False,
    ) -> dict:
        """Run literature retrieval and the selected experiment workflow."""

        if (
            not isinstance(research_question, str)
            or not research_question.strip()
        ):
            raise ValueError("research_question cannot be empty")

        should_ingest = (
            self.enable_ingestion if ingest is None else ingest
        )

        ingestion_report = None

        if should_ingest:
            ingestion_report = run_ingestion(
                research_question.strip(),
                self.store,
                max_per_source=50,
            )

        initial_state: ResearchState = {
            "research_question": research_question.strip(),
            "failure_state": None,
            "error": None,
        }

        if dataset_path is not None:
            initial_state["dataset_path"] = dataset_path

        if target_column is not None:
            initial_state["target_column"] = target_column

        if task_override is not None:
            initial_state["task_override"] = task_override

        app = self._build_app(experiment_mode)
        config = new_config(thread_id)

        if resume:
            snapshot = app.get_state(config)
            if not snapshot or not snapshot.values:
                raise ValueError(
                    f"No saved workflow state found for thread: {thread_id}"
                )
            final_state = app.invoke(None, config=config)
        else:
            final_state = app.invoke(initial_state, config=config)

        return {
            "run_id": config["configurable"]["thread_id"],
            "experiment_mode": experiment_mode,
            "ingestion_report": (
                {
                    "fetched": ingestion_report.fetched,
                    "after_dedupe": ingestion_report.after_dedupe,
                    "chunks_written": ingestion_report.chunks_written,
                    "failures": ingestion_report.failures,
                }
                if ingestion_report is not None
                else None
            ),
            "state": final_state,
            "workflow_thread_id": config["configurable"]["thread_id"],
        }
