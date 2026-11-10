
"""Offline demo: python -m src.workflow.demo

EVERYTHING HERE IS SCRIPTED. Papers are synthetic samples and the LLM is a FakeLLM,
so the output demonstrates plumbing only. It is NOT research and NOT a result.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from src.agents.code_debugger import CodeDebugger
from src.agents.csv_experiment_agent import CSVExperimentAgent
from src.agents.statistical_critic import StatisticalCriticAgent
from src.agents.code_executor import CodeExecutor
from src.agents.experiment_designer import ExperimentDesigner
from src.agents.hypothesis_generator import HypothesisGenerator
from src.agents.literature_reviewer import LiteratureReviewer
from src.llm.client import FakeLLM
from src.schemas.paper import Paper
from src.utils.citation import make_citation_key
from src.vectorstore.chroma_store import ChromaStore
from src.vectorstore.embeddings import FakeEmbedder
from src.vectorstore.indexing import index_papers
from src.workflow.graph import build_graph, new_config


SAMPLE_PAPERS = [
    Paper(
        paper_id="arxiv:demo.0001",
        source="arxiv",
        title="Sample study of regularization in small classifiers",
        authors=["Ada Example"],
        abstract=(
            "SYNTHETIC SAMPLE. Regularization and generalization "
            "in small classifiers are discussed."
        ),
        date="2025-01-10",
        publication_status="preprint",
    ),
    Paper(
        paper_id="pmid:demo0002",
        source="pubmed",
        title="Sample note on classification model performance",
        authors=["Bo Sample"],
        abstract=(
            "SYNTHETIC SAMPLE. Classification performance and "
            "model stability are discussed."
        ),
        date="2024-06",
        publication_status="peer_reviewed",
    ),
]


def main() -> None:
    tmp = Path(tempfile.mkdtemp())

    # 1. Index synthetic papers
    store = ChromaStore(tmp / "chroma", FakeEmbedder())
    print("index:", index_papers(SAMPLE_PAPERS, store))

    # 2. Generate citation keys
    keys, used = [], set()

    for paper in SAMPLE_PAPERS:
        key = make_citation_key(paper, used)
        used.add(key)
        keys.append(key)

    # 3. Scripted hypothesis
    hypothesis = {
        "hypothesis": (
            "Reducing Logistic Regression parameter C from 1.0 to 0.1 "
            "improves classification accuracy through stronger regularization."
        ),
        "motivation": "Regularization may improve generalization.",
        "supporting_evidence": [keys[0]],
        "contradicting_evidence": [],
        "novelty_reasoning": "Scripted demonstration hypothesis.",
        "experiment": (
            "Compare Logistic Regression with C=1.0 and C=0.1 "
            "using 20 paired random seeds."
        ),
        "success_metric": "accuracy",
        "falsification_condition": (
            "The treatment does not improve mean accuracy over the baseline."
        ),
    }

    # 4. Scripted responses:
    #    Response 1 -> Literature Reviewer
    #    Response 2 -> Hypothesis Generator
    #    Response 3 -> Experiment Designer
    experiment_plan_response = {
        "null_hypothesis": (
            "Changing Logistic Regression parameter C does not change accuracy."
        ),
        "dataset": "breast_cancer",
        "model": "logreg",
        "metric": "accuracy",
        "independent_variable": "C",
        "baseline_value": 1.0,
        "treatment_value": 0.1,
        "expected_direction": "higher",
        "fixed_params": {},
        "falsification_criteria": (
            "Treatment accuracy does not improve over baseline."
        ),
    }

    llm = FakeLLM([
        f"Scripted summary citing [{keys[0]}] (preprint, not peer reviewed).",
        json.dumps({"hypotheses": [hypothesis]}),
        json.dumps(experiment_plan_response),
    ])

    # 5. Build the complete five-stage workflow
    graph = build_graph(
        reviewer=LiteratureReviewer(store, llm, top_k=2),
        hypothesis=HypothesisGenerator(llm),
        executor=CodeExecutor(),
        debugger=CodeDebugger(llm),
        critic=StatisticalCriticAgent(),
        designer=ExperimentDesigner(llm),
        checkpoint_path=str(tmp / "ck.db"),
    )

    # 6. Execute workflow
    config = new_config()

    initial_state = {
        "research_question": (
            "Does stronger regularization improve Logistic Regression "
            "classification accuracy?"
        )
    }

    for update in graph.stream(
        initial_state,
        config=config,
        stream_mode="updates",
    ):
        for node, delta in update.items():
            print(f"\n[{node}]")
            print("Updated keys:", sorted(delta))

            if "failure_state" in delta:
                print("Failure:", delta.get("failure_state"))
                print("Error:", delta.get("error"))

    # 7. Read final workflow state
    final = graph.get_state(config).values

    print("\n========== FINAL WORKFLOW RESULT ==========")

    print("Failure state:", final.get("failure_state"))
    print("Error:", final.get("error"))

    if final.get("selected_hypothesis"):
        print(
            "Selected hypothesis:",
            final["selected_hypothesis"]["hypothesis"],
        )

    if final.get("experiment_plan"):
        print(
            "Experiment plan:",
            json.dumps(final["experiment_plan"], indent=2),
        )

    if final.get("experiment_results"):
        print(
            "Experiment result label:",
            final["experiment_results"].get("result_label"),
        )

    if final.get("statistical_results"):
        print(
            "Statistical results:",
            json.dumps(final["statistical_results"], indent=2, default=str),
        )

    if final.get("critic_feedback"):
        print(
            "Critic feedback:",
            json.dumps(final["critic_feedback"], indent=2, default=str),
        )

    print("\nNOTE: FakeLLM and synthetic papers are used.")
    print("This demo verifies workflow plumbing, not scientific findings.")


if __name__ == "__main__":
    main()
