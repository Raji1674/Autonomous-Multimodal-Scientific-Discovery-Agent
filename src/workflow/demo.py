
"""
Live LLM demo:
    python -m src.workflow.demo

Uses the real Groq LLM and persistent literature database.
"""

from __future__ import annotations

import json
import tempfile

from dotenv import load_dotenv

from src.agents.code_debugger import CodeDebugger
from src.agents.code_executor import CodeExecutor
from src.agents.experiment_designer import ExperimentDesigner
from src.agents.hypothesis_generator import HypothesisGenerator
from src.agents.literature_reviewer import LiteratureReviewer
from src.agents.statistical_critic import StatisticalCriticAgent
from src.agents.synthesis import SynthesisAgent

from src.experiments.research_memory import ResearchMemory
from src.llm.providers import make_client
from src.utils.config import load_config

from src.vectorstore.chroma_store import ChromaStore
from src.vectorstore.embeddings import SentenceTransformerEmbedder

from src.workflow.graph import build_graph, new_config


def main() -> None:

    load_dotenv()

    print("\n========== INITIALIZING REAL LLM ==========")

    app_config = load_config()
    llm = make_client(app_config.models)

    print("Provider:", app_config.models.provider)
    print("Model:", app_config.models.model)

    response = llm.complete(
        "You are a helpful assistant.",
        "Reply with exactly: REAL LLM CONNECTION SUCCESS",
    )

    print("Connection test:", response)

    # --------------------------------------------------
    # 1. LONG-TERM RESEARCH MEMORY
    # --------------------------------------------------

    memory = ResearchMemory(
        "data/memory/research_memory.sqlite"
    )

    print("\n========== LONG-TERM MEMORY ==========")
    print("Existing research records:", memory.count())

    # --------------------------------------------------
    # 2. CONNECT TO REAL PERSISTENT LITERATURE DATABASE
    # --------------------------------------------------

    store = ChromaStore(
        app_config.retrieval.chroma_path,
        SentenceTransformerEmbedder(
            app_config.retrieval.embedding_model
        ),
    )

    print("\n========== REAL LITERATURE DATABASE ==========")
    print("Chroma database:", app_config.retrieval.chroma_path)
    print("Stored chunks:", store.count())

    research_question = (
        "Does stronger regularization improve "
        "Logistic Regression classification accuracy?"
    )

    retrieved = store.query(
        research_question,
        k=max(app_config.retrieval.top_k * 4, 20),
    )

    print(
        "Retrieved literature candidates:",
        len(retrieved),
    )

    for hit in retrieved:
        metadata = hit.chunk.metadata

        print(
            "Paper ID:", hit.chunk.paper_id,
            "| Source:", metadata.get("source", "unknown"),
            "| Status:", metadata.get(
                "publication_status", "unknown"
            ),
            "| Title:", metadata.get("title", "untitled"),
        )

    if store.count() == 0:
        raise RuntimeError(
            "The persistent literature database is empty. "
            "Run src.ingestion.pipeline before starting the demo."
        )

    # --------------------------------------------------
    # 3. INITIALIZE ALL AGENTS
    # --------------------------------------------------

    reviewer = LiteratureReviewer(
        store,
        llm,
        top_k=app_config.retrieval.top_k,
    )

    hypothesis_agent = HypothesisGenerator(llm)

    designer = ExperimentDesigner(llm)

    executor = CodeExecutor()

    debugger = CodeDebugger(llm)

    critic = StatisticalCriticAgent()

    synthesis_agent = SynthesisAgent(llm)

    # --------------------------------------------------
    # 4. BUILD LANGGRAPH WORKFLOW
    # --------------------------------------------------

    checkpoint_dir = tempfile.mkdtemp(
        prefix="scientific_demo_"
    )

    checkpoint_path = f"{checkpoint_dir}/checkpoints.sqlite"

    graph = build_graph(
        reviewer=reviewer,
        hypothesis=hypothesis_agent,
        executor=executor,
        debugger=debugger,
        critic=critic,
        designer=designer,
        synthesis=synthesis_agent,
        checkpoint_path=checkpoint_path,
    )

    # --------------------------------------------------
    # 5. INITIAL RESEARCH QUESTION
    # --------------------------------------------------

    initial_state = {
        "research_question": research_question,
    }

    config = new_config()

    # --------------------------------------------------
    # 6. RETRIEVE PREVIOUS RESEARCH MEMORY
    # --------------------------------------------------

    previous_research = memory.retrieve(
        research_question,
        limit=3,
        exclude_thread_id=config["configurable"]["thread_id"],
    )

    initial_state["memory_context"] = previous_research

    initial_state["memory_retrieval"] = {
        "records_found": len(previous_research),
        "retrieval_method": "keyword_overlap",
    }

    print("\n========== MEMORY RETRIEVAL ==========")

    print(
        "Records retrieved:",
        len(previous_research),
    )

    for item in previous_research:

        print(
            "\nPrevious question:",
            item["research_question"],
        )

        print(
            "Previous hypothesis:",
            item["hypothesis"],
        )

        print(
            "Previous verdict:",
            item["verdict"],
        )

    # --------------------------------------------------
    # 7. EXECUTE COMPLETE RESEARCH WORKFLOW
    # --------------------------------------------------

    print(
        "\n========== STARTING LIVE RESEARCH WORKFLOW =========="
    )

    for update in graph.stream(
        initial_state,
        config=config,
        stream_mode="updates",
    ):

        for node, delta in update.items():

            print(f"\n[{node}]")

            print(
                "Updated keys:",
                sorted(delta),
            )

            if "failure_state" in delta:

                print(
                    "Failure:",
                    delta.get("failure_state"),
                )

                print(
                    "Error:",
                    delta.get("error"),
                )

    # --------------------------------------------------
    # 8. GET FINAL WORKFLOW STATE
    # --------------------------------------------------

    final = graph.get_state(config).values

    # --------------------------------------------------
    # 9. SAVE RESEARCH TO MEMORY
    # --------------------------------------------------

    memory_record_id = memory.save(
        final,
        thread_id=config["configurable"]["thread_id"],
    )

    print("\n========== MEMORY SAVE ==========")

    if memory_record_id:

        print("Memory saved: True")

        print(
            "Memory record ID:",
            memory_record_id,
        )

    else:

        print("Memory saved: False")

        print(
            "Reason: Workflow did not meet "
            "memory success criteria."
        )

    # --------------------------------------------------
    # 10. FINAL WORKFLOW RESULTS
    # --------------------------------------------------

    print("\n========== FINAL WORKFLOW RESULT ==========")

    print(
        "Failure state:",
        final.get("failure_state"),
    )

    print(
        "Error:",
        final.get("error"),
    )

    if final.get("selected_hypothesis"):

        print(
            "\nSelected hypothesis:",
            final["selected_hypothesis"]["hypothesis"],
        )

    if final.get("experiment_plan"):

        print(
            "\nExperiment plan:",
            json.dumps(
                final["experiment_plan"],
                indent=2,
                default=str,
            ),
        )

    if final.get("experiment_results"):

        print(
            "\nExperiment results:",
            json.dumps(
                final["experiment_results"],
                indent=2,
                default=str,
            ),
        )

    if final.get("statistical_results"):

        print(
            "\nStatistical results:",
            json.dumps(
                final["statistical_results"],
                indent=2,
                default=str,
            ),
        )

    if final.get("critic_feedback"):

        print(
            "\nCritic feedback:",
            json.dumps(
                final["critic_feedback"],
                indent=2,
                default=str,
            ),
        )

    # --------------------------------------------------
    # 11. HUMAN-PAPER BENCHMARK
    # --------------------------------------------------

    print("\n========== HUMAN-PAPER BENCHMARK ==========")

    benchmark = final.get("human_paper_benchmark")

    if benchmark:

        print(
            json.dumps(
                benchmark,
                indent=2,
                default=str,
            )
        )

    else:

        print("Benchmark result is unavailable.")

    # --------------------------------------------------
    # 12. FINAL RESEARCH REPORT
    # --------------------------------------------------

    print("\n========== FINAL RESEARCH REPORT ==========")

    if final.get("final_report"):

        print(final["final_report"])

    else:

        print("Final report was not generated.")

    # --------------------------------------------------
    # 13. LATEX ABSTRACT
    # --------------------------------------------------

    print("\n========== LATEX ABSTRACT ==========")

    if final.get("latex_abstract"):

        print(final["latex_abstract"])

    else:

        print("LaTeX abstract was not generated.")

    print(
        "\nLaTeX abstract path:",
        final.get("latex_abstract_path"),
    )

    # --------------------------------------------------
    # 14. MEMORY SUMMARY
    # --------------------------------------------------

    print("\n========== MEMORY SUMMARY ==========")

    print(
        "Total stored research records:",
        memory.count(),
    )

    print(
        "Records retrieved:",
        final.get("memory_retrieval", {}).get(
            "records_found", 0
        ),
    )

    print(
        "Memory context received:",
        len(initial_state["memory_context"]),
    )

    # --------------------------------------------------
    # 15. FINAL STATUS
    # --------------------------------------------------

    print("\n========== EXECUTION SUMMARY ==========")

    if final.get("failure_state") or final.get("error"):

        print("Workflow completed with an error.")

        print(
            "Failure:",
            final.get("failure_state"),
        )

        print(
            "Error:",
            final.get("error"),
        )

    else:

        print("Workflow completed without reported errors.")

    print(
        "\nLLM Provider:",
        app_config.models.provider,
    )

    print(
        "LLM Model:",
        app_config.models.model,
    )

    print(
        "\nNOTE: The live workflow uses the persistent "
        "Chroma database and SentenceTransformer embeddings. "
        "ArXiv preprints and records with unknown publication "
        "status must not be presented as verified peer-reviewed papers."
    )


if __name__ == "__main__":
    main()
