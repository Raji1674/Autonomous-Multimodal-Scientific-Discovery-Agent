
import streamlit as st
import pandas as pd
import tempfile
import os
import json
import traceback

from pathlib import Path
from src.workflow.service import ResearchService

st.set_page_config(
    page_title="Autonomous Scientific Discovery Agent",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Autonomous Multimodal Scientific Discovery Agent")
st.caption(
    "AI-powered literature review, hypothesis generation "
    "and ML experimentation"
)

# =====================================================
# RESEARCH CONFIGURATION
# =====================================================

st.sidebar.header("Research Configuration")

question = st.sidebar.text_area(
    "Research Question",
    value="Does increasing the number of trees from 100 to 200 improve Random Forest classification accuracy on the Breast Cancer Wisconsin dataset?",
    height=130
)

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

# =====================================================
# CSV DATASET RESEARCH WORKFLOW
# =====================================================

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
        st.dataframe(df.head(10), use_container_width=True)

        target_column = st.selectbox(
            "Select Target Column",
            options=df.columns.tolist()
        )

        if st.button(
            "🚀 Start Autonomous Research",
            type="primary"
        ):

            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".csv",
                delete=False
            ) as temp:

                temp.write(uploaded_file.getvalue())
                dataset_path = temp.name

            try:
                with st.spinner(
                    "Running autonomous research workflow..."
                ):

                    service = ResearchService(
                        enable_ingestion=False
                    )

                    result = service.run(
                        question,
                        dataset_path=dataset_path,
                        target_column=target_column,
                        task_override="classification",
                        experiment_mode="csv"
                    )

                state = result.get("state", {})

                st.success("Research workflow completed!")

                st.subheader("Research Execution")

                st.write(
                    "Run ID:",
                    result.get("run_id", "Not available")
                )

                if result.get("ingestion_report"):
                    st.subheader("Literature Ingestion")
                    st.json(result["ingestion_report"])

                st.subheader("Research Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Workflow Failure",
                        str(state.get("failure_state") or "None")
                    )

                with col2:
                    st.metric(
                        "Workflow Error",
                        str(state.get("error") or "None")
                    )

                # Final Research Report
                st.subheader("Final Research Report")

                report = state.get("final_report")

                if report:
                    st.markdown(str(report))
                else:
                    st.info("No final report was returned.")

                # Experiment Results
                st.subheader("Experiment Results")

                experiment_results = state.get(
                    "experiment_results"
                )

                if experiment_results:
                    st.json(experiment_results)

                # Human Paper Benchmark
                st.subheader("Human-Paper Benchmark")

                benchmark = state.get(
                    "human_paper_benchmark", {}
                )

                if benchmark:

                    st.write(
                        "Status:",
                        benchmark.get("status")
                    )

                    if benchmark.get("status") == "contextual_comparison":

                        col1, col2 = st.columns(2)

                        col1.metric(
                            "Agent Accuracy",
                            f"{benchmark['agent_baseline_accuracy']:.2%}"
                        )

                        col2.metric(
                            "Published Accuracy",
                            f"{benchmark['published_accuracy']:.2%}"
                        )

                        st.write(
                            "Difference (percentage points):",
                            benchmark.get(
                                "difference_percentage_points"
                            )
                        )

                        st.json(
                            benchmark.get("reference_paper", {})
                        )

                        st.warning(
                            "Contextual comparison only; "
                            "not an independent reproduction."
                        )

                    else:
                        st.info(
                            benchmark.get(
                                "reason",
                                "Benchmark unavailable."
                            )
                        )

                else:
                    st.info(
                        "Human-paper benchmark is not available."
                    )

                # Statistical Critic
                st.subheader("Statistical Critic")

                critic = state.get("critic_feedback")

                if critic:
                    st.json(critic)

                elif (
                    isinstance(experiment_results, dict)
                    and experiment_results.get("critic")
                ):
                    st.json(experiment_results["critic"])

                else:
                    st.info(
                        "Critic details are included in the "
                        "experiment results or report, if generated."
                    )

                # Reinforcement Feedback Policy
                st.divider()

                st.subheader(
                    "🧠 Reinforcement Feedback Policy"
                )

                policy_stats = state.get(
                    "debug_policy_stats", []
                )

                reward_history = state.get(
                    "debug_reward_history", []
                )

                if policy_stats:

                    st.write(
                        "Debugging Strategy Performance"
                    )

                    policy_df = pd.DataFrame(policy_stats)

                    st.dataframe(
                        policy_df,
                        use_container_width=True
                    )

                    if "q_value" in policy_df.columns:

                        st.bar_chart(
                            policy_df.set_index(
                                "strategy"
                            )["q_value"]
                        )

                else:
                    st.info(
                        "No policy statistics available. "
                        "They will appear after a debugging attempt."
                    )

                if reward_history:

                    st.write("Execution Reward History")

                    st.dataframe(
                        pd.DataFrame(reward_history),
                        use_container_width=True
                    )

                st.metric(
                    "Debug Attempts",
                    state.get("debug_attempts", 0)
                )

                # Complete Workflow State
                st.subheader("Workflow State")

                with st.expander(
                    "View complete workflow output"
                ):

                    st.json({
                        key: str(value)[:10000]
                        for key, value in state.items()
                    })

            except Exception as exc:

                st.error(
                    f"Research workflow failed: {exc}"
                )

                st.code(traceback.format_exc())

            finally:

                if os.path.exists(dataset_path):
                    os.remove(dataset_path)

    except Exception as exc:

        st.error(f"Could not read CSV: {exc}")

# =====================================================
# SAMPLE DATASET WORKFLOW
# =====================================================

else:

    st.info(
        "Upload a CSV file to inspect the dataset "
        "and start an experiment."
    )

    sample_path = Path("data/sample.csv")

    if sample_path.exists():

        if st.button("Use Project Sample Dataset"):
            st.session_state["use_sample"] = True

    if st.session_state.get("use_sample"):

        df = pd.read_csv(sample_path)

        st.subheader("Sample Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        target_column = st.selectbox(
            "Select Target Column",
            options=df.columns.tolist(),
            index=(
                df.columns.tolist().index("target")
                if "target" in df.columns else 0
            )
        )

        if st.button(
            "🚀 Run Research on Sample Dataset",
            type="primary"
        ):

            try:
                with st.spinner(
                    "Running autonomous research workflow..."
                ):

                    service = ResearchService(
                        enable_ingestion=False
                    )

                    result = service.run(
                        question,
                        dataset_path=str(sample_path),
                        target_column=target_column,
                        task_override="classification",
                        experiment_mode="csv"
                    )

                state = result.get("state", {})

                st.success("Research workflow completed!")

                st.write(
                    "Run ID:",
                    result.get("run_id")
                )

                st.subheader("Final Research Report")

                st.markdown(
                    str(
                        state.get(
                            "final_report",
                            "No report returned."
                        )
                    )
                )

                st.subheader("Experiment Results")

                st.json(
                    state.get("experiment_results", {})
                )

                st.subheader("Human-Paper Benchmark")

                benchmark = state.get(
                    "human_paper_benchmark", {}
                )

                if benchmark:
                    st.json(benchmark)
                else:
                    st.info("Benchmark not available.")

                # Reinforcement Feedback Policy
                st.divider()

                st.subheader(
                    "🧠 Reinforcement Feedback Policy"
                )

                policy_stats = state.get(
                    "debug_policy_stats", []
                )

                reward_history = state.get(
                    "debug_reward_history", []
                )

                if policy_stats:

                    policy_df = pd.DataFrame(policy_stats)

                    st.dataframe(
                        policy_df,
                        use_container_width=True
                    )

                    if "q_value" in policy_df.columns:

                        st.bar_chart(
                            policy_df.set_index(
                                "strategy"
                            )["q_value"]
                        )

                else:
                    st.info(
                        "No debugging policy statistics available yet."
                    )

                if reward_history:

                    st.write("Reward History")

                    st.dataframe(
                        pd.DataFrame(reward_history),
                        use_container_width=True
                    )

                st.metric(
                    "Debug Attempts",
                    state.get("debug_attempts", 0)
                )

                st.subheader("Complete Workflow State")

                with st.expander("View details"):

                    st.json({
                        key: str(value)[:10000]
                        for key, value in state.items()
                    })

            except Exception as exc:

                st.error(
                    f"Research workflow failed: {exc}"
                )

                st.code(traceback.format_exc())


# =====================================================
# PDF RESEARCH PAPER PROCESSING
# =====================================================

st.divider()

st.header("📄 Research Paper PDF Analysis")

pdf_file = st.file_uploader(
    "Upload Research Paper (PDF)",
    type=["pdf"],
    key="research_pdf"
)

if pdf_file is not None:

    if st.button("Extract PDF Content"):

        from src.ingestion.pdf_processor import extract_pdf_content
        from src.ingestion.pdf_indexer import index_pdf_content
        from src.utils.config import load_config
        from src.vectorstore.chroma_store import ChromaStore
        from src.vectorstore.embeddings import SentenceTransformerEmbedder

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False
        ) as temp_pdf:

            temp_pdf.write(pdf_file.getvalue())
            temp_pdf_path = temp_pdf.name

        try:

            # Extract PDF
            with st.spinner(
                "Extracting PDF text and images..."
            ):

                pdf_result = extract_pdf_content(
                    temp_pdf_path
                )
                pdf_result["pdf_name"] = pdf_file.name

            config = load_config()

            # Initialize Chroma
            store = ChromaStore(
                config.retrieval.chroma_path,
                SentenceTransformerEmbedder(
                    config.retrieval.embedding_model
                )
            )

            # Index PDF Text
            index_report = index_pdf_content(
                pdf_result,
                store
            )

            st.subheader(
                "PDF Text Vector Database Indexing"
            )

            st.json(index_report)

            # =================================================
            # VLM IMAGE ANALYSIS
            # =================================================

            st.subheader("Vision Model Analysis")

            visual_findings = []

            visual_index_report = {
                "images_received": len(pdf_result["images"]),
                "images_analyzed": 0,
                "visual_chunks_indexed": 0,
            }

            if pdf_result["images"]:

                try:

                    from src.agents.pdf_vision_analyzer import PDFVisionAnalyzer
                    from src.schemas.paper import Chunk

                    analyzer = PDFVisionAnalyzer(
                        config.vision
                    )

                    with st.spinner(
                        "Analysing PDF figures using Groq Vision Model..."
                    ):

                        visual_findings = analyzer.analyze_pdf_images(
                            pdf_result["images"]
                        )

                    successful_findings = [
                        item
                        for item in visual_findings
                        if "error" not in item
                    ]

                    visual_index_report["images_analyzed"] = len(
                        successful_findings
                    )

                    visual_chunks = []

                    pdf_id = (
                        f"pdf:{Path(pdf_result['pdf_name']).stem}"
                    )

                    # Convert VLM findings into Chroma chunks
                    for idx, finding in enumerate(
                        successful_findings
                    ):

                        visual_text = "\n".join([
                            f"Figure type: {finding.get('figure_type', 'unknown')}",
                            f"Title: {finding.get('title', '')}",
                            f"Description: {finding.get('description', '')}",
                            f"Axes: {json.dumps(finding.get('axes', []))}",
                            f"Key observations: {json.dumps(finding.get('key_observations', []))}",
                            f"Numerical values: {json.dumps(finding.get('numerical_values', []))}",
                            f"Scientific interpretation: {finding.get('scientific_interpretation', '')}",
                            f"Limitations: {json.dumps(finding.get('limitations', []))}",
                        ])

                        page_number = int(
                            finding["page_number"]
                        )

                        visual_chunks.append(
                            Chunk(
                                chunk_id=(
                                    f"{pdf_id}#visual#page{page_number}#{idx}"
                                ),
                                paper_id=pdf_id,
                                text=visual_text,
                                metadata={
                                    "title": pdf_result["pdf_name"],
                                    "source": "uploaded_pdf_visual",
                                    "page_number": page_number,
                                    "chunk_index": idx,
                                    "citation_key": pdf_id,
                                    "figure_type": str(
                                        finding.get(
                                            "figure_type",
                                            "unknown"
                                        )
                                    ),
                                },
                            )
                        )

                    # Index visual findings
                    if visual_chunks:

                        written = store.upsert(
                            visual_chunks
                        )

                        visual_index_report[
                            "visual_chunks_indexed"
                        ] = written

                    st.json(visual_index_report)

                    # Display VLM analysis
                    for idx, finding in enumerate(
                        visual_findings
                    ):

                        with st.expander(
                            f"Figure {idx + 1} | Page "
                            f"{finding.get('page_number', 'N/A')}"
                        ):

                            if "error" in finding:

                                st.error(
                                    finding["error"]
                                )

                            else:

                                st.json(finding)

                except Exception as vision_exc:

                    st.warning(
                        "Vision analysis could not be completed. "
                        "PDF text indexing is still available."
                    )

                    st.error(str(vision_exc))

            else:

                st.info(
                    "No embedded images found. "
                    "Vision analysis was skipped."
                )

            # =================================================
            # RETRIEVE TEXT EVIDENCE
            # =================================================

            st.subheader("Retrieved Text Evidence")

            retrieved = store.query(
                question,
                k=5,
                where={
                    "source": "uploaded_pdf"
                }
            )

            if retrieved:

                for item in retrieved:

                    with st.expander(
                        f"Page {item.chunk.metadata.get('page_number', 'N/A')} "
                        f"| Similarity: {item.score:.4f}"
                    ):

                        st.write(
                            item.chunk.text
                        )

            else:

                st.info(
                    "No relevant text chunks retrieved."
                )

            # =================================================
            # RETRIEVE VISUAL EVIDENCE
            # =================================================

            st.subheader("Retrieved Visual Evidence")

            visual_retrieved = store.query(
                question,
                k=5,
                where={
                    "source": "uploaded_pdf_visual"
                }
            )

            if visual_retrieved:

                for item in visual_retrieved:

                    with st.expander(
                        f"Visual evidence | Page "
                        f"{item.chunk.metadata.get('page_number', 'N/A')} "
                        f"| Similarity: {item.score:.4f}"
                    ):

                        st.write(
                            item.chunk.text
                        )

            else:

                st.info(
                    "No relevant visual evidence retrieved."
                )

            # =================================================
            # PDF SUMMARY
            # =================================================

            st.success(
                "PDF extraction completed!"
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Total Pages",
                pdf_result["total_pages"]
            )

            col2.metric(
                "Total Words",
                pdf_result["total_words"]
            )

            # Display extracted page text
            st.subheader("Extracted Page Content")

            for page in pdf_result["pages"]:

                with st.expander(
                    f"Page {page['page_number']} - "
                    f"{page['word_count']} words"
                ):

                    st.text_area(
                        "Extracted Text",
                        page["text"],
                        height=250,
                        key=f"pdf_page_{page['page_number']}"
                    )

            # Display extracted images
            st.subheader("Extracted Images")

            if pdf_result["images"]:

                for image_info in pdf_result["images"]:

                    st.image(
                        image_info["image_path"],
                        caption=(
                            f"Page {image_info['page_number']}"
                        ),
                        width=400
                    )

            else:

                st.info(
                    "No embedded images found in this PDF."
                )

        except Exception as exc:

            st.error(
                f"PDF processing failed: {exc}"
            )

            st.code(
                traceback.format_exc()
            )

        finally:

            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Autonomous Multimodal Scientific Discovery Agent | Research prototype"
)
