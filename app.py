import streamlit as st
import pandas as pd
import tempfile
import os
import json
import traceback

from pathlib import Path
from src.workflow.service import ResearchService


# =====================================================
# PAGE CONFIGURATION
# =====================================================

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
# RESEARCH QUESTION AND OPTIONAL DATASET
# =====================================================

st.header("Research Configuration")

question = st.text_area(
    "Enter your Research Question",
    placeholder=(
        "Example: Can machine learning predict cardiovascular disease "
        "using clinical risk factors?"
    ),
    height=130,
    key="research_question_input"
)

uploaded_file = st.file_uploader(
    "Upload CSV Dataset (Optional — the agent can search for one)",
    type=["csv"],
    key="research_csv"
)

target_column = None
csv_is_valid = True


# =====================================================
# OPTIONAL CSV PREVIEW
# =====================================================

if uploaded_file is not None:

    try:
        preview_df = pd.read_csv(uploaded_file)

        if preview_df.empty or len(preview_df.columns) < 2:

            st.error(
                "The uploaded CSV must contain rows and at least two columns."
            )

            csv_is_valid = False

        else:

            st.subheader("Uploaded Dataset Preview")

            st.write(
                f"Rows: {preview_df.shape[0]} | "
                f"Columns: {preview_df.shape[1]}"
            )

            st.dataframe(
                preview_df.head(),
                use_container_width=True
            )

            columns = preview_df.columns.tolist()

            default_index = (
                columns.index("target")
                if "target" in columns
                else 0
            )

            target_column = st.selectbox(
                "Select Target Column",
                options=columns,
                index=default_index,
                key="uploaded_csv_target"
            )

            target_unique_count = preview_df[
                target_column
            ].nunique(dropna=True)

            st.write("Selected target:", target_column)

            st.write(
                "Unique target values:",
                target_unique_count
            )

            if target_unique_count < 2:

                st.error(
                    "The selected target must contain at least two unique values."
                )

                csv_is_valid = False

    except Exception as exc:

        st.error(f"Could not read CSV: {exc}")

        csv_is_valid = False


# =====================================================
# START AUTONOMOUS RESEARCH
# =====================================================

if st.button(
    "🚀 Start Autonomous Research",
    type="primary",
    disabled=(
        not question.strip()
        or not csv_is_valid
    )
):

    dataset_path = None

    try:

        # Save optional uploaded CSV temporarily
        if uploaded_file is not None:

            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".csv",
                delete=False
            ) as temp:

                temp.write(
                    uploaded_file.getvalue()
                )

                dataset_path = temp.name

        # Execute workflow
        with st.spinner(
            "Running research: literature review, hypothesis generation, "
            "dataset discovery/validation and ML experiments..."
        ):

            service = ResearchService(
                enable_ingestion=True
            )

            result = service.run(
                question.strip(),
                dataset_path=dataset_path,
                target_column=target_column,
                experiment_mode="csv"
            )

        state = result.get("state", {}) or {}

        st.success(
            "Research workflow execution finished."
        )

        # =================================================
        # RESEARCH EXECUTION
        # =================================================

        st.subheader("Research Execution")

        st.write(
            "Run ID:",
            result.get("run_id", "Not available")
        )

        ingestion_report = result.get("ingestion_report")

        if ingestion_report:

            st.subheader("Literature Ingestion")

            st.json(ingestion_report)

        # =================================================
        # WORKFLOW STATUS
        # =================================================

        st.subheader("Workflow Status")

        col1, col2 = st.columns(2)

        col1.metric(
            "Workflow Failure",
            str(state.get("failure_state") or "None")
        )

        col2.metric(
            "Workflow Error",
            str(state.get("error") or "None")
        )

        # =================================================
        # LITERATURE REVIEW
        # =================================================

        st.subheader("Literature Review")

        if state.get("literature_review"):

            st.write(
                state["literature_review"]
            )

        elif state.get("literature_evidence"):

            st.json(
                state["literature_evidence"]
            )

        else:

            st.info(
                "Literature details are included in the workflow "
                "state/report when available."
            )

        # =================================================
        # HYPOTHESIS GENERATION
        # =================================================

        st.subheader("Generated Hypotheses")

        hypotheses = (
            state.get("hypotheses")
            or state.get("generated_hypotheses")
        )

        if hypotheses:

            st.write(hypotheses)

        elif state.get("selected_hypothesis"):

            st.json(
                state["selected_hypothesis"]
            )

        else:

            st.info(
                "No hypothesis details were returned in a separate field."
            )

        # =================================================
        # DATASET DISCOVERY AND VALIDATION
        # =================================================

        st.subheader(
            "Dataset Discovery and Validation"
        )

        if state.get("selected_dataset"):

            st.json(
                state["selected_dataset"]
            )

        if state.get("dataset_metadata"):

            st.json(
                state["dataset_metadata"]
            )

        if state.get("dataset_validation"):

            st.json(
                state["dataset_validation"]
            )

        if not any(
            state.get(key)
            for key in (
                "selected_dataset",
                "dataset_metadata",
                "dataset_validation"
            )
        ):

            st.info(
                "Dataset information is not available. "
                "Check the workflow error above."
            )

        # =================================================
        # FINAL RESEARCH REPORT
        # =================================================

        st.subheader("Final Research Report")

        report = state.get("final_report")

        if report:

            st.markdown(
                str(report)
            )

        else:

            st.info(
                "No final report was returned."
            )

        # =================================================
        # EXPERIMENT RESULTS
        # =================================================

        st.subheader("Experiment Results")

        experiment_results = state.get(
            "experiment_results"
        )

        if experiment_results:

            st.json(
                experiment_results
            )

        else:

            st.info(
                "No experiment results were returned."
            )

        # =================================================
        # HUMAN PAPER BENCHMARK
        # =================================================

        st.subheader("Human-Paper Benchmark")

        benchmark = (
            state.get("human_paper_benchmark", {})
            or {}
        )

        if benchmark:

            st.json(benchmark)

        else:

            st.info(
                "Human-paper benchmark is not available."
            )

        # =================================================
        # STATISTICAL CRITIC
        # =================================================

        st.subheader("Statistical Critic")

        critic = state.get("critic_feedback")

        if critic:

            st.json(critic)

        elif (
            isinstance(experiment_results, dict)
            and experiment_results.get("critic")
        ):

            st.json(
                experiment_results["critic"]
            )

        else:

            st.info(
                "No separate critic details were returned."
            )

        # =================================================
        # REINFORCEMENT FEEDBACK
        # =================================================

        st.divider()

        st.subheader(
            "🧠 Reinforcement Feedback Policy"
        )

        policy_stats = (
            state.get("debug_policy_stats", [])
            or []
        )

        reward_history = (
            state.get("debug_reward_history", [])
            or []
        )

        if policy_stats:

            policy_df = pd.DataFrame(
                policy_stats
            )

            st.dataframe(
                policy_df,
                use_container_width=True
            )

            if (
                "q_value" in policy_df.columns
                and "strategy" in policy_df.columns
            ):

                st.bar_chart(
                    policy_df.set_index("strategy")["q_value"]
                )

        else:

            st.info(
                "No policy statistics available."
            )

        if reward_history:

            st.subheader("Execution Reward History")

            st.dataframe(
                pd.DataFrame(reward_history),
                use_container_width=True
            )

        st.metric(
            "Debug Attempts",
            state.get("debug_attempts", 0)
        )

        # =================================================
        # COMPLETE WORKFLOW STATE
        # =================================================

        st.subheader("Complete Workflow State")

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

        st.code(
            traceback.format_exc()
        )

    finally:

        if (
            dataset_path is not None
            and os.path.exists(dataset_path)
        ):

            os.remove(dataset_path)


elif not question.strip():

    st.info(
        "Enter a research question. CSV upload is optional."
    )


# =====================================================
# PDF RESEARCH PAPER PROCESSING
# =====================================================

st.divider()

st.header(
    "📄 Research Paper PDF Analysis"
)

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

            temp_pdf.write(
                pdf_file.getvalue()
            )

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

            # Index PDF text
            index_report = index_pdf_content(
                pdf_result,
                store
            )

            st.subheader(
                "PDF Text Vector Database Indexing"
            )

            st.json(index_report)

            # =================================================
            # VISION MODEL ANALYSIS
            # =================================================

            st.subheader(
                "Vision Model Analysis"
            )

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

                    st.json(
                        visual_index_report
                    )

                    # Display vision findings
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
            # RETRIEVE TEXT AND VISUAL EVIDENCE
            # =================================================

            if question.strip():

                st.subheader(
                    "Retrieved Text Evidence"
                )

                retrieved = store.query(
                    question.strip(),
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

                st.subheader(
                    "Retrieved Visual Evidence"
                )

                visual_retrieved = store.query(
                    question.strip(),
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

            else:

                st.info(
                    "PDF indexed successfully. Enter a research question "
                    "to retrieve relevant evidence."
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
            st.subheader(
                "Extracted Page Content"
            )

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
            st.subheader(
                "Extracted Images"
            )

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