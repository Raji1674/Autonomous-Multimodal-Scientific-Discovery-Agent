
import os
import uuid
from pathlib import Path

import pandas as pd
from src.utils.config import load_config
from src.llm.providers import make_client
from src.vectorstore.chroma_store import ChromaStore
from src.vectorstore.embeddings import SentenceTransformerEmbedder
from src.ingestion.pipeline import run_ingestion
from src.agents.literature_reviewer import LiteratureReviewer
from src.agents.hypothesis_generator import HypothesisGenerator
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from src.agents.csv_experiment_agent import CSVExperimentAgent
from src.experiments.dataset_search import search_datasets
from src.experiments.csv_trainer import train_csv_dataset
from src.experiments.csv_critic import evaluate_csv_with_critic


app = Flask(__name__, template_folder="templates")
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "local-development-secret-change-before-deployment",
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_uploaded_file():
    """Return only the CSV assigned to the current session."""
    filename = session.get("uploaded_csv")

    if not filename or Path(filename).name != filename:
        return None

    file_path = UPLOAD_DIR / filename

    if not file_path.is_file():
        return None

    return file_path


def render_dashboard(**kwargs):
    """Render the dashboard with common session information."""
    file_path = get_uploaded_file()

    defaults = {
        "message": None,
        "columns": [],
        "preview": [],
        "row_count": None,
        "uploaded": file_path is not None,
        "train_result": None,
        "evaluation_result": None,
        "keyword": "",
        "datasets": [],
        "research_result": None,
    }
    defaults.update(kwargs)

    return render_template("index.html", **defaults)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_dashboard()

    uploaded_file = request.files.get("csv_file")

    if not uploaded_file or not uploaded_file.filename:
        return render_dashboard(message="Please select a CSV file.")

    if Path(uploaded_file.filename).suffix.lower() != ".csv":
        return render_dashboard(message="Only CSV files are supported.")

    try:
        # Generate a server-side filename; never trust the uploaded filename.
        filename = f"{uuid.uuid4().hex}.csv"
        file_path = UPLOAD_DIR / filename

        uploaded_file.save(file_path)

        df = pd.read_csv(file_path)

        if df.empty:
            file_path.unlink(missing_ok=True)
            return render_dashboard(message="The CSV file is empty.")

        if len(df.columns) < 2:
            file_path.unlink(missing_ok=True)
            return render_dashboard(
                message="The CSV must contain at least two columns."
            )

        session["uploaded_csv"] = filename

        return render_dashboard(
            message="CSV uploaded successfully. Select the target column.",
            columns=df.columns.tolist(),
            preview=df.head(5).fillna("").to_dict("records"),
            row_count=len(df),
        )

    except Exception:
        return render_dashboard(
            message="Could not read the CSV. Check its format and try again."
        )


@app.route("/train", methods=["POST"])
def train():
    file_path = get_uploaded_file()
    target_column = request.form.get("target_column", "").strip()

    if file_path is None:
        return render_dashboard(
            message="Upload a CSV dataset before training."
        )

    try:
        df = pd.read_csv(file_path, nrows=0)

        if target_column not in df.columns:
            return render_dashboard(
                message="Please select a valid target column."
            )

        result = train_csv_dataset(
            file_path=file_path,
            target_column=target_column,
        )

        return render_dashboard(
            message="Training completed." if result.get("success")
            else "Training could not be completed.",
            columns=df.columns.tolist(),
            train_result=result,
        )

    except Exception:
        return render_dashboard(
            message="An error occurred during training. Check the dataset."
        )


@app.route("/evaluate", methods=["POST"])
def evaluate():
    file_path = get_uploaded_file()
    target_column = request.form.get("target_column", "").strip()

    if file_path is None:
        return render_dashboard(
            message="Upload a CSV dataset before evaluation."
        )

    try:
        df = pd.read_csv(file_path, nrows=0)

        if target_column not in df.columns:
            return render_dashboard(
                message="Please select a valid target column."
            )

        n_seeds = int(request.form.get("n_seeds", "20"))
        test_size = float(request.form.get("test_size", "0.2"))
        expected_direction = request.form.get(
            "expected_direction", "higher"
        )

        if not 2 <= n_seeds <= 100:
            raise ValueError("Number of runs must be between 2 and 100.")

        if not 0.1 <= test_size <= 0.5:
            raise ValueError("Test size must be between 0.1 and 0.5.")

        if expected_direction not in {"higher", "lower"}:
            raise ValueError("Invalid expected direction.")

        result = evaluate_csv_with_critic(
            file_path=str(file_path),
            target_column=target_column,
            n_seeds=n_seeds,
            test_size=test_size,
            expected_direction=expected_direction,
        )

        return render_dashboard(
            message="Evaluation completed." if result.get("success")
            else "Evaluation could not be completed.",
            columns=df.columns.tolist(),
            evaluation_result=result,
        )

    except (ValueError, TypeError):
        return render_dashboard(
            message="Invalid evaluation settings. Check the values and try again."
        )
    except Exception:
        return render_dashboard(
            message="An error occurred during evaluation. Check the dataset."
        )

@app.route("/experiment", methods=["POST"])
def experiment():
    file_path = get_uploaded_file()
    target_column = request.form.get("target_column", "").strip()
    task_override = request.form.get("task_override", "").strip() or None

    if file_path is None:
        return render_dashboard(message="Upload a CSV dataset first.")

    try:
        agent = CSVExperimentAgent()

        state = {
            "dataset_path": str(file_path),
            "target_column": target_column or None,
            "task_override": task_override,
            "selected_hypothesis": {
                "statement": "Evaluate the selected dataset using an appropriate ML task."
            },
        }

        result = agent.run(state)

        return render_dashboard(
            message="Experiment completed."
            if not result.get("failure_state")
            else result.get("error", "Experiment failed."),
            evaluation_result=result,
        )

    except Exception as exc:
        app.logger.exception("CSV experiment failed")
        return render_dashboard(
            message=f"Experiment failed: {exc}"
        )


@app.route("/search")
def search():
    keyword = request.args.get("keyword", "").strip()
    datasets = search_datasets(keyword) if keyword else []

    return render_dashboard(
        keyword=keyword,
        datasets=datasets,
    )


@app.route("/research", methods=["POST"])
def research():
    topic = request.form.get("research_question", "").strip()

    if not topic:
        return render_dashboard(
            message="Please enter a research topic."
        )

    try:
        app.logger.info("Starting research for topic: %s", topic)

        # Load configuration and initialize real components
        cfg = load_config()
        llm = make_client(cfg.models)

        store = ChromaStore(
            cfg.retrieval.chroma_path,
            SentenceTransformerEmbedder(cfg.retrieval.embedding_model),
        )

        # Step 1: Retrieve and index real research papers
        ingestion_report = run_ingestion(
            topic,
            store,
            max_per_source=10,
        )

        if ingestion_report.chunks_written <= 0:
            return render_dashboard(
                message="No research papers could be indexed. Check ArXiv/PubMed connectivity.",
                research_result={
                    "research_question": topic,
                    "error": "No literature evidence was indexed.",
                    "ingestion": {
                        "fetched": ingestion_report.fetched,
                        "after_dedupe": ingestion_report.after_dedupe,
                        "chunks_written": ingestion_report.chunks_written,
                        "failures": ingestion_report.failures,
                    },
                },
            )

        # Step 2: Literature review using retrieved papers
        state = {"research_question": topic}

        reviewer = LiteratureReviewer(
            store,
            llm,
            top_k=cfg.retrieval.top_k,
        )

        literature_result = reviewer.run(state)

        if literature_result.get("failure_state"):
            return render_dashboard(
                message="Literature review failed.",
                research_result={
                    **literature_result,
                    "research_question": topic,
                },
            )

        state.update(literature_result)

        # Step 3: Generate hypotheses from literature
        generator = HypothesisGenerator(
            llm,
            max_retries=cfg.agents.max_hypothesis_retries,
        )

        hypothesis_result = generator.run(state)

        final_result = {
            **literature_result,
            **hypothesis_result,
            "research_question": topic,
            "ingestion": {
                "fetched": ingestion_report.fetched,
                "after_dedupe": ingestion_report.after_dedupe,
                "chunks_written": ingestion_report.chunks_written,
                "failures": ingestion_report.failures,
            },
        }

        return render_dashboard(
            message="Research pipeline completed."
            if not hypothesis_result.get("failure_state")
            else "Literature review completed, but hypothesis generation failed.",
            research_result=final_result,
        )

    except Exception:
        app.logger.exception("Autonomous research pipeline failed")
        return render_dashboard(
            message="Research failed. Check the Flask terminal logs.",
            research_result={
                "research_question": topic,
                "error": "An unexpected error occurred during research.",
            },
        )

@app.errorhandler(413)
def file_too_large(_error):
    return render_dashboard(
        message="File is too large. Maximum upload size is 16 MB."
    ), 413


if __name__ == "__main__":
    app.run(debug=True)
