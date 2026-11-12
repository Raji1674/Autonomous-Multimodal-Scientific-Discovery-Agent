
import os
import uuid
from pathlib import Path

import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    session,
)

from src.agents.csv_experiment_agent import CSVExperimentAgent
from src.experiments.dataset_search import search_datasets
from src.experiments.csv_trainer import train_csv_dataset
from src.experiments.csv_critic import evaluate_csv_with_critic
from src.workflow.service import ResearchService


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
    """Render dashboard with common session information."""

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
        return render_dashboard(
            message="Please select a CSV file."
        )

    if Path(uploaded_file.filename).suffix.lower() != ".csv":
        return render_dashboard(
            message="Only CSV files are supported."
        )

    file_path = None

    try:
        filename = f"{uuid.uuid4().hex}.csv"
        file_path = UPLOAD_DIR / filename

        uploaded_file.save(file_path)

        df = pd.read_csv(file_path)

        if df.empty:
            file_path.unlink(missing_ok=True)
            return render_dashboard(
                message="The CSV file is empty."
            )

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
        app.logger.exception("CSV upload failed")

        if file_path is not None:
            file_path.unlink(missing_ok=True)

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
            message=(
                "Training completed."
                if result.get("success")
                else "Training could not be completed."
            ),
            columns=df.columns.tolist(),
            train_result=result,
        )

    except Exception:
        app.logger.exception("Model training failed")

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
            "expected_direction",
            "higher",
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
            message=(
                "Evaluation completed."
                if result.get("success")
                else "Evaluation could not be completed."
            ),
            columns=df.columns.tolist(),
            evaluation_result=result,
        )

    except (ValueError, TypeError):
        return render_dashboard(
            message="Invalid evaluation settings. Check the values and try again."
        )

    except Exception:
        app.logger.exception("CSV evaluation failed")

        return render_dashboard(
            message="An error occurred during evaluation. Check the dataset."
        )


@app.route("/experiment", methods=["POST"])
def experiment():

    file_path = get_uploaded_file()
    target_column = request.form.get("target_column", "").strip()

    task_override = (
        request.form.get("task_override", "").strip() or None
    )

    if file_path is None:
        return render_dashboard(
            message="Upload a CSV dataset first."
        )

    try:
        agent = CSVExperimentAgent()

        state = {
            "dataset_path": str(file_path),
            "target_column": target_column or None,
            "task_override": task_override,
            "selected_hypothesis": {
                "statement": (
                    "Evaluate the selected dataset using an appropriate ML task."
                )
            },
        }

        result = agent.run(state)

        return render_dashboard(
            message=(
                "Experiment completed."
                if not result.get("failure_state")
                else result.get("error", "Experiment failed.")
            ),
            evaluation_result=result,
        )

    except Exception:
        app.logger.exception("CSV experiment failed")

        return render_dashboard(
            message="Experiment failed. Check the Flask terminal logs."
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

    topic = request.form.get(
        "research_question",
        "",
    ).strip()

    if not topic:
        return render_dashboard(
            message="Please enter a research topic."
        )

    try:
        app.logger.info(
            "Starting autonomous research workflow."
        )

        # Initialize the integrated research service.
        service = ResearchService(enable_ingestion=False)

        # Run the complete protocol workflow.
        result = service.run(
            research_question=topic,
            experiment_mode="protocol",
        )

        state = result.get("state", {})

        research_result = {
            **state,
            "research_question": topic,
            "run_id": result.get("run_id"),
            "experiment_mode": result.get("experiment_mode"),
            "ingestion": result.get("ingestion_report"),
        }

        failure_state = state.get("failure_state")

        if failure_state:
            message = (
                "Research workflow stopped: "
                + str(state.get("error") or failure_state)
            )
        else:
            message = "Autonomous research workflow completed."

        return render_dashboard(
            message=message,
            research_result=research_result,
        )

    except Exception:
        app.logger.exception(
            "Autonomous research workflow failed."
        )

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

