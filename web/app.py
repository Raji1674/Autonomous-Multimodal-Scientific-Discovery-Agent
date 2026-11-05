
from src.experiments.dataset_search import search_datasets
from pathlib import Path
from src.experiments.dataset_validator import validate_csv_dataset
import pandas as pd
from flask import Flask, render_template, request
import uuid

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the production research service lazily.
_research_service = None


def get_research_service():
    global _research_service

    if _research_service is None:
        from src.workflow.service import ResearchService
        _research_service = ResearchService()

    return _research_service


@app.route("/", methods=["GET", "POST"])
def home():
    message = None
    columns = []
    preview = []
    row_count = None
    upload_id = None
    validated_upload_id = None
    validated_target_column = None

    if request.method == "POST":
        submitted_upload_id = request.form.get("upload_id", "").strip()

        # Handle target-column validation
        if submitted_upload_id:
            try:
                parsed_id = uuid.UUID(hex=submitted_upload_id)
                if parsed_id.hex != submitted_upload_id:
                    raise ValueError("Invalid upload ID")
            except ValueError:
                message = "Invalid upload reference."
            else:
                saved_path = UPLOAD_DIR / f"{submitted_upload_id}.csv"
                target_column = request.form.get("target_column", "").strip()
                upload_id = submitted_upload_id

                if not saved_path.is_file():
                    message = "Uploaded CSV not found. Please upload it again."
                elif not target_column:
                    message = "Please select a target column."
                else:
                    validation = validate_csv_dataset(
                        saved_path,
                        target_column,
                    )

                    columns = validation.get("column_names", [])
                    row_count = validation.get("rows")

                    if validation.get("valid"):
                        df = pd.read_csv(saved_path, nrows=5)
                        preview = df.fillna("").to_dict("records")

                        validated_upload_id = submitted_upload_id
                        validated_target_column = target_column

                        message = (
                            f"Dataset validated successfully. "
                            f"Target column: {target_column}"
                        )
                    else:
                        issues = validation.get("issues", [])
                        message = "Dataset validation failed: " + "; ".join(issues)

        # Handle a new CSV upload
        else:
            file = request.files.get("csv_file")

            if not file or not file.filename:
                message = "Please select a CSV file."
            elif Path(file.filename).suffix.lower() != ".csv":
                message = "Only CSV files are supported."
            else:
                upload_id = uuid.uuid4().hex
                saved_path = UPLOAD_DIR / f"{upload_id}.csv"

                try:
                    file.save(saved_path)
                    df = pd.read_csv(saved_path, nrows=5)

                    if df.empty:
                        saved_path.unlink(missing_ok=True)
                        upload_id = None
                        message = "The CSV file is empty."
                    else:
                        columns = df.columns.tolist()
                        preview = df.fillna("").to_dict("records")
                        message = (
                            "CSV uploaded successfully. "
                            "Select the target column to validate it."
                        )

                except Exception:
                    saved_path.unlink(missing_ok=True)
                    upload_id = None
                    message = "Could not read the CSV. Please check the file."

    return render_template(
        "index.html",
        message=message,
        columns=columns,
        preview=preview,
        upload_id=upload_id,
        row_count=row_count,
        validated_upload_id=validated_upload_id,
        validated_target_column=validated_target_column,
    )


@app.route("/search")
def search():
    keyword = request.args.get("keyword", "").strip()
    datasets = []

    if keyword:
        datasets = search_datasets(keyword)

    return render_template(
        "index.html",
        keyword=keyword,
        datasets=datasets,
    )


@app.route("/research", methods=["POST"])
def research():
    question = request.form.get("research_question", "").strip()
    upload_id = request.form.get("upload_id", "").strip()
    target_column = request.form.get("target_column", "").strip()

    result = {"ok": False}

    if not question:
        result = {
            "ok": False,
            "error": "Please enter a research question."
        }

    elif len(question) > 500:
        result = {
            "ok": False,
            "error": "Research questions must be 500 characters or fewer."
        }

    elif bool(upload_id) != bool(target_column):
        result = {
            "ok": False,
            "error": "Invalid dataset reference. Please validate your CSV again."
        }

    else:
        try:
            service = get_research_service()

            if upload_id:
                try:
                    parsed_id = uuid.UUID(hex=upload_id)
                    if parsed_id.hex != upload_id:
                        raise ValueError("Invalid upload ID")
                except ValueError:
                    result = {
                        "ok": False,
                        "error": "Invalid dataset reference."
                    }
                else:
                    saved_path = UPLOAD_DIR / f"{upload_id}.csv"

                    if not saved_path.is_file():
                        result = {
                            "ok": False,
                            "error": "Uploaded CSV not found. Please upload it again."
                        }
                    else:
                        validation = validate_csv_dataset(
                            saved_path,
                            target_column
                        )

                        if not validation.get("valid"):
                            result = {
                                "ok": False,
                                "error": "Dataset validation failed: "
                                + "; ".join(validation.get("issues", []))
                            }
                        else:
                            result = service.run(
                                question,
                                dataset_path=str(saved_path),
                                target_column=target_column
                            )
            else:
                result = service.run(question)

        except Exception:
            result = {
                "ok": False,
                "error": (
                    "Research execution failed. "
                    "Check the application configuration and logs."
                )
            }

    return render_template(
        "index.html",
        research_question=question,
        research_result=result,
        validated_upload_id=upload_id if result.get("ok") else None,
        validated_target_column=target_column if result.get("ok") else None,
    )

if __name__ == "__main__":
    app.run(debug=True)
