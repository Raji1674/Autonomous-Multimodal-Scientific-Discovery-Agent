from src.experiments.dataset_search import search_datasets
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def home():
    message = None
    columns = []
    preview = []
    row_count = None

    if request.method == "POST":
        file = request.files.get("csv_file")

        if not file or not file.filename:
            message = "Please select a CSV file."
        elif Path(file.filename).suffix.lower() != ".csv":
            message = "Only CSV files are supported."
        else:
            try:
                df = pd.read_csv(file, nrows=5)

                if df.empty:
                    message = "The CSV file is empty."
                else:
                    columns = df.columns.tolist()
                    preview = df.head(5).fillna("").to_dict("records")
                    row_count = len(df)
                    message = (
                        "CSV loaded successfully. "
                        "Showing the first 5 rows for validation."
                    )
            except Exception as exc:
                message = f"Could not read the CSV: {exc}"

    return render_template(
        "index.html",
        message=message,
        columns=columns,
        preview=preview,
        row_count=row_count,
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
if __name__ == "__main__":
    app.run(debug=True)
