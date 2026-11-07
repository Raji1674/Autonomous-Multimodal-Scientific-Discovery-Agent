
from pathlib import Path

import pandas as pd


def validate_csv_dataset(
    file_path: str | Path,
    target_column: str,
    task_type: str | None = None,
) -> dict:
    """Validate a CSV dataset for classification or regression."""

    path = Path(file_path)

    if not path.is_file():
        return {"valid": False, "issues": ["CSV file not found"]}

    if path.suffix.lower() != ".csv":
        return {"valid": False, "issues": ["Only CSV files are supported"]}

    try:
        df = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        return {"valid": False, "issues": [f"Cannot read CSV: {exc}"]}

    issues = []

    if df.empty:
        issues.append("Dataset is empty")

    if len(df.columns) < 2:
        issues.append("At least one feature and one target column are required")

    if not df.columns.is_unique:
        issues.append("Duplicate column names found")

    if target_column not in df.columns:
        issues.append(f"Target column {target_column!r} not found")

    if df.isna().any().any():
        issues.append("Missing values found; preprocessing is required")

    if df.duplicated().any():
        issues.append("Duplicate rows found")

    detected_task = None
    target_classes = None

    if target_column in df.columns:
        target = df[target_column]
        unique_count = int(target.nunique(dropna=True))
        target_classes = unique_count

        if unique_count < 2:
            issues.append("Target must contain at least two unique values")

        if task_type is not None and task_type not in {
            "classification", "regression"
        }:
            issues.append("task_type must be classification or regression")

        elif task_type is not None:
            detected_task = task_type

        elif pd.api.types.is_numeric_dtype(target) and unique_count > 10:
            detected_task = "regression"

        else:
            detected_task = "classification"

        if detected_task == "regression" and not pd.api.types.is_numeric_dtype(target):
            issues.append("Regression target must be numeric")

    return {
        "valid": not issues,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "target_column": target_column,
        "target_classes": target_classes,
        "task_type": detected_task,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "issues": issues,
    }
