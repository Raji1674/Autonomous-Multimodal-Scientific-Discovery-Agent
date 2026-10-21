
from pathlib import Path

import pandas as pd


def validate_csv_dataset(
    file_path: str | Path,
    target_column: str,
) -> dict:
    """Validate a CSV dataset before a classification experiment."""

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

    if target_column in df.columns:
        target = df[target_column]

        if target.nunique(dropna=True) < 2:
            issues.append("Target must contain at least two classes")

        if target.nunique(dropna=True) > 50:
            issues.append(
                "Target has more than 50 unique values; "
                "check whether this is a classification task"
            )

    return {
        "valid": not issues,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "target_column": target_column,
        "target_classes": (
            int(df[target_column].nunique(dropna=True))
            if target_column in df.columns
            else None
        ),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "issues": issues,
    }
