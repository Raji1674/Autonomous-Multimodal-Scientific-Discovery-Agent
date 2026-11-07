
from pathlib import Path
import csv
import pandas as pd


def validate_csv_dataset(
    file_path: str | Path,
    target_column: str | None = None,
) -> dict:
    """Validate CSV datasets for supervised and unsupervised ML."""

    path = Path(file_path)

    if not path.is_file():
        return {"valid": False, "issues": ["CSV file not found"]}

    if path.suffix.lower() != ".csv":
        return {"valid": False, "issues": ["Only CSV files are supported"]}

    try:
        # Check original CSV headers before pandas renames duplicates.
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            header = next(csv.reader(file), [])

        if len(header) != len(set(header)):
            return {
                "valid": False,
                "issues": ["Duplicate column names found"],
            }

        df = pd.read_csv(path)

    except (
        OSError,
        UnicodeError,
        csv.Error,
        pd.errors.ParserError,
        ValueError,
    ) as exc:
        return {"valid": False, "issues": [f"Cannot read CSV: {exc}"]}

    issues = []

    if df.empty:
        issues.append("Dataset is empty")

    if len(df.columns) < 2:
        issues.append("At least one feature and one target column are required")

    if not df.columns.is_unique:
        issues.append("Duplicate column names found")

    missing_values = int(df.isna().sum().sum())
    missing_feature_values = 0
    missing_target_values = 0
    target_classes = None

    if target_column is not None:
        if target_column not in df.columns:
            issues.append(f"Target column {target_column!r} not found")
        else:
            features = df.drop(columns=[target_column])
            target = df[target_column]

            missing_feature_values = int(features.isna().sum().sum())
            missing_target_values = int(target.isna().sum())

            all_null_features = features.columns[
                features.isna().all()
            ].tolist()

            if all_null_features:
                issues.append(
                    f"Features contain only missing values: {all_null_features}"
                )

            labelled_target = target.dropna()
            target_classes = int(labelled_target.nunique())

            if target_classes < 2:
                issues.append(
                    "Target must contain at least two classes after removing missing labels"
                )

            if target_classes > 50:
                issues.append(
                    "Target has more than 50 unique values; "
                    "check whether this is a classification task"
                )
    else:
        all_null_columns = df.columns[df.isna().all()].tolist()

        if all_null_columns:
            issues.append(
                f"Columns contain only missing values: {all_null_columns}"
            )

    duplicate_rows = int(df.duplicated().sum())

    if duplicate_rows:
        issues.append("Duplicate rows found")

    return {
        "valid": not issues,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "target_column": target_column,
        "target_classes": target_classes,
        "missing_values": missing_values,
        "missing_feature_values": missing_feature_values,
        "missing_target_values": missing_target_values,
        "duplicate_rows": duplicate_rows,
        "issues": issues,
    }
