
import pandas as pd


def detect_task(
    file_path: str,
    target_column: str | None = None,
    task_override: str | None = None,
) -> dict:
    """Detect classification, regression, or clustering task."""

    allowed_tasks = {"classification", "regression", "clustering"}

    if task_override is not None:
        if task_override not in allowed_tasks:
            return {
                "success": False,
                "error": f"Task must be one of {sorted(allowed_tasks)}",
            }

        return {
            "success": True,
            "task": task_override,
            "reason": "User-selected task",
        }

    try:
        df = pd.read_csv(file_path)

        if df.empty or len(df.columns) < 2:
            return {
                "success": False,
                "error": "Dataset must contain rows and at least two columns",
            }

        if target_column is None:
            return {
                "success": True,
                "task": "clustering",
                "reason": "No target column provided",
                "requires_confirmation": True,
            }

        if target_column not in df.columns:
            return {
                "success": False,
                "error": f"Target column '{target_column}' not found",
            }

        target = df[target_column].dropna()

        if target.nunique() < 2:
            return {
                "success": False,
                "error": "Target must contain at least two unique values",
            }

        if (
            pd.api.types.is_numeric_dtype(target)
            and target.nunique() > 20
        ):
            return {
                "success": True,
                "task": "regression",
                "reason": "Numeric target with more than 20 unique values",
                "requires_confirmation": True,
            }

        return {
            "success": True,
            "task": "classification",
            "reason": "Target has a limited number of distinct classes",
            "requires_confirmation": True,
        }

    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return {
            "success": False,
            "error": str(exc),
        }
