
import re
import pandas as pd


TARGET_ALIASES = {
    "target", "label", "class", "outcome", "result",
    "diagnosis", "disease", "condition", "status",
    "prediction", "y", "survived", "heart_disease",
    "cardiovascular_disease", "diabetes", "default",
}

STOP_WORDS = {
    "can", "the", "a", "an", "is", "are", "be", "using",
    "use", "machine", "learning", "model", "predict",
    "prediction", "predicting", "based", "on", "for",
    "with", "from", "does", "will", "improve",
}


def _normalise(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _infer_target(df, research_question):
    question_words = {
        word for word in _normalise(research_question).split("_")
        if len(word) > 2 and word not in STOP_WORDS
    }

    candidates = []

    for column in df.columns:
        name = _normalise(column)
        words = set(name.split("_"))

        if name.endswith("_id") or name in {"id", "index", "uuid"}:
            continue

        values = df[column].dropna()
        unique_count = values.nunique()

        if unique_count < 2 or unique_count >= len(values):
            continue

        score = 0

        if name in TARGET_ALIASES:
            score += 5

        if words & question_words:
            score += 4

        if any(alias in name for alias in TARGET_ALIASES):
            score += 2

        if 2 <= unique_count <= 20:
            score += 1

        if score:
            candidates.append((score, column))

    if not candidates:
        return None, "No suitable target column could be inferred"

    candidates.sort(key=lambda item: item[0], reverse=True)

    if (
        len(candidates) > 1
        and candidates[0][0] == candidates[1][0]
    ):
        return None, "Multiple possible target columns were found"

    return candidates[0][1], "Target inferred from dataset columns and research question"


def detect_task(
    file_path: str,
    target_column: str | None = None,
    task_override: str | None = None,
    research_question: str | None = None,
) -> dict:
    """Detect ML task and infer a target when a question is supplied."""

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
            "target_column": target_column,
            "reason": "User-selected task",
        }

    try:
        df = pd.read_csv(file_path)

        if df.empty or len(df.columns) < 2:
            return {
                "success": False,
                "error": "Dataset must contain rows and at least two columns",
            }

        if target_column is None and research_question:
            target_column, reason = _infer_target(df, research_question)

            if target_column is None:
                return {
                    "success": False,
                    "error": reason,
                    "requires_confirmation": True,
                }
        else:
            reason = "Provided target column" if target_column else "No target provided"

        if target_column is None:
            return {
                "success": True,
                "task": "clustering",
                "target_column": None,
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
            task = "regression"
        else:
            task = "classification"

        return {
            "success": True,
            "task": task,
            "target_column": target_column,
            "reason": reason,
            "requires_confirmation": True,
        }

    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return {
            "success": False,
            "error": str(exc),
        }
