
"""Run controlled CSV experiments inside the Docker sandbox."""

import json
import math
from pathlib import Path

from src.sandbox.executor import execute_python, SandboxExecutionError
from src.experiments.dataset_validator import validate_csv_dataset
from src.agents.statistical_critic import StatisticalCritic


ALLOWED_PARAMS = {
    "logistic_regression": {"C", "max_iter"},
    "random_forest": {"n_estimators", "max_depth"},
    "xgboost": {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
    },
}


def _failure(stage: str, message: str) -> dict:
    """Return a consistent experiment failure result."""
    return {
        "success": False,
        "stage": stage,
        "issues": [message],
    }


def validate_sandbox_inputs(dataset_path, n_seeds):
    """Validate dataset location and seed count.

    Raises ValueError for invalid paths and seed counts.
    """

    allowed_root = (Path.cwd() / "data" / "downloaded").resolve()
    dataset = Path(dataset_path).resolve()

    try:
        dataset.relative_to(allowed_root)
    except ValueError:
        raise ValueError(
            "Dataset must be located inside the data/downloaded directory."
        )

    if (
        isinstance(n_seeds, bool)
        or not isinstance(n_seeds, int)
        or not 2 <= n_seeds <= 100
    ):
        raise ValueError(
            "n_seeds must be an integer between 2 and 100."
        )

    if not dataset.exists():
        raise ValueError(f"Dataset does not exist: {dataset}")

    if not dataset.is_file() or dataset.suffix.lower() != ".csv":
        raise ValueError("Dataset must be an existing CSV file.")

    return dataset


def _validate_params(
    params: dict,
    name: str,
    algorithm: str = "random_forest",
) -> dict:
    """Validate parameters for the selected classifier."""

    if not isinstance(params, dict):
        raise ValueError(f"{name} must be a dictionary")

    if algorithm not in ALLOWED_PARAMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    unknown = set(params) - ALLOWED_PARAMS[algorithm]

    if unknown:
        raise ValueError(
            f"Unsupported parameters in {name}: {unknown}"
        )

    result = {}

    for key, value in params.items():

        if key == "C":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    "C must be a finite positive number"
                )

        elif key == "max_iter":
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 100 <= value <= 10000
            ):
                raise ValueError(
                    "max_iter must be an integer between 100 and 10000"
                )

        elif key in {"n_estimators", "max_depth"}:
            maximum = 500 if key == "n_estimators" else 100

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= maximum
            ):
                raise ValueError(
                    f"{key} must be an integer between 1 and {maximum}"
                )

        elif key in {
            "learning_rate",
            "subsample",
            "colsample_bytree",
        }:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                or (
                    key in {"subsample", "colsample_bytree"}
                    and value > 1
                )
            ):
                raise ValueError(f"Invalid value for {key}")

        result[key] = value

    return result


def run_sandbox_csv_experiment(
    file_path: str | Path,
    target_column: str,
    baseline_params: dict | None = None,
    treatment_params: dict | None = None,
    n_seeds: int = 20,
    test_size: float = 0.2,
    expected_direction: str = "higher",
    n_tests: int = 1,
    script_override: str | None = None,
    baseline_algorithm: str = "random_forest",
    treatment_algorithm: str = "random_forest",
) -> dict:
    """Execute paired CSV experiments in Docker and evaluate results."""

    # ---------------------------------------------------------
    # 1. Validate dataset path and seed count
    # ---------------------------------------------------------

    path = validate_sandbox_inputs(file_path, n_seeds)

    # ---------------------------------------------------------
    # 2. Validate experiment configuration
    # ---------------------------------------------------------

    if not isinstance(target_column, str) or not target_column.strip():
        return _failure(
            "validation",
            "A valid target column is required",
        )

    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, (int, float))
        or not math.isfinite(test_size)
        or not 0.1 <= test_size <= 0.5
    ):
        return _failure(
            "configuration",
            "test_size must be between 0.1 and 0.5",
        )

    if expected_direction not in {"higher", "lower"}:
        return _failure(
            "configuration",
            "expected_direction must be higher or lower",
        )

    if (
        isinstance(n_tests, bool)
        or not isinstance(n_tests, int)
        or n_tests < 1
    ):
        return _failure(
            "configuration",
            "n_tests must be a positive integer",
        )

    if baseline_algorithm not in ALLOWED_PARAMS:
        return _failure(
            "configuration",
            "Invalid baseline algorithm",
        )

    if treatment_algorithm not in ALLOWED_PARAMS:
        return _failure(
            "configuration",
            "Invalid treatment algorithm",
        )

    # ---------------------------------------------------------
    # 3. Validate dataset contents
    # ---------------------------------------------------------

    try:
        validation = validate_csv_dataset(path, target_column)

    except Exception as exc:
        return _failure(
            "validation",
            f"Dataset validation raised {type(exc).__name__}: {exc}",
        )

    if not validation.get("valid"):
        return {
            "success": False,
            "stage": "validation",
            "issues": validation.get(
                "issues",
                ["Dataset validation failed"],
            ),
            "validation": validation,
        }

    # ---------------------------------------------------------
    # 4. Set baseline and treatment defaults
    # ---------------------------------------------------------

    baseline_defaults = {
        "logistic_regression": {
            "C": 1.0,
            "max_iter": 2000,
        },
        "random_forest": {
            "n_estimators": 100,
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
        },
    }

    treatment_defaults = {
        "logistic_regression": {
            "C": 0.1,
            "max_iter": 2000,
        },
        "random_forest": {
            "n_estimators": 200,
        },
        "xgboost": {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.1,
        },
    }

    try:
        base_params = baseline_defaults[baseline_algorithm].copy()
        base_params.update(
            _validate_params(
                baseline_params or {},
                "baseline_params",
                baseline_algorithm,
            )
        )

        treat_params = treatment_defaults[treatment_algorithm].copy()
        treat_params.update(
            _validate_params(
                treatment_params or {},
                "treatment_params",
                treatment_algorithm,
            )
        )

    except (TypeError, ValueError) as exc:
        return _failure("configuration", str(exc))

    if (
        baseline_algorithm == treatment_algorithm
        and base_params == treat_params
    ):
        return _failure(
            "configuration",
            "Baseline and treatment configurations must differ",
        )

    # ---------------------------------------------------------
    # 5. Build experiment configuration
    # ---------------------------------------------------------

    config = {
        "target_column": target_column,
        "baseline_params": base_params,
        "treatment_params": treat_params,
        "baseline_algorithm": baseline_algorithm,
        "treatment_algorithm": treatment_algorithm,
        "n_seeds": n_seeds,
        "test_size": float(test_size),
    }

    try:
        config_json = json.dumps(config, allow_nan=False)

    except (TypeError, ValueError) as exc:
        return _failure(
            "configuration",
            f"Invalid configuration: {exc}",
        )

    # ---------------------------------------------------------
    # 6. Generate sandbox experiment script
    # ---------------------------------------------------------

    script = '''
import json
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder,
)

from xgboost import XGBClassifier

config = json.loads(CONFIG_JSON)

df = pd.read_csv("/input/dataset.csv")
df = df.dropna(subset=[config["target_column"]])

X = df.drop(columns=[config["target_column"]])
y = df[config["target_column"]]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

class_counts = pd.Series(y).value_counts()

if len(class_counts) < 2 or class_counts.min() < 2:
    raise ValueError(
        "Insufficient target classes for stratified evaluation"
    )

numeric = X.select_dtypes(include="number").columns.tolist()
categorical = X.select_dtypes(exclude="number").columns.tolist()


def build_preprocessor():
    transformers = []

    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ))

    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))

    if not transformers:
        raise ValueError("Dataset has no usable feature columns")

    return ColumnTransformer(transformers)


baseline_scores = []
treatment_scores = []

last_train = None
last_test = None

num_classes = len(class_counts)
eval_metric = "logloss" if num_classes == 2 else "mlogloss"

for seed in range(config["n_seeds"]):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=seed,
        stratify=y,
    )

    last_train = len(X_train)
    last_test = len(X_test)

    experiment_arms = (
        (
            config["baseline_algorithm"],
            config["baseline_params"],
            baseline_scores,
        ),
        (
            config["treatment_algorithm"],
            config["treatment_params"],
            treatment_scores,
        ),
    )

    for algorithm, params, scores in experiment_arms:

        if algorithm == "logistic_regression":
            classifier = LogisticRegression(
                **params,
                random_state=seed,
                solver="liblinear" if num_classes == 2 else "lbfgs",
            )

        elif algorithm == "random_forest":
            classifier = RandomForestClassifier(
                **params,
                random_state=seed,
                n_jobs=1,
            )

        elif algorithm == "xgboost":
            classifier = XGBClassifier(
                **params,
                random_state=seed,
                n_jobs=1,
                tree_method="hist",
                eval_metric=eval_metric,
            )

        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        model = Pipeline([
            ("preprocessing", build_preprocessor()),
            ("classifier", classifier),
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        scores.append(
            float(accuracy_score(y_test, predictions))
        )


result = {
    "result_label": "executed",
    "baseline": baseline_scores,
    "treatment": treatment_scores,
    "n_seeds": config["n_seeds"],
    "n_train": last_train,
    "n_test": last_test,
    "baseline_algorithm": config["baseline_algorithm"],
    "treatment_algorithm": config["treatment_algorithm"],
}

print(json.dumps(result, allow_nan=False))
'''.replace("CONFIG_JSON", repr(config_json))

    if script_override is not None:
        if (
            not isinstance(script_override, str)
            or not script_override.strip()
        ):
            return _failure(
                "configuration",
                "script_override must be non-empty Python code",
            )

        script = script_override

    # ---------------------------------------------------------
    # 7. Execute experiment inside Docker
    # ---------------------------------------------------------

    try:
        docker_result = execute_python(
            script,
            dataset_path=path,
        )

    except SandboxExecutionError as exc:
        message = str(exc)

        return {
            "success": False,
            "stage": "sandbox",
            "issues": [message],
            "debug_feedback": message,
            "generated_code": {
                "code": script,
                "experiment_name": "paired_classifier_experiment",
            },
        }

    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"

        return {
            "success": False,
            "stage": "sandbox",
            "issues": [message],
            "debug_feedback": message,
            "generated_code": {
                "code": script,
                "experiment_name": "paired_classifier_experiment",
            },
        }

    if not isinstance(docker_result, dict):
        return _failure(
            "sandbox",
            "Sandbox executor returned an invalid response",
        )

    if docker_result.get("output_truncated"):
        return _failure(
            "result_validation",
            "Docker output was truncated; experiment result cannot be trusted",
        )

    if docker_result.get("result_label") != "executed":
        message = str(docker_result.get("output", ""))[-2000:]

        return {
            "success": False,
            "stage": "sandbox",
            "issues": [message or "Sandbox execution failed"],
            "debug_feedback": message or "Sandbox execution failed",
            "generated_code": {
                "code": script,
                "experiment_name": "paired_classifier_experiment",
            },
        }

    # ---------------------------------------------------------
    # 8. Parse sandbox JSON output
    # ---------------------------------------------------------

    try:
        output = docker_result.get("output", "").strip()

        if not output:
            raise ValueError("Sandbox returned empty output")

        result = json.loads(output.splitlines()[-1])

        if not isinstance(result, dict):
            raise ValueError(
                "Experiment output must be a JSON object"
            )

    except (
        json.JSONDecodeError,
        IndexError,
        AttributeError,
        ValueError,
    ) as exc:
        return {
            "success": False,
            "stage": "result_parsing",
            "issues": [
                f"Docker did not return valid experiment JSON: {exc}"
            ],
            "debug_feedback": str(exc),
            "generated_code": {
                "code": script,
                "experiment_name": "paired_classifier_experiment",
            },
        }

    # ---------------------------------------------------------
    # 9. Validate experiment result
    # ---------------------------------------------------------

    if result.get("result_label") != "executed":
        return _failure(
            "result_validation",
            "Experiment result is not labelled executed",
        )

    returned_baseline = result.get("baseline_algorithm")
    returned_treatment = result.get("treatment_algorithm")

    if (
        returned_baseline is not None
        and returned_baseline != baseline_algorithm
    ):
        return _failure(
            "result_validation",
            "Executed baseline algorithm does not match the requested algorithm",
        )

    if (
        returned_treatment is not None
        and returned_treatment != treatment_algorithm
    ):
        return _failure(
            "result_validation",
            "Executed treatment algorithm does not match the requested algorithm",
        )

    for key in ("baseline", "treatment"):

        scores = result.get(key)

        if not isinstance(scores, list) or len(scores) != n_seeds:
            return _failure(
                "result_validation",
                f"Invalid {key} score array",
            )

        if any(
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
            for score in scores
        ):
            return _failure(
                "result_validation",
                f"{key} scores must be finite accuracy values between 0 and 1",
            )

    for key in ("n_train", "n_test"):

        value = result.get(key)

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            return _failure(
                "result_validation",
                f"{key} must be a positive integer",
            )

    returned_seeds = result.get("n_seeds")

    if (
        isinstance(returned_seeds, bool)
        or not isinstance(returned_seeds, int)
        or returned_seeds != n_seeds
    ):
        return _failure(
            "result_validation",
            "Returned seed count does not match requested seed count",
        )

    # ---------------------------------------------------------
    # 10. Statistical evaluation
    # ---------------------------------------------------------

    try:
        critic = StatisticalCritic().evaluate_scores(
            result,
            expected_direction=expected_direction,
            n_tests=n_tests,
        )

    except (ValueError, TypeError, KeyError) as exc:
        return _failure("critic", str(exc))

    # ---------------------------------------------------------
    # 11. Return experiment results and protocol
    # ---------------------------------------------------------

    return {
        "success": True,
        "dataset": path.name,
        "target_column": target_column,
        "validation": validation,
        "experiment": result,
        "critic": critic,
        "experiment_protocol": {
            "algorithm": (
                f"{baseline_algorithm} vs {treatment_algorithm}"
            ),
            "baseline_algorithm": baseline_algorithm,
            "treatment_algorithm": treatment_algorithm,
            "baseline_params": base_params,
            "treatment_params": treat_params,
            "metric": "accuracy",
            "n_seeds": n_seeds,
            "test_size": float(test_size),
            "n_train": result["n_train"],
            "n_test": result["n_test"],
        },
    }
