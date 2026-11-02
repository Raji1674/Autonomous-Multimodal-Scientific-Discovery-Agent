
"""Run controlled CSV experiments inside the Docker sandbox."""

import json
import math
from pathlib import Path

from src.sandbox.executor import execute_python, SandboxExecutionError
from src.experiments.dataset_validator import validate_csv_dataset
from src.agents.statistical_critic import StatisticalCritic


ALLOWED_PARAMS = {"n_estimators", "max_depth"}


def _validate_params(params: dict, name: str) -> dict:
    """Validate permitted Random Forest parameters."""

    if not isinstance(params, dict):
        raise ValueError(f"{name} must be a dictionary")

    unknown = set(params) - ALLOWED_PARAMS

    if unknown:
        raise ValueError(f"Unsupported parameters in {name}: {unknown}")

    result = {}

    for key, value in params.items():
        if key == "n_estimators":
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 500:
                raise ValueError("n_estimators must be an integer between 1 and 500")

        if key == "max_depth":
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 100
            ):
                raise ValueError("max_depth must be None or an integer between 1 and 100")

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
) -> dict:
    """Execute paired CSV experiments in Docker and evaluate their results."""

    path = Path(file_path).resolve(strict=True)

    # Restrict datasets to the project's downloaded-data directory.
    allowed_root = Path("data/downloaded").resolve()

    if not path.is_relative_to(allowed_root):
        raise ValueError("Only datasets inside data/downloaded are permitted")

    if not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError("A valid CSV dataset is required")

    if not isinstance(n_seeds, int) or isinstance(n_seeds, bool) or not 2 <= n_seeds <= 100:
        raise ValueError("n_seeds must be between 2 and 100")

    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, (int, float))
        or not math.isfinite(test_size)
        or not 0.1 <= test_size <= 0.5
    ):
        raise ValueError("test_size must be between 0.1 and 0.5")

    if expected_direction not in {"higher", "lower"}:
        raise ValueError("expected_direction must be higher or lower")

    if not isinstance(n_tests, int) or isinstance(n_tests, bool) or n_tests < 1:
        raise ValueError("n_tests must be a positive integer")

    validation = validate_csv_dataset(path, target_column)

    if not validation["valid"]:
        return {
            "success": False,
            "stage": "validation",
            "issues": validation["issues"],
        }

    base_params = {"n_estimators": 100}
    base_params.update(_validate_params(baseline_params or {}, "baseline_params"))

    treat_params = {"n_estimators": 200}
    treat_params.update(_validate_params(treatment_params or {}, "treatment_params"))

    if base_params == treat_params:
        return {
            "success": False,
            "stage": "configuration",
            "issues": ["Baseline and treatment parameters must differ"],
        }

    config = {
        "target_column": target_column,
        "baseline_params": base_params,
        "treatment_params": treat_params,
        "n_seeds": n_seeds,
        "test_size": float(test_size),
    }

    config_json = json.dumps(config, allow_nan=False)

    script = '''
import json
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

config = json.loads(CONFIG_JSON)

df = pd.read_csv("/input/dataset.csv")
df = df.dropna(subset=[config["target_column"]])

X = df.drop(columns=[config["target_column"]])
y = df[config["target_column"]]

if y.nunique() < 2 or y.value_counts().min() < 2:
    raise ValueError("Insufficient target classes for stratified evaluation")

numeric = X.select_dtypes(include="number").columns.tolist()
categorical = X.select_dtypes(exclude="number").columns.tolist()

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

baseline_scores = []
treatment_scores = []
last_train = None
last_test = None

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

    for params, scores in (
        (config["baseline_params"], baseline_scores),
        (config["treatment_params"], treatment_scores),
    ):
        model = Pipeline([
            ("preprocessing", ColumnTransformer(transformers)),
            ("classifier", RandomForestClassifier(
                **params,
                random_state=seed,
            )),
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        scores.append(float(accuracy_score(y_test, predictions)))

result = {
    "result_label": "executed",
    "baseline": baseline_scores,
    "treatment": treatment_scores,
    "n_seeds": config["n_seeds"],
    "n_train": last_train,
    "n_test": last_test,
}

print(json.dumps(result, allow_nan=False))
'''.replace("CONFIG_JSON", repr(config_json))

    try:
        docker_result = execute_python(script, dataset_path=path)
    except SandboxExecutionError as exc:
        return {
            "success": False,
            "stage": "sandbox",
            "issues": [str(exc)],
        }
    if docker_result.get("output_truncated"):
        return {
            "success": False,
            "stage": "result_validation",
            "issues": ["Docker output was truncated; experiment result cannot be trusted"],
        }

    if docker_result["result_label"] != "executed":
        return {
            "success": False,
            "stage": "sandbox_execution",
            "issues": [docker_result["output"][-2000:]],
        }

    try:
        result = json.loads(docker_result["output"].strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {
            "success": False,
            "stage": "result_parsing",
            "issues": ["Docker did not return valid experiment JSON"],
        }

    if result.get("result_label") != "executed":
        return {
            "success": False,
            "stage": "result_validation",
            "issues": ["Experiment result is not labelled executed"],
        }
        # Validate accuracy scores returned by the Docker container.
    for key in ("baseline", "treatment"):
        scores = result.get(key)

        if not isinstance(scores, list) or len(scores) != n_seeds:
            return {
                "success": False,
                "stage": "result_validation",
                "issues": [f"Invalid {key} score array"],
            }

        if any(
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
            for score in scores
        ):
            return {
                "success": False,
                "stage": "result_validation",
                "issues": [f"{key} scores must be finite accuracy values between 0 and 1"],
            }

    for key in ("n_train", "n_test"):
        value = result.get(key)

        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return {
                "success": False,
                "stage": "result_validation",
                "issues": [f"{key} must be a positive integer"],
            }

    if (
        not isinstance(result.get("baseline"), list)
        or not isinstance(result.get("treatment"), list)
        or len(result["baseline"]) != n_seeds
        or len(result["treatment"]) != n_seeds
    ):
        return {
            "success": False,
            "stage": "result_validation",
            "issues": ["Invalid paired score arrays returned by Docker"],
        }

    try:
        critic = StatisticalCritic().evaluate_scores(
            result,
            expected_direction=expected_direction,
            n_tests=n_tests,
        )
    except (ValueError, TypeError, KeyError) as exc:
        return {
            "success": False,
            "stage": "critic",
            "issues": [str(exc)],
        }

    return {
        "success": True,
        "dataset": path.name,
        "target_column": target_column,
        "validation": validation,
        "experiment": result,
        "critic": critic,
    }
