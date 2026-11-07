
"""Vetted experiment code: shared by in-process and Docker execution."""
from __future__ import annotations

import inspect
import json

from src.experiments.protocol import ExperimentPlan, validate_result


def _experiment(
    plan: dict,
    dataset_path: str | None = None,
    target_column: str | None = None,
) -> dict:
    import platform
    import time

    import numpy as np
    import pandas as pd
    import sklearn

    from sklearn import datasets
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import (
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import (
        LinearRegression,
        LogisticRegression,
        Ridge,
    )
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
    )

    task_type = plan.get("task_type", "classification")
    metric = plan["metric"]
    excluded_columns = list(plan.get("excluded_columns") or [])

    dataset_metadata = {
        "dataset_source": "builtin",
        "dataset_name": plan["dataset"],
        "target_column": None,
        "excluded_columns": [],
    }

    if dataset_path is not None:
        if not target_column:
            raise ValueError("A target column is required for CSV experiments")

        if plan["dataset"] != "uploaded_csv":
            raise ValueError(
                "Uploaded CSV experiments require dataset='uploaded_csv'"
            )

        df = pd.read_csv(dataset_path)

        if target_column not in df.columns:
            raise ValueError("Selected target column was not found")

        if target_column in excluded_columns:
            raise ValueError("The target column cannot be excluded from the dataset")

        missing_exclusions = [
            column for column in excluded_columns
            if column not in df.columns
        ]
        if missing_exclusions:
            raise ValueError(
                f"Excluded columns were not found: {missing_exclusions}"
            )

        drop_columns = [target_column, *excluded_columns]
        X = df.drop(columns=drop_columns)
        y = df[target_column]

        if X.empty or X.shape[1] == 0:
            raise ValueError("CSV must contain at least one predictor feature")

        if y.empty or y.isna().any():
            raise ValueError("CSV target contains no samples or missing values")

        if task_type == "regression":
            y = pd.to_numeric(y, errors="raise")
            if not np.isfinite(y.to_numpy(dtype=float)).all():
                raise ValueError("Regression target contains non-finite values")
            if y.nunique() < 2:
                raise ValueError("Regression target must contain at least two values")

        elif task_type == "classification":
            if y.nunique() < 2 or y.nunique() > 50:
                raise ValueError(
                    "Classification target must contain between 2 and 50 classes"
                )
            if min(y.value_counts()) < 2:
                raise ValueError("Each target class needs at least two samples")
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

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

        preprocessing = ColumnTransformer(transformers)

        dataset_metadata = {
            "dataset_source": "uploaded_csv",
            "dataset_name": "uploaded_csv",
            "target_column": target_column,
            "excluded_columns": excluded_columns,
            "feature_columns": X.columns.tolist(),
        }

    else:
        if task_type != "classification":
            raise ValueError(
                "Regression experiments currently require an uploaded CSV"
            )

        loaders = {
            "iris": datasets.load_iris,
            "wine": datasets.load_wine,
            "breast_cancer": datasets.load_breast_cancer,
            "digits": datasets.load_digits,
        }

        if plan["dataset"] not in loaders:
            raise ValueError(f"Unsupported built-in dataset: {plan['dataset']}")

        X, y = loaders[plan["dataset"]](return_X_y=True)
        preprocessing = None

    if len(y) == 0 or pd.isna(y).any():
        raise ValueError("Dataset target contains no samples or missing values")

    if task_type == "classification" and min(
        pd.Series(y).value_counts()
    ) < 2:
        raise ValueError("Each target class needs at least two samples")

    def build(value, seed):
        kw = dict(plan.get("fixed_params") or {})
        kw[plan["independent_variable"]] = value
        name = plan["model"]

        if name == "logreg":
            model = LogisticRegression(max_iter=2000, **kw)
        elif name == "svm":
            model = SVC(random_state=seed, **kw)
        elif name == "knn":
            model = KNeighborsClassifier(**kw)
        elif name == "tree":
            model = DecisionTreeClassifier(random_state=seed, **kw)
        elif name == "random_forest":
            model = RandomForestClassifier(
                random_state=seed, n_jobs=1, **kw
            )
        elif name == "linear_regression":
            model = LinearRegression(**kw)
        elif name == "decision_tree_regressor":
            model = DecisionTreeRegressor(random_state=seed, **kw)
        elif name == "random_forest_regressor":
            model = RandomForestRegressor(
                random_state=seed, n_jobs=1, **kw
            )
        elif name == "ridge":
            model = Ridge(**kw)
        else:
            raise ValueError(f"Unsupported model: {name}")

        if preprocessing is not None:
            return Pipeline([
                ("preprocessing", clone(preprocessing)),
                ("estimator", model),
            ])

        if name in ("logreg", "svm", "knn"):
            return make_pipeline(StandardScaler(), model)

        return model

    def score(model, Xs, ys):
        pred = model.predict(Xs)

        if metric == "accuracy":
            return float(accuracy_score(ys, pred))
        if metric == "f1_macro":
            return float(f1_score(ys, pred, average="macro"))
        if metric == "mae":
            return float(mean_absolute_error(ys, pred))
        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(ys, pred)))
        if metric == "r2":
            return float(r2_score(ys, pred))

        raise ValueError(f"Unsupported metric: {metric}")

    t0 = time.time()

    out = {
        "baseline": [],
        "treatment": [],
        "baseline_train": [],
        "treatment_train": [],
    }

    for seed in range(plan["n_seeds"]):
        split_options = {
            "test_size": plan["test_size"],
            "random_state": seed,
        }

        if task_type == "classification":
            split_options["stratify"] = y

        Xtr, Xte, ytr, yte = train_test_split(
            X,
            y,
            **split_options,
        )

        for arm, value in (
            ("baseline", plan["baseline_value"]),
            ("treatment", plan["treatment_value"]),
        ):
            model = build(value, seed).fit(Xtr, ytr)
            out[arm].append(score(model, Xte, yte))
            out[arm + "_train"].append(score(model, Xtr, ytr))

    return {
        "result_label": "executed",
        **out,
        **dataset_metadata,
        "task_type": task_type,
        "metric": metric,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "seeds": list(range(plan["n_seeds"])),
        "runtime_s": round(time.time() - t0, 3),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
    }


def run_plan(
    plan: ExperimentPlan,
    dataset_path: str | None = None,
    target_column: str | None = None,
) -> dict:
    """Run our vetted template in-process."""
    return validate_result(
        _experiment(
            plan.model_dump(mode="json"),
            dataset_path,
            target_column,
        )
    )


def render_script(
    plan: ExperimentPlan,
    dataset_path: str | None = None,
    target_column: str | None = None,
) -> str:
    """Render a standalone script for the Docker sandbox."""
    plan_json = json.dumps(plan.model_dump(mode="json"))
    path_json = repr(dataset_path)
    target_json = repr(target_column)

    return (
        "import json\n\n"
        + inspect.getsource(_experiment)
        + f"\n\nprint(json.dumps(_experiment({plan_json}, {path_json}, {target_json})))\n"
    )
