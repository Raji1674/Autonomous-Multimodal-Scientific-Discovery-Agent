
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.experiments.dataset_validator import validate_csv_dataset
from src.statistics.significance import paired_analysis


def evaluate_csv_models(
    file_path: str | Path,
    target_column: str,
    baseline_params: dict | None = None,
    treatment_params: dict | None = None,
    n_seeds: int = 20,
    test_size: float = 0.2,
) -> dict:
    """Compare two Random Forest configurations on paired CSV splits."""

    if n_seeds < 2:
        raise ValueError("n_seeds must be at least 2")
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5")

    validation = validate_csv_dataset(file_path, target_column)
    if not validation["valid"]:
        return {
            "success": False,
            "stage": "validation",
            "issues": validation["issues"],
        }

    df = pd.read_csv(file_path)
    df = df.dropna(subset=[target_column])
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if y.nunique() < 2:
        return {
            "success": False,
            "stage": "validation",
            "issues": ["Target must contain at least two classes"],
        }

    # Require enough samples in every class for repeated stratified splits.
    if y.value_counts().min() < 2:
        return {
            "success": False,
            "stage": "validation",
            "issues": ["Each target class needs at least two samples"],
        }

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

    base_params = {"n_estimators": 100}
    base_params.update(baseline_params or {})
    treat_params = {"n_estimators": 200}
    treat_params.update(treatment_params or {})

    if base_params == treat_params:
        return {
            "success": False,
            "stage": "configuration",
            "issues": ["Baseline and treatment parameters must differ"],
        }

    baseline_scores = []
    treatment_scores = []

    for seed in range(n_seeds):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )

        for params, scores in (
            (base_params, baseline_scores),
            (treat_params, treatment_scores),
        ):
            model = Pipeline([
                ("preprocessing", clone(preprocessing)),
                ("classifier", RandomForestClassifier(
                    **params,
                    random_state=seed,
                )),
            ])
            model.fit(X_train, y_train)
            scores.append(float(accuracy_score(
                y_test, model.predict(X_test)
            )))

    analysis = paired_analysis(
        baseline_scores,
        treatment_scores,
        n_train=len(X_train),
        n_test=len(X_test),
    )

    return {
        "success": True,
        "result_label": "executed",
        "dataset": Path(file_path).name,
        "target_column": target_column,
        "baseline_params": base_params,
        "treatment_params": treat_params,
        "baseline": baseline_scores,
        "treatment": treatment_scores,
        "n_seeds": n_seeds,
        "analysis": analysis,
    }
