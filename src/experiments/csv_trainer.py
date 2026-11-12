
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from src.experiments.dataset_validator import validate_csv_dataset


def train_csv_dataset(
    file_path: str | Path,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Validate a CSV and train a baseline classification model."""

    validation = validate_csv_dataset(file_path, target_column)

    if not validation["valid"]:
        return {
            "success": False,
            "stage": "validation",
            "issues": validation["issues"],
        }

    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Remove rows with missing target labels.
    labelled = y.notna()
    X, y = X.loc[labelled], y.loc[labelled]

    if y.nunique() < 2:
        return {
            "success": False,
            "stage": "validation",
            "issues": ["Target must contain at least two classes after cleaning"],
        }

    # Avoid stratification when a class has too few samples.
    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessing = ColumnTransformer(transformers)

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
        )),
    ])

    try:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    except (ValueError, TypeError) as exc:
        return {
            "success": False,
            "stage": "training",
            "issues": [str(exc)],
        }

    return {
        "success": True,
        "dataset": Path(file_path).name,
        "target_column": target_column,
        "model": "RandomForestClassifier",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(
            y_test, predictions, average="macro", zero_division=0
        )),
        "validation": validation,
    }
def train_all_models(
    file_path: str | Path,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train and compare multiple classification models."""

    validation = validate_csv_dataset(file_path, target_column)

    if not validation["valid"]:
        return {
            "success": False,
            "stage": "validation",
            "issues": validation["issues"],
        }

    try:
        df = pd.read_csv(file_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        labelled = y.notna()
        X, y = X.loc[labelled], y.loc[labelled]

        if y.nunique() < 2:
            return {
                "success": False,
                "stage": "validation",
                "issues": ["Target must contain at least two classes"],
            }

        stratify = y if y.value_counts().min() >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        numeric_columns = X.select_dtypes(include="number").columns.tolist()
        categorical_columns = X.select_dtypes(
            exclude="number"
        ).columns.tolist()

        transformers = []

        if numeric_columns:
            transformers.append((
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_columns,
            ))

        if categorical_columns:
            transformers.append((
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_columns,
            ))

        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "DecisionTree": DecisionTreeClassifier(
                random_state=random_state
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
            ),
            "SVM": SVC(),
        }

        results = []
        failures = []

        for name, classifier in models.items():
            model = Pipeline([
                ("preprocessing", ColumnTransformer(transformers)),
                ("classifier", classifier),
            ])

            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                results.append({
                    "model": name,
                    "accuracy": float(
                        accuracy_score(y_test, predictions)
                    ),
                    "f1_macro": float(
                        f1_score(
                            y_test,
                            predictions,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                })

            except (ValueError, TypeError) as exc:
                failures.append({
                    "model": name,
                    "error": str(exc),
                })

        if not results:
            return {
                "success": False,
                "stage": "training",
                "issues": failures,
            }

        results.sort(
            key=lambda item: item["f1_macro"],
            reverse=True,
        )

        return {
            "success": True,
            "dataset": Path(file_path).name,
            "target_column": target_column,
            "models": results,
            "best_model": results[0]["model"],
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "failures": failures,
            "validation": validation,
        }

    except (ValueError, TypeError, OSError) as exc:
        return {
            "success": False,
            "stage": "training",
            "issues": [str(exc)],
        }
