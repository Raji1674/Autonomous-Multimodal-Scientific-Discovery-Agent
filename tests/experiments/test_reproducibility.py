
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def run_experiment(seed: int) -> dict:
    """Run Logistic Regression using a reproducible random seed."""

    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )

    model = LogisticRegression(
        C=1.0,
        max_iter=5000,
        random_state=seed,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "seed": seed,
        "accuracy": accuracy_score(y_test, predictions),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "predictions": predictions.tolist(),
    }


def test_same_seed_produces_same_results():
    first = run_experiment(seed=42)
    second = run_experiment(seed=42)

    assert first["accuracy"] == second["accuracy"]
    assert first["train_rows"] == second["train_rows"]
    assert first["test_rows"] == second["test_rows"]
    assert first["predictions"] == second["predictions"]


def test_different_seeds_are_executable():
    first = run_experiment(seed=42)
    second = run_experiment(seed=100)

    assert first["seed"] != second["seed"]
    assert 0 <= first["accuracy"] <= 1
    assert 0 <= second["accuracy"] <= 1


def test_dataset_split_has_no_overlap():
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data

    X_train, X_test = train_test_split(
        X,
        test_size=0.30,
        random_state=42,
    )

    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert len(X_train) + len(X_test) == len(X)
