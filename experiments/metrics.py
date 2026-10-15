
import json
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def calculate_metrics():
    # Load the sample dataset
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    # Split dataset consistently
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Train the model
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )
    model.fit(X_train, y_train)

    # Predict test records
    predictions = model.predict(X_test)

    # Calculate metrics
    results = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "model": "Logistic Regression",
        "accuracy": round(
            float(accuracy_score(y_test, predictions)), 4
        ),
        "precision": round(
            float(precision_score(y_test, predictions, zero_division=0)), 4
        ),
        "recall": round(
            float(recall_score(y_test, predictions, zero_division=0)), 4
        ),
        "f1_score": round(
            float(f1_score(y_test, predictions, zero_division=0)), 4
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
    }

    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "metrics.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("Evaluation completed successfully!")
    print(json.dumps(results, indent=4))
    print(f"Metrics saved to: {output_path}")


if __name__ == "__main__":
    calculate_metrics()
