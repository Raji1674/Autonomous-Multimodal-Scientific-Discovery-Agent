
from pathlib import Path
import json

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def run_experiment():
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    results = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "model": "Logistic Regression",
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "accuracy": round(float(accuracy), 4),
        "random_state": 42,
    }

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "experiment_results.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("Experiment completed successfully!")
    print(json.dumps(results, indent=4))
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    run_experiment()
