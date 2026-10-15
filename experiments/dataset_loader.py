
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_sample_dataset():
    """Load a sample dataset for initial ML testing."""
    dataset = load_breast_cancer(as_frame=True)

    df = dataset.data.copy()
    df["target"] = dataset.target

    return df, "target"


def inspect_dataset(df, target_column):
    """Check the dataset before running an experiment."""
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    return {
        "rows": int(df.shape[0]),
        "features": int(df.shape[1] - 1),
        "target_column": target_column,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {
            str(key): int(value)
            for key, value in df[target_column]
            .value_counts().items()
        },
    }


if __name__ == "__main__":
    df, target = load_sample_dataset()
    summary = inspect_dataset(df, target)

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "sample.csv"
    df.to_csv(output_path, index=False)

    print("Dataset loaded successfully!")
    print(f"Saved to: {output_path}")
    print(summary)
    print("\nFirst five rows:")
    print(df.head())
