
import pandas as pd


def validate_dataset(df, target_column):
    issues = []

    # 1. Check whether dataset is empty
    if df.empty:
        issues.append("Dataset is empty.")

    # 2. Check target column
    if target_column not in df.columns:
        issues.append(
            f"Target column '{target_column}' does not exist."
        )
        return {
            "valid": False,
            "issues": issues
        }
    
    # Check whether the dataset has feature columns
    if df.shape[1] < 2:
        issues.append(
            "Dataset must contain at least one feature "
            "column and one target column."
        )


    # 3. Check missing values
    missing_count = int(df.isnull().sum().sum())
    if missing_count > 0:
        issues.append(
            f"Dataset contains {missing_count} missing values."
        )

    # 4. Check duplicate rows
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        issues.append(
            f"Dataset contains {duplicate_count} duplicate rows."
        )

    # 5. Check missing target labels
    missing_targets = int(df[target_column].isnull().sum())
    if missing_targets > 0:
        issues.append(
            f"Target column has {missing_targets} missing labels."
        )

    # 6. Check whether target has enough classes
    class_count = int(df[target_column].nunique())
    if class_count < 2:
        issues.append(
            "Target must have at least two distinct classes."
        )

    return {
        "valid": len(issues) == 0,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": target_column,
        "missing_values": missing_count,
        "duplicate_rows": duplicate_count,
        "target_classes": class_count,
        "issues": issues
    }


if __name__ == "__main__":
    from experiments.dataset_loader import load_sample_dataset

    df, target = load_sample_dataset()
    result = validate_dataset(df, target)

    print("Dataset Validation Result")
    print(result)
