import pandas as pd

from src.experiments.dataset_validator import validate_csv_dataset


def test_valid_csv_dataset(tmp_path):
    file_path = tmp_path / "valid.csv"
    pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "target": [0, 0, 1, 1],
    }).to_csv(file_path, index=False)

    result = validate_csv_dataset(file_path, "target")

    assert result["valid"] is True
    assert result["rows"] == 4
    assert result["columns"] == 2
    assert result["target_classes"] == 2
    assert result["issues"] == []


def test_missing_file():
    result = validate_csv_dataset("missing.csv", "target")

    assert result["valid"] is False
    assert "CSV file not found" in result["issues"]


def test_invalid_file_extension(tmp_path):
    file_path = tmp_path / "data.txt"
    file_path.write_text("feature,target\n1,0\n2,1\n")

    result = validate_csv_dataset(file_path, "target")

    assert result["valid"] is False
    assert "Only CSV files are supported" in result["issues"]


def test_missing_target_column(tmp_path):
    file_path = tmp_path / "data.csv"
    pd.DataFrame({"feature": [1, 2, 3]}).to_csv(file_path, index=False)

    result = validate_csv_dataset(file_path, "target")

    assert result["valid"] is False
    assert any("Target column" in issue for issue in result["issues"])


def test_missing_feature_values_are_allowed(tmp_path):
    file_path = tmp_path / "data.csv"

    pd.DataFrame({
        "feature": [1, None, 3],
        "target": [0, 1, 1],
    }).to_csv(file_path, index=False)

    result = validate_csv_dataset(file_path, "target")

    assert result["valid"] is True
    assert result["missing_values"] == 1
    assert result["missing_feature_values"] == 1
    assert result["missing_target_values"] == 0


def test_duplicate_rows(tmp_path):
    file_path = tmp_path / "data.csv"
    pd.DataFrame({
        "feature": [1, 1, 3],
        "target": [0, 0, 1],
    }).to_csv(file_path, index=False)

    result = validate_csv_dataset(file_path, "target")

    assert result["valid"] is False
    assert result["duplicate_rows"] == 1


def test_target_with_one_class(tmp_path):
    file_path = tmp_path / "data.csv"
    pd.DataFrame({
        "feature": [1, 2, 3],
        "target": [1, 1, 1],
    }).to_csv(file_path, index=False)

    result = validate_csv_dataset(file_path, "target")

    assert result["valid"] is False
    assert any("at least two classes" in issue for issue in result["issues"])