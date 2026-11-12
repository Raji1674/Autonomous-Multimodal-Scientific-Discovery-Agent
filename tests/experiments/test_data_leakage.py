
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def create_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])


def test_train_test_split_has_no_overlap():
    X = pd.DataFrame({
        "age": range(100),
        "cholesterol": range(200, 300),
    })

    y = pd.Series([0, 1] * 50)

    X_train, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_preprocessing_is_fitted_only_on_training_data():
    X_train = pd.DataFrame({
        "feature": [10.0, 20.0, 30.0, np.nan],
    })

    X_test = pd.DataFrame({
        "feature": [1000.0, 2000.0],
    })

    pipeline = create_pipeline()
    pipeline.fit(X_train)

    # Imputer must learn the training mean, not the combined mean.
    learned_mean = pipeline.named_steps["imputer"].statistics_[0]

    assert learned_mean == 20.0


def test_test_data_does_not_change_fitted_scaler():
    X_train = pd.DataFrame({
        "feature": [10.0, 20.0, 30.0],
    })

    X_test = pd.DataFrame({
        "feature": [1000.0, 2000.0],
    })

    pipeline = create_pipeline()
    pipeline.fit(X_train)

    learned_mean = pipeline.named_steps["scaler"].mean_[0]
    learned_scale = pipeline.named_steps["scaler"].scale_[0]

    assert learned_mean == 20.0
    assert np.isclose(learned_scale, np.std([10.0, 20.0, 30.0]))


def test_pipeline_transforms_test_data_without_refitting():
    X_train = pd.DataFrame({
        "feature": [10.0, 20.0, 30.0],
    })

    X_test = pd.DataFrame({
        "feature": [40.0, 50.0],
    })

    pipeline = create_pipeline()
    pipeline.fit(X_train)

    original_mean = pipeline.named_steps["scaler"].mean_.copy()

    transformed_test = pipeline.transform(X_test)

    assert transformed_test.shape == (2, 1)
    assert np.array_equal(
        pipeline.named_steps["scaler"].mean_,
        original_mean,
    )
