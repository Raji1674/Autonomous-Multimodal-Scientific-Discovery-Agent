
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


def train_regression_models(
    file_path: str | Path,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train and compare seven regression algorithms."""

    try:
        df = pd.read_csv(file_path)

        if target_column not in df.columns:
            return {
                "success": False,
                "stage": "validation",
                "issues": [f"Target column '{target_column}' not found"],
            }

        df = df.dropna(subset=[target_column])

        if len(df) < 5:
            return {
                "success": False,
                "stage": "validation",
                "issues": ["At least 5 labelled rows are required"],
            }

        X = df.drop(columns=[target_column])
        y = pd.to_numeric(df[target_column], errors="coerce")

        valid_target = y.notna()
        X, y = X.loc[valid_target], y.loc[valid_target]

        if len(y) < 5:
            return {
                "success": False,
                "stage": "validation",
                "issues": ["At least 5 numeric target values are required"],
            }

        if X.shape[1] == 0:
            return {
                "success": False,
                "stage": "validation",
                "issues": ["At least one feature column is required"],
            }

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

        if not transformers:
            return {
                "success": False,
                "stage": "validation",
                "issues": ["No usable feature columns found"],
            }

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "DecisionTree": RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
            ),
            "RandomForest": RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
            ),
            "GradientBoosting": GradientBoostingRegressor(
                random_state=random_state,
            ),
            "SVR": SVR(),
        }

        # Replace the DecisionTree entry with an actual decision tree.
        from sklearn.tree import DecisionTreeRegressor
        models["DecisionTree"] = DecisionTreeRegressor(
            random_state=random_state
        )

        results = []
        failures = []

        for name, estimator in models.items():
            pipeline = Pipeline([
                ("preprocessing", ColumnTransformer(transformers)),
                ("regressor", estimator),
            ])

            try:
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)

                mse = mean_squared_error(y_test, predictions)

                results.append({
                    "model": name,
                    "mae": float(mean_absolute_error(y_test, predictions)),
                    "rmse": float(mse ** 0.5),
                    "r2": float(r2_score(y_test, predictions)),
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

        results.sort(key=lambda item: item["rmse"])

        return {
            "success": True,
            "dataset": Path(file_path).name,
            "target_column": target_column,
            "task": "regression",
            "models": results,
            "best_model": results[0]["model"],
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "failures": failures,
        }

    except (ValueError, TypeError, OSError) as exc:
        return {
            "success": False,
            "stage": "training",
            "issues": [str(exc)],
        }
