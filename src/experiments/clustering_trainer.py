
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def train_clustering_models(
    file_path: str | Path,
    n_clusters: int = 3,
    random_state: int = 42,
) -> dict:
    """Run and compare four unsupervised clustering algorithms."""

    try:
        df = pd.read_csv(file_path)

        if len(df) < 3:
            return {
                "success": False,
                "stage": "validation",
                "issues": ["At least 3 rows are required"],
            }

        if n_clusters < 2 or n_clusters >= len(df):
            return {
                "success": False,
                "stage": "validation",
                "issues": ["n_clusters must be between 2 and rows - 1"],
            }

        # Remove columns with no useful information.
        X = df.dropna(axis=1, how="all")
        X = X.loc[:, X.nunique(dropna=True) > 1]

        if X.empty:
            return {
                "success": False,
                "stage": "validation",
                "issues": ["No usable feature columns found"],
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
                "issues": ["No usable features found"],
            }

        preprocessing = ColumnTransformer(transformers)
        X_processed = preprocessing.fit_transform(X)

        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()

        X_processed = np.asarray(X_processed, dtype=float)

        if not np.isfinite(X_processed).all():
            return {
                "success": False,
                "stage": "validation",
                "issues": ["Preprocessed data contains non-finite values"],
            }

        models = {
            "KMeans": KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10,
            ),
            "DBSCAN": DBSCAN(
                eps=0.5,
                min_samples=5,
            ),
            "Agglomerative": AgglomerativeClustering(
                n_clusters=n_clusters,
            ),
            "GaussianMixture": GaussianMixture(
                n_components=n_clusters,
                random_state=random_state,
            ),
        }

        results = []
        failures = []

        for name, model in models.items():
            try:
                if name == "GaussianMixture":
                    model.fit(X_processed)
                    labels = model.predict(X_processed)
                else:
                    labels = model.fit_predict(X_processed)

                labels = np.asarray(labels)
                unique_labels = set(labels)
                cluster_count = len(unique_labels - {-1})

                # DBSCAN uses -1 to identify noise.
                valid_mask = labels != -1
                valid_labels = labels[valid_mask]
                valid_data = X_processed[valid_mask]

                if (
                    len(set(valid_labels)) < 2
                    or len(valid_labels) <= len(set(valid_labels))
                ):
                    failures.append({
                        "model": name,
                        "error": "Not enough valid clusters for evaluation",
                    })
                    continue

                silhouette = silhouette_score(
                    valid_data, valid_labels
                )
                davies_bouldin = davies_bouldin_score(
                    valid_data, valid_labels
                )
                calinski_harabasz = calinski_harabasz_score(
                    valid_data, valid_labels
                )

                results.append({
                    "model": name,
                    "clusters": cluster_count,
                    "noise_points": int(np.sum(labels == -1)),
                    "silhouette_score": float(silhouette),
                    "davies_bouldin_score": float(davies_bouldin),
                    "calinski_harabasz_score": float(calinski_harabasz),
                })

            except (ValueError, TypeError, FloatingPointError) as exc:
                failures.append({
                    "model": name,
                    "error": str(exc),
                })

        if not results:
            return {
                "success": False,
                "stage": "clustering",
                "issues": failures,
            }

        results.sort(
            key=lambda item: item["silhouette_score"],
            reverse=True,
        )

        return {
            "success": True,
            "dataset": Path(file_path).name,
            "task": "clustering",
            "models": results,
            "best_model": results[0]["model"],
            "rows": len(X_processed),
            "failures": failures,
        }

    except (ValueError, TypeError, OSError) as exc:
        return {
            "success": False,
            "stage": "clustering",
            "issues": [str(exc)],
        }
