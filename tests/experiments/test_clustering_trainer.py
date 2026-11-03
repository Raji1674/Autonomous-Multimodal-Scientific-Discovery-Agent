import pandas as pd

from src.experiments.clustering_trainer import train_clustering_models


def test_clustering_trainer_returns_comparison(tmp_path):
    csv_path = tmp_path / "clusters.csv"

    data = []
    for i in range(30):
        data.append({
            "feature1": i,
            "feature2": i * 1.2,
        })

    for i in range(30):
        data.append({
            "feature1": 100 + i,
            "feature2": 120 + i * 1.2,
        })

    for i in range(30):
        data.append({
            "feature1": 200 + i,
            "feature2": 240 + i * 1.2,
        })

    pd.DataFrame(data).to_csv(csv_path, index=False)

    result = train_clustering_models(csv_path, n_clusters=3)

    assert result["success"] is True
    assert result["task"] == "clustering"
    assert result["best_model"] in [
        "KMeans",
        "DBSCAN",
        "Agglomerative",
        "GaussianMixture",
    ]
    assert len(result["models"]) >= 1

    for model in result["models"]:
        assert -1 <= model["silhouette_score"] <= 1
        assert model["davies_bouldin_score"] >= 0
        assert model["calinski_harabasz_score"] >= 0


def test_clustering_trainer_rejects_small_dataset(tmp_path):
    csv_path = tmp_path / "small.csv"

    pd.DataFrame({
        "feature1": [1, 2],
        "feature2": [3, 4],
    }).to_csv(csv_path, index=False)

    result = train_clustering_models(csv_path)

    assert result["success"] is False
    assert result["stage"] == "validation"