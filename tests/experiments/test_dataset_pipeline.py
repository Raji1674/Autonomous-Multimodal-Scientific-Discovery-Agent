
from src.experiments import dataset_pipeline


def test_pipeline_reports_no_datasets(monkeypatch, tmp_path):
    monkeypatch.setattr(
        dataset_pipeline,
        "search_datasets",
        lambda keyword, limit=10: [],
    )

    result = dataset_pipeline.download_and_train(
        keyword="sample",
        target_column="target",
        output_dir=tmp_path,
    )

    assert result["success"] is False
    assert result["stage"] == "search"
    assert result["issues"]


def test_pipeline_can_return_search_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        dataset_pipeline,
        "search_datasets",
        lambda keyword, limit=10: [],
    )

    result = dataset_pipeline.download_and_train(
        keyword="unknown",
        target_column="label",
        output_dir=tmp_path,
    )

    assert result["success"] is False
    assert "No datasets found" in result["issues"][0]


def test_pipeline_downloads_and_trains(monkeypatch, tmp_path):
    from src.experiments import dataset_pipeline


    class FakeResponse:
        content = b"feature,target\n1,0\n2,0\n3,0\n4,0\n5,1\n6,1\n7,1\n8,1\n"

        def raise_for_status(self):
            pass

        def json(self):
            return [{"path": "data.csv", "type": "file"}]

    monkeypatch.setattr(
        dataset_pipeline,
        "search_datasets",
        lambda keyword, limit=10: [
            {
                "name": "test/sample",
                "downloads": 10,
                "url": "https://huggingface.co/datasets/test/sample",
            }
        ],
    )
    monkeypatch.setattr(
        dataset_pipeline.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )
    monkeypatch.setattr(
        dataset_pipeline,
        "train_csv_dataset",
        lambda path, target_column: {
            "success": True,
            "accuracy": 0.9,
            "issues": [],
        },
    )

    result = dataset_pipeline.download_and_train(
        keyword="sample",
        target_column="target",
        output_dir=tmp_path,
    )

    assert result["success"] is True
    assert result["dataset_id"] == "test/sample"
    assert result["accuracy"] == 0.9
    assert not list(tmp_path.glob("*.csv"))
