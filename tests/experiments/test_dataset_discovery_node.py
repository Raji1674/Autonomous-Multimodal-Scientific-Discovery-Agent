
from src.workflow import dataset_discovery


def test_discovery_downloads_and_validates_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_discovery, "DOWNLOAD_DIR", tmp_path)

    monkeypatch.setattr(
        dataset_discovery,
        "search_datasets",
        lambda keyword, limit=10: [
            {
                "name": "test/sample",
                "downloads": 100,
                "url": "https://huggingface.co/datasets/test/sample",
            }
        ],
    )

    class FakeResponse:
        def __init__(self, content=None, data=None):
            self.content = content
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    responses = [
        FakeResponse(data=[
            {"path": "sample.csv", "type": "file"}
        ]),
        FakeResponse(
            content=(
                b"feature,target\n"
                b"1,0\n2,0\n3,0\n4,0\n"
                b"5,1\n6,1\n7,1\n8,1\n"
            )
        ),
    ]

    monkeypatch.setattr(
        dataset_discovery.requests,
        "get",
        lambda *args, **kwargs: responses.pop(0),
    )

    result = dataset_discovery.dataset_discovery_node({
        "research_question": "Can features predict the target?",
        "selected_hypothesis": {
            "statement": "Features are associated with the target"
        },
        "target_column": "target",
    })

    assert result["failure_state"] is None
    assert result["dataset_validation"]["valid"] is True
    assert result["selected_dataset"]["name"] == "test/sample"
    assert result["dataset_path"].endswith("test__sample__sample.csv")


def test_discovery_allows_missing_target_column(monkeypatch):
    monkeypatch.setattr(
        dataset_discovery,
        "search_datasets",
        lambda keyword, limit=10: [],
    )

    result = dataset_discovery.dataset_discovery_node({
        "research_question": "Study a classification problem"
    })

    assert result["failure_state"] == "DATASET_DISCOVERY_FAILED"
    assert "Target column" not in result["error"]
    assert "No Hugging Face datasets were found" in result["error"]


def test_discovery_handles_no_datasets(monkeypatch):
    monkeypatch.setattr(
        dataset_discovery,
        "search_datasets",
        lambda keyword, limit=10: [],
    )

    result = dataset_discovery.dataset_discovery_node({
        "research_question": "Study a classification problem",
        "target_column": "target",
    })

    assert result["failure_state"] == "DATASET_DISCOVERY_FAILED"
    assert result["dataset_candidates"] == []


def test_download_rejects_oversized_content_length(tmp_path):
    destination = tmp_path / "large.csv"

    class FakeResponse:
        headers = {"Content-Length": str(
            dataset_discovery.MAX_DOWNLOAD_BYTES + 1
        )}

    import pytest

    with pytest.raises(ValueError, match="100 MB"):
        dataset_discovery._download_limited(
            FakeResponse(), destination
        )

    assert not destination.exists()


def test_download_removes_partial_file_on_size_limit(tmp_path):
    destination = tmp_path / "large.csv"

    class FakeResponse:
        headers = {}

        def iter_content(self, chunk_size):
            yield b"a" * (dataset_discovery.MAX_DOWNLOAD_BYTES + 1)

    import pytest

    with pytest.raises(ValueError, match="100 MB"):
        dataset_discovery._download_limited(
            FakeResponse(), destination
        )

    assert not destination.exists()

