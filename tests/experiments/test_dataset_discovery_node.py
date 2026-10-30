
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


def test_discovery_rejects_missing_target_column():
    result = dataset_discovery.dataset_discovery_node({
        "research_question": "Study a classification problem"
    })

    assert result["failure_state"] == "DATASET_DISCOVERY_FAILED"
    assert "Target column" in result["error"]


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
