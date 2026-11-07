import requests
import pytest

from src.experiments.dataset_search import search_datasets


def test_empty_keyword():
    assert search_datasets("") == []


def test_whitespace_keyword():
    assert search_datasets("   ") == []


def test_successful_dataset_search(monkeypatch):
    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                {"id": "user/dataset1", "downloads": 100},
                {"id": "user/dataset2", "downloads": 50},
            ]

    def mock_get(*args, **kwargs):
        assert kwargs["params"]["search"] == "heart disease"
        assert kwargs["params"]["limit"] == 10
        assert kwargs["timeout"] == 20
        return MockResponse()

    monkeypatch.setattr(
        "src.experiments.dataset_search.requests.get", mock_get
    )

    result = search_datasets(" heart disease ")

    assert len(result) == 2
    assert result[0]["name"] == "user/dataset1"
    assert result[0]["downloads"] == 100
    assert result[0]["url"] == (
        "https://huggingface.co/datasets/user/dataset1"
    )


@pytest.mark.parametrize("response_data", [
    None,
    {},
    "invalid response",
    123,
])
def test_invalid_api_response(monkeypatch, response_data):
    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return response_data

    monkeypatch.setattr(
        "src.experiments.dataset_search.requests.get",
        lambda *args, **kwargs: MockResponse(),
    )

    assert search_datasets("health") == []


def test_invalid_dataset_entries_are_skipped(monkeypatch):
    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                None,
                "invalid",
                {},
                {"id": ""},
                {"id": 123},
                {"id": "valid/dataset"},
            ]

    monkeypatch.setattr(
        "src.experiments.dataset_search.requests.get",
        lambda *args, **kwargs: MockResponse(),
    )

    result = search_datasets("health")

    assert len(result) == 1
    assert result[0]["name"] == "valid/dataset"
    assert result[0]["downloads"] == 0


@pytest.mark.parametrize(
    "given_limit, expected_limit",
    [
        (0, 1),
        (-5, 1),
        (10, 10),
        (100, 50),
    ],
)
def test_limit_is_clamped(monkeypatch, given_limit, expected_limit):
    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return []

    def mock_get(*args, **kwargs):
        assert kwargs["params"]["limit"] == expected_limit
        return MockResponse()

    monkeypatch.setattr(
        "src.experiments.dataset_search.requests.get", mock_get
    )

    assert search_datasets("health", limit=given_limit) == []


def test_http_error_returns_empty_list(monkeypatch):
    def mock_get(*args, **kwargs):
        raise requests.HTTPError("Server error")

    monkeypatch.setattr(
        "src.experiments.dataset_search.requests.get", mock_get
    )

    assert search_datasets("health") == []


def test_network_error_returns_empty_list(monkeypatch):
    def mock_get(*args, **kwargs):
        raise requests.ConnectionError("Connection failed")

    monkeypatch.setattr(
        "src.experiments.dataset_search.requests.get", mock_get
    )

    assert search_datasets("health") == []


def test_invalid_json_returns_empty_list(monkeypatch):
    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            raise ValueError("Invalid JSON")

    monkeypatch.setattr(
        "src.experiments.dataset_search.requests.get",
        lambda *args, **kwargs: MockResponse(),
    )

    assert search_datasets("health") == []


def test_type_error_returns_empty_list(monkeypatch):
    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            raise TypeError("Invalid response type")

    monkeypatch.setattr(
        "src.experiments.dataset_search.requests.get",
        lambda *args, **kwargs: MockResponse(),
    )

    assert search_datasets("health") == []