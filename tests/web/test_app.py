import re

import io

import pytest

from web.app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("web.app.UPLOAD_DIR", tmp_path)
    app.config.update(
        TESTING=True,
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    )

    with app.test_client() as test_client:
        yield test_client


def test_home_page_loads(client):
    response = client.get("/")

    assert response.status_code == 200


def test_upload_rejects_missing_file(client):
    response = client.post("/", data={})

    assert response.status_code == 200
    assert b"Please select a CSV file" in response.data


def test_upload_rejects_non_csv_file(client):
    response = client.post(
        "/",
        data={
            "csv_file": (io.BytesIO(b"not a csv"), "dataset.txt"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Only CSV files are supported" in response.data


def test_upload_rejects_invalid_csv(client):
    response = client.post(
        "/",
        data={
            "csv_file": (
                io.BytesIO(b"feature,target\n1,0\n2,0\n"),
                "dataset.csv",
            ),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Select target column" in response.data

    html = response.get_data(as_text=True)
    match = re.search(
        r'name="upload_id" value="([a-f0-9]{32})"',
        html,
    )
    assert match is not None

    upload_id = match.group(1)

    validation_response = client.post(
        "/",
        data={
            "upload_id": upload_id,
            "target_column": "target",
        },
    )

    assert validation_response.status_code == 200
    assert b"at least two classes" in validation_response.data.lower()

def test_research_without_csv(client, monkeypatch):
    class FakeService:
        def run(self, question):
            return {"ok": True, "report": "Research completed"}

    monkeypatch.setattr(
        "web.app.get_research_service",
        lambda: FakeService(),
    )

    response = client.post(
        "/research",
        data={"research_question": "Predict student performance"},
    )

    assert response.status_code == 200
    assert b"Research completed" in response.data


def test_research_with_validated_csv(client, monkeypatch):
    from web.app import UPLOAD_DIR

    upload_id = "a" * 32
    csv_path = UPLOAD_DIR / f"{upload_id}.csv"

    csv_path.write_text(
        "study_hours,target\n2,0\n4,0\n6,1\n8,1\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "web.app.validate_csv_dataset",
        lambda path, target: {"valid": True, "issues": []},
    )

    class FakeService:
        def run(self, question, dataset_path=None, target_column=None):
            assert question == "Predict student performance"
            assert dataset_path == str(csv_path)
            assert target_column == "target"
            return {"ok": True, "report": "CSV experiment completed"}

    monkeypatch.setattr(
        "web.app.get_research_service",
        lambda: FakeService(),
    )

    response = client.post(
        "/research",
        data={
            "research_question": "Predict student performance",
            "upload_id": upload_id,
            "target_column": "target",
        },
    )

    assert response.status_code == 200
    assert b"CSV experiment completed" in response.data


def test_research_rejects_invalid_upload_id(client):
    response = client.post(
        "/research",
        data={
            "research_question": "Predict student performance",
            "upload_id": "invalid-id",
            "target_column": "target",
        },
    )

    assert response.status_code == 200
    assert b"Invalid dataset reference" in response.data


def test_research_rejects_missing_csv(client, monkeypatch):
    upload_id = "b" * 32

    class FakeService:
        def run(self, question):
            return {"ok": True}

    monkeypatch.setattr(
        "web.app.get_research_service",
        lambda: FakeService(),
    )

    response = client.post(
        "/research",
        data={
            "research_question": "Predict student performance",
            "upload_id": upload_id,
            "target_column": "target",
        },
    )

    assert response.status_code == 200
    assert b"Uploaded CSV not found" in response.data