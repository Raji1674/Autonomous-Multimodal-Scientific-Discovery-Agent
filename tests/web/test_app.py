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
