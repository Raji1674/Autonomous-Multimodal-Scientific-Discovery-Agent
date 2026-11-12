
from unittest.mock import patch

from src.ingestion.scheduler import make_job


def test_make_job_runs_all_queries():
    queries = ["machine learning", "cardiovascular disease"]

    with patch("src.ingestion.scheduler.run_ingestion") as mock_run:
        job = make_job(store="fake-store", queries=queries)
        job()

    assert mock_run.call_count == 2


def test_make_job_passes_store_and_query():
    store = "fake-store"
    queries = ["AI research"]

    with patch("src.ingestion.scheduler.run_ingestion") as mock_run:
        job = make_job(store=store, queries=queries)
        job()

    mock_run.assert_called_once_with(
        "AI research",
        store,
        max_per_source=50,
    )


def test_make_job_custom_max_per_source():
    with patch("src.ingestion.scheduler.run_ingestion") as mock_run:
        job = make_job(
            store="fake-store",
            queries=["deep learning"],
            max_per_source=10,
        )
        job()

    mock_run.assert_called_once_with(
        "deep learning",
        "fake-store",
        max_per_source=10,
    )


def test_make_job_with_empty_queries():
    with patch("src.ingestion.scheduler.run_ingestion") as mock_run:
        job = make_job(store="fake-store", queries=[])
        job()

    mock_run.assert_not_called()
