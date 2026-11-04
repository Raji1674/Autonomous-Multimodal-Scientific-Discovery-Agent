"""Daily ingestion job (cron-style) using the `schedule` library."""
from __future__ import annotations

import time

import schedule

from src.ingestion.pipeline import run_ingestion
from src.vectorstore.base import VectorStore


def make_job(store: VectorStore, queries: list[str], max_per_source: int = 50):
    def job() -> None:
        for q in queries:
            run_ingestion(q, store, max_per_source=max_per_source)
    return job


def run_forever(store: VectorStore, queries: list[str], at: str = "02:00") -> None:  # pragma: no cover
    job = make_job(store, queries)
    job()  # run once at start
    schedule.every().day.at(at).do(job)
    while True:
        schedule.run_pending()
        time.sleep(60)
