from unittest.mock import call

from src.ingestion import scheduler
import pytest

def test_make_job_runs_ingestion_for_each_query(monkeypatch):
    calls = []

    def fake_run_ingestion(query, store, max_per_source):
        calls.append((query, store, max_per_source))

    monkeypatch.setattr(scheduler, "run_ingestion", fake_run_ingestion)

    store = object()
    job = scheduler.make_job(
        store,
        queries=["student performance", "learning analytics"],
        max_per_source=10,
    )

    job()

    assert calls == [
        ("student performance", store, 10),
        ("learning analytics", store, 10),
    ]


def test_make_job_with_no_queries_does_nothing(monkeypatch):
    calls = []

    monkeypatch.setattr(
        scheduler,
        "run_ingestion",
        lambda *args, **kwargs: calls.append(args),
    )

    job = scheduler.make_job(object(), queries=[])
    job()

    assert calls == []


def test_run_forever_runs_immediately_and_registers_daily_job(monkeypatch):
    calls = []
    scheduled = {}

    class StopLoop(Exception):
        pass

    class FakeSchedule:
        @property
        def day(self):
            return self

        def at(self, time):
            scheduled["time"] = time
            return self

        def do(self, job):
            scheduled["job"] = job
            return job

    monkeypatch.setattr(
        scheduler.schedule,
        "every",
        lambda: FakeSchedule(),
    )

    def stop_loop():
        raise StopLoop

    monkeypatch.setattr(scheduler.schedule, "run_pending", stop_loop)

    monkeypatch.setattr(
        scheduler,
        "run_ingestion",
        lambda query, store, max_per_source: calls.append(query),
    )

    with pytest.raises(StopLoop):
        scheduler.run_forever(
            store=object(),
            queries=["student performance"],
            at="03:30",
        )

    # The job runs once immediately when the scheduler starts.
    assert calls == ["student performance"]

    # The job is also registered for the requested daily time.
    assert scheduled["time"] == "03:30"
    assert callable(scheduled["job"])

    # Verify the registered job performs ingestion when called.
    scheduled["job"]()
    assert calls == ["student performance", "student performance"]

def test_make_job_continues_after_one_query_fails(monkeypatch):
    calls = []

    def fake_run_ingestion(query, store, max_per_source):
        calls.append(query)

        if query == "failing query":
            raise RuntimeError("Simulated ingestion failure")

    monkeypatch.setattr(scheduler, "run_ingestion", fake_run_ingestion)

    job = scheduler.make_job(
        store=object(),
        queries=["failing query", "working query"],
    )

    job()

    assert calls == ["failing query", "working query"]
