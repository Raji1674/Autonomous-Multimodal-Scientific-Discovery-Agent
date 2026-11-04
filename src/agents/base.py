"""BaseAgent: logging, timing and error-to-failure-state mapping for every agent."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

from src.schemas.state import FailureState, ResearchState
from src.utils.logging import get_logger, log_event


class BaseAgent(ABC):
    name: str = "agent"
    failure_on_error: FailureState = FailureState.EXECUTION_FAILED

    def __init__(self) -> None:
        self.log = get_logger(f"agent.{self.name}")

    @abstractmethod
    def run(self, state: ResearchState) -> dict:
        """Return ONLY the state keys this agent changes."""

    def __call__(self, state: ResearchState) -> dict:
        start = time.perf_counter()
        try:
            update = self.run(state)
        except Exception as exc:  # noqa: BLE001 - an agent must never crash the pipeline
            update = {"failure_state": self.failure_on_error.value, "error": f"{type(exc).__name__}: {exc}"}
            log_event(self.log, self.name, "run", "error", time.perf_counter() - start, error=str(exc))
            return update
        status = "failed" if update.get("failure_state") else "ok"
        log_event(self.log, self.name, "run", status, time.perf_counter() - start,
                  error=update.get("error"), failure_state=update.get("failure_state"))
        return update
