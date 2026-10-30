"""Structured JSON logging with secret redaction."""
from __future__ import annotations

import json
import logging
import re
import sys
import time

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"gsk_[A-Za-z0-9]{10,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*\S+"),
]


def redact(text: str) -> str:
    """Replace anything that looks like an API key or secret."""
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "event", {}))
        return redact(json.dumps(payload, default=str))


def get_logger(name: str = "autosci") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, agent: str, action: str, status: str,
              duration: float | None = None, error: str | None = None, **extra) -> None:
    """Log one structured agent event."""
    event = {"agent": agent, "action": action, "status": status,
             "duration": None if duration is None else round(duration, 4), "error": error}
    event.update(extra)
    logger.info(f"{agent}:{action}:{status}", extra={"event": event})
