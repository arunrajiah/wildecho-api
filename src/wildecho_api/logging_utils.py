"""Structured logging: a per-request ID and a JSON line formatter.

Every request gets an ID: reused from the client's request-ID header if it sent
one, otherwise freshly generated. It is bound to a :class:`~contextvars.ContextVar`
for the lifetime of the request by the middleware in ``main.py``, attached to
*every* log line emitted anywhere during that request via a ``logging.Filter`` (no
need to thread it through individual log calls), returned as a response header,
and included in the body of ``POST /v1/identify``.

That last part is what this exists for: a client can hand the ID from an
identification back to ``POST /v1/feedback`` when correcting it, and an operator
debugging a misidentification can grep the JSON logs for that exact ID to see the
whole request's server-side trail.

ASGI servers run each request in its own ``asyncio.Task``, and a fresh Task gets an
independent copy of the current context, so concurrent requests never see each
other's request ID even though this is process-wide, mutable module state.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextvars import ContextVar

_request_id_var: ContextVar[str | None] = ContextVar("wildecho_request_id", default=None)

#: What untraced log lines (startup, background tasks) show instead of an ID.
NO_REQUEST_ID = "-"


def new_request_id() -> str:
    """A fresh opaque ID. Not a UUID for cryptographic reasons, just for uniqueness."""
    return uuid.uuid4().hex


def set_request_id(request_id: str) -> None:
    _request_id_var.set(request_id)


def get_request_id() -> str | None:
    return _request_id_var.get()


class RequestIdFilter(logging.Filter):
    """Attaches the currently-bound request ID to every :class:`LogRecord`."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or NO_REQUEST_ID  # type: ignore[attr-defined]
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line. What hosted log aggregators (Fly, Render, ...) want."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", NO_REQUEST_ID),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable console output for local development."""

    def __init__(self) -> None:
        super().__init__(fmt="%(asctime)s %(levelname)-8s [%(request_id)s] %(name)s: %(message)s")


def configure_logging(log_level: str, log_format: str) -> None:
    """Set up the root logger once at process startup.

    Replaces any prior handlers so repeated calls (e.g. across test runs that
    re-enter the app lifespan) do not accumulate duplicate log lines.
    """
    formatter: logging.Formatter = JsonFormatter() if log_format == "json" else TextFormatter()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)
