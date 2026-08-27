"""Structured logging configuration.

Development renders human-readable coloured lines; production emits one JSON object per
event so that logs are queryable in Loki/CloudWatch without regex parsing. The
``request_id`` context variable is bound by middleware and therefore appears on every log
line emitted while handling a request, including from deep inside services.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_ctx: ContextVar[str | None] = ContextVar("user_id", default=None)


def _inject_context(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Attach ambient request/user identifiers to every event."""
    if (rid := request_id_ctx.get()) is not None:
        event_dict.setdefault("request_id", rid)
    if (uid := user_id_ctx.get()) is not None:
        event_dict.setdefault("user_id", uid)
    return event_dict


def configure_logging(*, level: str = "INFO", json_output: bool = False) -> None:
    """Initialise structlog and route stdlib logging through it."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
        force=True,
    )

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _inject_context,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Uvicorn's own access log duplicates our middleware log; silence it.
    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.access").propagate = False


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for ``name``."""
    return structlog.stdlib.get_logger(name)
