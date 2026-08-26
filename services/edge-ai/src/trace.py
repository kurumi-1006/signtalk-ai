from __future__ import annotations

import logging
from typing import Any


def trace(
    logger: logging.Logger,
    trace_id: str,
    step: str,
    title: str,
    **details: Any,
) -> None:
    """Print one readable, multi-line prediction trace block to the console."""
    logger.info('[TRACE][%s] STEP %s | %s', trace_id, step, title)
    for key, value in details.items():
        logger.info('[TRACE][%s]          %s = %s', trace_id, key, value)
