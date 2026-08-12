"""Centralized logging configuration.

Every module calls ``get_logger(__name__)`` instead of using the root logger,
so log lines are tagged with their origin module.  ``setup_logging()`` is
called exactly once, from ``main.py`` at startup.
"""
from __future__ import annotations

import logging

LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once at startup."""
    logging.basicConfig(level=level, format=LOG_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
