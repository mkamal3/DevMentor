"""Logging configuration helpers for DevMentor."""

import logging
from typing import Optional


def setup_logger(name: str = "devmentor", level: str = "INFO") -> logging.Logger:
    """Configure and return an application logger.

    Args:
        name: Logger name.
        level: Logging level string (e.g., INFO, DEBUG).

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return an existing logger by name."""
    return logging.getLogger(name or "devmentor")
