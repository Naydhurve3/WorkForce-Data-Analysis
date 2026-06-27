"""Logging configuration using loguru."""

import os
import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    logger.remove()

    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan> | <level>{message}</level>",
        level=level,
        colorize=True,
    )

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}",
            level="DEBUG",
            rotation=rotation,
            retention=retention,
        )


def get_logger(name: str):
    return logger.bind(name=name)
