"""Logging configuration with loguru."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    name: str = "yolo26-asl",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
    colorize: bool = True,
) -> "logger":
    """
    Configure and return a logger instance.

    Args:
        name: Logger name (used as prefix in messages).
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.
        rotation: When to rotate the log file.
        retention: How long to keep old log files.
        colorize: Whether to colorize console output.

    Returns:
        Configured logger instance.
    """
    # Remove default handler
    logger.remove()

    # Console handler with rich formatting
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=colorize,
        backtrace=True,
        diagnose=True,
    )

    # File handler (if specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )

        logger.add(
            log_file,
            format=file_format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
        )

    return logger.bind(name=name)


def get_logger(name: str = "yolo26-asl") -> "logger":
    """Get a logger instance with the given name."""
    return logger.bind(name=name)


class LoggerContext:
    """Context manager for temporary logger configuration."""

    def __init__(self, level: str = "DEBUG"):
        self.level = level
        self.handler_id = None

    def __enter__(self):
        self.handler_id = logger.add(
            sys.stderr,
            level=self.level,
            format="{level} | {message}",
        )
        return logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handler_id is not None:
            logger.remove(self.handler_id)
        return False
