"""
Logging Configuration
=====================
Centralized logging setup with rotating file handlers, colored console output,
and configurable log levels. All modules should use:

    from src.logging_config import get_logger
    logger = get_logger(__name__)
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path


# =============================================================================
# Constants
# =============================================================================

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = LOG_DIR / "fraud_detection.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5


# =============================================================================
# Custom Formatter
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Console formatter with ANSI color codes for different log levels."""

    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# =============================================================================
# Logger Factory
# =============================================================================

_loggers = {}


def get_logger(name: str, level: str = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Parameters
    ----------
    name : str
        Logger name (typically __name__).
    level : str, optional
        Log level override. Defaults to LOG_LEVEL env var or INFO.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    log_level = getattr(logging, (level or DEFAULT_LOG_LEVEL), logging.INFO)
    logger.setLevel(log_level)

    # Avoid duplicate handlers on re-import
    if logger.handlers:
        _loggers[name] = logger
        return logger

    # Console handler with colors — use UTF-8 wrapper on Windows
    utf8_stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8',
                       closefd=False, errors='replace')
    console_handler = logging.StreamHandler(utf8_stdout)
    console_handler.setLevel(log_level)
    console_fmt = ColoredFormatter(
        "%(asctime)s | %(levelname)-18s | %(name)-25s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # Rotating file handler (plain text, no colors)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)  # File captures everything
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-30s | %(funcName)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    _loggers[name] = logger
    return logger


def set_global_level(level: str):
    """Set log level for all registered loggers."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    for logger in _loggers.values():
        logger.setLevel(log_level)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.handlers.RotatingFileHandler
            ):
                handler.setLevel(log_level)
