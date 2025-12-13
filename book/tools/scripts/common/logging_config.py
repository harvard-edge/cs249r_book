"""
Centralized logging configuration for MLSysBook tools.

This module provides a standardized logging setup with support for structured
logging, different output formats, and configurable log levels.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

from .config import get_config


class StructuredFormatter(logging.Formatter):
    """Custom formatter that adds structured information to log records."""

    def __init__(self, include_context: bool = True) -> None:
        """Initialize the formatter.

        Args:
            include_context: Whether to include additional context in log records
        """
        super().__init__()
        self.include_context = include_context

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with structured information."""
        # Add timestamp
        record.timestamp = datetime.utcnow().isoformat()

        # Add context information if available
        if self.include_context and hasattr(record, 'context'):
            record.context_str = f" | Context: {record.context}"
        else:
            record.context_str = ""

        # Create the formatted message
        if record.levelno >= logging.ERROR and record.exc_info:
            # For errors with exceptions, include the full traceback
            formatted = (
                f"[{record.timestamp}] {record.levelname:8} | "
                f"{record.name} | {record.getMessage()}{record.context_str}\n"
                f"Exception: {self.formatException(record.exc_info)}"
            )
        else:
            formatted = (
                f"[{record.timestamp}] {record.levelname:8} | "
                f"{record.name} | {record.getMessage()}{record.context_str}"
            )

        return formatted


class ProgressAwareHandler(RichHandler):
    """Rich handler that works well with progress bars and other rich output."""

    def __init__(self, *args, **kwargs) -> None:
        # Create a separate console for logging to avoid conflicts
        console = Console(stderr=True, force_terminal=True)
        kwargs['console'] = console
        kwargs['show_time'] = False  # We'll handle time in our formatter
        kwargs['show_path'] = False
        super().__init__(*args, **kwargs)


def setup_logging(
    name: Optional[str] = None,
    level: Optional[str] = None,
    log_to_file: Optional[bool] = None,
    log_file_path: Optional[Path] = None,
    enable_rich_tracebacks: bool = True,
    context: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """Set up logging configuration for MLSysBook tools.

    Args:
        name: Logger name (defaults to calling module)
        level: Log level (defaults to config value)
        log_to_file: Whether to log to file (defaults to config value)
        log_file_path: Path for log file (defaults to config value)
        enable_rich_tracebacks: Whether to enable rich exception formatting
        context: Additional context to include in all log messages

    Returns:
        Configured logger instance
    """
    config = get_config()

    # Use provided values or fall back to config
    level = level or config.log_level
    log_to_file = log_to_file if log_to_file is not None else config.log_to_file
    log_file_path = log_file_path or config.log_file_path

    # Set up rich tracebacks if enabled
    if enable_rich_tracebacks:
        install(show_locals=True)

    # Create logger
    logger = logging.getLogger(name or __name__)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear any existing handlers
    logger.handlers.clear()

    # Add console handler with rich formatting
    console_handler = ProgressAwareHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter("%(name)s | %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Add file handler if requested
    if log_to_file and log_file_path:
        # Ensure log directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # Always debug level for files
        file_formatter = StructuredFormatter(include_context=True)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Add context if provided
    if context:
        logger = LoggerAdapter(logger, context)

    return logger


def get_logger(
    name: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """Get a logger instance with optional context.

    Args:
        name: Logger name (defaults to calling module)
        context: Additional context to include in log messages

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name or __name__)

    # If logger doesn't have handlers, set it up
    if not logger.handlers:
        logger = setup_logging(name)

    # Add context if provided
    if context:
        logger = LoggerAdapter(logger, context)

    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Adapter that adds context information to log records."""

    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]) -> None:
        """Initialize the adapter with context information.

        Args:
            logger: Base logger instance
            extra: Context information to add to log records
        """
        super().__init__(logger, extra)

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process a log message to add context information."""
        if 'extra' not in kwargs:
            kwargs['extra'] = {}

        # Add our context to the extra information
        kwargs['extra'].update(self.extra)
        kwargs['extra']['context'] = self.extra

        return msg, kwargs


class ProgressLogger:
    """Logger that works well with progress bars and other rich output."""

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize the progress logger.

        Args:
            name: Logger name
        """
        self.logger = get_logger(name)
        self._progress_active = False

    def start_progress(self) -> None:
        """Indicate that a progress operation is starting."""
        self._progress_active = True

    def end_progress(self) -> None:
        """Indicate that a progress operation has ended."""
        self._progress_active = False

    def log(self, level: str, message: str, **kwargs) -> None:
        """Log a message with progress awareness.

        Args:
            level: Log level (debug, info, warning, error, critical)
            message: Log message
            **kwargs: Additional keyword arguments for logging
        """
        if self._progress_active:
            # For progress operations, use a simpler format
            print(f"[{level.upper()}] {message}", file=sys.stderr)
        else:
            # Use normal logging
            getattr(self.logger, level.lower())(message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        self.log('debug', message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        self.log('info', message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        self.log('warning', message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        self.log('error', message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message."""
        self.log('critical', message, **kwargs)
