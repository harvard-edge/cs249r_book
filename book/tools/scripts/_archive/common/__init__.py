"""
Common utilities and shared components for MLSysBook tools.

This package provides shared functionality across all tools in the MLSysBook project,
including base classes, configuration management, logging setup, and common utilities.

Modules:
    base_classes: Abstract base classes for tools and processors
    config: Configuration management and environment handling
    exceptions: Custom exception definitions
    logging_config: Centralized logging configuration
    validators: Input validation utilities
    file_utils: File and path operation utilities
"""

from .exceptions import MLSysBookError, ConfigurationError, ValidationError
from .config import get_config, Config
from .logging_config import setup_logging, get_logger

__version__ = "1.0.0"
__all__ = [
    "MLSysBookError",
    "ConfigurationError",
    "ValidationError",
    "get_config",
    "Config",
    "setup_logging",
    "get_logger",
]
