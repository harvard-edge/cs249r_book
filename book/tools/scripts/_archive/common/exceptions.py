"""
Custom exception definitions for MLSysBook tools.

This module defines a hierarchy of custom exceptions to provide clear error handling
and better debugging information across all tools in the MLSysBook project.
"""

from typing import Optional, Any, Dict


class MLSysBookError(Exception):
    """Base exception for all MLSysBook-related errors.

    This is the root exception class that all other custom exceptions inherit from.
    It provides consistent error handling and optional context information.

    Args:
        message: Human-readable error message
        context: Optional dictionary containing error context information
        original_error: Original exception that caused this error (if any)
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.original_error = original_error

    def __str__(self) -> str:
        """Return a formatted error message with context."""
        error_msg = self.message

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            error_msg += f" (Context: {context_str})"

        if self.original_error:
            error_msg += f" (Caused by: {self.original_error})"

        return error_msg


class ConfigurationError(MLSysBookError):
    """Raised when there are configuration-related errors.

    This includes missing configuration files, invalid configuration values,
    environment variable issues, and other configuration problems.
    """
    pass


class ValidationError(MLSysBookError):
    """Raised when input validation fails.

    This includes invalid file paths, malformed data, missing required fields,
    and other input validation issues.
    """
    pass


class FileOperationError(MLSysBookError):
    """Raised when file operations fail.

    This includes file not found, permission denied, disk space issues,
    and other file system related errors.
    """
    pass


class ProcessingError(MLSysBookError):
    """Raised when content processing fails.

    This includes parsing errors, conversion failures, and other content
    processing related issues.
    """
    pass


class APIError(MLSysBookError):
    """Raised when external API calls fail.

    This includes HTTP errors, authentication failures, rate limiting,
    and other API-related issues.
    """
    pass


class ToolExecutionError(MLSysBookError):
    """Raised when tool execution fails.

    This is a general execution error for tools that encounter unexpected
    conditions during their main operation.
    """
    pass


class DependencyError(MLSysBookError):
    """Raised when required dependencies are missing or incompatible.

    This includes missing Python packages, external tools, or version
    compatibility issues.
    """
    pass
