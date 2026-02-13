"""
Abstract base classes for MLSysBook tools.

This module provides abstract base classes that define common interfaces
and patterns for tools in the MLSysBook project.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import argparse
from dataclasses import dataclass

from .config import get_config
from .logging_config import get_logger
from .exceptions import ToolExecutionError, ValidationError


@dataclass
class ToolResult:
    """Standard result object for tool operations.

    Attributes:
        success: Whether the operation was successful
        message: Human-readable result message
        data: Optional result data
        errors: List of error messages
        warnings: List of warning messages
        metadata: Additional metadata about the operation
    """
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class BaseTool(ABC):
    """Abstract base class for all MLSysBook tools.

    This class provides a common interface and shared functionality for
    command-line tools and processing utilities.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Any] = None
    ) -> None:
        """Initialize the tool.

        Args:
            name: Tool name (defaults to class name)
            description: Tool description
            config: Configuration object (defaults to global config)
        """
        self.name = name or self.__class__.__name__
        self.description = description or self.__class__.__doc__ or "MLSysBook tool"
        self.config = config or get_config()
        self.logger = get_logger(f"tools.{self.name}")

        # Initialize state
        self._initialized = False
        self._results: List[ToolResult] = []

    @abstractmethod
    def run(self, *args, **kwargs) -> ToolResult:
        """Run the tool with the provided arguments.

        This is the main entry point for tool execution.

        Returns:
            ToolResult object with operation results
        """
        pass

    def validate_inputs(self, *args, **kwargs) -> None:
        """Validate tool inputs.

        Override this method to provide input validation specific to your tool.

        Raises:
            ValidationError: If validation fails
        """
        pass

    def initialize(self) -> None:
        """Initialize the tool.

        Override this method to perform one-time setup operations.
        """
        if self._initialized:
            return

        self.logger.debug(f"Initializing tool: {self.name}")
        self._initialized = True

    def cleanup(self) -> None:
        """Clean up resources used by the tool.

        Override this method to perform cleanup operations.
        """
        self.logger.debug(f"Cleaning up tool: {self.name}")

    def add_result(self, result: ToolResult) -> None:
        """Add a result to the tool's result history.

        Args:
            result: ToolResult to add
        """
        self._results.append(result)

    def get_results(self) -> List[ToolResult]:
        """Get all results from tool execution.

        Returns:
            List of ToolResult objects
        """
        return self._results.copy()

    def get_last_result(self) -> Optional[ToolResult]:
        """Get the most recent result.

        Returns:
            Most recent ToolResult or None if no results
        """
        return self._results[-1] if self._results else None

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


class FileProcessorTool(BaseTool):
    """Base class for tools that process files.

    This class provides common functionality for tools that work with
    files and directories.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the file processor tool."""
        super().__init__(*args, **kwargs)
        self.processed_files: List[Path] = []
        self.failed_files: List[Path] = []

    @abstractmethod
    def process_file(self, file_path: Path) -> ToolResult:
        """Process a single file.

        Args:
            file_path: Path to the file to process

        Returns:
            ToolResult object with processing results
        """
        pass

    def process_directory(
        self,
        directory: Path,
        pattern: str = "**/*",
        recursive: bool = True
    ) -> ToolResult:
        """Process all matching files in a directory.

        Args:
            directory: Directory to process
            pattern: Glob pattern for file matching
            recursive: Whether to process subdirectories

        Returns:
            ToolResult object with overall processing results
        """
        self.logger.info(f"Processing directory: {directory}")

        try:
            if recursive:
                files = list(directory.rglob(pattern))
            else:
                files = list(directory.glob(pattern))

            files = [f for f in files if f.is_file()]

            results = []
            for file_path in files:
                try:
                    result = self.process_file(file_path)
                    results.append(result)

                    if result.success:
                        self.processed_files.append(file_path)
                    else:
                        self.failed_files.append(file_path)

                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")
                    self.failed_files.append(file_path)
                    results.append(ToolResult(
                        success=False,
                        message=f"Failed to process {file_path}: {e}",
                        errors=[str(e)]
                    ))

            successful = len(self.processed_files)
            failed = len(self.failed_files)

            return ToolResult(
                success=failed == 0,
                message=f"Processed {successful} files, {failed} failed",
                data={
                    "processed_files": self.processed_files,
                    "failed_files": self.failed_files,
                    "results": results
                },
                metadata={
                    "total_files": len(files),
                    "successful_count": successful,
                    "failed_count": failed
                }
            )

        except Exception as e:
            error_msg = f"Failed to process directory {directory}: {e}"
            self.logger.error(error_msg)
            return ToolResult(
                success=False,
                message=error_msg,
                errors=[str(e)]
            )


class CLITool(BaseTool):
    """Base class for command-line interface tools.

    This class provides argument parsing and common CLI patterns.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the CLI tool."""
        super().__init__(*args, **kwargs)
        self.parser = self.create_argument_parser()

    def create_argument_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser for this tool.

        Override this method to add tool-specific arguments.

        Returns:
            Configured ArgumentParser
        """
        parser = argparse.ArgumentParser(
            prog=self.name,
            description=self.description
        )

        # Add common arguments
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose output"
        )

        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be done without making changes"
        )

        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
            help="Set the logging level"
        )

        return parser

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command-line arguments.

        Args:
            args: Arguments to parse (defaults to sys.argv)

        Returns:
            Parsed arguments namespace
        """
        return self.parser.parse_args(args)

    def run_cli(self, args: Optional[List[str]] = None) -> ToolResult:
        """Run the tool as a CLI application.

        Args:
            args: Command-line arguments

        Returns:
            ToolResult object
        """
        try:
            parsed_args = self.parse_args(args)

            # Update logging level if specified
            if hasattr(parsed_args, 'log_level'):
                self.logger.setLevel(parsed_args.log_level)

            # Run input validation
            self.validate_inputs(**vars(parsed_args))

            # Run the tool
            return self.run(**vars(parsed_args))

        except ValidationError as e:
            error_msg = f"Input validation failed: {e}"
            self.logger.error(error_msg)
            return ToolResult(
                success=False,
                message=error_msg,
                errors=[str(e)]
            )
        except Exception as e:
            error_msg = f"Tool execution failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            return ToolResult(
                success=False,
                message=error_msg,
                errors=[str(e)]
            )


class ContentProcessor(ABC):
    """Abstract base class for content processing components.

    This class defines the interface for components that process
    textbook content (markdown, images, etc.).
    """

    @abstractmethod
    def can_process(self, content_type: str, file_path: Path) -> bool:
        """Check if this processor can handle the given content.

        Args:
            content_type: Type of content (e.g., 'markdown', 'image')
            file_path: Path to the content file

        Returns:
            True if this processor can handle the content
        """
        pass

    @abstractmethod
    def process(self, file_path: Path, **kwargs) -> ToolResult:
        """Process the content.

        Args:
            file_path: Path to the content file
            **kwargs: Additional processing options

        Returns:
            ToolResult object with processing results
        """
        pass

    def get_priority(self) -> int:
        """Get the processor priority.

        Higher numbers indicate higher priority. When multiple processors
        can handle the same content, the one with the highest priority is used.

        Returns:
            Priority value (default: 0)
        """
        return 0
