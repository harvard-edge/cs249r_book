"""
Base command class for TinyTorch CLI.
"""

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Optional
from pathlib import Path
import logging
import sys
import os
from contextlib import contextmanager

from ..core.config import CLIConfig
from ..core.virtual_env_manager import get_venv_path
from ..core.console import get_console
from ..core.exceptions import TinyTorchCLIError

logger = logging.getLogger(__name__)

@contextmanager
def suppress_output():
    """Context manager to suppress stdout temporarily."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

class BaseCommand(ABC):
    """Base class for all CLI commands."""

    # Command metadata - override in subclasses
    category: str = "other"  # "essential", "workflow", "tracking", "community", "shortcut", "developer"
    hidden: bool = False  # Set to True to hide from main help

    def __init__(self, config: CLIConfig):
        """Initialize the command with configuration."""
        self.config = config
        self.console = get_console()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the command name."""
        pass

    @property
    def venv_path(self) -> Path:
        """Return the command name."""
        return get_venv_path()

    @property
    @abstractmethod
    def description(self) -> str:
        """Return the command description."""
        pass

    @abstractmethod
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add command-specific arguments to the parser."""
        pass

    @abstractmethod
    def run(self, args: Namespace) -> int:
        """Execute the command and return exit code."""
        pass

    def validate_args(self, args: Namespace) -> None:
        """Validate command arguments. Override in subclasses if needed."""
        pass

    def execute(self, args: Namespace) -> int:
        """Execute the command with error handling."""
        try:
            self.validate_args(args)
            return self.run(args)
        except TinyTorchCLIError as e:
            logger.error(f"Command failed: {e}")
            self.console.print(f"[red]❌ {e}[/red]")
            return 1
        except Exception as e:
            logger.exception(f"Unexpected error in command {self.name}")
            self.console.print(f"[red]❌ Unexpected error: {e}[/red]")
            return 1
