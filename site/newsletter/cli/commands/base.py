"""Base class all news CLI commands inherit from."""

from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace

from ..core.config import Config
from ..core.console import get_console


class BaseCommand(ABC):
    """Every news subcommand subclasses this."""

    # Category used in help-text grouping.
    # One of: "draft", "publish", "archive", "info".
    category: str = "info"

    def __init__(self, config: Config):
        self.config = config
        self.console = get_console()

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Override to add subcommand arguments."""
        return None

    @abstractmethod
    def run(self, args: Namespace) -> int: ...
