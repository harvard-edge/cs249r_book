"""
CLI Commands package.

Each command is implemented as a separate module with proper separation of concerns.
Commands are organized into logical groups: system, module, and package.
"""

from .base import BaseCommand

# Individual commands
from .test import TestCommand
from .export import ExportCommand
from .src import SrcCommand
from .nbgrader import NBGraderCommand
from .benchmark import BenchmarkCommand
from .community import CommunityCommand

# Command groups (with subcommands organized in subfolders)
from .system import SystemCommand
from .module import ModuleWorkflowCommand
from .package import PackageCommand

__all__ = [
    'BaseCommand',
    # Individual commands
    'TestCommand',
    'ExportCommand',
    'SrcCommand',
    'NBGraderCommand',
    'BenchmarkCommand',
    'CommunityCommand',
    # Command groups
    'SystemCommand',
    'ModuleWorkflowCommand',
    'PackageCommand',
]
