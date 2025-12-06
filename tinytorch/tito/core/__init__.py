"""
Core CLI functionality and shared utilities.
"""

from .console import get_console
from .exceptions import TinyTorchCLIError, ValidationError, ExecutionError
from .config import CLIConfig

__all__ = [
    'get_console',
    'TinyTorchCLIError',
    'ValidationError', 
    'ExecutionError',
    'CLIConfig'
] 