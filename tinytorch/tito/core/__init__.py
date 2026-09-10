"""
Core CLI functionality and shared utilities.
"""

from .console import get_console
from .exceptions import TinyTorchCLIError, ValidationError, ExecutionError
from .config import CLIConfig
from .modules import (
    get_module_mapping,
    get_module_name,
    get_module_display_name,
    get_next_module,
    normalize_module_number,
    get_total_modules,
    module_exists,
    clear_cache,
)

__all__ = [
    'get_console',
    'TinyTorchCLIError',
    'ValidationError',
    'ExecutionError',
    'CLIConfig',
    # Module utilities
    'get_module_mapping',
    'get_module_name',
    'get_module_display_name',
    'get_next_module',
    'normalize_module_number',
    'get_total_modules',
    'module_exists',
    'clear_cache',
]
