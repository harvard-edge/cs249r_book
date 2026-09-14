"""
Exception hierarchy for TinyTorch CLI.
"""

class TinyTorchCLIError(Exception):
    """Base exception for all CLI errors."""
    pass

class ValidationError(TinyTorchCLIError):
    """Raised when validation fails."""
    pass

class ExecutionError(TinyTorchCLIError):
    """Raised when command execution fails."""
    pass

class EnvironmentError(TinyTorchCLIError):
    """Raised when environment setup is invalid."""
    pass

class ModuleNotFoundError(TinyTorchCLIError):
    """Raised when a requested module is not found."""
    pass
