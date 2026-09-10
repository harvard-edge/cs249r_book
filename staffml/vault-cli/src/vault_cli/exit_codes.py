"""Stable exit-code taxonomy for the ``vault`` CLI.

Documented in ``vault-cli/docs/EXIT_CODES.md`` and referenced from
ARCHITECTURE.md §4.6. Never renumber an existing code — scripts pin to these.
"""

from enum import IntEnum


class ExitCode(IntEnum):
    """Process exit codes emitted by ``vault`` subcommands.

    Values match ``sysexits.h`` conventions where applicable.
    """

    SUCCESS = 0
    # Validation or invariant failure (schema violation, content-hash mismatch,
    # registry inconsistency, etc.). Distinguishes data-problems from bugs.
    VALIDATION_FAILURE = 1
    # Bad flags, missing args, usage errors surfaced by Typer/Click.
    USAGE_ERROR = 2
    # Filesystem / I/O failure (permission denied, disk full, missing file).
    IO_ERROR = 3
    # Network / D1 / Worker / external-service failure.
    NETWORK_ERROR = 4
    # Operator cancelled an interactive confirmation.
    USER_ABORTED = 5


__all__ = ["ExitCode"]
