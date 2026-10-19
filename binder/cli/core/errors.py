"""Turn unexpected exceptions into messages that say what failed and where.

Binder is driven by agents and CI as much as by people, so an unexpected error
must name the exception type, its message, and the source line that raised it,
not just ``str(exc)`` (which is empty for many exceptions).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _short_path(filename: str) -> str:
    """Return *filename* relative to the repository root when it lies inside it."""
    try:
        return Path(filename).resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return filename


def describe_exception(exc: BaseException) -> str:
    """Return ``"Type: message (at path:line in function)"`` for *exc*.

    The location is the innermost traceback frame, which is where the error was
    raised. It is omitted when the exception carries no traceback.
    """
    message = str(exc).strip()
    text = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    frames = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
    if frames:
        frame = frames[-1]
        text += f" (at {_short_path(frame.filename)}:{frame.lineno} in {frame.name})"
    return text


def print_traceback(exc: BaseException) -> None:
    """Write the full traceback of *exc* to stderr, keeping stdout clean for ``--json``."""
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
