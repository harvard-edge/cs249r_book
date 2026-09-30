"""
"Try it" prompts that run after a milestone has passed.

A try-it prompt lets a student type their own input into the model they just
trained and watch it respond. It is play, not grading: it runs only after the
milestone's verdict is printed, it never changes the exit code, and it is
skipped whenever nobody is at the keyboard (CI, tests, piped runs,
`tito milestone run --non-interactive`, and the recorded demos, whose
recorder sets TINYTORCH_NON_INTERACTIVE).
"""

import os
import sys
from typing import Callable, Optional

from rich.console import Console
from rich.panel import Panel

EXIT_WORDS = ("exit", "quit", "q")


def try_it_enabled() -> bool:
    """True only in a real terminal session that has not opted out."""
    if (
        os.environ.get("TINYTORCH_NON_INTERACTIVE") == "1"
        or os.environ.get("CI") == "true"
    ):
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()


def try_it(
    console: Console,
    intro: str,
    label: str,
    respond: Callable[[str], None],
    read: Optional[Callable[[], str]] = None,
) -> int:
    """
    Read inputs until the student leaves, passing each one to ``respond``.

    Enter on an empty line, ``exit``, Ctrl-D, or Ctrl-C ends the loop. An
    error raised by ``respond`` is printed and the loop continues, so a bad
    input cannot crash a milestone that has already passed.

    Args:
        console: Where to print.
        intro: What to try, shown once in a panel.
        label: The input prompt, e.g. ``"Complete > "``.
        respond: Called with each stripped, non-empty input.
        read: Input source for tests; when omitted, the terminal is used and
            the loop is skipped unless ``try_it_enabled()``.

    Returns:
        The number of inputs handled.
    """
    if read is None:
        if not try_it_enabled():
            return 0
        read = lambda: console.input(label)  # noqa: E731

    console.print()
    console.print(Panel(
        intro + "\n\n[dim]Press Enter on an empty line (or type exit) to finish.[/dim]",
        title="🎮 Try it", border_style="magenta",
    ))
    handled = 0
    while True:
        try:
            text = read().strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in EXIT_WORDS:
            break
        try:
            respond(text)
        except KeyboardInterrupt:
            break
        except Exception as error:  # the milestone already passed; keep playing
            console.print(f"[yellow]That input did not run: {error}[/yellow]")
        handled += 1
    console.print()
    return handled
