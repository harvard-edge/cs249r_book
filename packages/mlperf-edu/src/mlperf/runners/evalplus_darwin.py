"""Run the pinned EvalPlus evaluator on macOS without modifying its source.

EvalPlus sandboxes each candidate solution in a subprocess that first calls
``reliability_guard``, which caps memory with ``RLIMIT_AS`` and ``RLIMIT_DATA``.
Darwin refuses both outright: ``setrlimit`` raises ``ValueError: current limit
exceeds maximum limit`` for any value, including a reduction from unlimited.
The guard therefore raises before the solution runs, the worker dies, and every
candidate is scored as a failure. Measured on this host, that is all 164 of
them, including the canonical reference solutions.

Upstream already special-cases Darwin one line further down, skipping
``RLIMIT_STACK`` on that platform. Extending the same treatment to the other
two limits is the whole fix.

We apply it here rather than editing the evaluator because the evaluator's
bytes are pinned and its digest is part of a result's provenance. Patching the
environment leaves the evaluated source identical to the container image's.

What is lost on macOS is the memory cap, not the verdict. A solution that would
have been killed for exceeding four gigabytes instead runs to completion or
fails on its own timeout. That is a weaker sandbox, and it is why a container
remains the preferred path where one is available.

The patch is applied at module import rather than under ``__main__`` because
EvalPlus workers start with the spawn method on macOS, and a spawned child
re-imports this module. Patching only in the parent would leave every worker
unprotected against the very failure this exists to prevent.
"""

from __future__ import annotations

import platform
import resource
import sys

_ORIGINAL_SETRLIMIT = resource.setrlimit

#: Limits Darwin will not accept at any value. Attempting them raises rather
#: than clamping, so the call has to be tolerated instead of adjusted.
_UNSUPPORTED_ON_DARWIN = {resource.RLIMIT_AS, resource.RLIMIT_DATA}


def _best_effort_setrlimit(resource_id: int, limits: tuple[int, int]) -> None:
    """Apply a limit, tolerating platforms that refuse it.

    Only the limits Darwin is known to reject are swallowed. Any other failure
    still propagates, so a genuine sandbox problem is not hidden.
    """
    try:
        _ORIGINAL_SETRLIMIT(resource_id, limits)
    except (ValueError, OSError):
        if resource_id not in _UNSUPPORTED_ON_DARWIN:
            raise


def install() -> bool:
    """Patch ``resource.setrlimit`` on Darwin. Returns whether it was applied."""
    if platform.system() != "Darwin":
        return False
    resource.setrlimit = _best_effort_setrlimit
    return True


APPLIED = install()


def main() -> int:
    from evalplus.evaluate import main as evalplus_main

    if APPLIED:
        print(
            "note: running EvalPlus with best-effort memory limits. Darwin "
            "refuses RLIMIT_AS and RLIMIT_DATA, so solutions execute without a "
            "memory cap. Use the container path for a sandboxed run.",
            file=sys.stderr,
        )
    evalplus_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
