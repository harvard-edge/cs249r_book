"""Binder-as-front-door invariant check for the pre-commit config.

This powers::

    ./book/binder check cli --scope binder-canonical

The rule it enforces, in one sentence: **every pre-commit hook that targets
book content must dispatch through ``./book/binder``, not call a raw script.**

Why this exists
---------------
Binder is the single front door for all book-content checks: one entry point,
one error format, one place to register a new scope. Over time it is tempting
to wire a quick ``entry: python3 some_script.py`` hook for a new check instead
of adding a Binder scope. That silently grows a second invocation path —
exactly the drift this check forbids. The day it caught its first real case:
``book-check-lego-units`` called ``lint_lego_units.py`` directly while Binder
already ran the same linter as ``code/lego-units`` (retired 2026-06-10).

What counts as a "book-content hook"
------------------------------------
A *local* hook (``repo: local``) with an explicit ``entry:`` whose ``files:``
pattern is scoped to ``book/quarto/contents/``. Repo-wide guards (link checks,
mirror sync), CI hygiene, third-party hooks (mdformat, codespell — no local
``entry:``), and separate subprojects (vault-cli) are intentionally NOT book
content and are not inspected.

How to satisfy it
-----------------
Route the hook through Binder: add a ``Scope(...)`` to the relevant
``binder check <group>`` and set the hook ``entry`` to ``./book/binder check
<group>``. If a book-content hook genuinely cannot be a Binder scope, add its
id to ``ALLOWLIST`` below with a one-line justification so the exception is
explicit and reviewed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

# Marker that a hook's `files:` pattern is scoped to book chapter content.
BOOK_CONTENT_MARKER = "book/quarto/contents"

# Prefix that marks an entry as dispatching through the Binder front door.
BINDER_ENTRY_RE = re.compile(r"^\.?/?book/binder\b")

# Book-content hooks that are deliberately allowed to bypass Binder.
# Format: hook-id -> justification. Empty by design: every book-content hook
# currently routes through Binder. Add an entry ONLY with a real reason.
ALLOWLIST: dict[str, str] = {}

CONFIG_REL = ".pre-commit-config.yaml"


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    code: str
    message: str
    context: str = ""
    suggestion: str = ""


def _hook_line(raw_lines: List[str], hook_id: str) -> int:
    """Best-effort line number of `- id: <hook_id>` for a clickable location."""
    needle = re.compile(rf"^\s*-\s*id:\s*{re.escape(hook_id)}\s*$")
    for i, line in enumerate(raw_lines, 1):
        if needle.match(line):
            return i
    return 1


def _targets_book_content(files_pattern: str) -> bool:
    return BOOK_CONTENT_MARKER in (files_pattern or "")


def _dispatches_through_binder(entry: str) -> bool:
    return bool(BINDER_ENTRY_RE.match((entry or "").strip()))


def run_canonical(repo_root: Path) -> List[Violation]:
    """Return a Violation for every book-content hook that bypasses Binder."""
    config_path = repo_root / CONFIG_REL
    violations: List[Violation] = []
    if not config_path.is_file():
        return [
            Violation(
                file=CONFIG_REL,
                line=1,
                code="BINDER-CANON-000",
                message=f"{CONFIG_REL} not found at repo root.",
                suggestion=f"Expected the pre-commit config at {config_path}.",
            )
        ]

    raw_lines = config_path.read_text(encoding="utf-8").splitlines()
    data = yaml.safe_load("\n".join(raw_lines)) or {}

    for repo in data.get("repos", []):
        # Only local hooks have an `entry:` we can audit. Third-party repos
        # (mdformat, codespell, ruff, …) are legitimately not Binder.
        if repo.get("repo") != "local":
            continue
        for hook in repo.get("hooks", []):
            hook_id = hook.get("id", "<unknown>")
            entry = hook.get("entry", "")
            files_pattern = hook.get("files", "")

            if not _targets_book_content(files_pattern):
                continue
            if _dispatches_through_binder(entry):
                continue
            if hook_id in ALLOWLIST:
                continue

            violations.append(
                Violation(
                    file=CONFIG_REL,
                    line=_hook_line(raw_lines, hook_id),
                    code="BINDER-CANON-001",
                    message=(
                        f"Hook '{hook_id}' targets book content but its entry "
                        f"bypasses Binder: {entry!r}"
                    ),
                    context=f"files: {files_pattern}",
                    suggestion=(
                        f"Route '{hook_id}' through Binder: add a scope to the "
                        f"relevant `binder check <group>` and set entry to "
                        f"`./book/binder check <group>`. If it genuinely cannot "
                        f"be a Binder scope, add '{hook_id}' to ALLOWLIST in "
                        f"book/cli/checks/binder_canonical.py with a justification."
                    ),
                )
            )

    return violations
