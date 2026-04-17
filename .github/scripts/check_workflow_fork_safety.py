#!/usr/bin/env python3
"""
Check that workflows triggered by pull_request events do not reference
repository variables that won't be available on fork PRs.

Why this exists
---------------
GitHub does not expose repository variables to workflows triggered by
``pull_request`` events from forks. This is a security default. When a
workflow references ``${{ vars.* }}`` and runs on a fork PR, the
reference resolves to an empty string, which almost always leads to
silent build failures that look like "file missing" errors instead of
the real configuration problem.

Secrets are an even stronger case: ``${{ secrets.* }}`` is stripped for
fork PRs with the sole exception of ``secrets.GITHUB_TOKEN``, which is
always available (GitHub provides it automatically). Workflows that
need real secrets on PR should use ``pull_request_target`` (which runs
with target-branch context) and understand the security implications.

Incident: PR #1344. Three validate workflows used
``${{ vars.LABS_ROOT }}`` / ``${{ vars.KITS_ROOT }}`` /
``${{ vars.MLSYSIM_DOCS }}``. Every fork PR that touched those
directories silently failed its CI for a week before the pattern was
identified. The fix was to move the constants into workflow-level
``env:`` blocks, which ARE available in all contexts.

What this script does
---------------------
1. Parse every workflow file under ``.github/workflows/`` as YAML.
2. If its ``on:`` section declares a ``pull_request`` trigger (not
   ``pull_request_target``, which has target-branch context), scan the
   raw file text for ``${{ vars.* }}`` and for ``${{ secrets.* }}``
   references other than ``secrets.GITHUB_TOKEN``.
3. Skip comment lines and YAML block-scalar comments.
4. Fail with file:line:token pointers.

How to fix a violation
----------------------
Replace the ``vars.*`` reference with a workflow-level ``env:`` block::

    env:
      LABS_ROOT: labs

    jobs:
      ...
        steps:
          - working-directory: ${{ env.LABS_ROOT }}

Workflow-level env vars ARE exposed to fork PRs.

Exit codes
----------
0   All clean.
1   At least one violation found.
2   Script error (could not parse a workflow, missing directory, etc).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "error: PyYAML is required. install with: pip install pyyaml",
        file=sys.stderr,
    )
    sys.exit(2)

WORKFLOW_DIR = Path(".github/workflows")

# Match ${{ vars.NAME }} or ${{ secrets.NAME }} with optional whitespace.
UNSAFE_PATTERN = re.compile(r"\$\{\{\s*(vars|secrets)\.(\w+)\s*\}\}")

# GITHUB_TOKEN is the one secret that IS available on fork PRs.
ALWAYS_AVAILABLE_SECRETS = {"GITHUB_TOKEN"}


def triggers_on_pull_request(workflow_yaml: dict) -> bool:
    """True if the workflow's on: section declares a pull_request trigger.

    We treat pull_request_target as safe because it runs with
    target-branch context and DOES have access to repo vars/secrets.
    Only plain pull_request is fork-unsafe.
    """
    on_section = workflow_yaml.get("on", workflow_yaml.get(True))
    # PyYAML parses the bare keyword `on` as the Python boolean True
    # in some YAML contexts, so we also check the True key.

    if on_section is None:
        return False
    if isinstance(on_section, str):
        return on_section == "pull_request"
    if isinstance(on_section, list):
        return "pull_request" in on_section
    if isinstance(on_section, dict):
        return "pull_request" in on_section
    return False


def strip_comment(line: str) -> str:
    """Remove YAML inline comment. Preserves # inside quoted strings.

    Good enough for our purposes - we're scanning for ${{ ... }}
    patterns and any # inside a ${{ }} would be unusual.
    """
    in_single_quote = False
    in_double_quote = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif ch == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif ch == "#" and not in_single_quote and not in_double_quote:
            return line[:i]
    return line


def find_violations(path: Path) -> list[tuple[int, str, str]]:
    """Return [(line_number, token, context), ...] for fork-unsafe refs.

    Returns empty list if the workflow does not trigger on pull_request.
    """
    text = path.read_text()
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise RuntimeError(f"could not parse {path} as YAML: {e}")

    if not isinstance(parsed, dict):
        return []
    if not triggers_on_pull_request(parsed):
        return []

    violations: list[tuple[int, str, str]] = []
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        # Skip lines that are entirely comment. Inline comments after
        # the match still get scanned (but strip_comment removes the
        # comment portion first).
        stripped_leading = raw_line.lstrip()
        if stripped_leading.startswith("#"):
            continue
        code_part = strip_comment(raw_line)
        for match in UNSAFE_PATTERN.finditer(code_part):
            kind = match.group(1)  # "vars" or "secrets"
            name = match.group(2)
            if kind == "secrets" and name in ALWAYS_AVAILABLE_SECRETS:
                continue
            token = f"{kind}.{name}"
            violations.append((lineno, token, raw_line.strip()))
    return violations


def format_violation(path: Path, lineno: int, token: str, context: str) -> str:
    kind = token.split(".", 1)[0]
    return (
        f"  {path}:{lineno}\n"
        f"    token:   ${{{{ {token} }}}}\n"
        f"    context: {context}\n"
        f"    fix:     replace {kind}.* with a workflow-level env: block.\n"
        f"             see .github/scripts/check_workflow_fork_safety.py for why.\n"
    )


def main() -> int:
    if not WORKFLOW_DIR.is_dir():
        print(f"error: {WORKFLOW_DIR} not found", file=sys.stderr)
        return 2

    total_violations = 0
    exposed_workflows = 0
    scanned = 0

    workflow_paths = sorted(
        list(WORKFLOW_DIR.glob("*.yml")) + list(WORKFLOW_DIR.glob("*.yaml"))
    )

    for workflow_path in workflow_paths:
        scanned += 1
        try:
            violations = find_violations(workflow_path)
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        except OSError as e:
            print(f"error: could not read {workflow_path}: {e}", file=sys.stderr)
            return 2

        try:
            parsed = yaml.safe_load(workflow_path.read_text())
            if isinstance(parsed, dict) and triggers_on_pull_request(parsed):
                exposed_workflows += 1
        except yaml.YAMLError:
            continue

        if violations:
            print(f"\nfork-unsafe references in {workflow_path}:")
            for lineno, token, context in violations:
                print(format_violation(workflow_path, lineno, token, context))
                total_violations += 1

    if total_violations == 0:
        print(
            f"ok: scanned {scanned} workflows; "
            f"{exposed_workflows} trigger on pull_request; "
            f"none reference vars.* or (non-GITHUB_TOKEN) secrets.*"
        )
        return 0

    print(
        f"\nfound {total_violations} fork-unsafe reference(s) across "
        f"{exposed_workflows} pull_request-exposed workflow(s).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
