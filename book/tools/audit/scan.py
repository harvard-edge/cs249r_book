#!/usr/bin/env python3
"""Pass 15 audit scanner.

Walks a configurable content tree, runs every registered check function
against every .qmd file, and writes an audit-ledger.json.

Usage:
    python3 book/tools/audit/scan.py --scope vol1
    python3 book/tools/audit/scan.py --scope vol2 --categories vs-period,percent-symbol
    python3 book/tools/audit/scan.py --scope vol2 --output my-ledger.json

The scanner is READ-ONLY. It never modifies files. It is the SCAN stage
of the five-stage cycle (see Pass 15 plan section 2).
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

# Make the audit package importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit.ledger import Issue, Ledger
from audit.accept_list import (
    DEFAULT_ACCEPT_LIST,
    DEFAULT_NOTATION_ACCEPT_LIST,
    apply_accept_list,
    format_report,
    format_stale_warnings,
    load_accept_list,
)

# ── Check registry ──────────────────────────────────────────────────────────

# All check modules. Order matters only for report readability; the scanner
# is otherwise commutative. Each entry is (module_name, category_name).
# The category name is used for the ledger category field, --categories
# filter, and the fix-script routing.
CHECK_REGISTRY: list[tuple[str, str]] = [
    ("audit.checks.vs_period", "vs-period"),
    ("audit.checks.compound_prefix", "compound-prefix-closeup"),
    ("audit.checks.percent_symbol", "percent-symbol"),
    ("audit.checks.lowercase_prose_references", "lowercase-prose-references"),
    ("audit.checks.acknowledgements_spelling", "acknowledgements-spelling"),
    ("audit.checks.binary_units", "binary-units-in-prose"),
    ("audit.checks.h3_titlecase", "h3-titlecase"),
    ("audit.checks.concept_term_capitalization", "concept-term-capitalization"),
    ("audit.checks.abbreviation_first_use", "abbreviation-first-use"),
    ("audit.checks.latin_running_text", "latin-running-text"),
    ("audit.checks.alt_text_style", "alt-text-style"),
    ("audit.checks.bibliography_hygiene", "bibliography-hygiene"),
    ("audit.checks.notation_consistency", "notation-consistency"),
]


def load_checks(categories: list[str] | None = None):
    """Load the registered check modules. Filter by category if requested."""
    loaded = []
    for module_name, category in CHECK_REGISTRY:
        if categories and category not in categories:
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            print(f"ERROR importing {module_name}: {e}", file=sys.stderr)
            continue
        loaded.append((category, module))
    return loaded


# ── Scope resolution ────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTENTS_ROOT = REPO_ROOT / "book" / "quarto" / "contents"


# File extensions the scanner walks. Per-check modules may declare a
# narrower subset via the `FILE_EXTENSIONS` module attribute; by default
# (absent the attribute) a check runs on `.qmd` files only. The
# bibliography-hygiene check declares `FILE_EXTENSIONS = (".bib",)` so
# it sees the `.bib` files and existing `.qmd`-only checks skip them.
_SCANNED_EXTENSIONS = ("*.qmd", "*.bib")
DEFAULT_CHECK_EXTENSIONS = (".qmd",)


def resolve_scope(scope: str) -> list[Path]:
    """Return the content files for the requested scope.

    Supported scopes:
      vol1          -> book/quarto/contents/vol1/**/*.{qmd,bib}
      vol2          -> book/quarto/contents/vol2/**/*.{qmd,bib}
      both          -> both volumes
      <path>        -> treat as a file or directory path
    """
    def _walk(root: Path) -> list[Path]:
        files: list[Path] = []
        for pattern in _SCANNED_EXTENSIONS:
            files.extend(root.rglob(pattern))
        return files

    if scope == "vol1":
        root = CONTENTS_ROOT / "vol1"
    elif scope == "vol2":
        root = CONTENTS_ROOT / "vol2"
    elif scope == "both":
        return sorted(
            _walk(CONTENTS_ROOT / "vol1") + _walk(CONTENTS_ROOT / "vol2")
        )
    else:
        root = Path(scope)

    if not root.exists():
        raise SystemExit(f"Scope path does not exist: {root}")

    if root.is_file():
        return [root]
    return sorted(_walk(root))


# ── Rule file SHA ───────────────────────────────────────────────────────────


def get_rule_file_sha() -> str:
    """Get the git SHA of book-prose-merged.md so the ledger is reproducible."""
    rule_file = (
        Path.home()
        / "GitHub"
        / "AIConfigs"
        / "projects"
        / "MLSysBook"
        / ".claude"
        / "rules"
        / "book-prose-merged.md"
    )
    if not rule_file.exists():
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "-C", str(rule_file.parent.parent.parent.parent.parent),
             "log", "-n", "1", "--pretty=format:%H", "--", str(rule_file)],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "unknown"
    except (subprocess.TimeoutExpired, OSError):
        pass
    return "unknown"


# ── Main scan loop ──────────────────────────────────────────────────────────


def scan(
    scope: str,
    categories: list[str] | None = None,
    verbose: bool = False,
    accept_list_path: Path | None = None,
    use_accept_list: bool = True,
) -> Ledger:
    """Run all checks against the scope and return the ledger.

    If `use_accept_list` is True (the default), after all checks run the
    persistent accept-list at `accept_list_path` (default:
    book/tools/audit/accepted_fps.json) is applied: any issue whose
    (category, repo-relative-file, raw-line) triple matches an entry is
    flipped from `open` to `accepted` and tagged with the §10.9 rule that
    justifies it. See book/tools/audit/accept_list.py.
    """
    files = resolve_scope(scope)
    if verbose:
        print(f"Scanning {len(files)} files in scope={scope}...", file=sys.stderr)

    checks = load_checks(categories)
    if not checks:
        raise SystemExit("No check modules loaded.")

    if verbose:
        print(f"  Loaded {len(checks)} check modules: "
              f"{', '.join(c for c, _ in checks)}", file=sys.stderr)

    scope_tag = scope if scope in ("vol1", "vol2", "both") else "scope"
    ledger = Ledger(scope=scope_tag, rule_file_sha=get_rule_file_sha())

    start = time.time()
    counter = 0
    for category, module in checks:
        cat_start = time.time()
        cat_count = 0
        # Each check module may declare FILE_EXTENSIONS to narrow which
        # file types it runs on. Default is `.qmd` only.
        allowed_exts = getattr(
            module, "FILE_EXTENSIONS", DEFAULT_CHECK_EXTENSIONS
        )
        for file_path in files:
            if file_path.suffix not in allowed_exts:
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                print(f"  SKIP {file_path}: {e}", file=sys.stderr)
                continue

            try:
                issues, counter = module.check(
                    file_path, text, scope_tag, counter
                )
            except Exception as e:
                print(
                    f"  ERROR {category} on {file_path}: {e}",
                    file=sys.stderr,
                )
                continue

            for issue in issues:
                ledger.add(issue)
                cat_count += 1

        if verbose:
            elapsed = time.time() - cat_start
            print(
                f"  {category:35s} {cat_count:6d} issues ({elapsed:.1f}s)",
                file=sys.stderr,
            )

    if verbose:
        total_elapsed = time.time() - start
        print(
            f"\nTotal: {len(ledger.issues)} issues across {len(files)} "
            f"files ({total_elapsed:.1f}s)",
            file=sys.stderr,
        )

    # Apply the persistent accept-list (Pass 16 Item A). This is the stage
    # where editorially-verified scanner FPs get flipped from `open` to
    # `accepted`. Runs after all check modules so the accept-list can match
    # issues from any category, and runs regardless of --categories filter
    # (matching is keyed on category so off-category entries are ignored).
    if use_accept_list:
        primary = accept_list_path or DEFAULT_ACCEPT_LIST
        try:
            entries = load_accept_list(primary)
        except SystemExit:
            raise
        # Also load the notation-specific accept list (if present).
        try:
            entries.extend(load_accept_list(DEFAULT_NOTATION_ACCEPT_LIST))
        except SystemExit:
            raise
        # Build the scope-aware file set so stale detection does not flag
        # out-of-scope accept-list entries (e.g. vol2 entries in a vol1 scan).
        scanned_files: set[str] = set()
        for f in files:
            try:
                scanned_files.add(
                    f.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
                )
            except ValueError:
                # File outside the repo — shouldn't happen, but don't crash.
                scanned_files.add(str(f))
        result = apply_accept_list(
            ledger, entries, REPO_ROOT, scanned_files=scanned_files
        )
        if verbose or result.total_entries > 0:
            print(f"\n{format_report(result)}", file=sys.stderr)
        for line in format_stale_warnings(result):
            print(line, file=sys.stderr)

    return ledger


def print_summary(ledger: Ledger) -> None:
    """Print a human-readable summary to stderr."""
    summary = ledger.summary()
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"AUDIT LEDGER SUMMARY (scope={ledger.scope})", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"Total issues: {summary['total_issues']}", file=sys.stderr)
    print(f"Scan date: {ledger.scan_date}", file=sys.stderr)
    print(f"Rule file SHA: {ledger.rule_file_sha[:12]}...", file=sys.stderr)
    print(f"\nBy category:", file=sys.stderr)
    for cat in sorted(summary["by_category"]):
        count = summary["by_category"][cat]
        print(f"  {cat:35s} {count:6d}", file=sys.stderr)
    nonzero_statuses = {
        k: v for k, v in summary["by_status"].items() if v > 0
    }
    if nonzero_statuses:
        print(f"\nBy status:", file=sys.stderr)
        for st, count in sorted(nonzero_statuses.items()):
            print(f"  {st:35s} {count:6d}", file=sys.stderr)


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pass 15 audit scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scope",
        required=True,
        help="Scope: vol1, vol2, both, or a path to a file or directory",
    )
    parser.add_argument(
        "--categories",
        default="",
        help="Comma-separated category names to run (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the ledger JSON (default: audit-ledger.json in cwd)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-check timing and counts to stderr",
    )
    parser.add_argument(
        "--accept-list",
        type=Path,
        default=None,
        help=(
            "Path to the persistent accept-list JSON (default: "
            "book/tools/audit/accepted_fps.json). Accept-list entries "
            "flip matching issues from `open` to `accepted`."
        ),
    )
    parser.add_argument(
        "--no-accept-list",
        action="store_true",
        help=(
            "Disable the persistent accept-list. All known-FP entries "
            "will report as `open` — use this to audit the accept-list "
            "itself or to reproduce pre-Pass-16 scanner behavior."
        ),
    )
    args = parser.parse_args()

    categories = (
        [c.strip() for c in args.categories.split(",") if c.strip()]
        if args.categories
        else None
    )

    ledger = scan(
        args.scope,
        categories,
        verbose=args.verbose,
        accept_list_path=args.accept_list,
        use_accept_list=not args.no_accept_list,
    )

    output = args.output or Path("audit-ledger.json")
    ledger.save(output)
    print(f"\nLedger written to {output}", file=sys.stderr)

    print_summary(ledger)
    return 0


if __name__ == "__main__":
    sys.exit(main())
