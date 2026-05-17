#!/usr/bin/env python3
"""
Post-render guard: fail the build if any unresolved cross-reference literal
(`?@xxx-yyy`) remains in the rendered output.

WHY THIS EXISTS
---------------
vol1/vol2 build as `project.type: website`, so Quarto cannot resolve
cross-chapter `@xref`s natively. `scripts/fix_cross_references.py` patches
them up using a manually-maintained label registry. That registry has had
gaps in the past — most recently, the regex only handled `sec-`/`pri-` and
silently dropped every cross-chapter `@fig-`/`@tbl-`/`@eq-`/`@lst-` ref,
shipping literal `?@fig-foo` text into the live HTML.

Static pre-commit checks (`./binder check labels --scope orphans`) cannot
catch this class of bug — they only verify a label exists in *some* source
file, not that Quarto's render-time crossref resolver actually wired it up.

This script is the render-truth backstop. It scans the build output for any
residual `?@` literal and exits non-zero. Wired into Quarto's post-render
hook chain so every build (local or CI) gates on it.

USAGE
-----
1. Post-render hook (Quarto invokes with no args): scans the most recent
   build directory under `_build/html*/` or `_build/epub*/`.
2. Manual: `python3 scripts/verify_rendered_xrefs.py [<build-dir>]` to scan
   a specific directory.

EXIT CODES
----------
0 — no residual `?@` references found.
1 — one or more residual references found (build should fail).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Matches the literal that survives when the post-render fix script can't
# resolve a reference. Quarto emits both `<strong>?@xxx-yyy</strong>` (the
# common shape in our website-mode builds) and bare `?@xxx-yyy` in some
# contexts; we look for the bare core pattern so both forms are caught.
RESIDUAL_XREF_RE = re.compile(r'\?@([a-z]+-[a-zA-Z0-9_-]+)')

# Files inside the build tree we never want to scan (third-party assets that
# can legitimately contain `?@` substrings in minified JS, search indexes,
# EPUB nav/cover/title pages, etc.)
SKIP_PATH_FRAGMENTS = (
    "site_libs/",
    "search.json",
    "sitemap.xml",
    "robots.txt",
    "nav.xhtml",
    "cover.xhtml",
    "title_page.xhtml",
)

CONTENT_GLOBS = ("*.html", "*.xhtml")


def _iter_files(build_dir: Path):
    """Yield content HTML/XHTML files under the build directory.

    Handles three build layouts:
      - HTML site:  build_dir/contents/.../*.html + build_dir/index.html
      - EPUB extracted: build_dir/EPUB/text/*.xhtml
      - Anything else: recursive walk picking up *.html / *.xhtml
    """
    seen = set()
    for pattern in CONTENT_GLOBS:
        for f in build_dir.rglob(pattern):
            if f in seen:
                continue
            seen.add(f)
            yield f


def scan_build_dir(build_dir: Path) -> dict[str, list[tuple[Path, int, str]]]:
    """Return mapping of unresolved-ref → list of (file, line_no, context).

    `context` is a short prose snippet for the user to locate the failure.
    """
    findings: dict[str, list[tuple[Path, int, str]]] = {}

    for f in _iter_files(build_dir):
        rel = f.relative_to(build_dir)
        if any(frag in str(rel) for frag in SKIP_PATH_FRAGMENTS):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for i, line in enumerate(text.splitlines(), start=1):
            for m in RESIDUAL_XREF_RE.finditer(line):
                ref = m.group(1)
                start = max(0, m.start() - 40)
                end = min(len(line), m.end() + 40)
                snippet = line[start:end].strip()
                findings.setdefault(ref, []).append((rel, i, snippet))

    return findings


def _pick_build_dirs(explicit: Path | None) -> list[Path]:
    if explicit is not None:
        return [explicit]
    # When invoked as a Quarto post-render hook, only check the build that was
    # just produced — Quarto sets QUARTO_PROJECT_OUTPUT_DIR for the active
    # project. Without this guard, a vol1 post-render run would also flag
    # stale leaks in a previously-built vol2 directory, conflating the failure.
    quarto_out = os.environ.get("QUARTO_PROJECT_OUTPUT_DIR")
    if quarto_out:
        path = Path(quarto_out)
        return [path] if path.is_dir() else []
    # Manual invocation with no args and no Quarto env: scan every html-* and
    # epub-* directory so the user can audit the full local build state at once.
    build_root = Path("_build")
    if not build_root.is_dir():
        return []
    dirs = []
    for pattern in ("html*", "epub*"):
        dirs.extend(sorted(build_root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True))
    return [d for d in dirs if d.is_dir()]


def _format_report(build_dir: Path, findings: dict) -> str:
    lines = [f"❌ Unresolved cross-references in {build_dir}:"]
    for ref in sorted(findings):
        occurrences = findings[ref]
        lines.append(f"  ?@{ref}  ({len(occurrences)} occurrence{'s' if len(occurrences) != 1 else ''})")
        # Show at most 3 sample locations per ref to keep the report scannable.
        for path, line_no, snippet in occurrences[:3]:
            lines.append(f"    {path}:{line_no}  ...{snippet}...")
        if len(occurrences) > 3:
            lines.append(f"    ({len(occurrences) - 3} more)")
    return "\n".join(lines)


def main() -> int:
    explicit = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    build_dirs = _pick_build_dirs(explicit)

    if not build_dirs:
        print("⚠️  No build directory found — skipping render-truth crossref check")
        return 0

    any_failures = False
    for build_dir in build_dirs:
        findings = scan_build_dir(build_dir)
        if findings:
            any_failures = True
            # Write failure details to stderr so they survive the binder build
            # wrapper's `subprocess.run(capture_output=True)` — that wrapper
            # surfaces stderr on non-zero exit but suppresses stdout.
            print(_format_report(build_dir, findings), file=sys.stderr)
            total = sum(len(v) for v in findings.values())
            print(
                f"  → {total} unresolved reference{'s' if total != 1 else ''} "
                f"across {len(findings)} distinct label{'s' if len(findings) != 1 else ''}.",
                file=sys.stderr,
            )
            print(
                "  Fix: define the missing labels in source, or extend "
                "scripts/fix_cross_references.py to handle the prefix.",
                file=sys.stderr,
            )
        else:
            print(f"✅ No unresolved cross-references in {build_dir}")

    return 1 if any_failures else 0


if __name__ == "__main__":
    sys.exit(main())
