#!/usr/bin/env python3
"""Tier 1 link checker: validate internal Markdown / Quarto links offline.

Scope on purpose:
  - Validate ONLY relative-path links and same-file anchor links inside
    `.md` / `.qmd` files.
  - DO NOT touch external URLs (http/https/mailto/tel/...). External
    reachability is Lychee's job in CI; doing it here would make
    pre-commit slow and network-flaky.

Why a separate tool from `book/binder`:
  - The book toolchain owns Quarto cross-references (`@fig-foo`,
    `@sec-bar`), bibliography keys, label hygiene, etc.
  - This tool owns plain Markdown link integrity and works repo-wide
    (every Quarto site, plus loose READMEs).

Usage:
  python3 shared/scripts/check-internal-links.py            # check the whole repo
  python3 shared/scripts/check-internal-links.py FILE...    # check named files
  python3 shared/scripts/check-internal-links.py --staged   # check git-staged files
  python3 shared/scripts/check-internal-links.py --quiet    # only print failures

Exit codes:
  0  every internal link resolves
  1  one or more broken internal links (printed file:line: detail)
  2  invocation error
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Directories we never validate links in.
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "_site",
    "_book",
    "_build",
    ".quarto",
    "htmlcov",
    "site-packages",
    # Per-stage build outputs and vendored extensions:
    "_freeze",
    "_extensions",
    # The dev preview mirror gets very large and is a copy of generated HTML.
    "_archive",
}

# File globs we walk when no explicit list is given.
DEFAULT_GLOBS = ("**/*.md", "**/*.qmd")

# Match Markdown inline links: [text](target) and image links: ![alt](target).
# - target may be wrapped in <...> for spaces (rare, but handle it).
# - Not greedy on the [...] part; nested brackets in link text are unusual
#   in our content and would need a pure-PEG parser to handle safely.
LINK_RE = re.compile(
    r"(?P<bang>!?)\[(?P<text>[^\]]*)\]\((?P<target><[^>]+>|[^)\s]+)(?:\s+\"[^\"]*\")?\)"
)

# Match a fenced-code-block opener/closer: ``` or ~~~ optionally followed by
# an info string. Quarto/Pandoc allow attribute braces (```{python}) too.
FENCE_OPEN_RE = re.compile(r"^(?P<indent>\s{0,3})(?P<fence>`{3,}|~{3,})(?P<info>.*)$")

# Strip inline code spans `...` so a backticked target like `[x](y)` inside
# a paragraph isn't treated as a link.
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")

# Match explicit Quarto / Pandoc anchor syntax inside headings:
#   ## My Header {#sec-foo}
ANCHOR_ATTR_RE = re.compile(r"\{#(?P<id>[^\s}]+)[^}]*\}")

# Match ATX headings to derive their slugified id.
HEADING_RE = re.compile(r"^(#{1,6})\s+(?P<heading>.+?)\s*$", re.MULTILINE)

# Strip Quarto callouts and cross-ref macros from headings before slugifying.
HEADING_CLEAN_RE = re.compile(r"\{[^}]*\}|`[^`]*`")

# Schemes that are external and out of scope for this checker.
EXTERNAL_SCHEMES = (
    "http://",
    "https://",
    "mailto:",
    "tel:",
    "ftp://",
    "ftps://",
    "ssh://",
    "git://",
    "data:",
    "javascript:",
)


@dataclass(frozen=True)
class Problem:
    file: Path
    line: int
    target: str
    detail: str

    def render(self, root: Path) -> str:
        try:
            rel = self.file.relative_to(root)
        except ValueError:
            rel = self.file
        return f"{rel}:{self.line}: broken link → {self.target!r} ({self.detail})"


def slugify(heading: str) -> str:
    """Approximate Pandoc's `--id-prefix=section` slug for an ATX heading.

    Matches Pandoc's `auto_identifiers` extension well enough for our content:
      - Strip leading non-alphanumerics.
      - Replace whitespace with single hyphens.
      - Lowercase.
      - Drop chars that aren't alnum, hyphen, underscore, period, or colon.
    """
    cleaned = HEADING_CLEAN_RE.sub("", heading).strip()
    # Pandoc strips leading non-letter characters from the slug.
    cleaned = re.sub(r"^[^A-Za-z]+", "", cleaned)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = re.sub(r"[^a-z0-9_\-.:]", "", cleaned)
    return cleaned


def collect_anchors(text: str) -> set[str]:
    """Return the set of anchor ids defined in the given Markdown source.

    Combines:
      - Explicit `{#id}` attributes anywhere in the document.
      - Auto-generated heading slugs.
    """
    anchors: set[str] = set()

    for m in ANCHOR_ATTR_RE.finditer(text):
        anchors.add(m.group("id"))

    for m in HEADING_RE.finditer(text):
        heading = m.group("heading").strip()
        # If a heading already declares {#id}, the explicit id wins; capture it
        # AND skip the slugified form because the auto slug isn't generated.
        explicit = ANCHOR_ATTR_RE.search(heading)
        if explicit:
            continue
        slug = slugify(heading)
        if slug:
            anchors.add(slug)

    return anchors


def is_external(target: str) -> bool:
    return target.startswith(EXTERNAL_SCHEMES)


def split_target(target: str) -> tuple[str, str]:
    """Split a link target into (path, anchor)."""
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if "#" not in target:
        return target, ""
    path, _, anchor = target.partition("#")
    return path, anchor


def candidate_paths(source: Path, raw: str) -> list[Path]:
    """Return possible filesystem paths a link target could resolve to.

    Quarto sites resolve relative links against the file's directory, but
    `.qmd` files often link to a sibling without the extension (the rendered
    output is `.html`). We also accept `index.qmd` shorthand for directory
    targets.
    """
    if not raw:
        return []
    base = source.parent
    if raw.startswith("/"):
        # Site-absolute paths are resolved at render time and depend on the
        # site's configured `site-url` / base path. We can't validate them
        # offline reliably, so skip with a soft pass.
        return []

    raw_path = Path(raw)
    candidates = [base / raw_path]
    # Sometimes authors write `foo` when they mean `foo.qmd` (rendered as
    # `foo.html`). Only meaningful when raw has no extension.
    if not raw_path.suffix:
        candidates.append((base / raw_path).with_suffix(".qmd"))
        candidates.append((base / raw_path).with_suffix(".md"))
        candidates.append((base / raw_path).with_suffix(".html"))
        candidates.append(base / raw_path / "index.qmd")
        candidates.append(base / raw_path / "index.md")
    elif raw_path.suffix == ".html":
        # `foo.html` in source most likely points at sibling `foo.qmd`.
        candidates.append((base / raw_path).with_suffix(".qmd"))
        candidates.append((base / raw_path).with_suffix(".md"))

    return candidates


def file_text_cache() -> "dict[Path, str]":
    return {}


def read(path: Path, cache: dict[Path, str]) -> str | None:
    if path in cache:
        return cache[path]
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    cache[path] = text
    return text


def check_file(path: Path, cache: dict[Path, str]) -> list[Problem]:
    if not path.exists():
        return [Problem(path, 0, "", "file does not exist")]
    text = read(path, cache)
    if text is None:
        return [Problem(path, 0, "", "file unreadable as UTF-8")]

    own_anchors = collect_anchors(text)
    problems: list[Problem] = []

    in_fence = False
    fence_marker: str | None = None  # The exact `` `... `` or `~~~...` that opened the block.

    for line_no, line in enumerate(text.splitlines(), start=1):
        # Track fenced code blocks (``` or ~~~). Inside a fence, skip all link
        # parsing — TikZ, raw LaTeX, and code samples otherwise produce piles of
        # false positives that look like Markdown links.
        fence_match = FENCE_OPEN_RE.match(line)
        if fence_match:
            marker = fence_match.group("fence")[0] * 3  # normalize length to 3
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif fence_marker is not None and line.lstrip().startswith(fence_marker):
                in_fence = False
                fence_marker = None
            continue
        if in_fence:
            continue

        # Strip inline code so backticked link-shaped strings don't trip us.
        scan_line = INLINE_CODE_RE.sub("", line)

        for m in LINK_RE.finditer(scan_line):
            target = m.group("target")
            if not target or is_external(target):
                continue
            path_part, anchor = split_target(target)

            if not path_part:
                # Pure same-file anchor.
                if anchor and anchor not in own_anchors:
                    problems.append(
                        Problem(path, line_no, target, "anchor not found in this file")
                    )
                continue

            cands = candidate_paths(path, path_part)
            if not cands:
                continue  # Site-absolute or unparseable; skip.

            resolved = next((c for c in cands if c.exists()), None)
            if resolved is None:
                problems.append(
                    Problem(path, line_no, target, f"no such file (checked {len(cands)} candidate paths)")
                )
                continue

            if anchor:
                target_text = read(resolved, cache)
                if target_text is None:
                    # File exists but isn't text we can scan (e.g. a binary).
                    # Treat anchor as opaque; don't flag.
                    continue
                target_anchors = collect_anchors(target_text)
                if anchor not in target_anchors:
                    problems.append(
                        Problem(path, line_no, target, f"anchor #{anchor} not found in {resolved.name}")
                    )

    return problems


def staged_files(root: Path) -> list[Path]:
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]
    try:
        out = subprocess.check_output(cmd, cwd=root, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        sys.stderr.write(f"check-internal-links: failed to list staged files: {exc}\n")
        sys.exit(2)

    files = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if not (line.endswith(".md") or line.endswith(".qmd")):
            continue
        files.append((root / line).resolve())
    return files


def discover_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for glob in DEFAULT_GLOBS:
        for candidate in root.glob(glob):
            if any(part in EXCLUDE_DIRS for part in candidate.parts):
                continue
            files.append(candidate)
    return files


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="Specific .md/.qmd files to check.")
    parser.add_argument("--staged", action="store_true", help="Check files staged for commit.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only print failures.")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="Exclude files whose repo-relative path matches this glob. Repeatable.",
    )
    args = parser.parse_args(argv)

    if args.staged and args.files:
        parser.error("--staged is mutually exclusive with explicit FILES.")

    root = REPO_ROOT
    if args.staged:
        files = staged_files(root)
    elif args.files:
        files = []
        for raw in args.files:
            p = Path(raw)
            if not p.is_absolute():
                p = (root / p).resolve()
            if p.suffix not in (".md", ".qmd"):
                continue
            if any(part in EXCLUDE_DIRS for part in p.parts):
                continue
            files.append(p)
    else:
        files = discover_files(root)

    if args.exclude:
        import fnmatch

        kept = []
        for f in files:
            try:
                rel = str(f.relative_to(root))
            except ValueError:
                rel = str(f)
            if any(fnmatch.fnmatch(rel, pat) for pat in args.exclude):
                continue
            kept.append(f)
        files = kept

    if not files:
        if not args.quiet:
            print("check-internal-links: no .md/.qmd files to check.")
        return 0

    cache: dict[Path, str] = {}
    all_problems: list[Problem] = []
    for path in sorted(set(files)):
        all_problems.extend(check_file(path, cache))

    if all_problems:
        for prob in all_problems:
            print(prob.render(root))
        print(f"\ncheck-internal-links: {len(all_problems)} broken internal link(s) in {len({p.file for p in all_problems})} file(s).", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"check-internal-links: OK ({len(files)} file(s) scanned).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
