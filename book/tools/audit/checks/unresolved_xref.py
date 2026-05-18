"""Check: unresolved Quarto cross-references (`?@` literal in rendered output).

Defect class: when a Quarto cross-reference like `@sec-foo` fails to resolve,
the rendered HTML/PDF shows the literal text `?@sec-foo` (with a question
mark prefix). Source-level scanners typically miss this because the source
`@sec-foo` looks syntactically valid — only the missing target ID gives it
away, and the target may live in a sibling file.

This check is CROSS-FILE: it must read every chapter in both volumes to
build the index of defined IDs before deciding whether any single
reference is unresolved. The per-file `check()` entrypoint required by
the registry therefore lazy-builds (and memoizes) the corpus-wide ID
index on first invocation, then validates the references in the current
file against it. The `scan_corpus()` entrypoint is provided for the
`binder check release` orchestrator which wants a single global pass.

Relationship to `./book/binder check refs`:

  The existing `binder check refs` (specifically the `--scope orphans`
  sub-scope, see book/cli/commands/validate.py `_run_unreferenced_labels`)
  already detects unresolved references. At HEAD `b68e665ba9` it returns
  zero broken refs. This scanner therefore exists to (a) surface the same
  defect class in the unified audit-ledger format so the release gate
  doesn't have to scrape multiple tool outputs, and (b) catch any future
  cases where `binder check refs` would miss a ref — typically references
  inside files or fenced regions that the binder's orphan scope filters
  out but that still render as `?@` in the published HTML/PDF.

Auto-fixable: NO. A missing target could be a typo in the reference, a
missing `{#sec-foo}` anchor on the intended target, or a stale reference
to deleted content. Each case requires human judgment.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from audit.ledger import Issue, make_issue_id

CATEGORY = "unresolved-xref"
RULE = "Quarto cross-reference must resolve to a defined ID"
RULE_TEXT = (
    "Every @sec-/@fig-/@tbl-/@eq-/@lst- reference must point to a "
    "{#sec-...}/{#fig-...}/etc. anchor defined somewhere in the corpus, "
    "otherwise the rendered output shows the literal `?@sec-foo` text."
)

# ── Patterns ────────────────────────────────────────────────────────────────

# ID-discovery: match every defined anchor in the corpus.
# Matches `{#sec-foo}`, `{#fig-bar-baz}`, `{#tbl-xyz .class}`, etc.
# Also picks up the Quarto YAML cell-option form `#| label: fig-xyz`.
_ID_ANCHOR_RE = re.compile(r"\{#((?:sec|fig|tbl|eq|lst)-[\w.:-]+)")
_ID_YAML_LABEL_RE = re.compile(
    r"^\s*#\|\s*label:\s*((?:sec|fig|tbl|eq|lst)-[\w.:-]+)"
)
_ID_JUPYTER_LABEL_RE = re.compile(
    r"^\s*%%\|\s*label:\s*((?:sec|fig|tbl|eq|lst)-[\w.:-]+)"
)

# Reference-discovery: match every @sec-/@fig-/@tbl-/@eq-/@lst- usage in prose.
# We allow letters, digits, hyphens, underscores, and dots in the slug —
# matching the LABEL_REF_PATTERN convention in book/cli/commands/validate.py.
# Trailing punctuation (`,`, `.`, `;`, `:`, `)`) is NOT part of the slug.
_XREF_RE = re.compile(
    r"(?<![\w@!])@((?:sec|fig|tbl|eq|lst)-[A-Za-z0-9][\w-]*)"
)

# Citation prefixes that look like xrefs but are bibliography keys. Quarto
# treats `[@foo2020]` as a citation; we only flag the cross-ref slugs above.
# The reference regex above already requires a sec-/fig-/tbl-/eq-/lst- prefix.


# ── Module-level corpus index (memoized) ────────────────────────────────────

# Cached set of defined IDs across both volumes. Built on first call.
# Keyed by the resolved root path so multiple roots don't collide.
_ID_INDEX_CACHE: dict[Path, frozenset[str]] = {}


def _content_root(file_path: Path) -> Optional[Path]:
    """Walk up from `file_path` to find the `book/quarto/contents` root.

    Returns the contents directory containing vol1/ and vol2/, or None
    if `file_path` is not inside such a tree.
    """
    for parent in file_path.resolve().parents:
        candidate = parent / "book" / "quarto" / "contents"
        if candidate.is_dir():
            return candidate
        if parent.name == "contents" and (parent / "vol1").is_dir():
            return parent
    return None


def _build_id_index(root: Path) -> frozenset[str]:
    """Walk vol1/ + vol2/ under `root` and return the set of defined IDs."""
    ids: set[str] = set()
    for vol in ("vol1", "vol2"):
        vol_root = root / vol
        if not vol_root.is_dir():
            continue
        for qmd in vol_root.rglob("*.qmd"):
            try:
                text = qmd.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            # `{#sec-foo}` style anchors anywhere in the file.
            for m in _ID_ANCHOR_RE.finditer(text):
                ids.add(m.group(1))
            # `#| label: fig-foo` cell-option labels (per-line).
            for line in text.splitlines():
                m = _ID_YAML_LABEL_RE.match(line)
                if m:
                    ids.add(m.group(1))
                m = _ID_JUPYTER_LABEL_RE.match(line)
                if m:
                    ids.add(m.group(1))
    return frozenset(ids)


def _get_id_index(root: Path) -> frozenset[str]:
    """Return the memoized ID index for `root`, building it if needed."""
    resolved = root.resolve()
    if resolved not in _ID_INDEX_CACHE:
        _ID_INDEX_CACHE[resolved] = _build_id_index(resolved)
    return _ID_INDEX_CACHE[resolved]


# ── Per-file check (registry entrypoint) ────────────────────────────────────


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan `file_path` for unresolved cross-references.

    The set of *defined* IDs is corpus-wide (vol1+vol2), built lazily and
    memoized. The set of *referenced* IDs is just this file's content.
    Any reference whose target is not in the corpus-wide index is flagged.

    Skips fenced code blocks and HTML comments — references inside those
    regions don't render as `?@` because they aren't rendered at all.
    """
    issues: list[Issue] = []
    counter = start_counter

    root = _content_root(file_path)
    if root is None:
        # Not inside a recognized content tree; can't validate.
        return issues, counter

    defined_ids = _get_id_index(root)

    # Walk lines, tracking simple code-fence + HTML-comment state. We do
    # this inline (rather than via LineWalker) because this check only
    # needs the two coarsest exclusions; any deeper protection (math,
    # inline code, etc.) is irrelevant because @sec-foo inside `code`
    # still doesn't render as a working ref, but it ALSO doesn't render
    # as the broken `?@sec-foo` text — so we let those through silently
    # rather than emit false positives.
    in_code_block = False
    in_html_comment = False

    for line_idx, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()

        # HTML comments first (may contain ``` lines).
        if not in_code_block:
            if "<!--" in line and "-->" not in line[line.index("<!--"):]:
                in_html_comment = True
                continue
            if in_html_comment:
                if "-->" in line:
                    in_html_comment = False
                continue

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        if "@" not in line:
            continue

        for m in _XREF_RE.finditer(line):
            slug = m.group(1)
            if slug in defined_ids:
                continue
            kind = slug.split("-", 1)[0]
            issues.append(
                Issue(
                    id=make_issue_id(scope, CATEGORY, counter),
                    category=CATEGORY,
                    rule=RULE,
                    rule_text=RULE_TEXT,
                    file=str(file_path),
                    line=line_idx,
                    col=m.start(),
                    before=line,
                    suggested_after="",  # No automatic fix; needs review
                    auto_fixable=False,
                    needs_subagent=False,
                    confidence="high",
                    reason=(
                        f"@{slug} has no matching {{#{slug}}} anchor in "
                        f"vol1/ or vol2/. Either fix the typo in the "
                        f"reference, or add a `{{#{slug}}}` anchor on the "
                        f"intended {kind.replace('sec', 'section').replace('fig', 'figure').replace('tbl', 'table').replace('eq', 'equation').replace('lst', 'listing')} target."
                    ),
                )
            )
            counter += 1

    return issues, counter


# ── Corpus-wide entrypoint (for `binder check release`) ─────────────────────


def scan_corpus(root: Path, scope: str = "both") -> list[Issue]:
    """Scan every .qmd in vol1/ + vol2/ and return unresolved-xref issues.

    `root` should be the `book/quarto/contents` directory (or any parent
    of it — we walk up to find the contents root). `scope` controls the
    Issue id prefix only; defaults to "both".

    This is the entrypoint the `binder check release` orchestrator calls.
    It deduplicates the cache, so subsequent per-file `check()` calls in
    the same process reuse the index.
    """
    # Locate the contents root.
    contents_root = None
    resolved = root.resolve()
    if (resolved / "vol1").is_dir() and (resolved / "vol2").is_dir():
        contents_root = resolved
    else:
        # Try descending: maybe `root` is the repo root.
        candidate = resolved / "book" / "quarto" / "contents"
        if candidate.is_dir():
            contents_root = candidate
    if contents_root is None:
        # Walk parents to find one.
        for parent in resolved.parents:
            candidate = parent / "book" / "quarto" / "contents"
            if candidate.is_dir():
                contents_root = candidate
                break
    if contents_root is None:
        return []

    # Force-build the index for this root.
    _get_id_index(contents_root)

    all_issues: list[Issue] = []
    counter = 0
    for vol in ("vol1", "vol2"):
        vol_root = contents_root / vol
        if not vol_root.is_dir():
            continue
        for qmd in sorted(vol_root.rglob("*.qmd")):
            try:
                text = qmd.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            issues, counter = check(qmd, text, scope, counter)
            all_issues.extend(issues)
    return all_issues
