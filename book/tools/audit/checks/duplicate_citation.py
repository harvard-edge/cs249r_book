r"""Check: duplicate Pandoc citation keys within a close window.

Rule: book-prose-merged.md section 10.X (citation density)

    The same `[@key]` cited 3+ times within a tight window (~50 lines)
    inside one file usually means the author over-cited a single source.
    The fix is editorial: collapse to one citation, rephrase so a single
    cite covers the cluster, or distribute the cites across distinct
    claims that need separate attribution. This is a *judgment* check —
    sometimes 3+ cites of the same key really are needed because each
    cite anchors a distinct claim that, on its own, would be unsupported.

Auto-fixable: NO. The fix is a rewrite decision, not a substitution.
Confidence: LOW. We expect a non-trivial false-positive rate because
some clusters legitimately re-cite the same key across distinct claims.
We surface for editor review without auto-fixing.

Protected contexts:
  - Code fences and YAML frontmatter (no citations to flag there).
  - Bibliography files (`.bib`) — they DEFINE keys, they don't cite them.
  - Footnote-DEFINITION blocks (`[^fn-id]: ...` lines and continuations).
    Footnotes routinely re-cite the source they footnote, and that
    pattern is not over-citation.

Citation forms recognized (Pandoc citeproc):
  [@key]
  [@key, @key2; @key3]
  [-@key]                    (suppress-author form)
  [@key, pp. 12-15]          (locator)
  @key                       (bare in-text "as @vaswani2017 showed")
  [see @key for ...]         (prefix form)

We treat any `@<key>` token NOT preceded by `\` and NOT inside a
`@(sec|fig|tbl|eq|lst)-...` cross-reference as a citation. The key is
everything matching `[\w][\w:.-]*` after the `@`. Cross-references use
the four reserved prefixes and are filtered out explicitly.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from audit.ledger import Issue, make_issue_id
from audit.protected_contexts import LineWalker

CATEGORY = "duplicate-citation"
RULE = "book-prose-merged.md section 10.X"
RULE_TEXT = (
    "The same citation key cited 3+ times within ~50 lines suggests "
    "over-citation; consider collapsing to one cite or rephrasing."
)

# Window size (lines) within which 3+ uses of the same key fire.
WINDOW_LINES = 50
# Minimum number of cites in the window to fire.
MIN_CITES = 3

# Reserved Quarto cross-reference prefixes — NOT citations.
_XREF_PREFIXES = ("sec-", "fig-", "tbl-", "eq-", "lst-")

# Any @<key> NOT escaped (no leading backslash), NOT preceded by an
# identifier char (avoid email-like `user@host`), and NOT preceded by `!`.
# The key body allows letters, digits, underscore, colon, hyphen, dot.
# The leading `-` is optional (suppress-author form, `[-@key]`).
_CITE_RE = re.compile(r"(?<![\\\w@!])-?@([A-Za-z][\w:.\-]*)")

# Footnote-definition opener: `[^fn-id]: ...`. Once we see one of these,
# the rest of that paragraph (until a blank line) is footnote body.
_FN_DEF_RE = re.compile(r"^\s*\[\^[^\]]+\]:\s")


def _should_skip_file(file_path: Path) -> bool:
    """Skip non-prose files where duplicate-citation counts are noisy.

    - `.bib` files DEFINE citation keys, they don't cite them.
    - Frontmatter / backmatter / glossary / parts / appendices contain
      either no citations at all or compact citation lists where 3+ uses
      of one key are intentional (e.g. a "further reading" appendix).
    """
    path = file_path.as_posix()
    if file_path.suffix == ".bib":
        return True
    for needle in (
        "/frontmatter/",
        "/backmatter/",
        "/parts/",
        "/glossary/",
        "/appendix/",
        "/appendices/",
        "/references",
    ):
        if needle in path:
            return True
    return False


def _extract_citations(line: str) -> list[tuple[int, str]]:
    """Return (col, key) tuples for every citation on the line.

    Filters out Quarto cross-references (@sec-, @fig-, @tbl-, @eq-, @lst-).
    """
    out: list[tuple[int, str]] = []
    for m in _CITE_RE.finditer(line):
        key = m.group(1)
        if key.startswith(_XREF_PREFIXES):
            continue
        out.append((m.start(), key))
    return out


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan for citation keys reused 3+ times in a 50-line window."""
    issues: list[Issue] = []
    counter = start_counter

    if _should_skip_file(file_path):
        return issues, counter

    # First pass: gather every (line_num, col, key) citation, skipping
    # code fences, YAML, display math, and footnote-definition bodies.
    cites: list[tuple[int, int, str, str]] = []  # (line_num, col, key, raw_line)

    walker = LineWalker(text)
    in_footnote_def = False

    for line, state, line_num in walker:
        # Skip block-protected contexts entirely.
        if state.in_code_fence or state.in_yaml or state.in_display_math:
            in_footnote_def = False
            continue

        # Footnote-definition tracking: opener resets to True; a blank
        # line ends the footnote body.
        if _FN_DEF_RE.match(line):
            in_footnote_def = True
            continue
        if in_footnote_def:
            if line.strip() == "":
                in_footnote_def = False
            # Either way, skip citations inside the footnote body.
            continue

        if "@" not in line:
            continue

        for col, key in _extract_citations(line):
            cites.append((line_num, col, key, line))

    if not cites:
        return issues, counter

    # Group citations by key.
    by_key: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for line_num, col, key, raw in cites:
        by_key[key].append((line_num, col, raw))

    # For each key with at least MIN_CITES total uses, find any
    # sliding 50-line window containing MIN_CITES+ uses. Emit ONE
    # issue per (file, key) — anchored at the first cite of the
    # densest qualifying cluster.
    for key, occurrences in by_key.items():
        if len(occurrences) < MIN_CITES:
            continue
        occurrences.sort()  # by line_num
        # Sliding window over occurrences: find the first index i for which
        # occurrences[i + MIN_CITES - 1].line_num - occurrences[i].line_num
        # <= WINDOW_LINES. That cluster is what we report.
        found_cluster: list[tuple[int, int, str]] | None = None
        for i in range(len(occurrences) - MIN_CITES + 1):
            window_start = occurrences[i][0]
            # Expand the window to include all occurrences within
            # WINDOW_LINES of window_start.
            j = i
            while (
                j < len(occurrences)
                and occurrences[j][0] - window_start <= WINDOW_LINES
            ):
                j += 1
            cluster = occurrences[i:j]
            if len(cluster) >= MIN_CITES:
                found_cluster = cluster
                break

        if not found_cluster:
            continue

        first_line, first_col, raw_line = found_cluster[0]
        last_line = found_cluster[-1][0]
        cite_count = len(found_cluster)
        line_nums = [str(o[0]) for o in found_cluster]

        issues.append(
            Issue(
                id=make_issue_id(scope, CATEGORY, counter),
                category=CATEGORY,
                rule=RULE,
                rule_text=RULE_TEXT,
                file=str(file_path),
                line=first_line,
                col=first_col,
                before=raw_line,
                suggested_after=(
                    f"consider collapsing to one cite or rephrasing "
                    f"(@{key} cited {cite_count}x within lines "
                    f"{first_line}-{last_line})"
                ),
                auto_fixable=False,
                needs_subagent=False,
                confidence="low",
                reason=(
                    f"@{key} cited {cite_count} times within "
                    f"{last_line - first_line + 1} lines "
                    f"(at lines {', '.join(line_nums)})"
                ),
            )
        )
        counter += 1

    return issues, counter
