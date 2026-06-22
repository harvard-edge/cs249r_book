"""Check: `\\index{}` placement — forbidden contexts.

Rule: index.md §1 (placement) + book-prose-merged.md §7 (LaTeX in attribute strings)

    "Place the tag immediately after the term IN BODY PROSE. Not before,
    not at the end of the paragraph."

This check enforces the converse: `\\index{...}` must NEVER appear in
contexts where LaTeX never sees it OR where it leaks as literal text
into the rendered output. Four forbidden contexts:

  1. Inside Python code fences (` ```{python} ... ``` ` blocks). LaTeX
     never indexes code-block content; the tag is dead weight at best
     and leaks as literal `\\index{...}` text in displayed code at worst
     (vol1/frameworks:2736 was the user-visible defect).

  2. Inside `$$..$$` block math. `\\index{}` inside `\\text{}` corrupts
     the makeindex key parsing (anti-pattern #4 'Math in key', plus the
     nested-brace anti-pattern #5).

  3. Inside `$..$` inline math. Same reasoning as block math.

  4. Inside attribute strings: `fig-cap=`, `tbl-cap=`, `lst-cap=`,
     `fig-alt=`, `tbl-alt=`, `title=`. Per book-prose-merged.md §7,
     these are extracted as plain text by Quarto for HTML title
     tooltips, PDF bookmarks, and EPUB metadata — so `\\index{...}`
     leaks as literal text.

The canonical fix per index.md §1: relocate the tag to body prose,
immediately after the term, in the first chapter that substantively
discusses the concept.

Auto-fixable: NO. The relocation requires reading the surrounding
prose to find the right body-prose mention; some tags should be
dropped (per §2 'Do NOT index passing mentions') rather than moved.
Marked `needs_subagent=False` because the per-site judgment is
quick once the operator sees the file:line.
"""

from __future__ import annotations

import re
from pathlib import Path

from audit.ledger import Issue, make_issue_id

CATEGORY = "index-placement"
RULE = "index.md §1 + book-prose-merged.md §7"
RULE_TEXT = "\\index{} must be placed in body prose only (not in code, math, or attribute strings)"

_INDEX_TAG_RE = re.compile(r"\\index\{[^}]+\}")
_FENCE_OPEN_RE = re.compile(r"^```(\{[^}]*\})?(.*)$")
_BLOCK_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)([^\$\n]{1,400})\$(?!\$)")

_ATTR_NAMES = ("fig-cap", "fig-alt", "tbl-cap", "tbl-alt", "lst-cap", "title")


def _emit(issues, counter, scope, file_path, line_no, line, context_label, reason):
    issues.append(
        Issue(
            id=make_issue_id(scope, CATEGORY, counter),
            category=CATEGORY,
            rule=RULE,
            rule_text=RULE_TEXT,
            file=str(file_path),
            line=line_no,
            col=0,
            before=line[:200],
            suggested_after="",  # relocation needs human judgment
            auto_fixable=False,
            needs_subagent=False,
            reason=f"\\index{{}} in {context_label}: {reason}",
        )
    )
    return counter + 1


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan for `\\index{}` in forbidden contexts."""
    issues: list[Issue] = []
    counter = start_counter
    lines = text.split("\n")

    # Pass 1 — line-by-line: code fences and attribute strings
    in_fence = False
    fence_kind = None  # 'python', 'tikz', or 'other'
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        m = _FENCE_OPEN_RE.match(stripped)
        if m and not in_fence:
            spec = (m.group(1) or "") + (m.group(2) or "")
            spec_lower = spec.lower()
            # A bare ``` (empty spec) is a CLOSE — not an open. Only treat as
            # open if the spec actually names a language. Otherwise skip the
            # line (orphan / stray close fence — common in malformed sources).
            if not spec.strip():
                continue
            if "python" in spec_lower:
                in_fence = True
                fence_kind = "python"
            elif "tikz" in spec_lower or "=latex" in spec_lower:
                in_fence = True
                fence_kind = "tikz"
            else:
                in_fence = True
                fence_kind = "other"
            continue
        elif m and in_fence:
            in_fence = False
            fence_kind = None
            continue

        if in_fence and fence_kind == "python":
            # Forbidden: \index{} inside Python code (anti-pattern #1)
            if r"\index{" in line:
                counter = _emit(
                    issues, counter, scope, file_path, line_num, line,
                    "Python code fence",
                    "LaTeX never indexes code; relocate tag to body prose",
                )
            continue

        if in_fence:
            # Skip tikz/other — those are LaTeX-rendered and \index{} is unusual but not strictly forbidden
            continue

        # Forbidden: \index{} inside attribute strings (anti-pattern #4)
        for attr in _ATTR_NAMES:
            attr_prefix = f'{attr}="'
            if attr_prefix not in line:
                continue
            idx = line.index(attr_prefix) + len(attr_prefix)
            end = line.find('"', idx)
            if end < 0:
                continue  # unterminated; skip
            if r"\index{" in line[idx:end]:
                counter = _emit(
                    issues, counter, scope, file_path, line_num, line,
                    f"{attr}= attribute string",
                    "leaks as literal text into HTML title tooltip / PDF bookmark; relocate to body prose",
                )

    # Pass 2 — block math $$..$$  (can span lines)
    for m in _BLOCK_MATH_RE.finditer(text):
        if r"\index{" in m.group(1):
            line_num = text[: m.start()].count("\n") + 1
            counter = _emit(
                issues, counter, scope, file_path, line_num,
                m.group(0)[:200].replace("\n", " "),
                "$$..$$ math block",
                "\\index{} inside math corrupts makeindex key; relocate to body prose",
            )

    # Pass 3 — inline math $..$  (single-line)
    for line_num, line in enumerate(lines, 1):
        for m in _INLINE_MATH_RE.finditer(line):
            if r"\index{" in m.group(1):
                counter = _emit(
                    issues, counter, scope, file_path, line_num, line,
                    "$..$ inline math",
                    "\\index{} inside math corrupts makeindex key; relocate to body prose",
                )

    return issues, counter
