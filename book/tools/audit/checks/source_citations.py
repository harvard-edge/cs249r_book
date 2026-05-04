"""Check: Source-attribution formatting in prose and captions.

This is the binder-native replacement for the old `manage_sources.py`
one-off. It enforces the small set of source-attribution conventions the
book uses in figure captions, table captions, listings, and the rare
body-prose source note:

- `Source:` is capitalized
- source notes end with a period
- academic sources use `Source: [@key]`
- source notes are not wrapped in stray asterisks
- source notes do not contain doubled periods or extra brackets

The check deliberately skips code fences, raw HTML/LaTeX blocks, YAML,
and similar protected contexts via `LineWalker`. It does not try to
rewrite content; it only flags formatting problems that should be fixed
in the source text itself.
"""

from __future__ import annotations

import re
from pathlib import Path

from audit.ledger import Issue, make_issue_id
from audit.protected_contexts import LineWalker, is_python_chunk_option

CATEGORY = "source-attribution"
RULE = "book-prose-merged.md section 6 / source note hygiene"
RULE_TEXT = "Source notes should be capitalized, punctuated, and bracketed consistently"

_ASTERISK_SOURCE_RE = re.compile(r"\*[Ss]ource:[^*]*\*")
_LOWERCASE_SOURCE_RE = re.compile(r"\bsource:")
_CAPITALIZED_SOURCE_RE = re.compile(r"\bSource:\s*")
_SOURCE_RE = re.compile(r"\b(?:Source|source):\s*")
_MALFORMED_ACADEMIC_RE = re.compile(r"\b[Ss]ource:\s*@[^[]")
_EXTRA_BRACKETS_RE = re.compile(r"\b[Ss]ource:\s*\[\[@")


def _looks_like_source_note(line: str, start: int) -> bool:
    """Heuristic guard so we only flag actual source notes.

    A plain prose phrase such as "common source:" should not be treated
    as an editorial source note. Real source notes usually start at a
    sentence boundary or after a structural delimiter in captions.
    """
    if start <= 0:
        return True
    prefix = line[:start].rstrip()
    if not prefix:
        return True
    last = prefix[-1]
    return last in ".:!?*\"')]}"


def _skip_line(line: str, state) -> bool:
    """Skip blocks that should never be scanned for source notes."""
    if state.in_yaml or state.in_code_fence or state.in_display_math:
        return True
    if state.in_html_style_block or state.in_html_comment:
        return True
    if is_python_chunk_option(line):
        return True
    return False


def _source_segment(line: str) -> tuple[int, str] | None:
    """Return the first source-note segment on the line, if present.

    The returned segment begins at `Source:` / `source:` and runs until
    the next double quote or end-of-line. This makes the check work both
    on body prose and on single-line Quarto attribute strings such as
    `fig-cap="... Source: [@key]."`.
    """
    m = _SOURCE_RE.search(line)
    if not m:
        return None
    start = m.start()
    rest = line[start:]
    # Stop at the next unescaped double quote if we are inside an attr
    # string; otherwise inspect the rest of the line.
    quote = rest.find('"')
    if quote != -1:
        segment = rest[:quote]
    else:
        segment = rest
    return start, segment


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan for source-attribution formatting issues."""
    issues: list[Issue] = []
    counter = start_counter

    walker = LineWalker(text)
    for line, state, line_num in walker:
        if _skip_line(line, state):
            continue

        # Stray asterisk wrapping is a stylistic error on the full line.
        for m in _ASTERISK_SOURCE_RE.finditer(line):
            issues.append(
                Issue(
                    id=make_issue_id(scope, "asterisk_sources", counter),
                    category="asterisk_sources",
                    rule=RULE,
                    rule_text=RULE_TEXT,
                    file=str(file_path),
                    line=line_num,
                    col=m.start(),
                    before=line,
                    auto_fixable=False,
                    needs_subagent=False,
                    reason="Source note is wrapped in asterisks",
                )
            )
            counter += 1

        # Lowercase source marker.
        lower = _LOWERCASE_SOURCE_RE.search(line)
        if lower and not _CAPITALIZED_SOURCE_RE.search(line) and _looks_like_source_note(line, lower.start()):
            issues.append(
                Issue(
                    id=make_issue_id(scope, "lowercase_sources", counter),
                    category="lowercase_sources",
                    rule=RULE,
                    rule_text=RULE_TEXT,
                    file=str(file_path),
                    line=line_num,
                    col=lower.start(),
                    before=line,
                    auto_fixable=False,
                    needs_subagent=False,
                    reason="Use capitalized 'Source:'",
                )
            )
            counter += 1

        segment_info = _source_segment(line)
        if segment_info is None:
            continue
        start, segment = segment_info
        if not _looks_like_source_note(line, start):
            continue

        # Missing terminal period on the source note itself.
        trimmed = segment.rstrip()
        if trimmed and not trimmed.endswith("."):
            issues.append(
                Issue(
                    id=make_issue_id(scope, "missing_periods", counter),
                    category="missing_periods",
                    rule=RULE,
                    rule_text=RULE_TEXT,
                    file=str(file_path),
                    line=line_num,
                    col=start,
                    before=line,
                    auto_fixable=False,
                    needs_subagent=False,
                    reason="Source note should end with a period",
                )
            )
            counter += 1

        # Double periods in the source note.
        if ".." in segment:
            issues.append(
                Issue(
                    id=make_issue_id(scope, "double_periods", counter),
                    category="double_periods",
                    rule=RULE,
                    rule_text=RULE_TEXT,
                    file=str(file_path),
                    line=line_num,
                    col=start,
                    before=line,
                    auto_fixable=False,
                    needs_subagent=False,
                    reason="Source note contains a double period",
                )
            )
            counter += 1

        # Academic source notes must use bracketed citations.
        if _MALFORMED_ACADEMIC_RE.search(segment):
            m = _MALFORMED_ACADEMIC_RE.search(segment)
            issues.append(
                Issue(
                    id=make_issue_id(scope, "malformed_citations", counter),
                    category="malformed_citations",
                    rule=RULE,
                    rule_text=RULE_TEXT,
                    file=str(file_path),
                    line=line_num,
                    col=start + (m.start() if m else 0),
                    before=line,
                    auto_fixable=False,
                    needs_subagent=False,
                    reason="Academic source should use bracketed citation syntax",
                )
            )
            counter += 1

        # Extra brackets around academic citations.
        if _EXTRA_BRACKETS_RE.search(segment):
            m = _EXTRA_BRACKETS_RE.search(segment)
            issues.append(
                Issue(
                    id=make_issue_id(scope, "extra_brackets", counter),
                    category="extra_brackets",
                    rule=RULE,
                    rule_text=RULE_TEXT,
                    file=str(file_path),
                    line=line_num,
                    col=start + (m.start() if m else 0),
                    before=line,
                    auto_fixable=False,
                    needs_subagent=False,
                    reason="Source note has extra citation brackets",
                )
            )
            counter += 1

    return issues, counter
