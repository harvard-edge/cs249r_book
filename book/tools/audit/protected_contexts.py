r"""Protected-context detection for the audit scanner.

This module is the load-bearing safety layer of Pass 15. Every check function
asks this module: "is this position safe to flag/edit?" The answer must
mirror the rules in book-prose-merged.md sections 9 and 10.

Two layers of protection:

1. LINE-LEVEL state (`LineWalker`): Tracks YAML frontmatter, fenced code
   blocks, display math, and HTML <style>/<script> blocks across lines.
   The walker is stateful and must process lines in order.

2. INLINE-LEVEL spans (`inline_protected_spans`): Returns a list of
   (start, end) byte ranges within a single line that are protected:
   inline math `$...$`, inline code `` `...` ``, `\index{...}`,
   `title="..."`, `fig-cap="..."`, `fig-alt="..."`, `lst-cap="..."`,
   `tbl-cap="..."`, `\ref{...}`, `@sec-...`, `@fig-...`, `@tbl-...`,
   `@eq-...`, `@lst-...` cross-references, citations `[@key]`, and
   anchor IDs `{#sec-...}`.

Plus a set of named context predicates that check functions compose:

- is_sentence_start(line, pos) — match position is at start of a sentence
- is_inside_bold(line, pos, end) — match is inside **bold** or ***triple***
- is_inside_callout_title(line, pos) — match is inside title="..."
- is_inside_protected_attribute(line, pos) — fig-cap, fig-alt, etc.
- is_table_header_row(line) — pipe-table row containing **bold** cells
- is_inside_index_entry(line, pos) — inside \index{...}

The choice of which protections apply is per-check: vs-period skips
code/math/index/anchors but not bold (because "**vs**" is wrong even in
bold); concept-term lowercase skips ALL of the above plus sentence start.
Check functions ask only for the protections they need.

References:
- book-prose-merged.md section 10.3 (concept term exceptions)
- book-prose-merged.md section 10.17 (lessons-learned safety patterns)
- book/tools/scripts/content/fix_capitalization.py (proven implementation)
- book/tools/scripts/content/fix_emdash.py (proven implementation)
- book/tools/scripts/content/fix_percent.py (proven implementation)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator


# ── Inline span patterns ─────────────────────────────────────────────────────
# These match exactly what the proven fix_*.py scripts protect.

# Inline math: $...$ (single $, not $$). Non-greedy. Allows escaped \$.
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)([^\n$]|\\\$)+?(?<!\\)\$(?!\$)")

# Inline code: `...` (single backticks). Non-greedy.
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

# \index{anything-including-bangs-and-spaces}
_INDEX_RE = re.compile(r"\\index\{[^}\n]*\}")

# Quoted attribute values: title="...", fig-cap="...", fig-alt="..." etc.
# We treat the entire attribute value as one protected span.
_PROTECTED_ATTRS = ("title", "fig-cap", "fig-alt", "lst-cap", "tbl-cap")
_ATTR_RE = re.compile(
    r"\b(" + "|".join(_PROTECTED_ATTRS) + r')="((?:[^"\\]|\\.)*?)"'
)

# Quarto cross-references: @sec-foo, @fig-foo, @tbl-foo, @eq-foo, @lst-foo
# Slug allows letters, digits, hyphens, underscores, dots.
_XREF_RE = re.compile(r"@(?:sec|fig|tbl|eq|lst)-[\w.-]+")

# Pandoc citations: [@key], [-@key], [@key1; @key2], [see @key pp. 33-35],
# and bare "As @vaswani2017 showed". We use two patterns:
# 1. Any [...] bracket span containing "@" is treated as a citation span.
# 2. Bare "@key" not already inside brackets, bounded by \b.
# Both are protected because citation keys can embed concept-term words.
_BRACKETED_CITE_RE = re.compile(r"\[[^\]\n]*@[\w:-]+[^\]\n]*\]")
_BARE_CITE_RE = re.compile(r"(?<![\w@!])-?@[\w:-]+")

# Footnote references: [^fn-anything-here] and [^fn-anything]: definitions.
# The reference form should be protected because the slug (e.g. fn-iron-law)
# may contain concept-term words. The definition form is also protected
# because it looks like a reference followed by a colon.
_FOOTNOTE_REF_RE = re.compile(r"\[\^[^\]\n]+\]")

# LaTeX \ref{...}, \cref{...}, \autoref{...}
_LATEX_REF_RE = re.compile(r"\\(?:auto)?(?:c)?ref\{[^}\n]*\}")

# Anchor IDs in headers / divs: {#sec-foo-bar-1234}
_ANCHOR_ID_RE = re.compile(r"\{#[\w.:-]+\}")


@dataclass
class LineState:
    """The block-level state at a given line.

    All five flags are independent and tracked across the line walker.
    A check function should treat the line as protected (skip entirely)
    if any flag is True for which it has not opted out.
    """

    in_yaml: bool = False
    in_code_fence: bool = False
    in_display_math: bool = False
    in_html_style_block: bool = False  # <style>...</style> or <script>...</script>
    in_html_comment: bool = False  # <!-- ... --> (multi-line); also <!---


class LineWalker:
    """Stateful walker over the lines of a QMD file.

    Use as an iterator. Each `next()` returns `(line, state, line_num)`
    where `state` is the LineState BEFORE processing this line — i.e. what
    block context this line lives inside. Line numbers are 1-indexed.

    Usage:
        walker = LineWalker(text)
        for line, state, line_num in walker:
            if state.in_code_fence:
                continue  # skip code blocks
            ...

    The walker handles all four block contexts the proven fix scripts
    track. Per fix_emdash.py and fix_capitalization.py:

    - YAML frontmatter is bounded by the first two `---` lines
    - Code fences are bounded by ``` (optionally with a language tag)
    - Display math is bounded by lines starting with $$
    - HTML style/script blocks are bounded by <style>/<script> ... </style>/</script>
    """

    _CODE_FENCE_RE = re.compile(r"^\s*```")
    _DISPLAY_MATH_RE = re.compile(r"^\s*\$\$")
    _HTML_OPEN_RE = re.compile(r"<(style|script)\b", re.IGNORECASE)
    _HTML_CLOSE_RE = re.compile(r"</(style|script)>", re.IGNORECASE)
    # Multi-line HTML comments. Quarto accepts both <!-- and <!--- (three
    # dashes) as the opener. The closer is always -->. A single-line comment
    # (open and close on same line) must NOT toggle state.
    _HTML_COMMENT_OPEN_RE = re.compile(r"<!--")
    _HTML_COMMENT_CLOSE_RE = re.compile(r"-->")

    def __init__(self, text: str):
        # Preserve trailing newlines so byte counts match the source
        self.lines = text.split("\n")
        self.state = LineState()
        self._yaml_seen = 0  # 0 = no ---, 1 = inside frontmatter, 2 = closed

    def __iter__(self) -> Iterator[tuple[str, LineState, int]]:
        for i, line in enumerate(self.lines):
            stripped = line.strip()

            # ── HTML comments (must come BEFORE code-fence handling) ──
            # If we're already inside a multi-line HTML comment, we must
            # NOT toggle code-fence state on ``` lines inside the comment.
            # This is the bug that the discarded bulk run did not handle.
            if self.state.in_html_comment:
                # Check if this line closes the comment
                if self._HTML_COMMENT_CLOSE_RE.search(line):
                    yield line, self.state, i + 1
                    self.state.in_html_comment = False
                else:
                    yield line, self.state, i + 1
                continue

            # Detect a comment that opens and closes on the same line
            # (single-line comment); state stays unchanged but the line
            # itself is treated as in-comment for protection purposes.
            opens = bool(self._HTML_COMMENT_OPEN_RE.search(line))
            closes = bool(self._HTML_COMMENT_CLOSE_RE.search(line))
            if opens and closes:
                # Single-line comment — protect this line, no state change
                saved = self.state.in_html_comment
                self.state.in_html_comment = True
                yield line, self.state, i + 1
                self.state.in_html_comment = saved
                continue
            if opens:
                # Multi-line comment opens here; line is in comment, state on
                self.state.in_html_comment = True
                yield line, self.state, i + 1
                continue

            # ── YAML frontmatter ──
            # The first --- opens; the second --- (while still in YAML)
            # closes. After that, --- is just a horizontal rule.
            if stripped == "---":
                if self._yaml_seen == 0:
                    self.state.in_yaml = True
                    self._yaml_seen = 1
                    # The opening --- itself is part of the frontmatter
                    yield line, self.state, i + 1
                    continue
                elif self._yaml_seen == 1 and self.state.in_yaml:
                    # The closing --- is still considered frontmatter for
                    # this iteration; toggle off afterwards.
                    yield line, self.state, i + 1
                    self.state.in_yaml = False
                    self._yaml_seen = 2
                    continue

            # ── Code fences ──
            # ``` toggles regardless of language tag. Must check before
            # display math because $$ inside a code block is just code.
            if self._CODE_FENCE_RE.match(line):
                if self.state.in_code_fence:
                    # Closing fence — yield this line as still in fence,
                    # then toggle off.
                    yield line, self.state, i + 1
                    self.state.in_code_fence = False
                    continue
                else:
                    # Opening fence — toggle on, then yield as in fence.
                    self.state.in_code_fence = True
                    yield line, self.state, i + 1
                    continue

            # Anything inside a code fence is protected by the fence flag,
            # no further processing needed.
            if self.state.in_code_fence:
                yield line, self.state, i + 1
                continue

            # ── Display math ──
            # Lines starting with $$ toggle the display-math flag, UNLESS
            # the line contains a closing $$ on the same line (single-line
            # display math like `$$ x = y $$`), in which case state is
            # unchanged but the line itself is protected.
            if self._DISPLAY_MATH_RE.match(line):
                dd_count = stripped.count("$$")
                if dd_count >= 2:
                    # Single-line display math. Yield as protected
                    # (we set the flag temporarily for this line only).
                    saved = self.state.in_display_math
                    self.state.in_display_math = True
                    yield line, self.state, i + 1
                    self.state.in_display_math = saved
                    continue
                else:
                    # Multi-line display math toggle.
                    if self.state.in_display_math:
                        yield line, self.state, i + 1
                        self.state.in_display_math = False
                    else:
                        self.state.in_display_math = True
                        yield line, self.state, i + 1
                    continue

            # ── HTML <style>/<script> blocks ──
            if self._HTML_OPEN_RE.search(line):
                self.state.in_html_style_block = True
            yield line, self.state, i + 1
            if self._HTML_CLOSE_RE.search(line):
                self.state.in_html_style_block = False


# ── Inline span computation ──────────────────────────────────────────────────


def inline_protected_spans(line: str) -> list[tuple[int, int]]:
    """Return a sorted, merged list of (start, end) protected spans in a line.

    Spans cover: inline code, inline math, \\index{...}, protected attribute
    values (title=, fig-cap=, fig-alt=, lst-cap=, tbl-cap=), Quarto
    @-references (sec-, fig-, tbl-, eq-, lst-), LaTeX \\ref{}/\\cref{}/\\autoref{},
    and anchor IDs ({#sec-...}).

    The result is sorted by start position and adjacent/overlapping spans
    are merged. Use `position_in_spans(pos, spans)` to test membership.

    Order of operations matters per book-prose-merged.md section 10.17 #8:
    "Stash code FIRST, then math, then refs, then index, then anchors,
    then HTML attrs — in that order, with non-greedy patterns." We do
    them all in one pass and merge, so order does not matter for the
    output, but the intent is the same.
    """
    spans: list[tuple[int, int]] = []

    for regex in (
        _INLINE_CODE_RE,
        _INLINE_MATH_RE,
        _INDEX_RE,
        _ATTR_RE,
        _XREF_RE,
        _LATEX_REF_RE,
        _ANCHOR_ID_RE,
        _BRACKETED_CITE_RE,
        _BARE_CITE_RE,
        _FOOTNOTE_REF_RE,
    ):
        for m in regex.finditer(line):
            spans.append((m.start(), m.end()))

    if not spans:
        return spans

    # Sort and merge overlapping/adjacent spans
    spans.sort()
    merged: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def position_in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    """True if `pos` falls inside any of the (start, end) spans.

    Half-open: [start, end). Linear scan; spans are typically <10 per line.
    """
    for start, end in spans:
        if start <= pos < end:
            return True
        if start > pos:
            return False  # spans are sorted
    return False


def range_in_spans(
    start: int, end: int, spans: list[tuple[int, int]]
) -> bool:
    """True if [start, end) overlaps any protected span."""
    for s_start, s_end in spans:
        if s_start < end and start < s_end:
            return True
        if s_start >= end:
            return False
    return False


# ── Inline context predicates ────────────────────────────────────────────────


def is_sentence_start(line: str, pos: int) -> bool:
    """True if position `pos` is at the start of a sentence on this line.

    Mirrors fix_capitalization.py:73-80. The test: nothing before pos
    (modulo whitespace and bullet/list markers), or the previous
    non-whitespace character is sentence-ending punctuation.

    "Start of line" includes lines that begin with `- `, `* `, `+ `,
    `1. `, `2. `, etc. (list markers) and bullet points like `* `.
    Section headers (`## `) are also treated as sentence starts.
    """
    before = line[:pos]
    # Strip list markers, bullet markers, and leading bold markers
    stripped = re.sub(
        r"^\s*(?:[-*+]\s+|\d+\.\s+|\*\*+|#+\s+)*", "", before
    ).rstrip()
    if not stripped:
        return True  # only whitespace, list marker, bold marker, or header before
    last_char = stripped[-1]
    return last_char in ".!?:"


def is_inside_bold(line: str, start: int, end: int) -> bool:
    """True if the [start, end) range is inside **...** or ***...***.

    Mirrors fix_capitalization.py:67-71. Looks for `**` or `***` immediately
    before start AND immediately after end. This is a conservative check
    that catches the common cases where bold wraps a single term tightly.
    """
    before = line[:start]
    after = line[end:]
    bold_before = before.endswith("**") or before.endswith("***")
    bold_after = after.startswith("**") or after.startswith("***")
    return bold_before and bold_after


def is_inside_attribute(line: str, pos: int, attr: str) -> bool:
    """True if `pos` falls inside an `attr="..."` value on this line.

    `attr` is one of "title", "fig-cap", "fig-alt", "lst-cap", "tbl-cap".
    Mirrors fix_capitalization.py:111-124.
    """
    needle = f'{attr}="'
    start = line.rfind(needle, 0, pos)
    if start < 0:
        return False
    # Find the closing quote, allowing for escaped \"
    end = start + len(needle)
    while end < len(line):
        if line[end] == "\\" and end + 1 < len(line):
            end += 2
            continue
        if line[end] == '"':
            return start < pos and pos < end
        end += 1
    return False


def is_inside_protected_attribute(line: str, pos: int) -> bool:
    """True if `pos` is inside any of the protected attribute values."""
    return any(
        is_inside_attribute(line, pos, attr) for attr in _PROTECTED_ATTRS
    )


def is_inside_index_entry(line: str, pos: int) -> bool:
    """True if `pos` is inside a \\index{...} entry."""
    idx_start = line.rfind("\\index{", 0, pos)
    if idx_start < 0:
        return False
    idx_end = line.find("}", idx_start)
    return idx_end >= 0 and pos < idx_end


def is_table_row(line: str) -> bool:
    """True if this line is a markdown pipe-table row.

    Includes both data rows and the separator row (`|---|---|`).
    """
    return line.lstrip().startswith("|")


def is_table_header_row(line: str) -> bool:
    """True if this line is a pipe-table row containing bold cells.

    Per book-prose-merged.md section 6, table column headers are always
    bold (`| **Column A** | **Column B** |`). The bulk-edit run that
    Pass 15 was created to prevent broke exactly these. We treat any
    pipe-table row with `**` as a header row to be safe.
    """
    return is_table_row(line) and "**" in line


def is_section_header(line: str) -> bool:
    """True if this line is an ATX section header (# ... through ###### ...).

    Per book-prose-merged.md section 10.9, H1/H2 use headline case (don't
    lowercase concept terms in them) and H3+ use sentence case. The
    h3_titlecase check uses heading_level() to distinguish.
    """
    return bool(re.match(r"^\s*#{1,6}\s", line))


def heading_level(line: str) -> int | None:
    """Return the heading level (1-6) for an ATX header line, or None."""
    m = re.match(r"^\s*(#{1,6})\s", line)
    return len(m.group(1)) if m else None


def is_div_attribute_line(line: str) -> bool:
    """True if this line is a Quarto div fence (`::: {...}` or `:::`).

    Per fix_percent.py:33-35.
    """
    stripped = line.lstrip()
    return stripped.startswith(":::")


def is_python_chunk_option(line: str) -> bool:
    """True if this line is a Python chunk option directive (`#| key: value`).

    Per fix_percent.py:31-32 and fix_capitalization.py:49-50.
    """
    return line.lstrip().startswith("#|")


def is_latex_command_line(line: str) -> bool:
    """True if this line begins with a LaTeX command (`\\begin{...}`, etc.)."""
    return line.lstrip().startswith("\\")


# ── Convenience: per-check default skip predicates ───────────────────────────


def default_line_skip(line: str, state: LineState) -> bool:
    """The line-skip predicate that almost every check should use.

    Skips: YAML frontmatter, code fences, display math, HTML style/script
    blocks, Python chunk options, div fences, LaTeX command lines.

    Does NOT skip: section headers, table rows. Checks that need to skip
    those should add their own line filter.
    """
    if state.in_yaml or state.in_code_fence or state.in_display_math:
        return True
    if state.in_html_style_block or state.in_html_comment:
        return True
    if is_python_chunk_option(line):
        return True
    if is_div_attribute_line(line):
        return True
    if is_latex_command_line(line):
        return True
    return False
