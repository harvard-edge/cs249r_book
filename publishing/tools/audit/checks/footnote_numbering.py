"""Check: footnote reference / definition integrity within a single file.

Quarto footnotes come in two shapes in this book:

  1. Named, two-part: a reference `[^id]` in body prose pointing at a
     definition `[^id]: explanatory text` later in the same file.
  2. Inline: `^[inline footnote text]` — self-contained, no separate id.

The book strongly prefers named footnotes with descriptive IDs (e.g.
`[^fn-amdahl]`) so that footnote text can be edited without renumbering
the whole chapter. Bare-numeric IDs (`[^1]`, `[^2]`) are typically the
fossil of an un-converted draft and should be either renamed or removed.

This check looks for four classes of defect within a single .qmd file:

  - footnote-missing-def   : `[^id]` reference with no matching definition.
  - footnote-orphan-def    : `[^id]: ...` definition with no body reference.
  - footnote-duplicate-def : the same `[^id]:` defined more than once.
  - footnote-numeric-id    : id is purely digits (`[^1]`, `[^42]`).

The first three overlap with `binder check footnotes --vol1 --vol2`
(`_run_footnote_refs` in book/cli/commands/validate.py). They are kept
here so the release-gate audit ledger has every footnote-integrity
defect in one place, and so the scanner still works if the binder check
output is suppressed. The fourth class — numeric IDs — is the actual
gap: `binder check footnotes` does not flag it.

Auto-fixable: NO. A missing definition can only be resolved by the
chapter author. An orphan definition might be intentional (some templates
generate definitions for IDs that will be referenced later). A numeric
ID needs a human to pick a descriptive name. Every issue is emitted with
`auto_fixable=False, needs_subagent=False`.

Protected contexts this check skips:
  - YAML frontmatter, fenced code blocks, display math, HTML comments
    (the default line-skip set). Footnote markers inside ``` code ``` are
    literal text and not real footnotes.
  - Inline code spans and inline math (per-line: a `[^id]` token inside
    `` `code` `` or `$math$` is not a real footnote).
"""

from __future__ import annotations

import re
from pathlib import Path

from audit.ledger import Issue, make_issue_id
from audit.protected_contexts import (
    LineWalker,
    position_in_spans,
)

# Narrow inline-protection: only inline code spans count as "not a
# real footnote." We CANNOT use `inline_protected_spans` here because
# that helper deliberately treats every `[^id]` as protected (useful for
# prose-style checks, fatal for a footnote check). We also avoid
# protecting inline math spans here because (a) `[^id]` inside `$...$`
# is essentially unheard of in this corpus and (b) the book's prose
# contains many literal `\$` dollar amounts that confuse a naive
# inline-math regex into swallowing entire sentences (including real
# footnote refs) into a single "math" span.
_CODE_SPAN_RE = re.compile(r"`[^`\n]+`")


def _local_protected_spans(line: str) -> list[tuple[int, int]]:
    """Spans on `line` where a `[^id]` token should NOT count as a footnote."""
    spans: list[tuple[int, int]] = []
    for m in _CODE_SPAN_RE.finditer(line):
        spans.append((m.start(), m.end()))
    if not spans:
        return spans
    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged

CATEGORY_MISSING = "footnote-missing-def"
CATEGORY_ORPHAN = "footnote-orphan-def"
CATEGORY_DUPLICATE = "footnote-duplicate-def"
CATEGORY_NUMERIC = "footnote-numeric-id"

RULE = "book-prose-merged.md footnote conventions"
RULE_TEXT_MISSING = "Every [^id] reference must have a matching [^id]: definition"
RULE_TEXT_ORPHAN = "Every [^id]: definition should have at least one [^id] reference"
RULE_TEXT_DUPLICATE = "An [^id] must be defined at most once per file"
RULE_TEXT_NUMERIC = (
    "Footnote IDs should be descriptive (e.g. [^fn-amdahl]), not bare numeric"
)

# A definition line starts at column 0 with "[^id]:" and a space + content.
# We match liberally on the id (anything that's not "]" or whitespace).
_FN_DEF_LINE_RE = re.compile(r"^\[\^([^\]\s]+)\]:\s")

# A footnote reference token "[^id]" anywhere on a line. Note that a
# definition line *also* contains "[^id]:" — we filter that case at
# the call site by checking the definition regex first.
_FN_REF_RE = re.compile(r"\[\^([^\]\s]+)\]")

# Numeric-only ids: "1", "42", "007". Treated as a separate defect class.
_NUMERIC_ID_RE = re.compile(r"^\d+$")


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan one file for footnote-integrity defects.

    Returns (issues, new_counter). Footnotes do not cross files in this
    book, so all matching is per-file.
    """
    issues: list[Issue] = []
    counter = start_counter

    # First pass: walk lines once, recording every definition and every
    # reference along with its line number and the raw line text.
    #   defs: id -> list of (line_num, raw_line)
    #   refs: id -> list of (line_num, raw_line, col)
    defs: dict[str, list[tuple[int, str]]] = {}
    refs: dict[str, list[tuple[int, str, int]]] = {}

    walker = LineWalker(text)
    for line, state, line_num in walker:
        # Custom line-skip: same as the default set MINUS the LaTeX-command
        # filter. Many body-prose lines in this book start with one or
        # more `\index{...}` macros immediately followed by the sentence
        # that contains a `[^id]` reference, and `default_line_skip`
        # treats any backslash-prefixed line as a LaTeX command line —
        # which would silently drop the real reference and produce false
        # "orphan definition" issues.
        if state.in_yaml or state.in_code_fence or state.in_display_math:
            continue
        if state.in_html_style_block or state.in_html_comment:
            continue
        stripped = line.lstrip()
        if stripped.startswith("#|") or stripped.startswith(":::"):
            continue
        # Cheap filter: skip lines that obviously have no footnote token.
        if "[^" not in line:
            continue

        # Is this a footnote definition line? If so, record the def and
        # do NOT also count the leading "[^id]" as a reference.
        def_m = _FN_DEF_LINE_RE.match(line)
        if def_m:
            fn_id = def_m.group(1)
            defs.setdefault(fn_id, []).append((line_num, line))
            # A definition line CAN legitimately reference other footnotes
            # in its body text — fall through and scan the rest of the
            # line for refs, but skip the leading "[^id]:" span.
            scan_start = def_m.end()
        else:
            scan_start = 0

        # Protected inline spans: only code and inline math. We must NOT
        # use the shared `inline_protected_spans` here because that helper
        # treats every `[^id]` as protected (see module docstring).
        spans = _local_protected_spans(line)

        for m in _FN_REF_RE.finditer(line, scan_start):
            if position_in_spans(m.start(), spans):
                continue
            fn_id = m.group(1)
            refs.setdefault(fn_id, []).append((line_num, line, m.start()))

    rel_file = str(file_path)

    # ── Missing definitions ────────────────────────────────────────────────
    # Every referenced id that has no matching definition. High confidence:
    # if the file references [^foo] but never defines it, Pandoc will
    # render the literal text "[^foo]" in the output — a visible defect.
    for fn_id in sorted(refs.keys() - defs.keys()):
        line_num, raw_line, col = refs[fn_id][0]
        issues.append(
            Issue(
                id=make_issue_id(scope, CATEGORY_MISSING, counter),
                category=CATEGORY_MISSING,
                rule=RULE,
                rule_text=RULE_TEXT_MISSING,
                file=rel_file,
                line=line_num,
                col=col,
                before=raw_line,
                suggested_after="",
                auto_fixable=False,
                needs_subagent=False,
                reason=(
                    f"footnote reference [^{fn_id}] has no matching "
                    f"[^{fn_id}]: definition in this file"
                ),
            )
        )
        counter += 1

    # ── Orphan definitions ─────────────────────────────────────────────────
    # Every defined id that is not referenced anywhere in the file body.
    # Medium confidence: occasionally chapters keep a definition staged for
    # a reference that will land in a later commit.
    for fn_id in sorted(defs.keys() - refs.keys()):
        line_num, raw_line = defs[fn_id][0]
        issues.append(
            Issue(
                id=make_issue_id(scope, CATEGORY_ORPHAN, counter),
                category=CATEGORY_ORPHAN,
                rule=RULE,
                rule_text=RULE_TEXT_ORPHAN,
                file=rel_file,
                line=line_num,
                col=0,
                before=raw_line,
                suggested_after="",
                auto_fixable=False,
                needs_subagent=False,
                reason=(
                    f"footnote definition [^{fn_id}]: has no matching "
                    f"reference in body prose (possibly stale)"
                ),
            )
        )
        counter += 1

    # ── Duplicate definitions ─────────────────────────────────────────────
    # Same id defined two or more times. High confidence: Pandoc behavior
    # is undefined (typically keeps only one), and the duplicate text is
    # almost certainly dead.
    for fn_id, occurrences in sorted(defs.items()):
        if len(occurrences) > 1:
            # Report each duplicate after the first, anchored on its own
            # line so the operator can diff them.
            first_line = occurrences[0][0]
            for line_num, raw_line in occurrences[1:]:
                issues.append(
                    Issue(
                        id=make_issue_id(scope, CATEGORY_DUPLICATE, counter),
                        category=CATEGORY_DUPLICATE,
                        rule=RULE,
                        rule_text=RULE_TEXT_DUPLICATE,
                        file=rel_file,
                        line=line_num,
                        col=0,
                        before=raw_line,
                        suggested_after="",
                        auto_fixable=False,
                        needs_subagent=False,
                        reason=(
                            f"footnote [^{fn_id}]: defined "
                            f"{len(occurrences)}x in this file "
                            f"(first definition at line {first_line})"
                        ),
                    )
                )
                counter += 1

    # ── Numeric IDs ───────────────────────────────────────────────────────
    # Bare-digit IDs are the fossil of un-converted draft footnotes. The
    # book's convention is descriptive named IDs ([^fn-amdahl], [^fn-tpu]).
    # We flag each numeric definition once (preferred — the def is the
    # canonical location), and each orphan numeric ref separately
    # (already covered by footnote-missing-def if there's no matching def).
    seen_numeric_def_ids: set[str] = set()
    for fn_id, occurrences in sorted(defs.items()):
        if not _NUMERIC_ID_RE.match(fn_id):
            continue
        seen_numeric_def_ids.add(fn_id)
        line_num, raw_line = occurrences[0]
        issues.append(
            Issue(
                id=make_issue_id(scope, CATEGORY_NUMERIC, counter),
                category=CATEGORY_NUMERIC,
                rule=RULE,
                rule_text=RULE_TEXT_NUMERIC,
                file=rel_file,
                line=line_num,
                col=0,
                before=raw_line,
                suggested_after="",
                auto_fixable=False,
                needs_subagent=False,
                reason=(
                    f"footnote id [^{fn_id}] is numeric — likely an "
                    f"unconverted draft footnote; rename to a descriptive id"
                ),
            )
        )
        counter += 1

    # Also flag numeric references whose definition is missing (so the
    # numeric-id signal isn't lost when the def is also missing). We avoid
    # double-flagging numeric ids that already have a flagged def.
    for fn_id in sorted(refs.keys()):
        if not _NUMERIC_ID_RE.match(fn_id):
            continue
        if fn_id in seen_numeric_def_ids:
            continue
        line_num, raw_line, col = refs[fn_id][0]
        issues.append(
            Issue(
                id=make_issue_id(scope, CATEGORY_NUMERIC, counter),
                category=CATEGORY_NUMERIC,
                rule=RULE,
                rule_text=RULE_TEXT_NUMERIC,
                file=rel_file,
                line=line_num,
                col=col,
                before=raw_line,
                suggested_after="",
                auto_fixable=False,
                needs_subagent=False,
                reason=(
                    f"footnote reference [^{fn_id}] uses a numeric id — "
                    f"likely an unconverted draft footnote"
                ),
            )
        )
        counter += 1

    return issues, counter
