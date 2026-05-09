"""Check: §10.3 concept terms must be lowercase in body prose.

Rule: book-prose-merged.md section 10.3

    "Concept terms that are NOT proper nouns must be lowercase in body
    prose, even when they were capitalized in pre-round-1 drafts."

The §10.3 lowercase list includes: iron law, degradation equation,
verification gap, data wall, compute wall, memory wall, power wall,
energy corollary, ML node, bitter lesson, scaling laws, information
roofline, data gravity, napkin math, starving accelerator, latency
cliff, four pillars framework, and several others.

This check ports the logic of
`book/tools/scripts/content/fix_capitalization.py`, which was used in
round 1 pass 4 to fix 243 violations in vol1. The port preserves the
proven exception logic while reusing the shared
`audit.protected_contexts` predicates so new protections added there
(e.g. footnote references, bracketed citations) are inherited
automatically.

§10.3 exception contexts — *headline-style* contexts where capitals stay:

    1. Start of sentence
    2. Inside **bold** (first definition)
    3. Inside ***triple bold*** (definition callout term)
    4. In H1 and H2 section headers (headline style)
    5. In `\\index{}` entries
    6. In bold table headers
    7. In bold structural labels inside callouts (e.g. `**The Iron Law Connection:**`)
    8. In table cells that list principles by name (pipe-table rows
       that reference a canonical principle anchor like `\\ref{pri-...}`)
    9. In `fig-cap` / `tbl-cap` / `lst-cap` bold-title position — the
       `**Bold Title**:` part *before* the colon. Detected via the bold-span
       logic, which sees the `**...**` markers regardless of whether the
       text sits in body prose or in an attribute value (Quarto renders
       both as bold).
    10. "Iron Law of Processor Performance" (H&P canonical reference)
    11. "Bitter Lesson" when used as Sutton's essay title

§10.3 *sentence-style* display contexts — concept terms STAY LOWERCASE:

    - Non-principle callout `title="..."` attributes ("box heads") —
      concept terms in ordinary callout titles follow the same lowercase
      rule as body prose. Per MIT Press style sheet: "Figure captions,
      table titles, and box heads are set sentence style."
    - `.callout-principle` titles are formal principle labels, not
      ordinary box heads, and stay Title Case.
    - `fig-cap` / `tbl-cap` descriptive text *after* the bold title.
    - `fig-alt` / `tbl-alt` (entire content, no bold-title position).
    - `lst-cap` descriptive text after bold title.

These contexts are NOT broadly protected — concept terms in them flag.
Only the `**Bold Title**:` portion within captions is protected (via the
bold-span detection at exception #9).

See `book-prose.md §10.3` (rule), `§10.3.2` (5-tier list), and `§10.3.3`
(audit and edit workflow). When the rule and this script disagree, the
rule wins; update this script to match.

Auto-fixable: YES. The lowercase replacement is deterministic and has
been validated against vol1 in round 1 pass 4. Detection and fix are
both pure string substitution after the exception checks.
"""

from __future__ import annotations

import re
from pathlib import Path

from audit.ledger import Issue, make_issue_id
from audit.protected_contexts import (
    LineWalker,
    heading_level,
    inline_protected_spans,
    is_inside_index_entry,
    is_python_chunk_option,
    is_sentence_start,
    is_table_header_row,
    position_in_spans,
)
# Direct import of the attribute regex so we can subtract attribute spans
# from the standard protected-span list — concept-caps treats callout
# `title=` and caption descriptive text as scan-able sentence-style
# contexts per §10.3.
from audit.protected_contexts import _ATTR_RE  # noqa: WPS437 (intentional)

CATEGORY = "concept-term-capitalization"
RULE = "book-prose-merged.md section 10.3"
RULE_TEXT = "§10.3 concept terms must be lowercase in body prose"


# Concept terms and their lowercase forms. This is the §10.3 subset that
# the proven fix_capitalization.py handled (10 terms) plus 7 low-risk
# additions from §10.3 that have been coined in the book and rarely
# appear as proper nouns.
#
# DELIBERATELY OMITTED (known ambiguous cases):
#   - "Transformer": proper-noun uses (Vision Transformer, GPT =
#     Generative Pre-trained Transformer, Transformer Engine NVIDIA
#     product) outnumber generic-architecture uses. Detection needs
#     per-occurrence judgment — best handled by a subagent lane.
#   - "Long Tail": too common as a statistical phrase ("the long tail
#     of the distribution") in other senses.
#   - "Machine Learning Operations": context-dependent with "MLOps"
#     acronym form; best left as a subagent-lane category.

_TERMS = {
    # Core ten from round 1 pass 4 (proven against vol1):
    "Iron Law":            "iron law",
    "Degradation Equation": "degradation equation",
    "Verification Gap":    "verification gap",
    "ML Node":             "ML node",
    "Bitter Lesson":       "bitter lesson",
    "Data Wall":           "data wall",
    "Compute Wall":        "compute wall",
    "Memory Wall":         "memory wall",
    "Power Wall":          "power wall",
    "Energy Corollary":    "energy corollary",
    # Low-risk additions from §10.3 (book-coined or domain-specific):
    "Scaling Laws":        "scaling laws",
    "Information Roofline": "information roofline",
    "Data Gravity":        "data gravity",
    "Napkin Math":         "napkin math",
    "Starving Accelerator": "starving accelerator",
    "Latency Cliff":       "latency cliff",
    "Four Pillars Framework": "four pillars framework",
    #
    # Tier 5 family extensions (walls, taxes, gaps, bottlenecks, cliffs
    # beyond the core list above) are NOT enforced auto-fixably here.
    # They follow the categorical-default lowercase rule per §10.3.2 but
    # have contextual exceptions the auto-fix cannot reliably distinguish:
    # scare-quoted canonical introductions ("Coordination Tax." at first
    # mention), pedagogical bullets in `.callout-learning-objectives` and
    # `.callout-checkpoint`, italic uniform structural-label patterns
    # ("**The Impediment**: *Physical Limits*" / "*The Coordination Tax*").
    # Use the manual grep + LLM-Edit workflow per §10.3.3 for Tier 5
    # family sweeps. Audit grep example:
    #
    #   grep -rEn '\b(Coordination Tax|Iteration Tax|...|Network Wall|
    #     Capacity Wall|Communication Wall)\b' book/quarto/contents \
    #     --include='*.qmd' \
    #     | grep -vE '(\\index\{|^[^:]*:[0-9]+:##|title=|...)'
}


# ── Span helper that excludes attribute values ────────────────────────────
#
# Per the §10.3 update of 2026-05-07, non-principle callout `title=`
# attributes and caption descriptive text are sentence-style display
# contexts that DO need concept-term scanning. The shared
# `inline_protected_spans` covers attribute spans (so the em-dash /
# quoting checks don't scan them); for concept-caps we subtract those
# attribute spans so terms inside ordinary titles and captions become
# visible to the audit.

def _spans_excluding_attributes(line: str) -> list[tuple[int, int]]:
    """Like `inline_protected_spans` but does NOT protect attribute
    values (non-principle callout title=, fig-cap, fig-alt, lst-cap, tbl-cap,
    tbl-alt).
    The bold-span check still handles `**Bold Title**:` portions inside
    captions; the descriptive text after the bold title and ordinary
    callout title attributes fall back to the sentence-style rule.
    """
    spans = inline_protected_spans(line)
    if not spans:
        return spans
    # Find attribute spans on this line
    attr_spans = [(m.start(), m.end()) for m in _ATTR_RE.finditer(line)]
    if not attr_spans:
        return spans
    # Subtract attribute spans from the protected-span list. Because
    # attribute spans completely contain any sub-spans we'd want to keep
    # protected (inline math `$...$`, code `` `...` ``, `\index{}` —
    # none of which appear inside `title=` / caption attributes in
    # practice), the simplest correct answer is to drop any protected
    # span whose interval falls fully inside an attribute span.
    result: list[tuple[int, int]] = []
    for s, e in spans:
        contained = any(a_s <= s and e <= a_e for a_s, a_e in attr_spans)
        if not contained:
            result.append((s, e))
    return result


# ── Bold-span predicate ────────────────────────────────────────────────────
#
# §10.3 exception #8 says "Bold structural labels inside callouts
# (e.g. `**The Iron Law Connection:**`)" are exempt from the lowercase
# rule. Matches inside the MIDDLE of a bold span — not just at the
# edges — must be skipped.
#
# The previous implementation used a generous OR test (`before` ends
# with `**` OR `after` starts with `**`), which correctly handled the
# edge cases `**Iron Law**` and `**Iron Law of ML Systems**` but
# missed the middle case `**The Iron Law Connection:**` where the
# term sits between bold markers but not adjacent to either one.
#
# The fix walks the line to find all `**...**` and `***...***` spans,
# then checks whether the match position falls inside any of them.

def _bold_spans(line: str) -> list[tuple[int, int]]:
    """Return (open_pos, close_pos_exclusive) for every bold span on the
    line. Both `**...**` and `***...***` count; single `*...*` italics
    are NOT bold and are not returned.

    Consecutive `**`/`***` markers are paired in order (open, close,
    open, close, ...). An unpaired trailing marker is ignored. Nested
    bold markers in the same line are rare and are handled by
    pair-order (not by stack depth).
    """
    markers: list[tuple[int, int]] = []  # (position, marker_length)
    i = 0
    n = len(line)
    while i < n:
        if i + 3 <= n and line[i:i+3] == "***":
            markers.append((i, 3))
            i += 3
        elif i + 2 <= n and line[i:i+2] == "**":
            markers.append((i, 2))
            i += 2
        else:
            i += 1

    spans: list[tuple[int, int]] = []
    for j in range(0, len(markers) - 1, 2):
        open_pos, _ = markers[j]
        close_pos, close_len = markers[j + 1]
        spans.append((open_pos, close_pos + close_len))
    return spans


def _is_inside_bold_span(
    line: str, start: int, end: int, spans: list[tuple[int, int]]
) -> bool:
    """True if [start, end) is fully enclosed by any bold span on the line."""
    for s, e in spans:
        if s <= start and end <= e:
            return True
    return False


# ── Preserved named-principle phrases ──────────────────────────────────────
#
# Book-coined formal principle names that include a concept term but are
# themselves proper nouns (parallel to "Amdahl's Law" in §10.9). Matches
# that fall inside any of these phrases are skipped regardless of any
# other rule — these are the explicit editorial exceptions to §10.3.

_PRESERVED_PHRASES = (
    "Data Gravity Invariant",
    "Data as Code Invariant",
)


def _is_inside_preserved_phrase(line: str, start: int, end: int) -> bool:
    """True if [start, end) is inside any preserved named-principle phrase."""
    for phrase in _PRESERVED_PHRASES:
        idx = line.find(phrase)
        while idx != -1:
            if idx <= start and end <= idx + len(phrase):
                return True
            idx = line.find(phrase, idx + 1)
    return False


# ── Per-term special cases (must stay capitalized even in body prose) ──────

def _is_hp_reference(line: str, match_end: int) -> bool:
    """True if `Iron Law` is actually `Iron Law of Processor Performance`.

    The Hennessy & Patterson canonical phrase is a proper-noun use and
    must remain capitalized, per §10.9. The check is a simple
    look-ahead — if the text immediately following the match is " of
    Processor Performance", the match is preserved.
    """
    return line[match_end:match_end + 25] == " of Processor Performance"


def _is_sutton_essay_title(line: str) -> bool:
    """True if the line appears to cite Sutton's essay 'The Bitter Lesson'.

    Two signals: (a) the line contains `"The Bitter Lesson"` or
    `"Bitter Lesson"` inside double quotes (essay-title reference), or
    (b) the line mentions "Sutton" (the author). Both mean the whole
    line should be left alone for this term.
    """
    if '"The Bitter Lesson"' in line or '"Bitter Lesson"' in line:
        return True
    if "Sutton" in line:
        return True
    return False


# ── Line-level filter ──────────────────────────────────────────────────────

def _skip_concept_term_line(line: str, state) -> bool:
    """Return True for lines where §10.3 concept terms must stay capitalized.

    Composes the block-level protections (YAML / code fence / display
    math / HTML style blocks / HTML comments) with concept-term-specific
    exclusions: any heading (H1-H6), div fences, Python chunk options,
    and bold table header rows.

    Deliberately does NOT inherit `default_line_skip`'s
    `is_latex_command_line` rule. That rule skips any line starting with
    `\\`, which is too strict for this check: lines like
    `\\index{foo!bar}To understand *why* the **Iron Law**...` contain
    real body prose and must be processed. The proven
    fix_capitalization.py (round 1 pass 4) did not have a
    leading-backslash skip, and parity with it requires the same.
    """
    # Block-level state (YAML, code fence, display math, HTML style/script,
    # HTML comments)
    if state.in_yaml or state.in_code_fence or state.in_display_math:
        return True
    if state.in_html_style_block or state.in_html_comment:
        return True
    # Python chunk option directive (`#| echo: false`, etc.)
    if is_python_chunk_option(line):
        return True
    # Principle callout titles are formal labels, not ordinary box heads.
    if ".callout-principle" in line:
        return True
    # Quarto div fence: skip only BARE closes / openers with no `title=`
    # attribute. Opener lines containing `title="..."` in non-principle
    # callouts are sentence-style display contexts that DO need to flag
    # concept terms per §10.3.
    stripped = line.lstrip()
    if stripped.startswith(":::") and 'title="' not in line:
        return True
    # H1 and H2 use headline-case; H3+ follow sentence-case but concept
    # terms in headings are still ambiguous (the book has many correct
    # H3 headings containing "iron law" — they should not be flagged).
    level = heading_level(line)
    if level is not None:
        return True
    # Bold table header rows
    if is_table_header_row(line):
        return True
    # §10.3 exception #9: pipe-table rows that list principles by name.
    # Detection: the row starts with `|` and contains a canonical
    # principle anchor reference (`\ref{pri-...}`). The principle-name
    # cell in such a row is a label, not body prose, so concept terms
    # like "Iron Law of ML Systems" stay Title Case.
    stripped = line.lstrip()
    if stripped.startswith("|") and "\\ref{pri-" in line:
        return True
    return False


# ── Match-level filter ─────────────────────────────────────────────────────

def _skip_match(
    line: str,
    start: int,
    end: int,
    term: str,
    spans: list[tuple[int, int]],
    bold_spans: list[tuple[int, int]],
) -> bool:
    """Return True if this specific (line, start, end) match should not flag.

    Composes the shared inline predicates plus the two §10.3 per-term
    special cases (H&P reference, Sutton essay title). Order of checks
    matches fix_capitalization.py for parity with the proven behavior.
    """
    # 1. Start of sentence — capital is correct at sentence start.
    if is_sentence_start(line, start):
        return True
    # 2. Inside any **bold** or ***triple bold*** span — first
    #    definition, longer bold phrase, bold-wrapped term, or §10.3
    #    exception #8 bold structural labels inside callouts
    #    (e.g. `**The Iron Law Connection:**`). Uses full-span
    #    containment, not just edge-adjacency.
    if _is_inside_bold_span(line, start, end, bold_spans):
        return True
    # 2b. Inside a preserved book-coined named-principle phrase
    #     (Data Gravity Invariant, Data as Code Invariant). These
    #     are formal principle names parallel to Amdahl's Law.
    if _is_inside_preserved_phrase(line, start, end):
        return True
    # 3. Inside inline math, inline code, \index{}, @sec-/fig-/tbl-, etc.
    if position_in_spans(start, spans):
        return True
    # 4. Inside \index{} (belt-and-braces; inline_protected_spans already
    #    covers this, but kept for clarity and parity with the proven script).
    if is_inside_index_entry(line, start):
        return True
    # 5. NOTE: non-principle callout title=, fig-cap, fig-alt, tbl-cap,
    #    tbl-alt, lst-cap are sentence-style display contexts per MIT Press
    #    style sheet and are NO LONGER broadly protected here. The bold-span
    #    check (rule 2) handles the `**Bold Title**:` portion of captions;
    #    the descriptive text after the bold title and ordinary callout
    #    `title=` attributes follow the same lowercase-concept-term rule as
    #    body prose.
    # 6. Per-term special cases
    if term == "Iron Law" and _is_hp_reference(line, end):
        return True
    if term == "Bitter Lesson" and _is_sutton_essay_title(line):
        return True
    return False


# ── Main check entry point ─────────────────────────────────────────────────

def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan for §10.3 concept terms that are wrongly capitalized.

    Returns (issues, next_counter). Every issue is auto_fixable — the
    fix is a deterministic lowercase substitution of the term. The
    suggested_after field holds the full line after all matches on that
    line have been replaced.
    """
    issues: list[Issue] = []
    counter = start_counter

    walker = LineWalker(text)
    for line, state, line_num in walker:
        if _skip_concept_term_line(line, state):
            continue

        spans = _spans_excluding_attributes(line)
        bold_spans = _bold_spans(line)

        # Collect every non-skipped match for every term on this line.
        # We record each match separately so the ledger reflects per-
        # violation granularity (consistent with compound_prefix.py).
        # The per-line `suggested_after` is then computed once after
        # walking all terms, by applying accepted edits right-to-left.
        matches: list[tuple[int, int, str, str]] = []  # (start, end, term, replacement)
        for term, replacement in _TERMS.items():
            offset = 0
            while True:
                idx = line.find(term, offset)
                if idx == -1:
                    break
                end = idx + len(term)
                if _skip_match(line, idx, end, term, spans, bold_spans):
                    offset = end
                    continue
                matches.append((idx, end, term, replacement))
                offset = end

        if not matches:
            continue

        # Compute the fixed line by applying edits right-to-left.
        matches.sort(key=lambda m: m[0])
        new_line = line
        for start, end, _term, replacement in reversed(matches):
            new_line = new_line[:start] + replacement + new_line[end:]

        # Emit one Issue per match.
        for start, end, term, replacement in matches:
            issues.append(
                Issue(
                    id=make_issue_id(scope, CATEGORY, counter),
                    category=CATEGORY,
                    rule=RULE,
                    rule_text=RULE_TEXT,
                    file=str(file_path),
                    line=line_num,
                    col=start,
                    before=line,
                    suggested_after=new_line,
                    auto_fixable=True,
                    reason=f"{term} -> {replacement}",
                )
            )
            counter += 1

    return issues, counter


# ── Adversarial self-test (Pass 16 Item C) ─────────────────────────────────
#
# Run with:
#     PYTHONPATH=book/tools python3 book/tools/audit/checks/concept_term_capitalization.py
#
# Two corpora:
#
#   POSITIVE_CASES (detector MUST flag) — hand-crafted lines where a §10.3
#   concept term is incorrectly capitalized in body prose.
#
#   NEGATIVE_CASES (detector MUST NOT flag) — §10.3 exception contexts
#   where the capital is correct.

_POSITIVE_LINES = [
    # Body prose with capitalized concept term
    "The Iron Law decomposes performance into three terms.",
    "We face a Memory Wall at the edge of the roofline.",
    "The Bitter Lesson of seventy years of AI research.",
    "Engineers hit a Compute Wall as batch sizes grow.",
    "A Power Wall limits datacenter density.",
    "The Verification Gap is the gap between testing and guaranteeing.",
    "The Degradation Equation describes silent reliability decay.",
    "The Scaling Laws predict model quality as a function of compute.",
    "A Data Wall constrains the training corpus.",
    # ML Node → ML node (acronym stays, noun lowercases)
    "Each ML Node runs a dedicated scheduler.",
    # New terms (concept term kept off position 0 — the detector is
    # conservative at sentence start, matching the proven script).
    "A quick Napkin Math tells you the order of magnitude.",
    "The Latency Cliff appears when batch size exceeds the budget.",
    "A Starving Accelerator is bottlenecked on data delivery.",
    "The Four Pillars Framework organizes our analysis.",
    "With Data Gravity, computation pulls toward large corpora.",
]

_NEGATIVE_LINES = [
    # Exception 1: start of sentence
    "Iron Law decomposes performance into three terms.",
    "Memory Wall is the binding constraint at this scale.",
    # Exception 2: inside **bold** (first definition)
    "The **Iron Law** of ML Systems is a three-term decomposition.",
    "Engineers encounter the **Memory Wall** whenever bandwidth saturates.",
    # Exception 2b: §10.3 #8 — bold structural labels in callouts
    # where the term is in the MIDDLE of the bold span, not adjacent
    # to the `**` markers. Requires full-span containment detection.
    "**The Iron Law Connection:**",
    "**The Memory Wall Implication:** bandwidth dominates.",
    # Preserved named-principle phrases (parallel to Amdahl's Law).
    "The Data Gravity Invariant determines where the model runs.",
    "Principle: the Data as Code Invariant governs reproducibility.",
    # Exception 3: triple bold (definition callout term)
    "***Memory Wall***\\index{Memory Wall!definition} is the point where...",
    # Exception 4: section headers (H1, H2, H3+ — all skipped by the line filter)
    "## The Iron Law of ML Systems",
    "# Memory Wall and Power Wall",
    "### The Bitter Lesson",
    # Exception 5: inside \index{}
    "Earlier discussion \\index{Iron Law!definition} established the framework.",
    # Exception 6: inside title="..."
    '::: {.callout-definition title="Iron Law"}',
    # Exception 7: table header row (bold-in-pipe)
    "| **Iron Law** | **Memory Wall** | **Power Wall** |",
    # Exception 8: H&P canonical reference
    "This analogy comes from the Iron Law of Processor Performance [@patterson2021hardware].",
    # Exception 9: Sutton essay title
    'Sutton\'s essay "The Bitter Lesson" argues that scale wins.',
    "Richard Sutton's Bitter Lesson is one of the most-cited arguments.",
    # Correct lowercase uses (already fine, nothing to flag)
    "The iron law decomposes performance into three terms.",
    "Engineers hit a memory wall when bandwidth saturates.",
    "The bitter lesson is that scale beats cleverness.",
]


def _self_test() -> int:
    """Run the adversarial test corpora and report failures."""
    from audit.protected_contexts import LineState

    pos_fail: list[str] = []
    neg_fail: list[str] = []
    state = LineState()  # fresh state — tests are single-line

    def _has_match(line: str) -> bool:
        if _skip_concept_term_line(line, state):
            return False
        spans = _spans_excluding_attributes(line)
        bold_spans = _bold_spans(line)
        for term in _TERMS:
            offset = 0
            while True:
                idx = line.find(term, offset)
                if idx == -1:
                    break
                end = idx + len(term)
                if not _skip_match(line, idx, end, term, spans, bold_spans):
                    return True
                offset = end
        return False

    for line in _POSITIVE_LINES:
        if not _has_match(line):
            pos_fail.append(line)

    for line in _NEGATIVE_LINES:
        if _has_match(line):
            neg_fail.append(line)

    total = len(_POSITIVE_LINES) + len(_NEGATIVE_LINES)
    failures = len(pos_fail) + len(neg_fail)
    passed = total - failures

    print(f"concept_term_capitalization self-test: {passed}/{total} passed")
    if pos_fail:
        print(f"\n{len(pos_fail)} POSITIVE case(s) NOT flagged (false negatives):")
        for line in pos_fail:
            print(f"  - {line}")
    if neg_fail:
        print(f"\n{len(neg_fail)} NEGATIVE case(s) flagged (false positives):")
        for line in neg_fail:
            print(f"  - {line}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(_self_test())
