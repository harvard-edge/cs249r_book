r"""Check: bare narrative attribution `Author et al. YEAR` without `[@cite]`.

Rule: book-prose-merged.md section 5 (Footnotes, Citations & Index → Citations
& References, "Bare Attributions are Forbidden")

    Hand-typed author/year text in body prose or footnote prose is the same
    anti-pattern as the manual+bracket duplicate, even when no `[@key]`
    follows. Forbidden shapes:

        Surname et al. {found, showed, …}      (no cite on line)
        Surname et al. YYYY                    (no cite on line)
        Surname et al. (YYYY)                  (no cite on line)
        Surname and Surname (YYYY)             (no cite on line)
        Surname, Surname, and Surname (YYYY)   (no cite on line)
        (Surname and Surname, YYYY)            (no cite on line)
        Surname (YYYY) {found, showed, …}      (no cite on line)

    The fix is to use narrative `@key` form or to anchor a `[@key]` at the
    fact. This check produces a `bare-attribution` issue per occurrence.

Provenance.  Audits N + O (`review/AUDIT_handtype_refs.md`,
`review/AUDIT_cite_form.md`) caught the original 14 sites in Waves 4 and 6;
this module promotes the ad-hoc detection logic to a reusable scanner.

Auto-fixable: NO.  Selecting the right `@key` requires reading the
references.bib and matching the surname + year to the canonical bib entry.
The scanner emits `confidence='medium'` and a `suggested_after` that
inserts `[@TODO-cite]` at the end of the attribution so the operator (or
a downstream subagent) sees exactly where the missing anchor belongs.

Protected contexts skipped via `LineWalker`:

  - YAML frontmatter, fenced code blocks, display math, HTML style/script
    blocks, HTML comments, Quarto chunk-option lines (`#| ...`), div fences,
    LaTeX command lines — all via `default_line_skip`.
  - Inline code, inline math, attribute strings (`fig-cap`, `fig-alt`,
    `tbl-cap`, `lst-cap`, `title`), `\index{...}` keys, `\ref{}` /
    `\cref{}`, anchor IDs, Quarto `@sec-/@fig-/@tbl-/@eq-/@lst-` refs,
    bracketed citations `[...@key...]`, bare `@key` cites, and
    footnote refs `[^fn-...]` — all via `inline_protected_spans`.

False-positive classes the check actively suppresses:

  1. `Product/event (YEAR)` annotations: `AlexNet (2012)`, `GPT-3 (2020)`,
     `EU AI Act (2024)`, `Dartmouth Conference (1956)`. The single
     `Surname (YYYY)` form is only flagged when an attribution verb
     (`found`, `showed`, `proposed`, `introduced`, etc.) appears within a
     short window after the parenthetical year.  A bare `Term (2012)` with
     no attribution verb is treated as a product/event annotation.
  2. Footnote bold-head `**Term (YEAR)**:` form: per book-prose.md §5, the
     parenthetical-year head IS the attribution; no cite is required and
     no bare attribution exists.  We detect this by checking whether the
     `(YYYY)` is immediately preceded by `**`.
  3. Acceptance line cite on the same line: if any `[@key]` or bare `@key`
     citation appears anywhere on the same line as the matched attribution
     phrase, the line is in-spec (the author tied the attribution to a
     specific cite).  We also check the immediately previous and next
     lines so that a wrapped footnote definition (`[^fn-foo]: ... Smith et
     al. found ...\n[@smith2020] ...`) does not falsely fire.
  4. Bibliography / references / parts / frontmatter / glossary files:
     skipped by file path (the same exclusion used by the existing
     bibliography_hygiene module and the `AUDIT_cite_form.md` greps).
"""

from __future__ import annotations

import re
from pathlib import Path

from audit.ledger import Issue, make_issue_id
from audit.protected_contexts import (
    LineWalker,
    default_line_skip,
    inline_protected_spans,
    position_in_spans,
)

CATEGORY = "bare-attribution"
RULE = "book-prose-merged.md section 5 (Citations & References)"
RULE_TEXT = (
    "Hand-typed `Author et al. YEAR` / `Author and Author (YEAR)` / "
    "`Author (YEAR) verb` without a `[@cite]` anchor is forbidden. "
    "Use narrative `@key` form or add `[@key]` at the fact."
)

# Attribution verbs that turn `Surname (YYYY)` from a product/event annotation
# into a bare author attribution.  Drawn from the corpus failure modes
# documented in review/AUDIT_cite_form.md (sections 1–6 of the violation list).
_ATTRIB_VERBS = (
    r"found|showed|proposed|introduced|discovered|demonstrated|"
    r"formalized|established|developed|invented|coined|defined|"
    r"observed|argued|showed|generalized|articulated|reported|"
    r"presented|published|studied|noted|trained|hypothesized|"
    r"conjectured|proved"
)

# 1. `Surname et al. YYYY` / `Surname et al. (YYYY)` / `Surname et al. in YYYY`.
#    Strongest signal.  Allow optional comma, optional "in ", and optional
#    parens around the year — all forms attested in the corpus per
#    review/AUDIT_cite_form.md security_privacy footnote findings.
_ET_AL_RE = re.compile(
    r"\b([A-Z][a-zA-Z'\-]+)\s+et\s+al\.?,?\s*(?:in\s+)?\(?([12][0-9]{3})\)?"
)

# 2. `Surname et al. {verb}` — "Smith et al. found that ..." with the year
#    omitted entirely.  Still a bare attribution because no cite anchor.
_ET_AL_VERB_RE = re.compile(
    rf"\b([A-Z][a-zA-Z'\-]+)\s+et\s+al\.?\s+(?:{_ATTRIB_VERBS})\b"
)

# 3. `Surname and Surname (YYYY)` / `Surname & Surname (YYYY)` — two-author
#    parenthetical with year.  Strong signal regardless of trailing verb.
_TWO_AUTHOR_RE = re.compile(
    r"\b([A-Z][a-zA-Z'\-]+)\s+(?:and|&)\s+([A-Z][a-zA-Z'\-]+)\s*\(([12][0-9]{3})\)"
)

# 4. `Surname, Surname, and Surname (YYYY)` — three-author form.
_THREE_AUTHOR_RE = re.compile(
    r"\b([A-Z][a-zA-Z'\-]+),\s+([A-Z][a-zA-Z'\-]+),?\s+and\s+([A-Z][a-zA-Z'\-]+)\s*\(([12][0-9]{3})\)"
)

# 5. `(Surname and Surname, YYYY)` — fully parenthetical multi-author attribution.
_PAREN_TWO_AUTHOR_RE = re.compile(
    r"\(([A-Z][a-zA-Z'\-]+)\s+(?:and|&)\s+([A-Z][a-zA-Z'\-]+),?\s+([12][0-9]{3})\)"
)

# 6. `Surname (YYYY)` — ambiguous form.  Only flag when followed (within ~80
#    chars) by an attribution verb.  Bare `Surname (YYYY)` with no verb is
#    overwhelmingly a product/event-year annotation per the corpus.
_SINGLE_AUTHOR_RE = re.compile(
    r"(?<!\*\*)\b([A-Z][a-zA-Z'\-]+)\s*\(([12][0-9]{3})\)"
)
_VERB_LOOKAHEAD_RE = re.compile(rf"\b(?:{_ATTRIB_VERBS})\b")

# Citation tokens that, if present anywhere on the same line, neutralize a
# match (the author already anchored the attribution to a cite).  We accept
# both bracketed and bare forms.
_HAS_CITE_RE = re.compile(r"\[@[\w:-]+|(?<![\w@!])@[\w:-]+")

# Surnames or capitalized words that frequently trigger false positives.
# These are common English words or compound model/product names that the
# regex would otherwise match as `Surname (YYYY)`.  Add to this set when a
# new false-positive class is identified.
_SURNAME_DENYLIST = frozenset(
    {
        # Common English words capitalized at sentence start (rare, but
        # `[A-Z][a-zA-Z]+` matches them).
        "The", "This", "These", "That", "Their", "Then", "Thus",
        "When", "Where", "Why", "How", "What", "Which",
        # Months (a date like `January (2020)` should not flag).
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    }
)


def _line_has_cite(line: str) -> bool:
    """True if the line contains any `[@key]` or bare `@key` citation."""
    return _HAS_CITE_RE.search(line) is not None


def _is_after_bold_head(line: str, match_start: int) -> bool:
    """True if the match is inside a footnote-style `**Term (YYYY)**:` head.

    Per book-prose.md §5, the bold parenthetical-year head IS the
    attribution and does not require a `[@cite]`.  We detect by looking
    for `**` between the start of the line and the `(YYYY)`, with no
    intervening `**` close before the paren.
    """
    prefix = line[:match_start]
    # Find the last `**` opener; if it appears more recently than a close
    # we are still inside a bold span.
    last_open = prefix.rfind("**")
    if last_open < 0:
        return False
    # Look for a matching close between the opener and the match position.
    rest = prefix[last_open + 2:]
    return "**" not in rest  # still inside the bold opener


def _is_product_or_event(line: str, surname: str, match_end: int) -> bool:
    """True if `Surname (YYYY)` is a product/event annotation, not an author.

    Heuristic per review/AUDIT_cite_form.md "Accepted convention":
    `Surname (YYYY)` is overwhelmingly used for product/event year
    annotations in this corpus (`AlexNet (2012)`, `EU AI Act (2024)`).
    We require an attribution verb within 80 characters AFTER the match
    to upgrade it to a bare-attribution flag.

    Also returns True for known false-positive surnames.
    """
    if surname in _SURNAME_DENYLIST:
        return True
    # Check for attribution verb in the lookahead window.
    window = line[match_end : match_end + 80]
    return _VERB_LOOKAHEAD_RE.search(window) is None


def _build_suggestion(line: str, match_start: int, match_end: int) -> str:
    """Insert `[@TODO-cite]` immediately after the matched attribution.

    The suggested_after field shows the operator (or a downstream
    subagent) exactly where the missing cite belongs.  The TODO marker
    is intentionally unmatched by any bib key so a render-time check
    will fail until a human resolves it.
    """
    return line[:match_end] + " [@TODO-cite]" + line[match_end:]


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan for bare narrative attributions without a `[@cite]` anchor.

    Returns (issues, next_counter).
    """
    posix = file_path.as_posix()
    # Skip non-body locations: bibliography, references, parts files,
    # frontmatter, glossary, backmatter.  These either contain bib entries
    # (citations live elsewhere) or are out-of-scope for body-prose rules
    # per the AUDIT_cite_form.md scope definition.
    if any(
        excl in posix
        for excl in (
            "/backmatter/",
            "/frontmatter/",
            "/glossary/",
            "/parts/",
            "/references.qmd",
            "/_bibliography",
        )
    ):
        return [], start_counter

    issues: list[Issue] = []
    counter = start_counter

    # Pre-split for adjacent-line lookups (footnote definitions sometimes
    # carry the cite on the line AFTER the attribution).
    raw_lines = text.split("\n")

    walker = LineWalker(text)
    for line, state, line_num in walker:
        if default_line_skip(line, state):
            continue
        # Fast bail — every pattern requires either "et al" or a 4-digit
        # year token starting with 19/20.  We use a loose digit test (not
        # `\([12][0-9]{3}\)`) because the parenthetical-multi-author form
        # `(Surname and Surname, YYYY)` has the year after a comma rather
        # than after an opening paren.
        if "et al" not in line and not re.search(r"\b[12][0-9]{3}\b", line):
            continue

        spans = inline_protected_spans(line)

        # Treat the line as cite-anchored if EITHER this line OR an
        # immediately adjacent line carries a cite.  This handles the
        # footnote-definition pattern where the cite sits on the line
        # above or below the attribution prose.
        line_cite = _line_has_cite(line)
        prev_cite = (
            line_num >= 2 and _line_has_cite(raw_lines[line_num - 2])
        )
        next_cite = (
            line_num < len(raw_lines)
            and _line_has_cite(raw_lines[line_num])
        )
        has_adjacent_cite = line_cite or prev_cite or next_cite

        line_matches: list[tuple[int, int, str]] = []

        # Pattern 1: Surname et al. YYYY / (YYYY)
        for m in _ET_AL_RE.finditer(line):
            if position_in_spans(m.start(), spans):
                continue
            line_matches.append((m.start(), m.end(), f"{m.group(1)} et al. {m.group(2)}"))

        # Pattern 2: Surname et al. {verb}  (no year)
        for m in _ET_AL_VERB_RE.finditer(line):
            if position_in_spans(m.start(), spans):
                continue
            # Avoid double-flagging if pattern 1 already caught this site.
            if any(s <= m.start() < e for s, e, _ in line_matches):
                continue
            line_matches.append((m.start(), m.end(), f"{m.group(1)} et al. (no year)"))

        # Pattern 3: Surname and Surname (YYYY)
        for m in _TWO_AUTHOR_RE.finditer(line):
            if position_in_spans(m.start(), spans):
                continue
            line_matches.append(
                (m.start(), m.end(), f"{m.group(1)} and {m.group(2)} ({m.group(3)})")
            )

        # Pattern 4: Surname, Surname, and Surname (YYYY)
        for m in _THREE_AUTHOR_RE.finditer(line):
            if position_in_spans(m.start(), spans):
                continue
            line_matches.append(
                (
                    m.start(),
                    m.end(),
                    f"{m.group(1)}, {m.group(2)}, and {m.group(3)} ({m.group(4)})",
                )
            )

        # Pattern 5: (Surname and Surname, YYYY)
        for m in _PAREN_TWO_AUTHOR_RE.finditer(line):
            if position_in_spans(m.start(), spans):
                continue
            line_matches.append(
                (m.start(), m.end(), f"({m.group(1)} and {m.group(2)}, {m.group(3)})")
            )

        # Pattern 6: Surname (YYYY) — only if attribution verb follows.
        for m in _SINGLE_AUTHOR_RE.finditer(line):
            if position_in_spans(m.start(), spans):
                continue
            surname = m.group(1)
            # Suppress product/event annotations and known false-positives.
            if _is_product_or_event(line, surname, m.end()):
                continue
            # Suppress footnote bold-head form `**Term (YYYY)**:` where the
            # head IS the attribution per book-prose §5.
            if _is_after_bold_head(line, m.start()):
                continue
            line_matches.append((m.start(), m.end(), f"{surname} ({m.group(2)})"))

        if not line_matches:
            continue

        # If a cite is already present on this line or an immediate
        # neighbor, the author has anchored the attribution.  Skip.
        if has_adjacent_cite:
            continue

        # Dedup overlapping matches.  Patterns 3/4/5 (multi-author) span
        # the same byte range as pattern 6 (singleton `Surname (YYYY)`)
        # for the last author in the list — e.g. `Frankle and Carbin
        # (2019)` triggers both pattern 3 and pattern 6 (on `Carbin
        # (2019)`).  Keep the widest match (the multi-author form), drop
        # the narrower singleton hit.
        line_matches.sort(key=lambda t: (t[0], -(t[1] - t[0])))
        deduped: list[tuple[int, int, str]] = []
        for start, end, phrase in line_matches:
            # Suppress if any earlier (wider) match already covers `start`.
            if any(s <= start < e for s, e, _ in deduped):
                continue
            deduped.append((start, end, phrase))

        # Emit one Issue per distinct match.
        for start, end, phrase in deduped:
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
                    suggested_after=_build_suggestion(line, start, end),
                    auto_fixable=False,
                    needs_subagent=True,  # picking the right bib key is judgment work
                    confidence="medium",
                    reason=(
                        f"Bare attribution `{phrase}` with no `[@cite]` on "
                        "this or an adjacent line"
                    ),
                )
            )
            counter += 1

    return issues, counter
