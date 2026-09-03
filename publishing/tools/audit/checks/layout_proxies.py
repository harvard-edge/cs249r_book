r"""Check: source-level proxies for rendered-PDF layout defects.

Scanner H detects source-level patterns that *correlate* with bad layout
in the rendered PDF/HTML. The defects this scanner targets are inherently
visual — page breaks, stranded headings, widow/orphan paragraphs, runt
callouts — so the source-level rules below are heuristic proxies, not
ground truth. The actual ground-truth scanner (Scanner H+) needs rendered
output; see DESIGN DOC at the bottom of this docstring.

Categories emitted:

  orphan-heading
      A heading (## / ### / ####) whose body section is only 1-2 prose
      lines before the next heading at the same or higher level. Likely
      renders as a stranded heading at page bottom in the PDF, with the
      body bleeding to the next page.

  figure-without-caption
      A `::: {#fig-...}` div whose attribute block lacks a `fig-cap=`
      attribute. Quarto renders these without a figure number, breaks
      the `@fig-foo` cross-reference, and prints with no caption — a
      visible defect even before page layout enters the picture.

  runt-callout
      A `::: {.callout-...}` block with fewer than 2 non-empty body
      lines. Almost always a draft fragment that the author forgot to
      finish. Note callouts are intentionally short, so we require
      *zero* body lines to flag (a truly empty callout, not a terse one).

  empty-section
      A heading immediately followed by another heading (same or higher
      level) with no body prose between them. Almost always a TOC
      stub that never got filled in.

`table-without-caption` is intentionally NOT handled here — see
`table_caption.py` for that detection.

Protected contexts:
  - YAML frontmatter, code fences, display math, HTML comments — anything
    `LineWalker.default_line_skip` would skip. Headings inside fenced
    code blocks (e.g. shell session output) are not real headings.
  - Frontmatter and backmatter files are exempt (no layout enforcement
    on prefaces, dedications, glossaries, etc.).

Auto-fixable: NO. Every category needs human judgment — orphan headings
might be deliberate one-paragraph sections, captionless figures need a
written caption, runt callouts need their missing prose, empty sections
need either content or removal.

──────────────────────────────────────────────────────────────────────
DESIGN DOC: Future Scanner H+ (render-output verification)
──────────────────────────────────────────────────────────────────────

The source-level proxies above catch most authoring-time mistakes but
they cannot see the actual page layout. Scanner H+ will run AFTER the
Quarto render and inspect the rendered artifacts:

  PDF inspection (via pdfplumber or pdftotext + bbox):
    - True orphans/widows: heading on page N whose first body paragraph
      starts on page N+1. The source-level proxy can't predict this —
      depends on figure heights, margin notes, and accumulated text
      above the heading.
    - Cross-page table fragments: a `tabular` that breaks mid-row, or
      a multi-row table whose header row appears on one page and the
      data rows on the next without a continuation header.
    - Caption-figure separation: a figure on page N whose caption
      landed on page N+1.
    - Final-line runt: a paragraph whose last printed line is a single
      short word ("the.", "it.", "etc."). Classic widow/orphan control
      target; LaTeX's \widowpenalty/\clubpenalty can be tuned to
      reduce these but the scanner needs to verify the result.

  HTML inspection (via lxml/BeautifulSoup):
    - <figure> elements with no <figcaption> child (the rendered
      equivalent of our source-level captionless-figure proxy, but
      ground truth — catches cases where the fig-cap was authored but
      Quarto dropped it).
    - Cross-reference resolution: every `@fig-foo` / `@tbl-foo` /
      `@sec-foo` in the source resolves to an anchor in the HTML. A
      broken xref renders as "?@fig-foo" in the HTML, which the
      source-level scanner can't see.
    - Heading hierarchy gaps in TOC: H2 → H4 with no intervening H3
      shows up in the rendered nav as a depth jump; the source-level
      scanner could in principle catch this, but the HTML TOC is the
      definitive view.

  Implementation sketch:
    Scanner H+ runs as a *post-render* check, gated on `quarto render
    --to pdf` and `--to html` having succeeded. It walks the
    `_book/*.pdf` and `_book/*.html` outputs, joins findings back to
    source line numbers via Quarto's source-mapping comments, and
    emits Issues with the same shape as the source-level proxies.

    Until that wiring exists, Scanner H (this module) ships as the
    best-effort source-level approximation.

──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
from pathlib import Path

from audit.ledger import Issue, make_issue_id
from audit.protected_contexts import LineWalker

RULE = "release-gate scanner-H (layout proxies)"
RULE_TEXT = (
    "Source-level patterns that correlate with rendered-PDF layout "
    "defects: orphan headings, captionless figures, runt callouts, "
    "empty sections."
)

CAT_ORPHAN_HEADING = "orphan-heading"
CAT_FIG_NO_CAP = "figure-without-caption"
CAT_RUNT_CALLOUT = "runt-callout"
CAT_EMPTY_SECTION = "empty-section"

# Min body lines before the next heading. Sections with <= this many
# substantive body lines fire as orphan-heading candidates. Tuned to
# 2: a single sentence section is almost always an orphan; three
# lines or more is generally a deliberate short section.
_ORPHAN_BODY_THRESHOLD = 2

# Min body lines inside a callout before it counts as non-runt. A
# callout with 0 body lines is unambiguously broken; 1+ may be a
# terse-but-valid note.
_RUNT_CALLOUT_BODY_THRESHOLD = 1

# Heading: 1-6 leading '#' followed by space and text.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(\S.*)$")

# Figure div open: `::: {#fig-... ...}` (attribute block on the open line).
# We tolerate any number of `:` (Quarto allows `::::` for nesting).
_FIG_OPEN_RE = re.compile(r"^\s*:::+\s*\{[^}]*#fig-[A-Za-z0-9_-]+[^}]*\}")

# Callout div open: `::: {.callout-foo ...}`.
_CALLOUT_OPEN_RE = re.compile(r"^\s*:::+\s*\{[^}]*\.callout-[A-Za-z0-9_-]+[^}]*\}")

# Generic div open / close. A bare ":::" line closes whatever is on top.
_DIV_OPEN_RE = re.compile(r"^\s*:::+\s*\{")
_DIV_CLOSE_RE = re.compile(r"^\s*:::+\s*$")

# Frontmatter / backmatter / glossary / parts files are exempt — they're
# structurally short on purpose.
_EXEMPT_PATH_FRAGMENTS = (
    "/frontmatter/",
    "/backmatter/",
    "/parts/",
    "/glossary/",
    "/appendix/",
)


def _is_path_exempt(file_path: Path) -> bool:
    posix = file_path.as_posix()
    return any(frag in posix for frag in _EXEMPT_PATH_FRAGMENTS)


def _heading_level(line: str) -> int | None:
    m = _HEADING_RE.match(line)
    if not m:
        return None
    return len(m.group(1))


def _is_substantive_body_line(line: str) -> bool:
    """A 'body line' for orphan-detection purposes.

    Blank lines, pure attribute lines, div fences, and HTML comments
    don't count toward the body-line tally. We want to know whether the
    author actually wrote prose under the heading.
    """
    stripped = line.strip()
    if not stripped:
        return False
    # Attribute / div fence lines.
    if stripped.startswith(":::"):
        return False
    # HTML comment-only line.
    if stripped.startswith("<!--") and stripped.endswith("-->"):
        return False
    # A line that is only an `\index{...}` tag is not substantive.
    if stripped.startswith("\\index{") and stripped.endswith("}"):
        return False
    return True


def _has_fig_cap_attr(attr_block: str) -> bool:
    """Does the attribute block contain a fig-cap= attribute?

    Quarto accepts `fig-cap="..."` or `fig-cap='...'`. We also accept the
    legacy `caption=` form on a `#fig-` div, since some older content
    uses it.
    """
    if "fig-cap" in attr_block:
        return True
    # Some legacy figures put the caption text after the closing `:::`
    # rather than in the attribute — those count as captionless from
    # the cross-reference standpoint; Quarto's `@fig-foo` ref still
    # needs `fig-cap=` to render properly.
    return False


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan for source-level layout-defect proxies."""
    if _is_path_exempt(file_path):
        return [], start_counter

    issues: list[Issue] = []
    counter = start_counter

    # First pass: collect line metadata using LineWalker so we know
    # which lines are inside code fences / YAML / math. Note: we do NOT
    # use `default_line_skip` here because it treats lines starting with
    # `\index{...}` as protected (LaTeX-command-line rule), but a prose
    # paragraph beginning with `\index{}` is real body content for the
    # purposes of orphan detection. We track only block-level protection
    # (code fences, YAML, display math, HTML blocks).
    walker = LineWalker(text)
    walked: list[tuple[str, bool, int]] = []
    for line, state, line_num in walker:
        protected = (
            state.in_yaml
            or state.in_code_fence
            or state.in_display_math
            or state.in_html_style_block
            or state.in_html_comment
        )
        walked.append((line, protected, line_num))

    n = len(walked)

    # ── orphan-heading and empty-section ───────────────────────────
    # Walk the file and, at each unprotected heading, count substantive
    # body lines until the next heading at the same or shallower level
    # (or EOF).
    for idx in range(n):
        line, protected, line_num = walked[idx]
        if protected:
            continue
        level = _heading_level(line)
        if level is None:
            continue
        # Skip H1 — that's the chapter title; orphan rules don't apply.
        if level <= 1:
            continue

        # Count substantive body lines until next heading at level <= this.
        body_count = 0
        next_heading_line: int | None = None
        for j in range(idx + 1, n):
            jline, jprotected, jnum = walked[j]
            if jprotected:
                # Inside a code fence / math block — don't count toward
                # body, but also don't stop scanning.
                continue
            jlevel = _heading_level(jline)
            if jlevel is not None and jlevel <= level:
                next_heading_line = jnum
                break
            if _is_substantive_body_line(jline):
                body_count += 1

        # Empty section: 0 body lines AND another heading follows.
        if body_count == 0 and next_heading_line is not None:
            issues.append(
                Issue(
                    id=make_issue_id(scope, CAT_EMPTY_SECTION, counter),
                    category=CAT_EMPTY_SECTION,
                    rule=RULE,
                    rule_text=RULE_TEXT,
                    file=str(file_path),
                    line=line_num,
                    col=0,
                    before=line.rstrip(),
                    suggested_after="",
                    auto_fixable=False,
                    needs_subagent=False,
                    reason=(
                        f"Heading at line {line_num} has no body prose "
                        f"before next heading at line {next_heading_line} "
                        "(empty section — TOC stub?)"
                    ),
                )
            )
            counter += 1
            continue  # don't also flag as orphan

        # Orphan heading: 1-2 body lines AND another heading follows.
        # If body_count > 0 but next_heading_line is None, this is the
        # last section — terse last sections are common (a closing
        # one-liner) and we don't flag them.
        if (
            0 < body_count <= _ORPHAN_BODY_THRESHOLD
            and next_heading_line is not None
        ):
            issues.append(
                Issue(
                    id=make_issue_id(scope, CAT_ORPHAN_HEADING, counter),
                    category=CAT_ORPHAN_HEADING,
                    rule=RULE,
                    rule_text=RULE_TEXT,
                    file=str(file_path),
                    line=line_num,
                    col=0,
                    before=line.rstrip(),
                    suggested_after="",
                    auto_fixable=False,
                    needs_subagent=False,
                    reason=(
                        f"Heading at line {line_num} followed by only "
                        f"{body_count} body line(s) before next heading "
                        f"at line {next_heading_line} (orphan-heading "
                        "risk in rendered PDF)"
                    ),
                )
            )
            counter += 1

    # ── figure-without-caption and runt-callout ───────────────────
    # Walk for div blocks. We need the multi-line attribute block: a
    # `::: {#fig-...` open may continue onto subsequent lines until the
    # closing `}` of the attribute set. So we accumulate attribute
    # text until the brace closes, then check for `fig-cap=`.
    i = 0
    while i < n:
        line, protected, line_num = walked[i]
        if protected:
            i += 1
            continue

        # Figure div open?
        if _FIG_OPEN_RE.match(line):
            # Accumulate the attribute block, which may span multiple
            # lines (Quarto allows wrapped attributes). We treat the
            # block as ending at the first line whose accumulated brace
            # count returns to zero.
            attr_lines = [line]
            open_braces = line.count("{") - line.count("}")
            j = i + 1
            while j < n and open_braces > 0:
                jline = walked[j][0]
                attr_lines.append(jline)
                open_braces += jline.count("{") - jline.count("}")
                j += 1
            attr_block = " ".join(attr_lines)
            if not _has_fig_cap_attr(attr_block):
                # Extract the fig id for the reason message.
                idm = re.search(r"#(fig-[A-Za-z0-9_-]+)", attr_block)
                fig_id = idm.group(1) if idm else "fig-?"
                issues.append(
                    Issue(
                        id=make_issue_id(scope, CAT_FIG_NO_CAP, counter),
                        category=CAT_FIG_NO_CAP,
                        rule=RULE,
                        rule_text=RULE_TEXT,
                        file=str(file_path),
                        line=line_num,
                        col=0,
                        before=line.rstrip()[:140],
                        suggested_after="",
                        auto_fixable=False,
                        needs_subagent=False,
                        reason=(
                            f"Figure div `{fig_id}` has no `fig-cap=` "
                            "attribute — Quarto will render it without "
                            "a figure number and break @fig- cross-refs"
                        ),
                    )
                )
                counter += 1
            i = j if j > i else i + 1
            continue

        # Callout div open?
        if _CALLOUT_OPEN_RE.match(line):
            # Find the matching closing `:::` at the same div depth.
            # Quarto uses depth-based fences (`:::` and `::::`); we
            # match a bare `:::` line that is not opening a nested div.
            # Body content includes prose lines AND any code-fence /
            # display-math / image lines — a callout with only a code
            # block is not "runt", it carries a code example.
            depth = 1
            body_lines = 0
            k = i + 1
            close_line = None
            while k < n:
                kline, kprotected, knum = walked[k]
                # Protected content (code fence, math, etc.) counts as
                # body — the callout has substantive non-prose content.
                if kprotected:
                    body_lines += 1
                    k += 1
                    continue
                if _DIV_OPEN_RE.match(kline):
                    depth += 1
                elif _DIV_CLOSE_RE.match(kline):
                    depth -= 1
                    if depth == 0:
                        close_line = knum
                        break
                elif _is_substantive_body_line(kline):
                    body_lines += 1
                k += 1

            # Truly empty callout (0 body lines) → flag.
            if body_lines < _RUNT_CALLOUT_BODY_THRESHOLD and close_line is not None:
                # Extract callout type for the reason message.
                cm = re.search(r"\.callout-([A-Za-z0-9_-]+)", line)
                callout_type = cm.group(1) if cm else "?"
                issues.append(
                    Issue(
                        id=make_issue_id(scope, CAT_RUNT_CALLOUT, counter),
                        category=CAT_RUNT_CALLOUT,
                        rule=RULE,
                        rule_text=RULE_TEXT,
                        file=str(file_path),
                        line=line_num,
                        col=0,
                        before=line.rstrip()[:140],
                        suggested_after="",
                        auto_fixable=False,
                        needs_subagent=False,
                        reason=(
                            f"Callout `.callout-{callout_type}` at line "
                            f"{line_num} has {body_lines} body line(s) "
                            f"before its close at line {close_line} "
                            "(runt callout — likely truncated draft)"
                        ),
                    )
                )
                counter += 1
            i = (k + 1) if close_line is not None else (i + 1)
            continue

        i += 1

    return issues, counter
