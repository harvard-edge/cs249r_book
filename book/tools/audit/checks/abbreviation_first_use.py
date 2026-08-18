"""Check: abbreviation first-use policy (§10.5).

Rule: book-prose-merged.md section 10.5 / abbreviations.md

    Most specialized abbreviations expand on first use in each chapter.

Each `.qmd` file is treated as one chapter for the ordinary chapter-level
abbreviation set. The check walks the file
once and for each abbreviation in the §10.5 canonical-forms table:

  1. Finds the first "canonical introduction" — a match for the
     template `<expansion> (<ABBREV>)` where <expansion> matches
     case-insensitively and allows singular/plural inflection
     (`convolutional neural networks (CNNs)`).

  2. Finds the first "bare use" — a word-boundary match of the
     abbreviation that is NOT inside a canonical introduction,
     NOT inside a protected context (inline code, math, index,
     citations, captions), and NOT inside a section heading.

  3. If the first bare use occurs BEFORE the first canonical
     introduction (or if no introduction exists anywhere in the
     file), flag the bare use.

A bare use on the SAME line as the canonical introduction is NOT
flagged — if the chapter introduces the abbreviation somewhere on
the line where it first appears bare, the reader sees the expansion
immediately and §10.5's intent is satisfied.

Fix: editorial judgment required. The typical fix is to insert the canonical
expansion at the first bare use or move an existing later expansion earlier.
Not auto-fixable; `needs_subagent=True`.

Reference: abbreviations.md (canonical forms dict).
"""

from __future__ import annotations

import re
from pathlib import Path

from audit.ledger import Issue, make_issue_id
from audit.protected_contexts import (
    LineWalker,
    heading_level,
    inline_protected_spans,
    is_div_attribute_line,
    is_inside_index_entry,
    is_inside_protected_attribute,
    is_python_chunk_option,
    is_table_caption_line,
    is_table_header_row,
    is_table_row,
    position_in_spans,
)

CATEGORY = "abbreviation-first-use"
RULE = "abbreviations.md"
RULE_TEXT = "Abbreviations must follow chapter-level first-use policy"

# Over-expansion is the other half of the policy: once a chapter has taught the
# abbreviation, later body prose uses the acronym alone.
CATEGORY_OVER = "abbreviation-over-expansion"
RULE_TEXT_OVER = "Do not repeat a full expansion after the chapter introduced it"


# ── §10.5 canonical forms table ────────────────────────────────────────────
#
# Each entry: (abbreviation, canonical expansion). The expansion is matched
# case-insensitively and allows trailing `s?` for singular/plural
# inflection. The abbreviation itself is matched case-sensitively so that
# `cnn` in lowercase prose doesn't falsely count as an introduction.
#
# Deliberately excluded from first-use checking:
#   - CUDA, cuDNN: §10.5 explicitly says "no expansion needed"
#   - i.i.d.: statistical convention, universally understood; has dots
#     which break standard word-boundary detection
#   - CI/CD: DevOps term with slash punctuation; commonly understood
#   - vs.: not an abbreviation in the expansion sense
#   - Baseline CS/ML abbreviations listed in §10.5's exempt list (Round 2,
#     2026-04-24). These are universally understood by the book's graduate
#     CS/ML audience and expanding them on every chapter's first use was
#     judged pedantic. The exempt list includes: CPU/GPU/TPU/ASIC/FPGA/DSA
#     hardware baseline; CNN/RNN/LSTM/MLP/LLM/ViT/GAN/VAE/MoE model
#     baseline; JIT/AOT/IR/ONNX compiler baseline; ReLU/Adam/SGD/AD/BPTT/
#     GEMM/BLAS numerical baseline; SIMD/RISC/MIPS architecture baseline;
#     NVMe/HBM/DRAM/SRAM memory baseline; JSON protocol baseline; SLA/ROC/
#     AUC/TCO operations baseline; GDPR/HIPAA legislation (proper nouns);
#     MAC/IOPS/NaN numerical baseline. See §10.5 for the full list and
#     rationale. These terms should be introduced once in the book (in
#     their canonical chapter) and may be used bare everywhere else.

import yaml

DATA_FILE = Path(__file__).resolve().parents[3] / "cli" / "data" / "abbreviations.yaml"
if not DATA_FILE.is_file():
    raise FileNotFoundError(f"Authoritative abbreviations data file missing: {DATA_FILE}")

with open(DATA_FILE, "r", encoding="utf-8") as _f:
    _DATA = yaml.safe_load(_f)

if not _DATA or "canonical_expansions" not in _DATA:
    raise ValueError(f"Invalid abbreviations data in {DATA_FILE}")

_CANONICAL = [
    (abbrev, tuple(exp) if isinstance(exp, list) else exp)
    for abbrev, exp in _DATA["canonical_expansions"].items()
]
_EXEMPT_BASELINES = set(_DATA.get("exempt_baselines", []))

# File-level exclusions. Files listed here are skipped entirely because
# their purpose is to define terms, not use them in running prose.
_EXCLUDED_FILES = (
    "glossary.qmd",
)

# A footnote definition head, e.g. `[^fn-mac]: **MAC (Multiply-Accumulate)**: …`
# (an optional `[offset=…]` layout directive may follow the colon).
#
# Footnote heads are EXEMPT from over-expansion. abbreviations.md currently says
# a footnote head "should usually use the acronym alone" once body prose has
# introduced it, but house practice has settled the other way: commit 284977f5
# (2026-08-17) deliberately restored full expansions across every footnote head
# in both volumes, treating the head as a self-contained glossary-style
# definition a reader meets cold in the margin. Practice wins here, so the check
# does not fight it. If the rule is ever reconciled to match, drop this exemption.
_FOOTNOTE_DEF_RE = re.compile(r"^\[\^[^\]]+\]:")


# Markup that may legitimately sit between an expansion and its `(ABBREV)`
# parenthetical. The house index convention (index.md) places the `\index{}`
# tag immediately after the term, which lands it in exactly this position:
#
#     neural architecture search\index{Neural Architecture Search} (NAS)
#
# Bold spans do the same when the term is a first definition (emphasis.md):
#
#     **Population Stability Index**\index{...} (PSI)
#
# Without this tolerance the checker reports a false "no introduction found"
# for every term that follows the book's own conventions. Each alternative
# consumes at least one character, so the group cannot loop on empty input.
# (Added 2026-08-17 after the probe in the abbreviation tooling audit showed
# the interposed-index form was silently unrecognized.)
_INTERPOSED = r"(?:\s|\*{1,3}|\\index\{[^{}]*\})*"


def _expansions(expansion) -> tuple:
    """Normalize an entry's expansion field to a tuple of alternatives."""
    return expansion if isinstance(expansion, tuple) else (expansion,)


def _expansion_body(expansion, anchored: bool) -> str:
    """Regex alternation matching any canonical expansion for one abbreviation.

    Each alternative allows flexible inter-word whitespace (line wrapping) and
    an optional trailing `s` for singular/plural inflection. `anchored` adds a
    leading word boundary, which the forward form needs and the reverse form
    (already inside parentheses) does not.
    """
    parts = []
    for exp in _expansions(expansion):
        tokens = re.split(r"\s+", exp)
        body = r"\s+".join(re.escape(t) for t in tokens) + r"s?"
        parts.append((r"\b" if anchored else "") + body)
    return "(?:" + "|".join(parts) + ")"


def _build_canonical_regex(abbrev: str, expansion) -> re.Pattern:
    """Regex for `<expansion> (<ABBREV>)` forward canonical introduction.

    The expansion part is case-insensitive and allows a trailing `s` for
    singular/plural inflection. The abbreviation part is case-sensitive
    and also allows `s?`. Whitespace between words in the expansion is
    flexible to handle line wrapping, and `\\index{}` / bold markup may sit
    between the expansion and the parenthetical (see `_INTERPOSED`).
    """
    # Case-insensitive expansion, case-sensitive abbreviation.
    expansion_re = _expansion_body(expansion, anchored=True)
    pattern = f"(?i:{expansion_re})" + _INTERPOSED + r"\(" + re.escape(abbrev) + r"s?\)"
    return re.compile(pattern)


def _build_canonical_regex_reverse(abbrev: str, expansion) -> re.Pattern:
    """Regex for `<ABBREV> (<expansion>)` reverse canonical introduction.

    Many footnote definitions and glossary entries use the reverse form:
    `**BLAS (Basic Linear Algebra Subprograms)**`. This is a valid
    introduction per §10.5's intent (the reader sees both the acronym
    and the expansion in one place).

    The parenthetical must match the exact canonical expansion
    (case-insensitive) so that asides like `MLP (Overhead Bound)` —
    where "Overhead Bound" is not the canonical expansion "multilayer
    perceptron" — do NOT count as introductions.
    """
    expansion_re = _expansion_body(expansion, anchored=False)
    pattern = (
        r"\b" + re.escape(abbrev) + r"s?" + _INTERPOSED + r"\(\s*"
        + f"(?i:{expansion_re})"
        + r"\s*\)"
    )
    return re.compile(pattern)


def _build_bare_regex(abbrev: str) -> re.Pattern:
    """Regex for bare word-boundary match.

    Case-sensitive. Matches both singular and plural (e.g. `CNN` and
    `CNNs`) as the same logical abbreviation. Excludes hyphenated prefixes
    and suffixes (e.g., `DP-SGD` is not treated as a bare occurrence of `SGD`).
    """
    return re.compile(r"(?<![-\w])" + re.escape(abbrev) + r"s?(?![-\w])")


# Precompile both regex sets at import time.
_CANONICAL_RE = {abbrev: _build_canonical_regex(abbrev, exp)
                 for abbrev, exp in _CANONICAL}
_CANONICAL_REVERSE_RE = {abbrev: _build_canonical_regex_reverse(abbrev, exp)
                         for abbrev, exp in _CANONICAL}
_BARE_RE = {abbrev: _build_bare_regex(abbrev) for abbrev, _ in _CANONICAL}
_EXPANSION_FOR = {abbrev: exp for abbrev, exp in _CANONICAL}


def _expansion_label(abbrev: str) -> str:
    """Human-readable expansion for an issue message, joining homonyms with 'or'."""
    return " or ".join(_expansions(_EXPANSION_FOR[abbrev]))


# ── Line-level filter ──────────────────────────────────────────────────────

def _skip_line_for_bare(line: str, state) -> bool:
    """Return True for lines where bare uses should NOT be flagged.

    Skips block-level protected contexts and several §10.5 protected
    editorial contexts where expanding an abbreviation would distort
    the format rather than clarify it:

    - YAML, code fences, display math, HTML style blocks, HTML comments
    - Python chunk options, div attribute lines
    - Headings (H1-H6) — per §10.9 headings use their own case rules
    - Table rows (both header rows with bold and data rows) — a cell
      is too constrained for a multi-word expansion
    - Table captions (`: **Title** ... {#tbl-...}`) — caption headers
      follow a distinct formatting contract
    - `.callout-tip` blocks (Learning Objectives) — protected per §9
      "Protected Content"; the Learning Objectives callout is the one
      place where bare abbreviations in bullets are expected
    - `.callout-checkpoint` blocks (self-check questions) — §9 protected

    Does NOT skip lines starting with `\\` — `\\index{foo!bar}` lines
    contain body prose (same rationale as
    concept_term_capitalization._skip_concept_term_line).

    Intro-finding uses a separate, looser filter so that introductions
    appearing inside headings still count as valid.
    """
    if state.in_yaml or state.in_code_fence or state.in_display_math or state.in_tikz:
        return True
    if state.in_html_style_block or state.in_html_comment:
        return True
    if state.in_tip_callout or state.in_checkpoint_callout:
        return True
    if state.in_definition_callout:
        return True
    if is_python_chunk_option(line):
        return True
    if is_div_attribute_line(line):
        return True
    if heading_level(line) is not None:
        return True
    if is_table_row(line):
        return True
    if is_table_caption_line(line):
        return True
    return False


def _skip_line_for_intro(line: str, state) -> bool:
    """Return True for lines where canonical introductions should NOT count.

    Introductions are allowed inside headings (a reader who reads the
    heading gets the expansion). We only skip lines that are truly
    non-prose: YAML frontmatter, code fences, display math, HTML style
    blocks/comments, Python chunk options, and div fences.
    """
    if state.in_yaml or state.in_code_fence or state.in_display_math or state.in_tikz:
        return True
    if state.in_html_style_block or state.in_html_comment:
        return True
    if is_python_chunk_option(line):
        return True
    if is_div_attribute_line(line):
        return True
    return False


# ── Match-level filter for bare uses ───────────────────────────────────────

def _is_bare_use_protected(
    line: str,
    start: int,
    spans: list[tuple[int, int]],
    intro_spans: list[tuple[int, int]],
) -> bool:
    """Return True if this bare-use match should not count as a bare use.

    Protected if: inside any inline span (code, math, index, citation,
    footnote ref, cross-reference, anchor id), inside an index entry,
    inside a protected attribute (title=, fig-cap=, fig-alt=), or
    inside a canonical-introduction span on the same line.
    """
    if position_in_spans(start, spans):
        return True
    if is_inside_index_entry(line, start):
        return True
    if is_inside_protected_attribute(line, start):
        return True
    for s, e in intro_spans:
        if s <= start < e:
            return True
    return False


# ── Main check entry point ─────────────────────────────────────────────────

def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan a file for §10.5 abbreviation-first-use violations.

    One-pass walk with two state dicts:
      - `intro_line`: abbrev → line number of first canonical
        introduction found (or absent if never introduced).
      - `first_bare`: abbrev → (line_num, col, line_text) of the
        first bare use found that is NOT protected and NOT on a
        line where the same abbrev is being introduced.

    After the walk, for each abbreviation where a first bare use was
    recorded AND the first introduction (if any) occurs strictly
    after the first bare use line, emit one issue at the bare-use
    line. Abbreviations that were introduced before or on the same
    line as their first bare use are silent.
    """
    issues: list[Issue] = []
    counter = start_counter

    # File-level exclusions: skip glossary files (the file is itself a
    # table of definitions, so every "first use" is really a definition
    # head, not a body-prose use).
    if file_path.name in _EXCLUDED_FILES:
        return issues, counter

    # Path-level exclusions: parts/ files are short volume dividers that
    # list concepts by abbreviation, not running prose that introduces
    # them. They inherit expansions from the chapters they divide.
    if "/parts/" in str(file_path):
        return issues, counter

    intro_line: dict[str, int] = {}
    first_bare: dict[str, tuple[int, int, str]] = {}
    # Body-prose introductions only, in document order, for over-expansion
    # (abbreviations.md: "the same chapter repeats the full expansion after the
    # acronym has already been introduced"). Restricting to body prose keeps
    # headings, table cells, captions, and callout scaffolding out of the count,
    # since those legitimately carry an expansion of their own.
    intro_body: dict[str, list[tuple[int, int, str]]] = {}

    walker = LineWalker(text)
    for line, state, line_num in walker:
        # Phase 1: find canonical introductions on this line.
        # Both forward (`expansion (ABBREV)`) and reverse (`ABBREV
        # (expansion)`) forms count as valid introductions per §10.5's
        # intent. Introductions count in headings too — a reader who
        # reads `## Convolutional neural network (CNN) architectures`
        # has seen the expansion.
        intro_spans_by_abbrev: dict[str, list[tuple[int, int]]] = {}
        is_body_prose = not _skip_line_for_bare(line, state)
        line_spans = inline_protected_spans(line)
        if not _skip_line_for_intro(line, state):
            for abbrev in _CANONICAL_RE:
                for pattern in (_CANONICAL_RE[abbrev],
                                _CANONICAL_REVERSE_RE[abbrev]):
                    for m in pattern.finditer(line):
                        # The span still suppresses a bare-use flag on this line
                        # even when it is not a reader-visible introduction.
                        intro_spans_by_abbrev.setdefault(abbrev, []).append(
                            (m.start(), m.end())
                        )
                        # An expansion that lives only inside an \index{} key,
                        # inline code, or math never reaches the reader, so it
                        # does not introduce the term. The house convention puts
                        # an index tag right after the term (index.md), which
                        # routinely duplicates the expansion inside the key —
                        # counting that as a second introduction reported a
                        # false over-expansion. (2026-08-17)
                        if is_inside_index_entry(line, m.start()):
                            continue
                        if position_in_spans(m.start(), line_spans):
                            continue
                        if abbrev not in intro_line:
                            intro_line[abbrev] = line_num
                        if is_body_prose and not _FOOTNOTE_DEF_RE.match(line.lstrip()):
                            intro_body.setdefault(abbrev, []).append(
                                (line_num, m.start(), line)
                            )

        # Phase 2: find bare uses on this line.
        # Bare uses are checked in body prose only: no headings, no
        # table headers, no block-level protected contexts.
        if not is_body_prose:
            continue

        spans = line_spans

        for abbrev, bare_re in _BARE_RE.items():
            # Already found the first bare use for this abbrev in this file.
            if abbrev in first_bare:
                continue
            # Introduced on a previous line — any bare use here is fine.
            if abbrev in intro_line and intro_line[abbrev] < line_num:
                continue
            intro_spans = intro_spans_by_abbrev.get(abbrev, [])
            for m in bare_re.finditer(line):
                if _is_bare_use_protected(line, m.start(), spans, intro_spans):
                    continue
                # Found a first bare use; record and stop scanning this abbrev.
                first_bare[abbrev] = (line_num, m.start(), line)
                break

    # Phase 3: emit issues for abbreviations whose first bare use comes
    # before their first canonical introduction (or that were never
    # introduced in this file at all).
    for abbrev, (line_num, col, line) in sorted(first_bare.items()):
        intro_ln = intro_line.get(abbrev)
        if intro_ln is not None and intro_ln <= line_num:
            continue
        reason = (
            f"First use of {abbrev!r} without canonical expansion "
            f"'{_expansion_label(abbrev)} ({abbrev})'"
        )
        if intro_ln is None:
            reason += " (no introduction found in this file)"
        else:
            reason += f" (first introduction is later, at line {intro_ln})"
        issues.append(
            Issue(
                id=make_issue_id(scope, CATEGORY, counter),
                category=CATEGORY,
                rule=RULE,
                rule_text=RULE_TEXT,
                file=str(file_path),
                line=line_num,
                col=col,
                before=line,
                suggested_after="",  # fix requires editorial judgment
                auto_fixable=False,
                needs_subagent=True,
                reason=reason,
            )
        )
        counter += 1

    # Phase 4: emit over-expansion issues. abbreviations.md requires the audit
    # to check BOTH directions; before 2026-08-17 only under-expansion existed,
    # so a chapter could re-teach the same expansion indefinitely with no gate.
    # Once a chapter has introduced an abbreviation in body prose, later body
    # prose uses the acronym alone.
    for abbrev, occurrences in sorted(intro_body.items()):
        for line_num, col, line in occurrences[1:]:
            issues.append(
                Issue(
                    id=make_issue_id(scope, CATEGORY_OVER, counter),
                    category=CATEGORY_OVER,
                    rule=RULE,
                    rule_text=RULE_TEXT_OVER,
                    file=str(file_path),
                    line=line_num,
                    col=col,
                    before=line,
                    suggested_after="",  # fix requires editorial judgment
                    auto_fixable=False,
                    needs_subagent=True,
                    reason=(
                        f"Repeat expansion of {abbrev!r} at line {line_num}; "
                        f"the chapter already introduced it at line "
                        f"{occurrences[0][0]}. Use the bare acronym here."
                    ),
                )
            )
            counter += 1

    return issues, counter


# ── Adversarial self-test ──────────────────────────────────────────────────
#
# Run with:
#     PYTHONPATH=book/tools python3 book/tools/audit/checks/abbreviation_first_use.py
#
# Each case is (name, text, expected) where `expected` is a set of
# "under:ABBREV" / "over:ABBREV" markers. The driver runs `check` on the text
# as if it were a single chapter file and compares.
#
# NOTE (2026-08-17): these cases previously exercised CNN / RNN / LLM / MLP,
# which moved to the §10.5 exempt baseline in the Round-2 change of 2026-04-24.
# The suite had been red for six of seventeen cases ever since, so a genuine
# regression would not have been noticed. Every case now uses an abbreviation
# the checker actually tracks.

_TESTS = [
    # ---- Positive: bare use before canonical introduction ----
    (
        "bare DAG with no introduction",
        "The DAG is rebuilt on every optimizer step.\n",
        {"under:DAG"},
    ),
    (
        "bare DAG with introduction only on a later line",
        "The DAG is rebuilt on every step.\n"
        "A directed acyclic graph (DAG) represents the computation.\n",
        {"under:DAG"},
    ),
    (
        "bare KWS never introduced",
        "This chapter discusses KWS wake-word pipelines.\n"
        "KWS inference is dominated by always-on power budget.\n",
        {"under:KWS"},
    ),
    (
        "plural bare use without introduction",
        "Modern DAGs encode operator dependencies explicitly.\n",
        {"under:DAG"},
    ),
    # ---- Negative: canonical introduction is present and early ----
    (
        "DAG introduced on first appearance",
        "A directed acyclic graph (DAG) is the workhorse representation.\n"
        "The DAG is then scheduled onto devices.\n",
        set(),
    ),
    (
        "introduction on the same line as first bare use",
        "The directed acyclic graph (DAG) discussion begins here.\n"
        "Subsequent DAG details follow.\n",
        set(),
    ),
    (
        "plural introduction covers plural bare use",
        "Directed acyclic graphs (DAGs) dominate graph compilers.\n"
        "The DAGs are then fused.\n",
        set(),
    ),
    (
        "introduction inside a heading counts",
        "## Directed acyclic graph (DAG) scheduling\n"
        "\n"
        "The DAG below has three stages.\n",
        set(),
    ),
    # ---- The \index{} interposition (house convention, index.md) ----
    (
        "expansion with an interposed \\index tag still counts",
        "The compiler builds a directed acyclic graph"
        "\\index{Directed Acyclic Graph} (DAG) for the model.\n"
        "The DAG is then optimized.\n",
        set(),
    ),
    (
        "expansion with bold and an interposed \\index tag still counts",
        "The **keyword spotting**\\index{Keyword Spotting!definition} (KWS) "
        "pipeline runs always-on.\n"
        "Later KWS mentions follow.\n",
        set(),
    ),
    # ---- Homonym: either canonical expansion introduces the term ----
    (
        "NAS as neural architecture search",
        "We apply neural architecture search (NAS) under a latency budget.\n"
        "The NAS run costs thousands of GPU-hours.\n",
        set(),
    ),
    (
        "NAS as network-attached storage",
        "A network-attached storage (NAS) appliance fronts the dataset.\n"
        "The NAS becomes the bottleneck at scale.\n",
        set(),
    ),
    (
        "bare NAS with neither expansion is still flagged",
        "The NAS budget dominates the experiment.\n",
        {"under:NAS"},
    ),
    # ---- Negative: bare use inside protected contexts ----
    (
        "bare DAG inside inline code",
        "The config sets `DAG=True` to enable graph mode.\n"
        "A directed acyclic graph (DAG) is then constructed.\n",
        set(),
    ),
    (
        "bare abbrev inside an index entry",
        "\\index{DAG!scheduling}A directed acyclic graph (DAG) is used.\n",
        set(),
    ),
    (
        "bare DAG in heading with later introduction",
        "### The DAG scheduler\n"
        "\n"
        "A directed acyclic graph (DAG) is the workhorse.\n",
        set(),
    ),
    # ---- Multi-abbreviation cases ----
    (
        "two abbrevs, both missing introduction",
        "The DAG and the ETL job disagree on ordering.\n",
        {"under:DAG", "under:ETL"},
    ),
    (
        "two abbrevs, one introduced, one not",
        "A directed acyclic graph (DAG) is paired with an ETL job.\n",
        {"under:ETL"},
    ),
    # ---- Reverse canonical form ----
    (
        "reverse form ABBREV (expansion) introduces the abbreviation",
        "Access follows RBAC (role-based access control) semantics.\n"
        "Modern RBAC policies are declarative.\n",
        set(),
    ),
    (
        "reverse form in footnote-definition style",
        "[^fn-ptx]: **PTX (Parallel Thread Execution)**: A virtual ISA.\n"
        "The PTX layer insulates code from silicon revisions.\n",
        set(),
    ),
    (
        "parenthetical that is NOT the canonical expansion must not count",
        "The XLA (Overhead Bound) path struggles on small batches.\n",
        {"under:XLA"},
    ),
    # ---- Over-expansion (the other half of abbreviations.md) ----
    (
        "repeating the full expansion in body prose is over-expansion",
        "A directed acyclic graph (DAG) represents the computation.\n"
        "Later, the directed acyclic graph (DAG) is scheduled.\n",
        {"over:DAG"},
    ),
    (
        "single expansion is not over-expansion",
        "A directed acyclic graph (DAG) represents the computation.\n"
        "Later, the DAG is scheduled.\n",
        set(),
    ),
    (
        "expansion in a heading plus one in body prose is not over-expansion",
        "## Directed acyclic graph (DAG) scheduling\n"
        "\n"
        "A directed acyclic graph (DAG) represents the computation.\n",
        set(),
    ),
    (
        # Regression: the house convention duplicates the expansion inside the
        # \index{} key, which is not reader-visible. Counting it reported a
        # false over-expansion on vol2/sustainable_ai:2281. (2026-08-17)
        "expansion duplicated inside its own \\index key is not over-expansion",
        "[^fn-nas]: **Neural Architecture Search (NAS) Carbon Cost**"
        "\\index{Neural Architecture Search (NAS) Carbon Cost!definition}: "
        "The reported figure is contested.\n",
        set(),
    ),
    (
        # House practice (commit 284977f5, 2026-08-17) keeps the full expansion
        # in every footnote definition head, so a head is never over-expansion.
        "footnote definition head repeating the expansion is exempt",
        "A directed acyclic graph (DAG) represents the computation.\n"
        "\n"
        "[^fn-dag]: **Directed Acyclic Graph (DAG)**: The scheduling structure.\n",
        set(),
    ),
    (
        "three body expansions flag the second and third",
        "A directed acyclic graph (DAG) is built.\n"
        "The directed acyclic graph (DAG) is optimized.\n"
        "The directed acyclic graph (DAG) is scheduled.\n",
        {"over:DAG"},
    ),
]

_KIND = {CATEGORY: "under", CATEGORY_OVER: "over"}


def _self_test() -> int:
    passed = 0
    failed = 0
    failures: list[str] = []

    for name, text, expected in _TESTS:
        issues, _ = check(Path("<test>"), text, "test", 0)
        got = {
            f"{_KIND[i.category]}:{i.reason.split(chr(39))[1]}"
            for i in issues
        }
        if got == expected:
            passed += 1
        else:
            failed += 1
            failures.append(
                f"{name}:\n    expected {sorted(expected)}\n    got      {sorted(got)}"
            )

    total = passed + failed
    print(f"abbreviation_first_use self-test: {passed}/{total} passed")
    for f in failures:
        print(f"\n  {f}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(_self_test())
