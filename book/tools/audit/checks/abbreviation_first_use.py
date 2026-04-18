"""Check: abbreviations must be expanded on first use per chapter (§10.5).

Rule: book-prose-merged.md section 10.5

    "Every abbreviation is expanded on its first use in each chapter.
    Expansion resets at every chapter boundary. Pattern:
    `convolutional neural network (CNN)` on first use, then `CNN`
    everywhere else in that chapter."

Each `.qmd` file is treated as one chapter. The check walks the file
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

Fix: editorial judgment required. The typical fix is to insert the
canonical expansion at the first bare use OR to move an existing
later expansion earlier. Not auto-fixable; `needs_subagent=True`.

Reference: book-prose-merged.md section 10.5 (canonical forms dict).
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
RULE = "book-prose-merged.md section 10.5"
RULE_TEXT = "Abbreviations must be expanded on first use per chapter"


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

_CANONICAL = [
    ("AST",    "abstract syntax tree"),
    ("AOT",    "ahead-of-time"),
    ("AUC",    "area under the curve"),  # §10.5 uses "area under the [ROC] curve"
    ("BPTT",   "backpropagation through time"),
    ("BLAS",   "Basic Linear Algebra Subprograms"),
    ("CNN",    "convolutional neural network"),
    ("CTM",    "continuous therapeutic monitoring"),
    ("DAG",    "directed acyclic graph"),
    ("DCE",    "dead-code elimination"),
    ("ELT",    "extract, load, transform"),
    ("ETL",    "extract, transform, load"),
    ("FFT",    "fast Fourier transform"),
    ("GDPR",   "General Data Protection Regulation"),
    ("GELU",   "Gaussian Error Linear Unit"),
    ("GEMM",   "general matrix multiply"),
    ("HIPAA",  "Health Insurance Portability and Accountability Act"),
    ("HOG",    "histogram of oriented gradients"),
    ("ICR",    "information-compute ratio"),
    ("ILSVRC", "ImageNet Large Scale Visual Recognition Challenge"),
    ("IOPS",   "input/output operations per second"),
    ("IR",     "intermediate representation"),
    ("JIT",    "just-in-time"),
    ("JSON",   "JavaScript Object Notation"),
    ("KWS",    "keyword spotting"),
    ("LLM",    "large language model"),
    ("MAC",    "multiply-accumulate"),
    ("MIPS",   "microprocessor without interlocked pipelined stages"),
    ("MLP",    "multilayer perceptron"),
    ("MoE",    "mixture-of-experts"),
    ("NAS",    "neural architecture search"),
    ("NaN",    "not a number"),
    ("NVMe",   "Non-Volatile Memory Express"),
    ("ONNX",   "Open Neural Network Exchange"),
    ("OTA",    "over-the-air"),
    ("PTX",    "Parallel Thread Execution"),
    ("RBAC",   "role-based access control"),
    ("ReLU",   "rectified linear unit"),
    ("RISC",   "reduced instruction set computer"),
    ("RNN",    "recurrent neural network"),
    ("ROC",    "receiver operating characteristic"),
    # SIFT is deliberately excluded: the §10.5 expansion is
    # "scale-invariant feature transform" (computer vision), but the
    # book also uses SIFT as "software-implemented fault tolerance" in
    # fault_tolerance.qmd — a different acronym that spells the same.
    # Homonym handling would require per-context disambiguation; skip
    # the term in Item D to avoid FPs.
    ("SIMD",   "single instruction, multiple data"),
    ("SLA",    "service level agreement"),
    ("SoC",    "system on chip"),
    ("SSA",    "static single-assignment"),
    ("TCO",    "total cost of ownership"),
    ("TFDV",   "TensorFlow Data Validation"),
    ("TPU",    "Tensor Processing Unit"),
    ("UAT",    "universal approximation theorem"),
    ("ViT",    "vision transformer"),
    ("Adam",   "Adaptive Moment Estimation"),
]

# File-level exclusions. Files listed here are skipped entirely because
# their purpose is to define terms, not use them in running prose.
_EXCLUDED_FILES = (
    "glossary.qmd",
)


def _build_canonical_regex(abbrev: str, expansion: str) -> re.Pattern:
    """Regex for `<expansion> (<ABBREV>)` forward canonical introduction.

    The expansion part is case-insensitive and allows a trailing `s` for
    singular/plural inflection. The abbreviation part is case-sensitive
    and also allows `s?`. Whitespace between words in the expansion is
    flexible to handle line wrapping.
    """
    # Tokenize the expansion on whitespace, escape each token, join with
    # `\s+` to allow flexible inter-word whitespace.
    expansion_tokens = re.split(r"\s+", expansion)
    expansion_re = r"\b" + r"\s+".join(re.escape(t) for t in expansion_tokens) + r"s?"
    # Case-insensitive expansion, case-sensitive abbreviation.
    pattern = f"(?i:{expansion_re})" + r"\s*\(" + re.escape(abbrev) + r"s?\)"
    return re.compile(pattern)


def _build_canonical_regex_reverse(abbrev: str, expansion: str) -> re.Pattern:
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
    expansion_tokens = re.split(r"\s+", expansion)
    expansion_re = r"\s+".join(re.escape(t) for t in expansion_tokens) + r"s?"
    pattern = (
        r"\b" + re.escape(abbrev) + r"s?\s*\(\s*"
        + f"(?i:{expansion_re})"
        + r"\s*\)"
    )
    return re.compile(pattern)


def _build_bare_regex(abbrev: str) -> re.Pattern:
    """Regex for bare `\\b<ABBREV>s?\\b` word-boundary match.

    Case-sensitive. Matches both singular and plural (e.g. `CNN` and
    `CNNs`) as the same logical abbreviation.
    """
    return re.compile(r"\b" + re.escape(abbrev) + r"s?\b")


# Precompile both regex sets at import time.
_CANONICAL_RE = {abbrev: _build_canonical_regex(abbrev, exp)
                 for abbrev, exp in _CANONICAL}
_CANONICAL_REVERSE_RE = {abbrev: _build_canonical_regex_reverse(abbrev, exp)
                         for abbrev, exp in _CANONICAL}
_BARE_RE = {abbrev: _build_bare_regex(abbrev) for abbrev, _ in _CANONICAL}
_EXPANSION_FOR = {abbrev: exp for abbrev, exp in _CANONICAL}


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
    if state.in_yaml or state.in_code_fence or state.in_display_math:
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
    if state.in_yaml or state.in_code_fence or state.in_display_math:
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

    intro_line: dict[str, int] = {}
    first_bare: dict[str, tuple[int, int, str]] = {}

    walker = LineWalker(text)
    for line, state, line_num in walker:
        # Phase 1: find canonical introductions on this line.
        # Both forward (`expansion (ABBREV)`) and reverse (`ABBREV
        # (expansion)`) forms count as valid introductions per §10.5's
        # intent. Introductions count in headings too — a reader who
        # reads `## Convolutional neural network (CNN) architectures`
        # has seen the expansion.
        intro_spans_by_abbrev: dict[str, list[tuple[int, int]]] = {}
        if not _skip_line_for_intro(line, state):
            for abbrev in _CANONICAL_RE:
                for pattern in (_CANONICAL_RE[abbrev],
                                _CANONICAL_REVERSE_RE[abbrev]):
                    for m in pattern.finditer(line):
                        intro_spans_by_abbrev.setdefault(abbrev, []).append(
                            (m.start(), m.end())
                        )
                        if abbrev not in intro_line:
                            intro_line[abbrev] = line_num

        # Phase 2: find bare uses on this line.
        # Bare uses are checked in body prose only: no headings, no
        # table headers, no block-level protected contexts.
        if _skip_line_for_bare(line, state):
            continue

        spans = inline_protected_spans(line)

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
            f"'{_EXPANSION_FOR[abbrev]} ({abbrev})'"
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

    return issues, counter


# ── Adversarial self-test (Pass 16 Item D) ─────────────────────────────────
#
# Run with:
#     PYTHONPATH=book/tools python3 book/tools/audit/checks/abbreviation_first_use.py
#
# Each test case is a (name, text, expected_flagged_abbrevs) triple. The
# driver runs `check` on each text as if it were a single chapter file
# and compares the set of flagged abbreviations to the expectation.

_TESTS = [
    # ---- Positive cases: bare use before canonical introduction ----
    (
        "bare CNN with no introduction",
        "The CNN achieves state-of-the-art accuracy on ImageNet.\n",
        {"CNN"},
    ),
    (
        "bare DAG with late introduction on a later line",
        "The DAG is rebuilt on every step.\n"
        "A directed acyclic graph (DAG) represents the computation.\n",
        {"DAG"},
    ),
    (
        "bare LLM never introduced",
        "This chapter discusses LLM serving strategies.\n"
        "LLM inference is dominated by memory bandwidth.\n",
        {"LLM"},
    ),
    (
        "plural bare use without introduction",
        "Modern CNNs use residual connections extensively.\n",
        {"CNN"},
    ),
    # ---- Negative cases: canonical introduction is present and early ----
    (
        "CNN introduced on first appearance",
        "A convolutional neural network (CNN) is the workhorse.\n"
        "The CNN is trained via SGD.\n",
        set(),
    ),
    (
        "CNN introduced on the same line as first bare use",
        "The convolutional neural network (CNN) discussion begins here.\n"
        "Subsequent CNN details follow.\n",
        set(),
    ),
    (
        "plural introduction covers plural bare use",
        "Convolutional neural networks (CNNs) dominate vision.\n"
        "The CNNs are trained end-to-end.\n",
        set(),
    ),
    (
        "introduction inside a heading counts",
        "## Convolutional neural network (CNN) architectures\n"
        "\n"
        "The CNN below has three layers.\n",
        set(),
    ),
    # ---- Negative: bare use inside protected contexts ----
    (
        "bare CNN inside inline code",
        "The config sets `CNN=True` to enable convolutions.\n"
        "A convolutional neural network (CNN) is then constructed.\n",
        set(),
    ),
    (
        "bare CNN inside citation key",
        "Prior work explores this direction [@krizhevsky2012cnn].\n"
        "A convolutional neural network (CNN) is constructed below.\n",
        set(),
    ),
    (
        "bare abbrev inside an index entry",
        "\\index{CNN!architecture}A convolutional neural network (CNN) is used.\n",
        set(),
    ),
    # ---- Negative: bare use in heading (skipped by bare-use line filter) ----
    (
        "bare CNN in heading with later introduction",
        "### The CNN architecture\n"
        "\n"
        "A convolutional neural network (CNN) is the workhorse.\n",
        set(),
    ),
    # ---- Multi-abbreviation cases ----
    (
        "two abbrevs, both missing introduction",
        "The CNN and the RNN dominate different modalities.\n",
        {"CNN", "RNN"},
    ),
    (
        "two abbrevs, one introduced, one not",
        "A convolutional neural network (CNN) is paired with an RNN.\n",
        {"RNN"},
    ),
    # ---- Reverse canonical form ----
    (
        "reverse form BLAS (expansion) introduces the abbreviation",
        "The name comes from the BLAS (Basic Linear Algebra Subprograms) specification.\n"
        "Modern BLAS implementations include OpenBLAS and MKL.\n",
        set(),
    ),
    (
        "reverse form in footnote-definition style",
        "[^fn-nvme]: **NVMe (Non-Volatile Memory Express)**: A storage protocol.\n"
        "The NVMe bandwidth ceiling is 7 GB/s per lane.\n",
        set(),
    ),
    (
        "parenthetical that is NOT the canonical expansion must not count",
        "The MLP (Overhead Bound) struggles on small batch sizes.\n",
        {"MLP"},
    ),
]


def _self_test() -> int:
    from audit.ledger import Ledger

    passed = 0
    failed = 0
    failures: list[str] = []

    for name, text, expected in _TESTS:
        issues, _ = check(Path("<test>"), text, "test", 0)
        got = {i.reason.split("'")[1] for i in issues}
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
