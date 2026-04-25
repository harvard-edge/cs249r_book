"""Check: notation consistency against the Notations chapter(s).

This is a conservative, high-signal scanner intended for CI:
- Emits OPEN issues only for high-confidence convention violations.
- Emits DEFERRED issues for context-dependent collisions (review-only).

Ground truth definitions live in:
  - book/quarto/contents/vol1/frontmatter/_notation_body.qmd (shared)
  - book/quarto/contents/vol2/frontmatter/_notation_distributed.qmd (vol2-only)

Rule: "Notation conventions" (see the Notations chapter in each volume).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from audit.ledger import Issue, STATUS_DEFERRED, make_issue_id
from audit.protected_contexts import LineWalker

CATEGORY = "notation-consistency"
RULE = "Notations chapter (vol1/vol2) — symbol collision conventions"
RULE_TEXT = (
    "Enforce book-wide notation conventions (e.g. BW not B for bandwidth; "
    "D_vol not D for bytes moved; R_peak not P for peak rate; L_lat not L for latency)."
)


# ── Definition parsing (from the Notations chapter tables) ────────────────────

_NOTATION_PATHS = (
    Path("book/quarto/contents/vol1/frontmatter/_notation_body.qmd"),
    Path("book/quarto/contents/vol2/frontmatter/_notation_distributed.qmd"),
)


def _strip_md(cell: str) -> str:
    s = cell.strip()
    # Remove bold/italic wrappers used in the tables.
    s = re.sub(r"^\*{1,3}", "", s)
    s = re.sub(r"\*{1,3}$", "", s)
    return s.strip()


def _strip_math_dollars(s: str) -> str:
    s = s.strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        return s[1:-1].strip()
    return s


def _canonical_symbol(raw: str) -> str:
    """Normalize a symbol string for matching.

    Keep LaTeX structure (subscripts) but normalize common wrappers.
    """
    s = _strip_md(raw)
    s = _strip_math_dollars(s)
    s = re.sub(r"\s+", "", s)
    # Normalize \mathrm{...} -> \text{...} for matching BW/MTBF/etc.
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\\text{\1}", s)
    return s


def _parse_symbol_defs(text: str) -> set[str]:
    """Extract the first column (Symbol) from markdown pipe tables."""
    lines = text.splitlines()
    symbols: set[str] = set()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.lstrip().startswith("|"):
            i += 1
            continue

        # Header row + separator row indicate a pipe table.
        if i + 1 >= len(lines):
            i += 1
            continue
        header = line
        sep = lines[i + 1]
        if not (sep.lstrip().startswith("|") and ":" in sep and "-" in sep):
            i += 1
            continue
        # Ensure this table is a notation table (header contains Symbol).
        if "Symbol" not in header:
            i += 1
            continue

        # Walk table rows until the pipe-table ends.
        i += 2
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            row = lines[i].strip()
            # Skip separator-like rows.
            if set(row.replace("|", "").strip()) <= set(":-"):
                i += 1
                continue
            parts = [p.strip() for p in row.strip("|").split("|")]
            if parts:
                sym_raw = parts[0]
                sym = _canonical_symbol(sym_raw)
                if sym:
                    symbols.add(sym)
            i += 1
        continue
    return symbols


_DEF_CACHE: set[str] | None = None


def _load_definition_registry(repo_root: Path) -> set[str]:
    global _DEF_CACHE
    if _DEF_CACHE is not None:
        return _DEF_CACHE

    symbols: set[str] = set()
    for rel in _NOTATION_PATHS:
        p = (repo_root / rel).resolve()
        if not p.exists():
            continue
        symbols |= _parse_symbol_defs(p.read_text(encoding="utf-8"))

    _DEF_CACHE = symbols
    return symbols


# ── Math extraction + token detection ─────────────────────────────────────────

_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)([^\n$]|\\\$)+?(?<!\\)\$(?!\$)")


def _display_delim_count(line: str) -> int:
    """Mirror LineWalker display-math delimiter logic (simplified)."""
    if "$$" not in line:
        return 0
    starts = bool(re.match(r"^\s*\$\$", line))
    ends = bool(re.search(r"\$\$\s*(?:\{[^}]*\}\s*)?$", line))
    after_end = bool(re.search(r"\\end\{[^}]+\}\s*\$\$", line))
    num = line.count("$$")
    if starts and ends and num >= 2:
        return 2
    if starts or ends or after_end:
        return 1
    return 0


@dataclass(frozen=True)
class MathOccurrence:
    line_num: int
    raw_math: str
    context: str  # nearby prose for heuristic disambiguation


def _extract_math_occurrences(text: str) -> list[MathOccurrence]:
    """Extract inline + display math from a QMD source (best-effort)."""
    lines = text.splitlines()
    walker = LineWalker(text)
    occ: list[MathOccurrence] = []

    in_display = False
    display_buf: list[str] = []
    display_start_line = 0

    for line, state, line_num in walker:
        # Skip YAML, code fences, style/script blocks, and HTML comments.
        if state.in_yaml or state.in_code_fence or state.in_html_style_block or state.in_html_comment:
            continue

        dd = _display_delim_count(line)
        if dd:
            # Delimiter line is considered part of math context; toggle display mode.
            if dd >= 2:
                # Single-line $$ ... $$ (rare in corpus but supported)
                inner = re.sub(r"^\s*\$\$", "", line)
                inner = re.sub(r"\$\$\s*(?:\{[^}]*\}\s*)?$", "", inner)
                ctx = _context_window(lines, line_num)
                occ.append(MathOccurrence(line_num=line_num, raw_math=inner, context=ctx))
            else:
                if not in_display:
                    in_display = True
                    display_buf = []
                    display_start_line = line_num
                else:
                    # closing
                    in_display = False
                    ctx = _context_window(lines, display_start_line)
                    occ.append(
                        MathOccurrence(
                            line_num=display_start_line,
                            raw_math="\n".join(display_buf),
                            context=ctx,
                        )
                    )
                    display_buf = []
                    display_start_line = 0
            continue

        if in_display:
            display_buf.append(line)
            continue

        # Inline math spans on the line.
        for m in _INLINE_MATH_RE.finditer(line):
            raw = m.group(0)
            inner = raw[1:-1]  # strip single $
            ctx = _context_window(lines, line_num)
            occ.append(MathOccurrence(line_num=line_num, raw_math=inner, context=ctx))

    # If a display block is unterminated, ignore it (structural issue is handled elsewhere).
    return occ


def _context_window(lines: list[str], line_num_1idx: int, radius: int = 1) -> str:
    i = line_num_1idx - 1
    start = max(0, i - radius)
    end = min(len(lines), i + radius + 1)
    return "\n".join(lines[start:end])


def _canon_math(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\\text{\1}", s)
    return s


def _has_defined_symbol(defs: set[str], sym: str) -> bool:
    return _canonical_symbol(sym) in defs


# Fast detectors for the conventions called out explicitly in the Notations chapter.
_HAS_BW_RE = re.compile(r"\\text\{BW\}")
_HAS_BW_BARE_RE = re.compile(r"(?<![A-Za-z\\])BW(?![A-Za-z])")
_HAS_DVOL_RE = re.compile(r"D_\{\\text\{vol\}\}")
_HAS_RPEAK_RE = re.compile(r"R_\{\\text\{peak\}\}")
_HAS_LLAT_RE = re.compile(r"L_\{\\text\{lat\}\}")
_HAS_LOSS_RE = re.compile(r"\\mathcal\{L\}")

_HAS_GREEK = {
    "eta": re.compile(r"\\eta\b"),
    "lambda": re.compile(r"\\lambda\b"),
    "alpha": re.compile(r"\\alpha\b"),
    "beta": re.compile(r"\\beta\b"),
}

# Single-letter var presence in math (avoid matching LaTeX commands like \Beta).
_HAS_SINGLE = {
    "B": re.compile(r"(?<![A-Za-z\\])B(?![A-Za-z])"),
    "D": re.compile(r"(?<![A-Za-z\\])D(?![A-Za-z])"),
    "P": re.compile(r"(?<![A-Za-z\\])P(?![A-Za-z])"),
    "L": re.compile(r"(?<![A-Za-z\\])L(?![A-Za-z])"),
    "d": re.compile(r"(?<![A-Za-z\\])d(?![A-Za-z])"),
    "N": re.compile(r"(?<![A-Za-z\\])N(?![A-Za-z])"),
}


def _ctx_has_any(ctx: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(p.search(ctx) for p in patterns)


_CTX_BANDWIDTH = [
    re.compile(r"\bbandwidth\b", re.IGNORECASE),
    re.compile(r"\bbytes/s\b", re.IGNORECASE),
    re.compile(r"\bGB/s\b", re.IGNORECASE),
    re.compile(r"\bTB/s\b", re.IGNORECASE),
    re.compile(r"\bGbps\b", re.IGNORECASE),
    re.compile(r"\blink\b", re.IGNORECASE),
    re.compile(r"\bthroughput\b", re.IGNORECASE),
    re.compile(r"\bnetwork\b", re.IGNORECASE),
]

_CTX_BATCH = [
    re.compile(r"\bbatch\b", re.IGNORECASE),
    re.compile(r"\bbatch\s+size\b", re.IGNORECASE),
    re.compile(r"\breserve\b", re.IGNORECASE),
]

_CTX_DATA_VOLUME = [
    re.compile(r"\bbytes?\b", re.IGNORECASE),
    re.compile(r"\btraffic\b", re.IGNORECASE),
    re.compile(r"\bmemory\b", re.IGNORECASE),
    re.compile(r"\bI/O\b", re.IGNORECASE),
    re.compile(r"\bmoved\b", re.IGNORECASE),
]

_CTX_PEAK_RATE = [
    re.compile(r"\bpeak\b", re.IGNORECASE),
    re.compile(r"\bFLOP/s\b", re.IGNORECASE),
    re.compile(r"\bTFLOP", re.IGNORECASE),
    re.compile(r"\broofline\b", re.IGNORECASE),
    re.compile(r"\brate\b", re.IGNORECASE),
    re.compile(r"\bthroughput\b", re.IGNORECASE),
]

_CTX_LATENCY = [
    re.compile(r"\blatency\b", re.IGNORECASE),
    re.compile(r"\bRTT\b", re.IGNORECASE),
    re.compile(r"\bms\b", re.IGNORECASE),
    re.compile(r"\bseconds?\b", re.IGNORECASE),
]


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    issues: list[Issue] = []
    counter = start_counter

    # Skip the notation definition sources themselves to avoid self-referential noise.
    posix = file_path.as_posix()
    if posix.endswith("/frontmatter/notation.qmd") or posix.endswith("/frontmatter/_notation_body.qmd") or posix.endswith("/frontmatter/_notation_distributed.qmd"):
        return issues, counter

    # Resolve repo root from file path (scan.py passes absolute file paths).
    repo_root = Path(__file__).resolve().parents[4]
    defs = _load_definition_registry(repo_root)

    occurrences = _extract_math_occurrences(text)
    if not occurrences:
        return issues, counter

    lines = text.splitlines()
    for occ in occurrences:
        m = _canon_math(occ.raw_math)
        ctx = occ.context
        line_raw = lines[occ.line_num - 1] if 1 <= occ.line_num <= len(lines) else ""

        # Common non-notation math uses: taxonomy/set intersection labels (e.g. D \cap M).
        # These are not dataset/data-volume variables and should not be fed into collision logic.
        if "\\cap" in m:
            continue

        # Enforce the book's explicit multi-letter convention for bandwidth: use \text{BW}.
        if _HAS_BW_BARE_RE.search(m) and not _HAS_BW_RE.search(m):
            issues.append(
                Issue(
                    id=make_issue_id(scope, CATEGORY, counter),
                    category=CATEGORY,
                    rule=RULE,
                    rule_text=RULE_TEXT,
                    file=str(file_path),
                    line=occ.line_num,
                    before=line_raw,
                    auto_fixable=False,
                    needs_subagent=True,
                    status=STATUS_DEFERRED,
                    reason="Bandwidth is recommended as \\text{BW} (not bare BW). Review for consistency.",
                )
            )
            counter += 1

        # Hard convention: bandwidth is \text{BW}, never bare B.
        if _HAS_SINGLE["B"].search(m) and not _HAS_BW_RE.search(m):
            # Avoid flagging lines that are explicitly *explaining* the convention
            # (e.g. "we reserve B for batch size and use BW for bandwidth").
            if _ctx_has_any(ctx, _CTX_BANDWIDTH) and not _ctx_has_any(ctx, _CTX_BATCH):
                # Prefer to only flag B-as-bandwidth when the math itself looks bandwidth-ish.
                if "\\frac" not in m and not _ctx_has_any(ctx, [re.compile(r"/s\b", re.IGNORECASE)]):
                    pass
                else:
                    issues.append(
                        Issue(
                            id=make_issue_id(scope, CATEGORY, counter),
                            category=CATEGORY,
                            rule=RULE,
                            rule_text=RULE_TEXT,
                            file=str(file_path),
                            line=occ.line_num,
                            before=line_raw,
                            auto_fixable=False,
                            needs_subagent=True,
                            reason="Possible bandwidth written as bare B; convention is \\text{BW}.",
                        )
                    )
                    counter += 1

        # Hard convention: data volume (bytes moved) is D_{text{vol}}, not D.
        if _HAS_SINGLE["D"].search(m) and not _HAS_DVOL_RE.search(m):
            # Heuristic: treat as bytes-moved context if BW / bytes / traffic appears nearby.
            if (
                ("\\frac" in m or "/" in m)
                and _ctx_has_any(ctx, _CTX_DATA_VOLUME)
                and (_HAS_BW_RE.search(m) or _HAS_BW_BARE_RE.search(m) or _ctx_has_any(ctx, _CTX_BANDWIDTH))
            ):
                issues.append(
                    Issue(
                        id=make_issue_id(scope, CATEGORY, counter),
                        category=CATEGORY,
                        rule=RULE,
                        rule_text=RULE_TEXT,
                        file=str(file_path),
                        line=occ.line_num,
                        before=line_raw,
                        auto_fixable=False,
                        needs_subagent=True,
                        reason="Possible bytes-moved written as bare D; convention is D_{\\text{vol}}.",
                    )
                )
                counter += 1

        # Hard convention: peak compute rate is R_{text{peak}}, not P.
        if _HAS_SINGLE["P"].search(m) and not _HAS_RPEAK_RE.search(m):
            if _ctx_has_any(ctx, _CTX_PEAK_RATE):
                issues.append(
                    Issue(
                        id=make_issue_id(scope, CATEGORY, counter),
                        category=CATEGORY,
                        rule=RULE,
                        rule_text=RULE_TEXT,
                        file=str(file_path),
                        line=occ.line_num,
                        before=line_raw,
                        auto_fixable=False,
                        needs_subagent=True,
                        reason="Possible peak performance written as bare P; convention is R_{\\text{peak}}.",
                    )
                )
                counter += 1

        # Hard convention: latency is L_{text{lat}}, not L; loss is \mathcal{L}.
        if _HAS_SINGLE["L"].search(m) and not _HAS_LLAT_RE.search(m) and not _HAS_LOSS_RE.search(m):
            if _ctx_has_any(ctx, _CTX_LATENCY):
                issues.append(
                    Issue(
                        id=make_issue_id(scope, CATEGORY, counter),
                        category=CATEGORY,
                        rule=RULE,
                        rule_text=RULE_TEXT,
                        file=str(file_path),
                        line=occ.line_num,
                        before=line_raw,
                        auto_fixable=False,
                        needs_subagent=True,
                        reason="Possible latency written as bare L; convention is L_{\\text{lat}} (loss is \\mathcal{L}).",
                    )
                )
                counter += 1

        # Soft collisions: context-dependent symbols flagged as DEFERRED (report-only).
        # These do NOT fail CI by default, but show up in the ledger for review.
        if _HAS_GREEK["eta"].search(m):
            issues.append(
                Issue(
                    id=make_issue_id(scope, CATEGORY, counter),
                    category=CATEGORY,
                    rule=RULE,
                    rule_text=RULE_TEXT,
                    file=str(file_path),
                    line=occ.line_num,
                    before=line_raw,
                    status=STATUS_DEFERRED,
                    needs_subagent=True,
                    reason="\\eta is context-dependent (learning rate vs efficiency). Review for collision.",
                )
            )
            counter += 1
        if _HAS_GREEK["lambda"].search(m):
            issues.append(
                Issue(
                    id=make_issue_id(scope, CATEGORY, counter),
                    category=CATEGORY,
                    rule=RULE,
                    rule_text=RULE_TEXT,
                    file=str(file_path),
                    line=occ.line_num,
                    before=line_raw,
                    status=STATUS_DEFERRED,
                    needs_subagent=True,
                    reason="\\lambda is context-dependent (sensitivity vs failure rate). Review for collision.",
                )
            )
            counter += 1
        if _HAS_GREEK["alpha"].search(m):
            issues.append(
                Issue(
                    id=make_issue_id(scope, CATEGORY, counter),
                    category=CATEGORY,
                    rule=RULE,
                    rule_text=RULE_TEXT,
                    file=str(file_path),
                    line=occ.line_num,
                    before=line_raw,
                    status=STATUS_DEFERRED,
                    needs_subagent=True,
                    reason="\\alpha is context-dependent (network latency vs learning rate). Review for collision.",
                )
            )
            counter += 1
        if _HAS_SINGLE["d"].search(m):
            issues.append(
                Issue(
                    id=make_issue_id(scope, CATEGORY, counter),
                    category=CATEGORY,
                    rule=RULE,
                    rule_text=RULE_TEXT,
                    file=str(file_path),
                    line=occ.line_num,
                    before=line_raw,
                    status=STATUS_DEFERRED,
                    needs_subagent=True,
                    reason="d is context-dependent (hidden dimension vs data parallelism). Review for collision.",
                )
            )
            counter += 1

    # Also: report frequent undefined symbols as DEFERRED candidates (optional, low noise threshold).
    # We only look for exact canonical matches of the notation-defined set, so this stays conservative.
    # (No issue emitted here if we can recognize at least one defined symbol in the file.)
    if defs:
        pass

    return issues, counter

