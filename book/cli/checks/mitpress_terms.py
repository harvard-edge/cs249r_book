#!/usr/bin/env python3
"""MIT Press canonical spelling dictionary (§10.7) — detection + fix.

The MIT Press editorial standard §10.7 fixes the canonical spelling of a long
list of terms (Webster's 11th first spelling). This module enforces the
**unambiguous** subset — terms with a single correct form regardless of
grammatical role — plus the proper-noun **capitalizations**. It is imported by
the `mitpress-spelling-dict` validator scope and the `mitpress-terms` format
target, so the check and the auto-fixer share one source of truth.

Deliberately EXCLUDED (context-sensitive — a naive rule false-positives):
  compute-bound / "is compute bound", memory-bound, open-source / "is open
  source", real-time (adj) / real time (n), round-trip / round trip,
  time-series, break-even, data-center (adj) / data center (n). Only the
  one-word error "datacenter" is flagged; the n/adj choice is left to the
  author.

Every hit carries an exact `replacement`, so output is LLM-actionable
("tradeoff → trade-off"). Matching happens on a MASKED copy of each line
(inline code, math, link targets, and Quarto attributes blanked to spaces of
equal length), so positions map 1:1 back onto the original and verbatim code
identifiers like `` `numpy` `` are never rewritten.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

CODE_FENCE_RE = re.compile(r"^\s*```")

# Glossary entries follow their own lowercase-key convention (§10.14 +
# glossary.md), so the §10.7 spelling dictionary does not apply there.
_SKIP_PATH_PARTS = ("/glossary/",)


def should_skip_file(path) -> bool:
    return any(part in str(path).replace("\\", "/") for part in _SKIP_PATH_PARTS)

# (regex, replacement-template) — spelling / hyphenation fixes. `\1` carries an
# optional plural 's' / inflection where relevant. Only the WRONG forms match.
SPELLING_TERMS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\btradeoff(s?)\b"), r"trade-off\1"),
    (re.compile(r"\bdata set(s?)\b"), r"dataset\1"),
    (re.compile(r"\bnon-zero\b"), "nonzero"),
    (re.compile(r"\bspeed-up(s?)\b"), r"speedup\1"),
    (re.compile(r"\bdatacenter(s?)\b"), r"data center\1"),
    (re.compile(r"\bpre-train(ed|ing)\b"), r"pretrain\1"),
    (re.compile(r"\bpre-process(ed|ing)?\b"), r"preprocess\1"),
    (re.compile(r"\bback[- ]propagation\b"), "backpropagation"),
    (re.compile(r"\bscatter plot(s?)\b"), r"scatterplot\1"),
    (re.compile(r"\bWiFi\b|\bwifi\b"), "Wi-Fi"),
    (re.compile(r"\bFaceID\b"), "Face ID"),
    (re.compile(r"\bGoogleNet\b|\bGooglenet\b"), "GoogLeNet"),
    (re.compile(r"\bGPU Direct\b|\bgpudirect\b"), "GPUDirect"),
]

# Proper nouns whose canonical capitalization is fixed: flag any case variant
# that is not exactly the canonical form.
CASE_TERMS = ["PyTorch", "TensorFlow", "NumPy", "TinyML", "MLflow", "MATLAB"]
_CASE_PATTERNS = [(re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE), t) for t in CASE_TERMS]


@dataclass(frozen=True)
class Hit:
    line: int
    match: str
    replacement: str
    context: str


def _mask(line: str) -> str:
    """Blank (with equal-length spaces) spans where a term is verbatim."""
    def blank(m):
        return " " * len(m.group())
    line = re.sub(r"`[^`]*`", blank, line)            # inline code
    line = re.sub(r"\$\$?[^$]*\$\$?", blank, line)    # math
    line = re.sub(r"\]\([^)]*\)", blank, line)        # link/image targets
    line = re.sub(r"\{[^}]*\}", blank, line)          # Quarto attributes
    line = re.sub(r"@[\w:.\-/]+", blank, line)        # cross-refs @sec-/@fig-/@tbl-…
    line = re.sub(r"\[\^[^\]]*\]", blank, line)       # footnote refs [^fn-…]
    line = re.sub(r"\bfn-[\w-]+", blank, line)        # bare footnote ids (def lines)
    return line


def _spans(masked: str) -> List[Tuple[int, int, str]]:
    """Return deduped (start, end, replacement) edits for one masked line.

    First-match-wins on overlapping spans, so a term that is both a spelling
    fix and a case fix (e.g. a hypothetical overlap) is only edited once.
    """
    raw: List[Tuple[int, int, str]] = []
    for pat, repl in SPELLING_TERMS:
        for m in pat.finditer(masked):
            fix = m.expand(repl) if "\\" in repl else repl
            raw.append((m.start(), m.end(), fix))
    for pat, canon in _CASE_PATTERNS:
        for m in pat.finditer(masked):
            if m.group() != canon:
                raw.append((m.start(), m.end(), canon))
    raw.sort()
    out: List[Tuple[int, int, str]] = []
    last_end = -1
    for s, e, r in raw:
        if s >= last_end:
            out.append((s, e, r))
            last_end = e
    return out


def _iter_text_lines(lines: List[str]):
    """Yield (idx, line) for lines that are prose (skip code fences, YAML, #|)."""
    in_code = False
    in_yaml = False
    for idx, line in enumerate(lines):
        s = line.strip()
        if idx == 0 and s == "---":
            in_yaml = True
            continue
        if in_yaml:
            if s == "---":
                in_yaml = False
            continue
        if CODE_FENCE_RE.match(line):
            in_code = not in_code
            continue
        if in_code or s.startswith("#|"):
            continue
        yield idx, line


def find_in_text(text: str) -> List[Hit]:
    lines = text.splitlines()
    hits: List[Hit] = []
    for idx, line in _iter_text_lines(lines):
        masked = _mask(line)
        for s, e, repl in _spans(masked):
            hits.append(Hit(idx + 1, line[s:e], repl,
                            line[max(0, s - 20): e + 12].strip()))
    return hits


def fix_text(text: str) -> Tuple[str, int]:
    lines = text.splitlines(keepends=True)
    raw = [ln.rstrip("\n") for ln in lines]
    eols = [ln[len(r):] for ln, r in zip(lines, raw)]
    changed = 0
    for idx, line in _iter_text_lines(raw):
        edits = _spans(_mask(line))
        if not edits:
            continue
        new = line
        for s, e, r in reversed(edits):  # right-to-left preserves indices
            new = new[:s] + r + new[e:]
        if new != line:
            raw[idx] = new
            changed += 1
    if not changed:
        return text, 0
    return "".join(r + e for r, e in zip(raw, eols)), changed


def audit(paths) -> List[tuple]:
    out: List[tuple] = []
    for rawp in paths:
        p = Path(rawp)
        files = p.rglob("*.qmd") if p.is_dir() else ([p] if p.suffix == ".qmd" else [])
        for f in files:
            if should_skip_file(f):
                continue
            for h in find_in_text(f.read_text(encoding="utf-8", errors="replace")):
                out.append((str(f), h.line, h.match, h.replacement))
    return out
