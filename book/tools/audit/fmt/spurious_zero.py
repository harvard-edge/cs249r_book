"""Shared detection of spurious trailing ``.0`` in narrative numbers.

Used by ``audit_prose.py`` (substituted QMD prose) and ``audit_html.py``
(rendered HTML). Filters encode known false positives: product version strings
(PCIe 4.0), unit specs (1.0 KFLOPs), LaTeX walkthrough literals, etc.
"""

from __future__ import annotations

import re

# Match whole numbers rendered with a redundant fractional part, e.g. 153.0, 739,726.0
SPURIOUS_ZERO = re.compile(r"\b[\d,]+\.0\b(?!\d)")

# Each pattern matches a context snippet around a *.0 hit that should be ignored.
_FALSE_POSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"Software\s+[\d,]+\.0",
        r"in\s+20\d{2}\.0",
        r"year\s+20\d{2}\.0",
        r"(?:Industry|PyTorch|TensorFlow|NVLink|CUDA|SciPy|PCIe|InfiniBand|DOI|CO;|CXL)\s*[\d,]*\.0",
        r"CRISP-DM\s+[\d,]+\.0",
        r"\bVersion\s+[\d,]+\.0\b",
        r"\bFamily\s+[\d,]+\.0,\s*Level\s+\d+",
        r"PyTorch\s+[\d,]+\.0",
        r"CO;\d+\.0",
        r"rounds to [\d,]+\.0",
        r"(?:AID-|::AID-)[^\s]*\d+\.0",
        r"\d+\.0 in less optimized",
        r"\d+\.0\s*(?:μs|us|percent vs|percent test error)",
        r" billion FLOPs\b",
        r"(?:Baseline|perfect|\[math\]).*\d+\.0",
        r"[\[\(=]\s*[\d,]+\.0",
        r"1\.0\s*\(perfect\)",
        r"\[0\.5,\s*2\.0\]",
        r"\\(?:approx|times|exp|lbrack|rbrack|hat|mathbf|partial|mathcal|frac|begin|end|cdot|;|,|\(|\[)",
        r"[\d,]+\.0\s*times\s*10",
        r"\\text\{[^}]*\d+\.0",
        r"PUE of [\d,]+\.0",
        r"PUE values of [\d,]+\.0",
        r"values above [\d,]+\.0",
        r"ideal of [\d,]+\.0",
        r"Section\s*[\d.]+\.0",
        r"(?:OAuth|MPI|NCCL)\s+[\d,]+\.0",
        r"[\d,]+\.0\s*kJ/kg/K",
        r"[\d,]+\.0\s*L/kWh",
        r"[\d,–-]+[\d,]+\.0\s*V\b",
        r"gradient norm [\d,]+\.0",
        r"\$[\d,]+\.0\s*million",
        r"magnitude threshold.*[\d,]+\.0",
        r"[\d,]+\.0 per (?:device|hour)",
        r"typically [\d,]+\.0",
        r"Exactly [\d,]+\.0",
        r"flat [\d,]+\.0",
        r"below [\d,]+\.0",
        r"to [\d,]+\.0, meaning",
        r"PUE of [\d,.]+\s+to\s+[\d,]+\.0",
    )
)


def is_spurious_zero_false_positive(context: str) -> bool:
    """Return True when a ``*.0`` match in *context* is a known false positive."""
    if any(p.search(context) for p in _FALSE_POSITIVE_PATTERNS):
        return True
    # CSS unit leaks (rare in HTML)
    if re.search(r"\b1\.0\b", context) and ("em" in context or "px" in context):
        return True
    if re.search(r"\b2\.0\b", context) and ("em" in context or "px" in context):
        return True
    return False


def find_spurious_zeros(text: str, *, context_chars: int = 40) -> list[tuple[str, str]]:
    """Return ``(value, context)`` pairs for suspicious ``*.0`` in *text*."""
    hits: list[tuple[str, str]] = []
    for match in SPURIOUS_ZERO.finditer(text):
        value = match.group(0)
        start = max(0, match.start() - context_chars)
        end = min(len(text), match.end() + context_chars)
        context = text[start:end].replace("\n", " ").strip()
        if is_spurious_zero_false_positive(context):
            continue
        hits.append((value, context))
    return hits
