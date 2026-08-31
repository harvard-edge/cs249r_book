"""Prose-integrity detectors: sentence starts, hand-typed attributions, italics.

Added 2026-08-14 after a tone-audit pass introduced nine banned section
meta-openers, two of which left a sentence starting with a lowercase word
("...energy budget. this section introduces..."). Nothing in the gate set
noticed. These three detectors cover failure classes that read as obviously
wrong to a human but pass every existing check.

Shared design: scan body prose only. Fenced code, inline code spans, math,
TikZ, index keys, YAML front matter, and table rows are skipped, because each
legitimately contains text that would otherwise look like a violation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, List

# ── shared prose masking ────────────────────────────────────────────────────

_INLINE_CODE = re.compile(r"`[^`]*`")
# Display math must be masked BEFORE inline math. `_MATH` alone matches the
# leading `$$` of a display block as an empty inline span, masking only the
# dollar signs and leaving the equation body exposed as "prose" -- so LaTeX
# subscripts (`U_{\text{platform}} ... \sum_{i}`) read as underscore italics.
# (Found 2026-08-14: ops_scale.qmd:3595 was a false positive from exactly this.)
_DISPLAY_MATH = re.compile(r"(?<!\\)\$\$.*?(?<!\\)\$\$", re.DOTALL)
_MATH = re.compile(r"(?<!\\)\$[^$\n]*(?<!\\)\$")
_INDEX = re.compile(r"\\index\{[^}]*\}")
_ATTR = re.compile(r'\b\w+(?:-\w+)*\s*=\s*"[^"]*"')
_URL = re.compile(r"https?://\S+|www\.\S+")


def _mask(line: str) -> str:
    """Blank out spans that are not body prose, preserving line length."""
    out = line
    for pat in (_INLINE_CODE, _DISPLAY_MATH, _MATH, _INDEX, _URL):
        out = pat.sub(lambda m: " " * len(m.group(0)), out)
    return out


def _is_skippable(stripped: str) -> bool:
    """Structural lines that are not running prose."""
    return (
        not stripped
        or stripped.startswith(("|", ">", "#", ":::", "::::", "---", "!["))
        or stripped.startswith(("- ", "* ", "+ "))
        or re.match(r"^\d+[.)]\s", stripped) is not None
        or stripped.startswith(("\\", "%", "<!--"))
    )


@dataclass
class Hit:
    line: int
    match: str
    context: str
    detail: str = ""


def _scan_prose(text: str) -> Iterator[tuple[int, str, str]]:
    """Yield (line_no, masked_line, raw_line) for body-prose lines only."""
    in_fence = False
    in_yaml = False
    in_display_math = False
    for i, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if i == 1 and stripped == "---":
            in_yaml = True
            continue
        if in_yaml:
            if stripped == "---":
                in_yaml = False
            continue
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        # A display-math block opened on its own line runs until the closing
        # `$$`; its body is LaTeX, not prose. An odd count of `$$` on one line
        # toggles the state (a balanced single-line `$$...$$` is handled by
        # `_DISPLAY_MATH` in `_mask` instead).
        if not in_fence and stripped.count("$$") % 2 == 1:
            in_display_math = not in_display_math
            continue
        if in_fence or in_display_math or _is_skippable(stripped):
            continue
        yield i, _mask(raw), raw


# ── A. sentence-start capitalization ────────────────────────────────────────

# Abbreviations that legitimately end in a period mid-sentence.
_ABBREV = (
    "e.g", "i.e", "cf", "vs", "etc", "al", "approx", "Fig", "fig", "Eq", "eq",
    "Sec", "sec", "Ch", "ch", "No", "no", "Dr", "Prof", "Mr", "Ms", "Mrs",
    "St", "Inc", "Ltd", "Co", "Jr", "Sr", "vol", "Vol", "pp", "p", "ed", "Eds",
)
_ABBREV_RE = re.compile(r"(?:^|[\s(\[])(?:" + "|".join(re.escape(a) for a in _ABBREV) + r")\.$")

# Terms that are legitimately lowercase at a sentence start
# (capitalization.md lowercase-main-entry allowlist).
_LOWERCASE_OK = {
    "nn", "torch", "jax", "tf", "autocast", "cublas", "cudnn", "cusparse",
    "onednn", "oneccl", "grpc", "vllm", "im2col", "mmap", "bfloat16",
    "k-anonymity", "k-center", "p50", "p95", "p99", "pj", "bitter",
    "coreboot", "iPhone", "eBPF", "gRPC", "cuDNN", "vLLM", "iOS",
}

# NOTE: no closing quote/paren allowed between the punctuation and the space.
# `"how fast can we learn?" to` ends a QUOTED question inside a larger
# sentence, not the sentence itself; same for `(...how many are correct?) and`.
# Exactly one or two spaces: a real sentence break. A long run of spaces is
# column padding in a prettified table, not prose.
_SENTENCE_BREAK = re.compile(r"[.!?](?:\*\*|\*|_)* {1,2}([a-z][\w-]*)")


def find_bad_sentence_starts(text: str) -> List[Hit]:
    hits: List[Hit] = []
    for line_no, masked, raw in _scan_prose(text):
        for m in _SENTENCE_BREAK.finditer(masked):
            word = m.group(1)
            if word.lower() in _LOWERCASE_OK:
                continue
            before = masked[: m.start() + 1]
            if _ABBREV_RE.search(before):
                continue
            # Multi-dot abbreviations: i.i.d., a.k.a., e.t.c.
            if re.search(r"(?:\b[A-Za-z]\.){2,}$", before):
                continue
            # An ellipsis is not a sentence end.
            if before.endswith("..") or before.endswith("\u2026."):
                continue
            hits.append(
                Hit(
                    line=line_no,
                    match=f"...{masked[max(0, m.start() - 24):m.end()].strip()}",
                    context=raw.strip()[:140],
                    detail=word,
                )
            )
    return hits


# ── B. hand-typed "et al." ──────────────────────────────────────────────────

_ET_AL = re.compile(r"\b([A-Z][A-Za-z\u00C0-\u017F'’-]+)\s+et\s+al\.")


# A suppressed-author cite renders as "(YEAR)" alone, so naming the author in
# prose beside it is the deliberate, correct pairing: "Han et al.'s pruning work
# [-@han2015deep]" renders as "Han et al.'s pruning work (2015)". That is not the
# citeproc-duplicate anti-pattern this detector targets, so exempt it.
# (Found 2026-08-14: model_compression.qmd:5792 and nn_architectures.qmd:2914
# were false positives from exactly this.)
_SUPPRESSED_CITE = re.compile(r"\[-@[\w:.-]+\]")


def find_manual_et_al(text: str) -> List[Hit]:
    hits: List[Hit] = []
    for line_no, masked, raw in _scan_prose(text):
        for m in _ET_AL.finditer(masked):
            if _SUPPRESSED_CITE.search(masked[m.end():m.end() + 60]):
                continue
            hits.append(
                Hit(
                    line=line_no,
                    match=m.group(0),
                    context=raw.strip()[:140],
                    detail=m.group(1),
                )
            )
    return hits


# ── C. underscore italics outside the Purpose hook ──────────────────────────

_UNDERSCORE_ITALIC = re.compile(r"(?<![\w\\{*])_([A-Za-z][^_\n]{2,}?)_(?![\w}*])")


def find_underscore_italics(text: str) -> List[Hit]:
    """`emphasis.md`: underscores are reserved for the Purpose hook question.

    The hook is a whole line wrapped in underscores and ending in `?`, so it is
    recognised structurally rather than by position.
    """
    hits: List[Hit] = []
    for line_no, masked, raw in _scan_prose(text):
        stripped = raw.strip()
        if stripped.startswith("_") and stripped.endswith(("_", "_?")) and "?" in stripped:
            continue  # Purpose hook question
        for m in _UNDERSCORE_ITALIC.finditer(masked):
            inner = m.group(1)
            if "_" in inner or inner.strip() != inner:
                continue  # snake_case identifier fragment
            hits.append(
                Hit(
                    line=line_no,
                    match=m.group(0)[:60],
                    context=raw.strip()[:140],
                    detail=inner[:40],
                )
            )
    return hits
