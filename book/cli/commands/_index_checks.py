"""Index check primitives for `binder check index` scopes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

INDEX_RE = re.compile(r"\\index\{([^{}]*(?:\{[^{}]*\}[^{}]*)?)\}")
SEEREF_RE = re.compile(r"^([^|]+)\|see(?:also)?\{([^}]+)\}$")

GENERIC_BARE = {
    "Algorithm", "Architecture", "Framework", "Implementation",
    "Memory", "Optimization", "Performance", "Pipeline", "Problem",
    "Scenario", "Conclusion", "Metric", "System", "Process", "Operation",
}

LOWERCASE_ALLOWLIST = {
    "bfloat16", "bitter lesson, The", "cuBLAS", "cuDNN", "cuSPARSE", "gRPC",
    "im2col", "k-Anonymity", "k-Center", "mmap", "nn.Module", "oneDNN",
    "p50 Latency", "p95 Latency", "torch.compile", "vLLM", "jax.grad",
    "tf.data", "tf.function", "autocast", "oneCCL",
}

PARENTHETICAL_ALLOWLIST = {"Precision (Metric)"}

_SKIP_PATH_PARTS = (
    "frontmatter/", "backmatter/", "/parts/", "/glossary/", "/appendix", "_shelved",
)


@dataclass(frozen=True)
class IndexIssue:
    file: str
    line: int
    code: str
    message: str
    severity: str = "error"


def _skip_file(path: Path, root: Path) -> bool:
    s = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    return any(part in s for part in _SKIP_PATH_PARTS)


def _iter_qmd_files(root: Path) -> list[Path]:
    contents = root / "quarto" / "contents"
    if not contents.is_dir():
        return []
    return sorted(
        p for p in contents.rglob("*.qmd")
        if not _skip_file(p, root)
    )


def check_anti_patterns(root: Path) -> list[IndexIssue]:
    """Corpus-level \\index{} anti-patterns from index.md §9."""
    keys: set[str] = set()
    for f in _iter_qmd_files(root):
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in INDEX_RE.finditer(text):
            k = m.group(1)
            if "|" not in k:
                keys.add(k)

    heads: set[str] = set()
    for k in keys:
        h = k.split("!", 1)[0]
        if "@" in h:
            h = h.split("@", 1)[1]
        heads.add(h)

    issues: list[IndexIssue] = []

    def _add(code: str, samples: list, label: str) -> None:
        if not samples:
            return
        preview = "; ".join(str(s) for s in samples[:3])
        extra = f" (+{len(samples) - 3} more)" if len(samples) > 3 else ""
        issues.append(IndexIssue(
            file="(corpus)",
            line=0,
            code=code,
            message=f"{label}: {len(samples)} hit(s) — e.g. {preview}{extra}",
        ))

    _add("sub_subentries", [k for k in keys if k.count("!") >= 2], "sub-subentries (A!B!C)")
    _add(
        "author_year",
        [k for k in keys if re.search(r"![A-Z][a-zA-Z]+ [12][0-9]{3}\}?$", k)],
        "author-year subentries",
    )
    _add("et_al", [k for k in keys if re.search(r"et al", k, re.IGNORECASE)], "et al. subentries")
    _add("generic_bare", [k for k in keys if k in GENERIC_BARE], "generic-bare entries")
    _add("inline_python", [k for k in keys if "`{python}" in k], "inline-Python in keys")
    _add(
        "underscore_in_key",
        [k for k in keys if "_" in k and "$" not in k],
        "underscore in keys",
    )
    _add("ampersand_unescaped", [k for k in keys if re.search(r"(?<!\\)&", k)], "unescaped &")
    _add(
        "article_leading",
        [k for k in keys if re.match(r"^(The|A|An) [A-Z]", k)],
        "article-leading mains",
    )
    plural_dups = [
        f"{h[:-1]} / {h}"
        for h in heads
        if h.endswith("s") and not h.endswith(("ss", "us", "is", "ous"))
        and len(h) > 3 and h[:-1] in heads
    ]
    _add("plural_duplicates", plural_dups, "plural-vs-singular duplicates")
    paren_heads = [
        h for h in heads
        if re.match(r".+\([A-Z][A-Z0-9]*\)$", h) and h not in PARENTHETICAL_ALLOWLIST
    ]
    _add("parenthetical_acronym_heads", paren_heads, "parenthetical-acronym headwords")
    lc_off = [h for h in heads if h and h[0].islower() and h not in LOWERCASE_ALLOWLIST]
    _add("lowercase_off_allowlist", lc_off, "lowercase mains off allowlist")

    return issues


def check_tag_placement(root: Path) -> list[IndexIssue]:
    """\\index{} inside bold/code/headings."""
    issues: list[IndexIssue] = []
    for f in _iter_qmd_files(root):
        rel = str(f.relative_to(root))
        text = f.read_text(encoding="utf-8", errors="replace")
        in_code_block = False
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block or "\\index{" not in line:
                continue
            if stripped.startswith("#") and not stripped.startswith("# │"):
                issues.append(IndexIssue(rel, i, "V4_heading", "\\index{} on heading line"))
                continue
            if stripped.startswith(":::"):
                continue
            for m in re.finditer(r"\*\*\\index\{", line):
                before = line[:m.start()]
                pairs = re.findall(r"\*\*[^*\n]+?\*\*", before)
                consumed_pairs = 0
                search_pos = 0
                for p in pairs:
                    idx = before.find(p, search_pos)
                    if idx >= 0:
                        consumed_pairs += 1
                        search_pos = idx + len(p)
                leftover = before[search_pos:]
                if leftover.count("**") % 2 == 0:
                    issues.append(IndexIssue(rel, i, "V1_inside_opening_bold", "\\index{} inside opening **"))
                    break
            for m in re.finditer(r"\\index\{", line):
                before = line[:m.start()]
                ticks = len(re.findall(r"(?<!`)`(?!`)", before))
                if ticks % 2 == 1:
                    issues.append(IndexIssue(rel, i, "V3_inside_code", "\\index{} inside code span"))
                    break
    return issues


def check_xref_resolves(root: Path) -> list[IndexIssue]:
    """Every |see / |seealso target resolves to a main entry."""
    main_heads: set[str] = set()
    see_refs: list[tuple[str, int, str, str]] = []

    for f in _iter_qmd_files(root):
        rel = str(f.relative_to(root))
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in INDEX_RE.finditer(text):
            k = m.group(1)
            sm = SEEREF_RE.match(k)
            if sm:
                line = text.count("\n", 0, m.start()) + 1
                see_refs.append((rel, line, sm.group(1).strip(), sm.group(2).strip()))
            else:
                h = k.split("!", 1)[0]
                if "@" in h:
                    h = h.split("@", 1)[1]
                main_heads.add(h)

    issues: list[IndexIssue] = []
    for rel, line, src, tgt in see_refs:
        if tgt not in main_heads:
            issues.append(IndexIssue(
                rel, line, "xref_unresolved",
                f"'{src}' -> '{tgt}' (target not found)",
            ))
    return issues
