#!/usr/bin/env python3
"""
book-index-anti-patterns hook.

Fails (exit 1) if the corpus contains any \index{} anti-pattern from
.claude/rules/index.md §9. Run from repo root.

Anti-patterns checked:
  - Sub-subentries (A!B!C)
  - Author-year subentries
  - et al. subentries
  - Generic-bare entries (Algorithm, Architecture, Memory, ...)
  - Inline-Python in keys (`{python}...`)
  - Underscore in keys (LuaLaTeX subscript bug)
  - Unescaped & in keys
  - Article-leading mains (The X)
  - Plural-vs-singular duplicates
  - Parenthetical-acronym headwords (off allowlist)
  - Lowercase mains off allowlist
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path.cwd()
CONTENT = ROOT / "book" / "quarto" / "contents"
INDEX_RE = re.compile(r"\\index\{([^{}]*(?:\{[^{}]*\}[^{}]*)?)\}")

GENERIC_BARE = {"Algorithm", "Architecture", "Framework", "Implementation",
    "Memory", "Optimization", "Performance", "Pipeline", "Problem",
    "Scenario", "Conclusion", "Metric", "System", "Process", "Operation"}

LOWERCASE_ALLOWLIST = {"bfloat16", "bitter lesson, The", "cuBLAS", "cuDNN",
    "cuSPARSE", "gRPC", "im2col", "k-Anonymity", "k-Center", "mmap",
    "nn.Module", "oneDNN", "p50 Latency", "p95 Latency", "torch.compile",
    "vLLM", "jax.grad", "tf.data", "tf.function", "autocast", "oneCCL"}

PARENTHETICAL_ALLOWLIST = {"Precision (Metric)"}


def main():
    keys = set()
    for f in sorted(CONTENT.rglob("*.qmd")):
        s = str(f)
        if any(x in s for x in ("frontmatter/", "backmatter/", "/parts/",
                                "/glossary/", "/appendix", "_shelved")):
            continue
        text = f.read_text(errors="replace")
        for m in INDEX_RE.finditer(text):
            k = m.group(1)
            if "|" not in k:
                keys.add(k)

    heads = set()
    for k in keys:
        h = k.split("!", 1)[0]
        if "@" in h:
            h = h.split("@", 1)[1]
        heads.add(h)

    failures = []

    sub_sub = [k for k in keys if k.count("!") >= 2]
    if sub_sub:
        failures.append(("sub_subentries", sub_sub[:5]))

    author_year = [k for k in keys if re.search(r"![A-Z][a-zA-Z]+ [12][0-9]{3}\}?$", k)]
    if author_year:
        failures.append(("author_year", author_year[:5]))

    et_al = [k for k in keys if re.search(r"et al", k, re.IGNORECASE)]
    if et_al:
        failures.append(("et_al", et_al[:5]))

    gb = [k for k in keys if k in GENERIC_BARE]
    if gb:
        failures.append(("generic_bare", gb[:5]))

    inline_py = [k for k in keys if "`{python}" in k]
    if inline_py:
        failures.append(("inline_python", inline_py[:5]))

    underscore = [k for k in keys if "_" in k and "$" not in k]
    if underscore:
        failures.append(("underscore_in_key", underscore[:5]))

    amp_unesc = [k for k in keys if re.search(r"(?<!\\)&", k)]
    if amp_unesc:
        failures.append(("ampersand_unescaped", amp_unesc[:5]))

    article_leading = [k for k in keys if re.match(r"^(The|A|An) [A-Z]", k)]
    if article_leading:
        failures.append(("article_leading", article_leading[:5]))

    plural_dups = [(h[:-1], h) for h in heads
                   if h.endswith("s") and not h.endswith(("ss", "us", "is", "ous"))
                   and len(h) > 3 and h[:-1] in heads]
    if plural_dups:
        failures.append(("plural_duplicates", plural_dups[:5]))

    paren_heads = [h for h in heads
                   if re.match(r".+\([A-Z][A-Z0-9]*\)$", h) and h not in PARENTHETICAL_ALLOWLIST]
    if paren_heads:
        failures.append(("parenthetical_acronym_heads", paren_heads[:5]))

    lc_off = [h for h in heads
              if h and h[0].islower() and h not in LOWERCASE_ALLOWLIST]
    if lc_off:
        failures.append(("lowercase_off_allowlist", lc_off[:5]))

    if failures:
        print("Index anti-pattern audit FAILED:")
        for name, samples in failures:
            print(f"  {name}: {len(samples)} sample(s):")
            for s in samples:
                print(f"    {s}")
        print("\nSee .claude/rules/index.md §9 for the full anti-pattern list.")
        return 1

    print(f"Index anti-pattern audit PASSED ({len(keys)} unique keys, {len(heads)} headwords)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
