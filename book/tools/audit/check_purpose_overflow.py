#!/usr/bin/env python3
"""Purpose-overflow check: every chapter's Purpose must fit on its opener page.

A teaching chapter opens with a cover image + the Purpose section (italic hook +
exactly one paragraph), then a forced \\newpage to the Learning Objectives. The
Purpose must NOT spill past the opener page (a 1-2 line overflow leaves the next
page nearly empty, which reads as broken in print).

This check renders-agnostically locates, in the built PDF, the page holding the
Purpose hook (start) and the page holding the Purpose paragraph's last words
(end). If end_page > start_page, the Purpose overflowed.

Usage:
  python3 book/tools/audit/check_purpose_overflow.py <pdf> --vol vol1
  (reads chapter Purpose text from book/quarto/contents/<vol>/*/*.qmd)
Exit 1 if any chapter overflows.
"""
import sys, re, glob, os, subprocess, argparse

def page_texts(pdf):
    # No -layout: -layout interleaves margin-note/running-header labels (e.g.
    # "Part I", "Facility") between paragraph words, which breaks multi-word
    # anchor matching. Plain reading-order keeps the Purpose paragraph contiguous.
    out = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True).stdout
    return out.split("\f")  # one entry per page (1-indexed via idx+1)

def _norm(s):
    s = re.sub(r'`\{python\}[^`]*`', '', s)        # drop inline-python (renders to a value)
    s = re.sub(r'[‘’“”`]', "'", s)  # curly quotes/backtick -> straight
    s = re.sub(r'[\\*_$\[\]{}#@|]', '', s)
    s = re.sub(r'[—–-]', ' ', s)                   # em/en/hyphen -> space
    return re.sub(r'\s+', ' ', s).strip()

def purpose_bits(qmd):
    t = open(qmd, encoding="utf-8").read()
    m = re.search(r'## Purpose.*?\n(.*?)(?:::: \{\.content-visible|::: \{\.callout-learning|\n## )', t, re.S)
    if not m: return None
    paras = [p.strip() for p in m.group(1).split("\n\n") if p.strip()]
    hook = next((p for p in paras if p.startswith("_") and p.rstrip().endswith("_")), None)
    prose = [p for p in paras if not p.startswith(("\\", "_", ":"))]
    if not hook or not prose: return None
    return _norm(hook).split(), _norm(prose[-1]).split()

def find_page(pages, words, head=True):
    """Find the page containing an anchor built from `words`; try decreasing
    lengths so a stray apostrophe/value in one token can't break the match."""
    def squash(s):
        return re.sub(r'\s+', ' ', re.sub(r"[‘’“”`—–-]", lambda m: "'" if m.group()[0] in "‘’“”`" else ' ', s)).strip().lower()
    sp = [squash(pg) for pg in pages]
    for n in (7, 6, 5, 4):
        anchor = " ".join(words[:n] if head else words[-n:])
        a = squash(anchor)
        if not a: continue
        for i, pg in enumerate(sp):
            if a in pg: return i + 1
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf"); ap.add_argument("--vol", required=True)
    a = ap.parse_args()
    pages = page_texts(a.pdf)
    bad = []
    chapters = sorted(glob.glob(f"book/quarto/contents/{a.vol}/*/*.qmd"))
    for qmd in chapters:
        if os.path.basename(qmd).startswith(("appendix_", "_")): continue
        bits = purpose_bits(qmd)
        if not bits: continue
        hook_words, para_words = bits
        sp = find_page(pages, hook_words, head=True)
        ep = find_page(pages, para_words, head=False)
        name = os.path.basename(qmd)
        if sp is None or ep is None:
            print(f"  ?? {name}: could not locate (hook={sp} end={ep})")
            continue
        status = "OVERFLOW" if ep > sp else "ok"
        if ep > sp:
            bad.append((name, sp, ep))
            print(f"  ✗ {name}: Purpose starts p{sp}, ends p{ep}  ({ep-sp} page overflow)")
        else:
            print(f"  ✓ {name}: Purpose fits on p{sp}")
    print(f"\n{'FAIL' if bad else 'PASS'}: {len(bad)} Purpose overflow(s) in {a.vol}")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
