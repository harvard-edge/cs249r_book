#!/usr/bin/env python3
"""
Volume-wide body-prose BOLD occurrence map -> AP2 candidates.

Phase 2 helper for the emphasis audit. Runs emphasis_extract.py over every chapter
of a volume IN CANONICAL ORDER (derived from the PDF config YAML), collects every
body-prose bold span, and reports any term bolded in >=2 distinct chapters -- the
only way an AP2 duplicate first-definition can occur. Earlier chapter = owner.

Usage:  python3 book/tools/audit/emphasis_ap2map.py vol1
        python3 book/tools/audit/emphasis_ap2map.py vol2 --watch inference,arithmetic\\ intensity

READ-ONLY. Flags candidates; ruling (P1 owner vs P3-ext/P4 vs real AP2) is a human read.
"""
import sys, re, json, subprocess, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))   # repo root
VOL  = sys.argv[1] if len(sys.argv) > 1 else 'vol1'
WATCH = []
if '--watch' in sys.argv:
    WATCH = [w.strip().lower() for w in sys.argv[sys.argv.index('--watch') + 1].split(',')]

def chapter_order(vol):
    """Canonical chapter order from the volume's PDF config (handles commented lines)."""
    cfg = os.path.join(ROOT, 'book', 'quarto', 'config', f'_quarto-pdf-{vol}.yml')
    pat = re.compile(rf'contents/{vol}/([a-z0-9_]+)/\1\.qmd')   # dir==stem -> a chapter (skips parts/, index)
    order, seen = [], set()
    for line in open(cfg, encoding='utf-8'):
        m = pat.search(line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1)); order.append(m.group(1))
    return order

def norm(span):
    t = span.strip().strip('*').strip()
    t = re.sub(r'\s+', ' ', t).lower()
    t = re.sub(r'^(the|a|an) ', '', t)
    return t.strip(' .,:;()')

order = chapter_order(VOL)
if not order:
    sys.exit(f"No chapters found for {VOL} in PDF config.")

occ = defaultdict(list)   # norm_term -> [(chap_idx, chap, line)]
extractor = os.path.join(HERE, 'emphasis_extract.py')
for ci, ch in enumerate(order):
    path = os.path.join(ROOT, 'book', 'quarto', 'contents', VOL, ch, f'{ch}.qmd')
    out = subprocess.run(['python3', extractor, path], capture_output=True, text=True).stdout
    for ln in out.splitlines():
        s = json.loads(ln)
        if s['type'] == 'bold':
            occ[norm(s['span'])].append((ci, ch, s['line']))

multi = {n: v for n, v in occ.items() if len({x[0] for x in v}) >= 2}
print(f"{VOL}: {len(order)} chapters, {len(occ)} distinct body-bold terms, "
      f"{len(multi)} bolded in >=2 chapters (AP2 candidates)\n")
for n in sorted(multi, key=lambda k: (multi[k][0][0], k)):
    rows = sorted(multi[n], key=lambda x: (x[0], x[2]))
    print(f"• {rows[0][2]:>6} {rows[0][1]:<22} {n!r}")
    for _, ch, ln in rows[1:]:
        print(f"  {ln:>6} {ch:<22} (later -> check vs owner)")

if WATCH:
    print("\n=== WATCHLIST ===")
    for w in WATCH:
        hits = occ.get(w, [])
        chs = ', '.join(f"{c}:L{l}" for _, c, l in hits)
        flag = "  <-- MULTI-CHAPTER" if len({h[0] for h in hits}) >= 2 else ""
        print(f"  {w:28} {len(hits)}x  {chs or '(not body-bolded)'}{flag}")
