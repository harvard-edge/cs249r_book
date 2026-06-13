#!/usr/bin/env python3
"""
Roll up per-chapter emphasis ledgers into a volume summary + fix-list.

Reads every .claude/_reviews/emphasis/<vol>/*.yml ledger (schema-tolerant: it finds
any dict carrying a 'recommendation' key, wherever the agent nested it) and prints
per-chapter counts, volume totals, and the full actionable fix-list (everything != keep).

Usage:  python3 book/tools/audit/emphasis_rollup.py vol1
"""
import sys, os, glob, re
from collections import Counter

try:
    import yaml
except ImportError:
    sys.exit("pip install pyyaml")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
VOL  = sys.argv[1] if len(sys.argv) > 1 else 'vol1'

def chapter_order(vol):
    cfg = os.path.join(ROOT, 'book', 'quarto', 'config', f'_quarto-pdf-{vol}.yml')
    pat = re.compile(rf'contents/{vol}/([a-z0-9_]+)/\1\.qmd')
    order, seen = [], set()
    for line in open(cfg, encoding='utf-8'):
        m = pat.search(line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1)); order.append(m.group(1))
    return order

def walk(node):
    if isinstance(node, dict):
        if 'recommendation' in node:
            yield node
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)

reviews = os.path.join(ROOT, '.claude', '_reviews', 'emphasis', VOL)
order = chapter_order(VOL) or [os.path.basename(p)[:-4] for p in sorted(glob.glob(f"{reviews}/*.yml"))]

rec_tot, actionable, p1_total, per_ch, missing = Counter(), [], 0, [], []
for ch in order:
    f = os.path.join(reviews, f"{ch}.yml")
    if not os.path.exists(f):
        missing.append(ch); continue
    data = yaml.safe_load(open(f))
    findings = list(walk(data))
    n_act = 0
    for x in findings:
        r = str(x.get('recommendation', '?')).lower().strip()
        rec_tot[r] += 1
        if r != 'keep':
            kind = x.get('kind') or x.get('type') or '?'
            actionable.append((ch, x.get('line', '?'), kind, str(x.get('span', ''))[:42], r))
            n_act += 1
    p1 = data.get('bold_P1_candidates') or []
    if isinstance(p1, dict): p1 = list(p1.values())
    p1n = len(p1) if isinstance(p1, list) else 0
    p1_total += p1n
    per_ch.append((ch, len(findings), n_act, p1n))

print(f"=== {VOL} EMPHASIS ROLL-UP  (chapter / spans / actionable / P1-candidates) ===")
for ch, tot, act, p1 in per_ch:
    print(f"  {'!' if act else '·'} {ch:22} {tot:4}   act={act:<3} P1={p1}")
if missing:
    print(f"  MISSING LEDGERS: {', '.join(missing)}")
print(f"\nTOTALS: {sum(t for _,t,_,_ in per_ch)} spans · {p1_total} P1-candidates · "
      f"recs={dict(rec_tot)} · actionable={len(actionable)}")
print("\nFIX-LIST (everything != keep):")
for ch, ln, kind, span, r in actionable:
    print(f"  [{r:11}] {ch:18} L{ln:<6} {kind:26} {span}")
