#!/usr/bin/env python3
"""Fairness-threshold lesson data, derived from the figure's own TikZ source.

The figure is an illustrative toy: sixteen drawn applicants, eight per subgroup,
four repayers (circles) and four defaulters (plus markers) in each. No LEGO cell
or MLSysIM value applies, and the LoanFairness credit-service metrics in
tbl-fairness-metrics-summary are deliberately NOT used here. Every value the
Explore view shows is a COUNT of drawn markers on one side of a boundary
(approved, misclassified, correct) or a ratio of those counts.

This script parses the marker positions and both candidate boundaries from
15_responsible_engr.qmd, maps them onto the rendered SVG with the transform
read from the verified artwork, checks every marker against its drawn path,
recomputes the per-step ledger, asserts the caption's endpoints (75 percent at
the right-hand line, 81.25 percent at the left-hand line; Subgroup A 3 -> 0
errors, Subgroup B 1 -> 3), sweeps every shared cutoff to confirm that 81.25
percent is the unique best shared accuracy, checks the Subgroup B cutoff that
equalizes TPR (the threshold-adjustment paragraph after the figure), and checks
that the constant embedded in the browser module matches.

Run from the repository root:
  python3 books/_extensions/mlsysbook-ext/figure-explore/original-v1-fairness-threshold.calc.py
"""
import argparse
import json
import re
from fractions import Fraction
from pathlib import Path

NAME = 'original-v1-fairness-threshold'
KEY = 'vol1:fig-fairness-threshold'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
QMD = ROOT / 'books/vol1/15_responsible_engr/15_responsible_engr.qmd'

p = argparse.ArgumentParser()
p.add_argument('--art-index', default='/tmp/figures-animation-art-current/index.json')
p.add_argument('--check', action='store_true', help='accepted for parity with sibling calcs; the embedded D is always checked')
args = p.parse_args()

# ---- LOAD: the figure's TikZ block, verbatim from the chapter ----
text = QMD.read_text()
start = text.index('{#fig-fairness-threshold')
tikz = text[start:text.index('\\end{tikzpicture}', start)]
lists = [[float(v) for v in m.split(',')] for m in re.findall(r'\\foreach \\x in\{([^}]+)\}', tikz)]
a_pos, a_neg, b_pos, b_neg = lists
assert a_pos == [0.05, 0.12, 0.19, 0.26] and a_neg == [0.38, 0.45, 0.52, 0.80], lists
assert b_pos == [0.08, 0.48, 0.56, 0.76] and b_neg == [0.69, 0.83, 0.90, 0.97], lists
bounds = [float(v) for v in re.findall(r'\\coordinate\(D[78]G\)at\(\$\(A1\)!([\d.]+)!\(A2\)\$\)', tikz)]
labels = re.findall(r'node\[above\](?:\(SE\))?\{([\d.]+)\\%\}', tikz)
# Left line (0.30) carries 81.25%, right line (0.63) carries 75% since commit ce4d703324.
assert bounds == [0.3, 0.63] and labels == ['81.25', '75'], (bounds, labels)
LEFT, RIGHT = bounds
LABEL = {LEFT: labels[0], RIGHT: labels[1]}

# Group 0 = Subgroup A, 1 = Subgroup B. Outcome 1 = repays (circle), 0 = defaults (plus).
markers = [{'g': g, 'pos': x, 'y': y} for g, pos, neg in [(0, a_pos, a_neg), (1, b_pos, b_neg)]
           for y, xs in [(1, pos), (0, neg)] for x in xs]
assert len(markers) == 16 and all(sum(m['g'] == g and m['y'] == y for m in markers) == 4 for g in (0, 1) for y in (0, 1))

# ---- LOAD: the verified artwork's transform and the sixteen drawn marker paths ----
art = json.load(open(args.art_index))[KEY]
svg = Path(art['asset']).read_text()
assert art['asset_sha1'] == '1c925a5d3062f7e01d90523c4d35e3b27660b87d', art['asset_sha1']
VIEWBOX = '0 0 292.214 87.545'
assert f'viewBox="{VIEWBOX}"' in svg
body = svg[svg.index('</defs>'):]
paths = re.findall(r'<path[^>]*>', body)
row = paths[0]
k, tx, ty = map(float, re.search(r'matrix\(([\d.]+), 0, 0, -[\d.]+, ([\d.]+), ([\d.]+)\)', row).groups())
u0, u1 = map(float, re.search(r'd="M ([-\d.]+) [-\d.]+ L ([\d.]+) ', row).groups())

def sx(pos):
    """Fractional TikZ position along a subgroup row -> verified SVG x."""
    return round(tx + k * (u0 + pos * (u1 - u0)), 4)

def centre(path):
    """Centre of one drawn marker in SVG coordinates (path bbox under its transform)."""
    a, ox, oy = map(float, re.search(r'matrix\(([\d.]+), 0, 0, -[\d.]+, ([-\d.]+), ([-\d.]+)\)', path).groups())
    nums = [float(v) for v in re.findall(r'-?[\d.]+', re.search(r' d="([^"]*)"', path).group(1))]
    xs, ys = nums[0::2], nums[1::2]
    return ox + a * (min(xs) + max(xs)) / 2, oy - a * (min(ys) + max(ys)) / 2

blue, red = 'rgb(0%, 0%, 69.999695%)', 'rgb(100%, 0%, 0%)'
drawn = [pth for pth in paths if blue in pth or red in pth]
assert len(drawn) == 20, len(drawn)  # sixteen applicants, then four legend glyphs
drawn = drawn[:16]
order = [(0, 1), (0, 0), (1, 1), (1, 0)]  # TikZ draw order: A circles, A pluses, B circles, B pluses
for i, pth in enumerate(drawn):
    g, y = order[i // 4]
    m = next(mm for mm in markers if mm['g'] == g and mm['y'] == y and 'x' not in mm)
    assert (blue if g == 0 else red) in pth and (('fill="rgb(100%' in pth) == (y == 1)), (i, pth[:120])
    cx, cy = centre(pth)
    m['x'], m['cy'] = round(cx, 4), round(cy, 4)
    assert abs(cx - sx(m['pos'])) < 0.05, (m, cx, sx(m['pos']))
row_y = [round(ty - k * float(v), 4) for v in re.findall(r'd="M [-\d.]+ ([-\d.]+) L', ''.join(paths[:2]))]
dashed = [pth for pth in paths if 'stroke-dasharray' in pth]
assert len(dashed) == 2
dash_x = [round(tx + k * float(re.search(r'd="M ([\d.]+) ', d).group(1)), 4) for d in dashed]
dash_y = [round(ty - k * float(v), 4) for v in re.search(r'd="M [\d.]+ ([-\d.]+) L [\d.]+ ([-\d.]+)', dashed[0]).groups()]
assert all(abs(x - sx(b)) < 0.01 for x, b in zip(dash_x, bounds)), (dash_x, bounds)
# Where the boundary splits into an A segment and a B segment: halfway between row A's
# axis and the top of row B's markers, so neither segment touches the other row.
b_top = min(m['cy'] for m in markers if m['g'] == 1) - 6
split_y = round((row_y[0] + b_top) / 2, 4)
assert row_y[0] < split_y < b_top

# ---- EXECUTE: ordered steps, one causal change each ----
# Shared phase: the boundary starts on the 75% line (right) and moves left to the
# 81.25% line, which is the direction the caption and prose describe. Stops between
# the lines are midpoints between consecutive drawn positions, so each move crosses
# exactly one marker.
positions = sorted({m['pos'] for m in markers})
assert len(positions) == 16  # no two markers share a position in the redesigned toy
crossed = sorted((x for x in positions if LEFT < x < RIGHT), reverse=True)
assert crossed == [0.56, 0.52, 0.48, 0.45, 0.38], crossed
mids = [round((a + b) / 2, 4) for a, b in zip(crossed, crossed[1:])]
assert mids == [0.54, 0.5, 0.465, 0.415], mids
# Split phase: Subgroup A stays on the 81.25% line; Subgroup B's own cutoff moves right
# (more lenient), crossing B's two middle repayers together, then one defaulter, then
# the last denied repayer. Stops are midpoints between consecutive Subgroup B positions.
b_positions = sorted(m['pos'] for m in markers if m['g'] == 1)
b_mid = lambda lo, hi: round((lo + hi) / 2, 4)
b_stops = [b_mid(0.56, 0.69), b_mid(0.69, 0.76), b_mid(0.76, 0.83)]
assert all(any(lo < s < hi for lo, hi in zip(b_positions, b_positions[1:])) for s in b_stops)
assert b_stops == [0.625, 0.725, 0.795], b_stops
STEPS = ([('none', None)] + [('decide', (RIGHT, RIGHT)), ('score', (RIGHT, RIGHT))]
         + [('score', (x, x)) for x in mids] + [('score', (LEFT, LEFT)), ('compare', (LEFT, LEFT))]
         + [('split', (LEFT, b)) for b in b_stops])

def name(b):
    if b is None:
        return None
    a, bb = b
    if a == bb:  # sweep-only cutoffs are never displayed and have no name
        return f'{LABEL[a]}% line' if a in LABEL else f'Move {mids.index(a) + 1}' if a in mids else None
    return f'B cutoff {b_stops.index(bb) + 1}'

def pct(n, d):
    v = Fraction(100 * n, d)
    s = f'{float(v):.2f}'.rstrip('0').rstrip('.')
    assert Fraction(s) == v, (n, d, s)  # exact: every value here terminates
    return s

def rate(n, d):
    """n/d as an exact two-place decimal (0.00, 0.25, 0.50, 0.75, 1.00)."""
    v = Fraction(n, d)
    s = f'{float(v):.2f}'
    assert Fraction(s) == v, (n, d, s)
    return s

def ledger(i, phase, b):
    L = {'step': i, 'phase': phase, 'boundary': list(b) if b else None, 'name': name(b)}
    if b is None:
        return L
    approve = lambda m: m['pos'] < b[m['g']]  # each boundary approves the markers to its left
    L['approved'] = [sum(approve(m) for m in markers if m['g'] == g) for g in (0, 1)]
    if phase == 'decide':
        return L
    L['repayersApproved'] = [sum(approve(m) for m in markers if m['g'] == g and m['y'] == 1) for g in (0, 1)]
    L['defaultersApproved'] = [sum(approve(m) for m in markers if m['g'] == g and m['y'] == 0) for g in (0, 1)]
    L['deniedRepayers'] = [4 - L['repayersApproved'][g] for g in (0, 1)]
    L['errors'] = [L['deniedRepayers'][g] + L['defaultersApproved'][g] for g in (0, 1)]
    L['wrong'] = [j for j, m in enumerate(markers) if approve(m) != (m['y'] == 1)]
    L['correct'] = 16 - sum(L['errors'])
    L['accuracyPct'] = pct(L['correct'], 16)
    # Rates beside the counts, so the ledger reads like the chapter's TPR/FPR table.
    L['tpr'] = [rate(L['repayersApproved'][g], 4) for g in (0, 1)]
    L['fpr'] = [rate(L['defaultersApproved'][g], 4) for g in (0, 1)]
    return L

ledgers = [ledger(i, ph, b) for i, (ph, b) in enumerate(STEPS)]
SHARED_STEPS = range(2, 8)   # 75% line .. 81.25% line
SPLIT_STEPS = range(9, 12)

# ---- GUARD: invariants, the caption's endpoints, and the sweep ----
for L in ledgers:
    if 'errors' in L:
        assert len(L['wrong']) == sum(L['errors']) and L['correct'] + sum(L['errors']) == 16
        for g in (0, 1):
            assert L['approved'][g] == L['repayersApproved'][g] + L['defaultersApproved'][g]
first, best_line = ledgers[2], ledgers[7]
assert first['name'] == '75% line' and best_line['name'] == '81.25% line'
# Caption: 75% -> 81.25%; A perfectly separated; B from one misclassification to three.
assert first['errors'] == [3, 1] and first['correct'] == 12 and first['accuracyPct'] == '75'
assert first['defaultersApproved'] == [3, 0] and first['deniedRepayers'] == [0, 1]
assert best_line['errors'] == [0, 3] and best_line['correct'] == 13 and best_line['accuracyPct'] == '81.25'
assert best_line['deniedRepayers'] == [0, 3] and best_line['defaultersApproved'] == [0, 0]
assert best_line['tpr'] == ['1.00', '0.25'] and best_line['fpr'] == ['0.00', '0.00']
assert [ledgers[s]['errors'] for s in SHARED_STEPS] == [[3, 1], [3, 2], [2, 2], [2, 3], [1, 3], [0, 3]]
assert [ledgers[s]['accuracyPct'] for s in SHARED_STEPS] == ['75', '68.75', '75', '68.75', '75', '81.25']
# Every shared move crosses exactly one marker, so exactly one ring appears or disappears.
for s in list(SHARED_STEPS)[1:]:
    assert len(set(ledgers[s - 1]['wrong']) ^ set(ledgers[s]['wrong'])) == 1, s
# Sweep every distinct shared cutoff: 81.25% is the unique best, everything else <= 75%.
cuts = [positions[0] - 0.01] + [(a + b) / 2 for a, b in zip(positions, positions[1:])] + [positions[-1] + 0.01]
sweep = [ledger(-1, 'score', (c, c)) for c in cuts]
best_correct = max(L['correct'] for L in sweep)
winners = [c for c, L in zip(cuts, sweep) if L['correct'] == best_correct]
runner_up = max(L['correct'] for L in sweep if L['correct'] < best_correct)
assert best_correct == 13 and len(winners) == 1 and 0.26 < winners[0] < 0.38 and 0.26 < LEFT < 0.38
assert runner_up == 12 and pct(runner_up, 16) == '75'
# Each subgroup's own best is not reachable by one shared cutoff at the same time.
own_best = [min(L['errors'][g] for L in sweep) for g in (0, 1)]
assert own_best == [0, 1] and not any(L['errors'] == own_best for L in sweep)
b_best_regions = [round(c, 4) for c, L in zip(cuts, sweep) if L['errors'][1] == 1]
# B reaches 1 error left of its 0.69 defaulter (one denied repayer) and right of its 0.76
# repayer (one approved defaulter; 0.80 is an A marker, so that B region spans two cuts).
assert b_best_regions == [0.625, 0.78, 0.815], b_best_regions
# Split steps: B moves right from the 81.25% line; A is unchanged throughout.
split = [ledgers[s] for s in SPLIT_STEPS]
assert all(L['errors'][0] == 0 and L['tpr'][0] == '1.00' and L['fpr'][0] == '0.00' for L in split)
assert [L['errors'][1] for L in split] == [1, 2, 1]
assert [L['tpr'][1] for L in split] == ['0.75', '0.75', '1.00'] and [L['fpr'][1] for L in split] == ['0.00', '0.25', '0.25']
assert [L['deniedRepayers'][1] for L in split] == [1, 1, 0] and [L['defaultersApproved'][1] for L in split] == [0, 1, 1]
assert set(ledgers[8]['wrong']) - set(split[0]['wrong']) and len(set(ledgers[8]['wrong']) ^ set(split[0]['wrong'])) == 2
for prev, cur in zip(split, split[1:]):
    assert len(set(prev['wrong']) ^ set(cur['wrong'])) == 1
# B cutoff 1 already beats every shared boundary with no false positive; the FPR cost
# appears only when B's cutoff moves far enough right to equalize TPR at 1.00.
assert split[0]['correct'] > best_correct and split[0]['fpr'][1] == '0.00' and split[0]['tpr'][1] != split[0]['tpr'][0]
assert split[1]['correct'] < split[0]['correct'] and split[2]['accuracyPct'] == split[0]['accuracyPct']
assert split[2]['tpr'][1] == split[2]['tpr'][0] and split[2]['fpr'][1] != '0.00'
last = ledgers[-1]
assert last['correct'] == 15 and last['accuracyPct'] == '93.75' and last['tpr'] == ['1.00', '1.00']

# Compare step carries the 75% line for before/after and the sweep result.
cmp = ledgers[8]
cmp['before'] = {k: first[k] for k in ['name', 'errors', 'correct', 'accuracyPct', 'deniedRepayers', 'defaultersApproved', 'tpr', 'fpr']}
cmp['sweep'] = {'cutoffs': len(cuts), 'bestCorrect': best_correct, 'bestPct': pct(best_correct, 16),
                'bestCount': len(winners), 'nextPct': pct(runner_up, 16),
                # Correct of 16 at every shared position, strictest (left) to most lenient (right).
                'correctByPosition': [L['correct'] for L in sweep]}
assert cmp['sweep']['correctByPosition'][[i for i, c in enumerate(cuts) if LEFT < c < 0.38][0]] == best_correct
# Final step carries the shared 81.25% line it departed from.
last['shared'] = {k: best_line[k] for k in ['name', 'errors', 'correct', 'accuracyPct', 'deniedRepayers', 'defaultersApproved', 'tpr', 'fpr']}
final = {k: last[k] for k in ['phase', 'name', 'boundary', 'errors', 'correct', 'accuracyPct', 'repayersApproved', 'defaultersApproved', 'deniedRepayers', 'tpr', 'fpr', 'shared']}

# ---- OUTPUT: drawing constants for the browser module, checked byte-for-byte ----
D = {
    'viewBox': VIEWBOX, 'stops': [list(b) if b else None for _, b in STEPS], 'phases': [ph for ph, _ in STEPS],
    'mids': mids, 'bStops': b_stops, 'bounds': bounds, 'labels': labels,
    'tx': tx, 'k': k, 'u0': u0, 'u1': u1, 'rowY': row_y, 'lineY': dash_y, 'splitY': split_y,
    'markers': [{key: m[key] for key in ['g', 'y', 'pos', 'x', 'cy']} for m in markers],
}
module = HERE / f'{NAME}.js'
if module.exists():
    found = re.search(r'const D=(\{.*?\});\n', module.read_text())
    actual = json.loads(found.group(1)) if found else None
    if actual != D:
        print(json.dumps(D, separators=(',', ':')))
        raise SystemExit('module constant D is stale; paste the "D" printed above')
print(json.dumps({'units': 'counts of drawn toy markers; positions are fractions of the drawn score axis, not scores',
                  'D': D, 'ledgers': ledgers, 'expected_final': final,
                  'sweep': [{'cut': round(c, 4), 'errors': L['errors'], 'accuracyPct': L['accuracyPct']} for c, L in zip(cuts, sweep)]},
                 separators=(',', ':')))
