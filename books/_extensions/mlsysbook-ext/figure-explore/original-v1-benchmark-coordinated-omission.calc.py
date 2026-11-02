#!/usr/bin/env python3
"""Coordinated-omission lesson data, derived from the figure's own TikZ source.

The figure is schematic: no LEGO cell or MLSysIM value applies. Every value the
Explore view shows is a COUNT of drawn intervals (requests sent, outstanding,
recorded) at a cursor position on the drawing's own time axis, in TikZ cm.
This script parses those intervals from 12_benchmarking.qmd, maps them onto the
rendered SVG with the transform read from the verified artwork, recomputes the
per-step ledger, asserts the endpoints, and checks that the constant embedded
in the browser module matches.

Run from the repository root:
  python3 books/_extensions/mlsysbook-ext/figure-explore/original-v1-benchmark-coordinated-omission.calc.py
"""
import argparse
import json
import re
from pathlib import Path

NAME = 'original-v1-benchmark-coordinated-omission'
KEY = 'vol1:fig-benchmark-coordinated-omission'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
QMD = ROOT / 'books/vol1/12_benchmarking/12_benchmarking.qmd'
BP_PER_CM = 72 / 2.54  # TikZ cm -> PostScript points in the dvisvgm output

p = argparse.ArgumentParser()
p.add_argument('--art-index', default='/tmp/figures-animation-art-current/index.json')
args = p.parse_args()

# ---- LOAD: the figure's TikZ block, verbatim from the chapter ----
text = QMD.read_text()
start = text.index('{#fig-benchmark-coordinated-omission')
tikz = text[start:text.index('\\end{tikzpicture}', start)]
pause = [float(v) for v in re.search(r'\\fill\[RedL!65\] \(([\d.]+),[\d.]+\) rectangle \(([\d.]+),[\d.]+\)', tikz).groups()]
open_rows = [tuple(map(float, t.split('/'))) for t in re.search(r'\\foreach \\x/\\end/\\y in \{([^}]+)\}', tikz).group(1).split(',')]
closed = [tuple(map(float, m)) for m in re.findall(r'\\draw\[arr,GreenLine\] \(([\d.]+),1\.95\)--\(([\d.]+),1\.95\)', tikz)]
axis = [float(v) for v in re.search(r'\\draw\[arr\] \(([\d.]+),\.05\)--\(([\d.]+),\.05\)', tikz).groups()]
assert pause == [4.0, 8.4], pause
assert [(a, e) for a, e, _ in open_rows] == [(4.3, 9.1), (5.5, 9.7), (6.7, 10.3), (7.9, 10.9)], open_rows
assert closed == [(4.3, 9.1), (9.6, 10.9)], closed
assert axis == [2.9, 12.8], axis

# ---- LOAD: the verified artwork's transform and per-interval arrowheads ----
art = json.load(open(args.art_index))[KEY]
svg = Path(art['asset']).read_text()
assert art['asset_sha1'] == 'cd6012765d36af7e26a203c202d6ba3e5a00dd28', art['asset_sha1']
assert 'viewBox="0 0 362.984 186.224"' in svg
scale, ox, oy = map(float, re.search(r'matrix\(([\d.]+), 0, 0, -[\d.]+, ([\d.]+), ([\d.]+)\)', svg).groups())
body = svg[svg.index('</defs>'):]
greens = [m.group(0) for m in re.finditer(r'<path[^>]*>', body) if '56.077576%' in m.group(0)]
heads = [float(re.search(r'matrix\([^,]+, 0, 0, [^,]+, ([\d.]+),', g).group(1)) for g in greens[2:18:3]]
line_ends = [float(re.search(r' L ([\d.]+) ', g).group(1)) for g in greens[1:18:3]]

def sx(cm):
    return round(scale * cm * BP_PER_CM + ox, 4)

def sy(cm):
    return round(oy - scale * cm * BP_PER_CM, 4)

intervals = [{'policy': 0, 'arrival': a, 'response': e, 'y': y} for a, e, y in open_rows]
intervals += [{'policy': 1, 'arrival': a, 'response': e, 'y': 1.95} for a, e in closed]
for iv, head, end in zip(intervals, heads, line_ends):
    iv['x0'], iv['x1'], iv['sy'] = sx(iv['arrival']), round(scale * end + ox, 4), sy(iv['y'])
    iv['head'] = head
    assert abs(iv['x0'] - sx(iv['arrival'])) < 1e-9
    # The drawn line stops short of the response coordinate (arrow shorten + tip).
    assert 0 < sx(iv['response']) - iv['x1'] < 5, iv

# ---- EXECUTE: ordered cursor endpoints, one causal change each ----
cursors = [axis[0], pause[0], 4.3, 7.9, pause[1], 9.1, 9.6, 10.3, 10.9, 10.9]

def ledger(t):
    inside = lambda a: pause[0] <= a < pause[1]
    out = {'cursor': t, 'server': 'paused' if pause[0] <= t < pause[1] else 'running'}
    for key in ['sent', 'sentInPause', 'outstanding', 'recorded', 'recordedFromPause']:
        out[key] = [0, 0]
    for iv in intervals:
        k = iv['policy']
        if iv['arrival'] <= t:
            out['sent'][k] += 1
            out['sentInPause'][k] += inside(iv['arrival'])
            if iv['response'] <= t:
                out['recorded'][k] += 1
                out['recordedFromPause'][k] += inside(iv['arrival'])
            else:
                out['outstanding'][k] += 1
    return out

ledgers = [ledger(t) for t in cursors]

# ---- GUARD: invariants and the endpoints the storyboard promises ----
for L in ledgers:
    for k in (0, 1):
        assert L['sent'][k] == L['outstanding'][k] + L['recorded'][k]
    assert L['outstanding'][1] <= 1, 'one waiting client never has two requests in flight'
assert max(L['outstanding'][0] for L in ledgers) == 4
assert ledgers[0]['sent'] == [0, 0] and ledgers[1]['server'] == 'paused'
assert ledgers[3]['outstanding'] == [4, 1] and ledgers[4]['server'] == 'running'
assert ledgers[5]['outstanding'] == [3, 0] and ledgers[6]['outstanding'] == [3, 1]
final = ledgers[-1]
assert final['sentInPause'] == [4, 1] and final['recorded'] == [4, 2]
assert final['recordedFromPause'] == [4, 1] and final['outstanding'] == [0, 0]

# ---- OUTPUT: drawing constants for the browser module, checked byte-for-byte ----
D = {
    'pause': pause, 'axis': axis, 'scale': scale, 'ox': ox, 'oy': oy, 'cursors': cursors,
    'intervals': [{k: iv[k] for k in ['policy', 'arrival', 'response', 'x0', 'x1', 'sy', 'head']} for iv in intervals],
}
module = HERE / f'{NAME}.js'
if module.exists():
    actual = json.loads(re.search(r'const D=(\{.*?\});\n', module.read_text()).group(1))
    assert actual == D, 'module constant D is stale; paste the "D" printed below'
print(json.dumps({'units': 'TikZ cm on the schematic time axis; counts of drawn intervals, not measurements',
                  'D': D, 'ledgers': ledgers, 'expected_final': final}, separators=(',', ':')))
