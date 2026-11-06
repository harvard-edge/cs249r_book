#!/usr/bin/env python3
"""Lesson data for vol1:fig-resnet-activations-histogram (10_model_compression.qmd).

Bar heights are read from the figure's own TikZ tables in the chapter source (a
digitization adapted from wu2020integer, normalized to a peak of 1; relative
frequencies, not counts), so every clipped share is illustrative. The quantizer
is the chapter's affine UINT8 grid (eq-quant-scale, LEGO QuantizationMathCalc):
MLSysIM's QuantizationGridScenario with bits=8 and alpha=0, because every plotted
activation is non-negative. That gives s = beta / (2^8 - 1), z = 0, codes 0..255.
Thresholds sit on the 0.1-wide bin edges, so a bin is either kept or clipped.
No KL, MSE, or accuracy optimum is computed.

Drawing coordinates come from the verified SVG's axis transform
(viewBox 0 0 458.13 193.829); asserted against drawn bar and line positions.

Run:  PYTHONPATH=mlsysim <python> <this file>            print the lesson data (JSON)
      PYTHONPATH=mlsysim <python> <this file> --check    also assert the factory's embedded D equals it
      PYTHONPATH=mlsysim <python> <this file> --write    rewrite the factory's embedded D block
"""
import json
import re
import sys
from pathlib import Path

from mlsysim import Scenarios

HERE = Path(__file__).resolve()
BOOKS = HERE.parents[3]
QMD = BOOKS / 'vol1/10_model_compression/10_model_compression.qmd'
FACTORY = HERE.with_name('original-v1-activation-clipping.js')

# ── Source histogram: the two TikZ tables inside the figure block ──────────────
text = QMD.read_text()
block = text[text.index('::: {#fig-resnet-activations-histogram'):]
block = block[:block.index('\n:::\n')]
tables = re.findall(r'table\[row sep=\\\\\]\{\s*x\s+y\\\\(.*?)\};', block, re.S)
assert len(tables) == 2, 'expected blue bulk and red tail tables'
rows = [[tuple(float(v) for v in r.split()) for r in re.findall(r'([\d.]+\s+[\d.]+)\\\\', t)] for t in tables]
bulk, tail = rows
assert len(bulk) == 12 and len(tail) == 9
BIN = 0.1
bins = [dict(center=x, height=y, tail=i >= len(bulk)) for i, (x, y) in enumerate(bulk + tail)]
for i, b in enumerate(bins):
    assert abs(b['center'] - (0.05 + BIN * i)) < 1e-9, b
    b['lo'], b['hi'] = round(b['center'] - BIN / 2, 10), round(b['center'] + BIN / 2, 10)
assert 'coordinates {(1.1,0) (1.1,1.0)}' in block, 'printed candidate threshold moved'
assert 'xmin=0, xmax=2.5' in block and 'bar width=15pt' in block
CANDIDATE, XMAX = 1.1, 2.5
TOP = bins[-1]['hi']                     # upper edge of the last occupied bin
assert TOP == 2.1
total = sum(b['height'] for b in bins)

# ── SVG axis transform (pgfplots → dvisvgm): matrix(K,0,0,-K,OX,OY) ──────────
K, OX, OY = 0.998105, 40.552989, 151.75835
X_AXIS_LOCAL, Y_ONE_LOCAL = 408.717799, 123.508075   # local length of x in [0, 2.5]; local height of y = 1
BAR_LOCAL = 15 * 72 / 72.27                          # 15 TeX pt in local big points
X = lambda v: OX + K * v * X_AXIS_LOCAL / XMAX
Y = lambda h: OY - K * h * Y_ONE_LOCAL
assert abs((8.173441 - BAR_LOCAL / 2) - 0.702248) < 0.01      # first drawn blue bar, left edge (local)
assert abs(X_AXIS_LOCAL / XMAX * 1.1 - 179.834749) < 0.01     # drawn green line (local)
assert abs(X_AXIS_LOCAL / XMAX * 2.05 + BAR_LOCAL / 2 - 342.619859) < 0.01   # last red bar, right edge
geometry = {'x0': round(X(0), 4), 'xPerUnit': round(K * X_AXIS_LOCAL / XMAX, 6), 'y0': round(Y(0), 4),
            'yPerUnit': round(K * Y_ONE_LOCAL, 6), 'yTop': round(Y(1.0), 4), 'barWidth': round(K * BAR_LOCAL, 4)}
# Printed annotation boxes (SVG getBBox of the glyph groups, verified artwork): the moving line breaks around them.
LABELS = {'candidate': [231.83, 36.66, 310.53, 55.44], 'outlier': [310.56, 88.68, 400.3, 116.21]}
assert LABELS['candidate'][0] > X(1.1) and LABELS['outlier'][0] > X(1.6)
bars = [dict(x=round(X(b['center']) - geometry['barWidth'] / 2, 4), y=round(Y(b['height']), 4),
             w=geometry['barWidth'], h=round(Y(0) - Y(b['height']), 4), lo=b['lo'], tail=b['tail']) for b in bins]


def sig(x, digits=3):
    """Plain decimal text with `digits` significant figures, e.g. 0.00824."""
    return f'{float(f"{x:.{digits}g}"):.{max(0, digits - 1 - int(f"{x:e}".split("e")[1]))}f}'


def threshold(beta):
    """Every reported quantity at one clipping threshold, via MLSysIM's grid."""
    grid = Scenarios.Uint8RangeMinusOneToThree.variant(
        name='Plotted ResNet-50 activations on a UINT8 grid', values=[b['center'] for b in bins],
        alpha=0.0, beta=beta)
    assert grid.bits == 8 and grid.qmin == 0 and grid.qmax == grid.steps == 255 and grid.zero_point == 0
    clipped = [i for i, b in enumerate(bins) if b['lo'] >= beta - 1e-9]
    # A clipped bin's centre saturates to the top code; a kept one never does unless it is the last code.
    assert all(grid.codes[i] == grid.qmax for i in clipped)
    share = sum(bins[i]['height'] for i in clipped) / total
    clip_err = max(0.0, TOP - beta)
    return {'beta': round(beta, 10), 'betaText': f'{beta:.1f}', 'step': grid.scale, 'stepText': sig(grid.scale),
            'halfStep': grid.max_rounding_error, 'halfStepText': sig(grid.max_rounding_error),
            'clippedBins': len(clipped), 'clippedShare': share, 'clippedShareText': f'{100 * share:.1f}%',
            'maxClipError': round(clip_err, 10), 'maxClipErrorText': f'{clip_err:.1f}' if clip_err else '0',
            'emptyRange': round(max(0.0, beta - TOP), 10), 'emptyRangeText': f'{max(0.0, beta - TOP):.1f}'}


edges = [round(BIN * k, 10) for k in range(1, int(round(XMAX / BIN)) + 1)]   # 0.1 … 2.5
table = [threshold(b) for b in edges]
by = {t['betaText']: t for t in table}
STORY = ['2.1', '1.6', '1.1', '0.5']                   # max method, toward the candidate, candidate, into the bulk

# ── Asserted endpoints ────────────────────────────────────────────────────────
mx, cand, deep = by['2.1'], by['1.1'], by['0.5']
assert mx['clippedBins'] == 0 and mx['clippedShare'] == 0 and mx['maxClipError'] == 0
assert abs(mx['step'] - 2.1 / 255) < 1e-12 and abs(mx['halfStep'] - 2.1 / 510) < 1e-12
assert cand['clippedBins'] == 10 and abs(cand['step'] - 1.1 / 255) < 1e-12   # 1.15 bulk bar + all nine red bars
assert by['1.6']['clippedBins'] == 5 and deep['clippedBins'] == 16
assert abs(cand['clippedShare'] - (0.12 + sum(y for _, y in tail)) / total) < 1e-12
steps = [t['step'] for t in table]; shares = [t['clippedShare'] for t in table]
assert steps == sorted(steps) and shares == sorted(shares, reverse=True)       # the trade-off is monotone
assert all(t['clippedShare'] == 0 for t in table if t['beta'] >= TOP)

D = {'geometry': geometry, 'bars': bars, 'candidate': CANDIDATE, 'top': TOP, 'xmax': XMAX, 'bin': BIN,
     'codes': 255, 'tickEvery': 17, 'labels': list(LABELS.values()), 'story': STORY, 'table': table}

PATTERN = re.compile(r'(// BEGIN calc\.py output\nconst D=)(\{.*\})(;\n// END calc\.py output)', re.S)
if __name__ == '__main__':
    payload = json.dumps(D, ensure_ascii=False, separators=(',', ':'))
    if '--write' in sys.argv:
        src = FACTORY.read_text()
        assert PATTERN.search(src), 'factory lacks BEGIN/END calc.py markers'
        FACTORY.write_text(PATTERN.sub(lambda m: m.group(1) + payload + m.group(3), src))
    if '--check' in sys.argv or '--write' in sys.argv:
        embedded = json.loads(PATTERN.search(FACTORY.read_text()).group(2))
        assert embedded == json.loads(payload), 'factory D differs from calc.py output; run --write'
    print(json.dumps({k: D[k] for k in ('geometry', 'candidate', 'top', 'codes', 'story')} |
                     {'story_rows': [by[b] for b in STORY]}, indent=1, ensure_ascii=False))
