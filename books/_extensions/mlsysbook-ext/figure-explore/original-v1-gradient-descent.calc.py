#!/usr/bin/env python3
"""Recompute the authored one-weight trace; no timing or empirical claims.

The constants/functions are executed from the figure's existing Python cell.
Pixel bounds are mapped to the verified PNG identified in the entry record.
Run with --write to update embedded data, or --check to detect source drift.
"""
import ast
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
QMD = HERE.parents[3] / 'vol1/05_nn_computation/05_nn_computation.qmd'
source = QMD.read_text().split('::: {#fig-gradient-descent-steps', 1)[1]
cell = source.split('```{python}', 1)[1].split('```', 1)[0]
start = cell.index('w_min, curvature =')
end = cell.index('w_axis =')
# The authored recurrence returns an ndarray; a list is sufficient here.
class Array:
    @staticmethod
    def array(values):
        return values
scope = {'np': Array}
exec(compile(ast.parse(cell[start:end]), str(QMD), 'exec'), scope)
rows = []
for rate in (scope['lr_good'], scope['lr_large']):
    weights = scope['descend'](rate)
    rows.append([{'weight': w, 'loss': scope['loss'](w), 'gradient': scope['grad'](w),
                  'delta': -rate * scope['grad'](w)} for w in weights])
for trace, rate in zip(rows, (scope['lr_good'], scope['lr_large'])):
    for a, b in zip(trace, trace[1:]):
        assert abs(a['weight'] + a['delta'] - b['weight']) < 1e-12
        assert b['loss'] < a['loss']
    assert len(trace) == scope['n_steps'] + 1
D = {'minimum': scope['w_min'], 'curvature': scope['curvature'],
     'rates': [scope['lr_good'], scope['lr_large']], 'rows': rows,
     'steps': scope['n_steps'], 'width': 3660, 'height': 2388,
     'axes': [[335, 188, 1908, 1821], [2027, 188, 3600, 1821]],
     'xlim': [-1.8, 5.8], 'ylim': [0, 12]}
factory = HERE.with_name('original-v1-gradient-descent.js')
pattern = re.compile(r'(// BEGIN calc.py output\nconst D=)(.*?)(;\n// END calc.py output)', re.S)
payload = json.dumps(D, separators=(',', ':'))
if '--write' in sys.argv:
    factory.write_text(pattern.sub(lambda m: m[1] + payload + m[3], factory.read_text()))
if '--check' in sys.argv:
    match = pattern.search(factory.read_text())
    assert match and json.loads(match[2]) == D, 'Recompute gradient-descent lesson data'
print(json.dumps(D, indent=2))
