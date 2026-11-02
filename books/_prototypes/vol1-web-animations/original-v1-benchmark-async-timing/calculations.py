"""Reproduce SVG coordinates and verify the schematic asynchronous boundaries."""
import json
from pathlib import Path
import re

SCALE = 0.998239
OFFSET = -4.776421
SOURCE = [62.362816, 104.883015, 124.726557, 300.477495, 325.987266]
geometry = dict(zip(['start', 'call', 'deviceStart', 'deviceStop', 'hostStop'], [round(x * SCALE + OFFSET, 6) for x in SOURCE]))
geometry['deviceMid'] = round((geometry['deviceStart'] + geometry['deviceStop']) / 2, 6)
assert geometry['start'] < geometry['call'] < geometry['deviceStart'] < geometry['deviceMid'] < geometry['deviceStop'] < geometry['hostStop']
trace = [dict(step=i, x=geometry[name], call='ready' if i == 0 else 'stopped', device='not started' if i < 2 else 'running' if i < 4 else 'stopped', synchronized='ready' if i == 0 else 'running' if i < 5 else 'stopped') for i,name in enumerate(['start','call','deviceStart','deviceMid','deviceStop','hostStop'])]
module = Path(__file__).resolve().parents[3] / '_extensions/mlsysbook-ext/figure-explore/original-v1-benchmark-async-timing.js'
if module.exists():
    actual=json.loads(re.search(r'const G=(\{[^;]+\});',module.read_text()).group(1))
    assert actual == geometry, (actual, geometry)
print(json.dumps({'geometry':geometry,'trace':trace,'units':'canonical SVG coordinates, not measured duration'}, indent=2))
