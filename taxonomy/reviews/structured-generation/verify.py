"""Check review inventory against its frozen input; no evaluation metrics."""
import gzip
import hashlib
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
packet = json.loads((HERE / 'baseline.json').read_text())
raw = (ROOT / packet['input']['path']).read_bytes()
if str(packet["input"]["path"]).endswith(".gz"):
    raw = gzip.decompress(raw)
if hashlib.sha256(raw).hexdigest() != packet['input']['sha256']:
    raise SystemExit('Source changed since review; compare snapshots before reusing findings.')
doc = yaml.safe_load(raw)
concepts = [c for c in doc['concepts'] if c.get('home') == packet['subject']['id']]
ids = {c['id'] for c in concepts}
edges = [e for e in doc['relationships'] if e['source'] in ids or e['target'] in ids]
assert concepts == packet['concepts']
assert edges == packet['incident_relationships']
assert packet['inventory_counts'] == {'concepts': len(concepts), 'incident_relationships': len(edges)}
report = (HERE / 'REVIEW.md').read_text()
for entry in concepts + edges:
    assert report.count('`' + entry['id'] + '`') == 1, entry['id']
print('Inventory matches the frozen source; every concept and incident relationship has a review entry.')
print(json.dumps(packet['inventory_counts'], sort_keys=True))
