"""Verify a proposed review packet without modifying the canonical vocabulary."""
import gzip
import hashlib
import json
import argparse
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("packet", nargs="?", type=Path, default=HERE / "decisions.json")
args = parser.parse_args()
packet = json.loads(args.packet.read_text())
raw = (ROOT / packet["input"]["path"]).read_bytes()
if str(packet["input"]["path"]).endswith(".gz"):
    raw = gzip.decompress(raw)
if hashlib.sha256(raw).hexdigest() != packet["input"]["sha256"]:
    raise SystemExit("Baseline changed; review the differences before reusing this packet.")
doc = yaml.safe_load(raw)
records = packet["records"]
ids = [r["id"] for r in records]
assert len(ids) == len(set(ids)), "Duplicate review record"
concept_ids = {r["id"] for r in records if r["kind"] == "concept"}
expected = [c for c in doc["concepts"] if c["id"] in concept_ids]
edges = [e for e in doc["relationships"]
         if e["source"] in concept_ids or e["target"] in concept_ids]
assert [r["before"] for r in records] == expected + edges
assert packet["inventory_counts"] == {
    "concepts": len(expected), "incident_relationships": len(edges)}
for record in records:
    assert record["id"] == record["before"]["id"]
    fields = record["proposed_fields"]
    assert set(fields) <= {"label", "definition", "aliases", "predicate", "conditions"}
    assert all(value != record["before"].get(key) for key, value in fields.items())
    assert bool(fields) == (record["disposition"] == "revise")
print("Frozen baseline and complete incident-edge inventory verified; see accepted reviews for application status.")
print(json.dumps(packet["inventory_counts"], sort_keys=True))
