"""Reproduce the v4 historical ledger migration from pinned evidence artifacts.

Writes a candidate file; never modifies the canonical ledger automatically.
Usage: python3 binder/tools/audit/migrate_taxonomy_ledger.py OUTPUT
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRIOR_SHA256 = 'cbc72fa6e769fec7aa8015fd6d154a425674e5124b379d35dd10724f4b48fef8'


def migrate(root=ROOT):
    directory = root / 'taxonomy/method-study'
    raw = gzip.decompress((directory / 'legacy-burned-clusters-v3.json.gz').read_bytes())
    if hashlib.sha256(raw).hexdigest() != PRIOR_SHA256:
        raise ValueError('Historical ledger archive differs from the pinned predecessor')
    ledger = json.loads(raw)
    inventory_raw = (directory / 'evaluation-reconstruction.json').read_bytes()
    ledger.update(
        schema_version=4,
        editor_coauthor_5yr_clusters=[],
        stage1_human_registrations=[],
        refresh_cycle_resets=[],
        active_refresh_cycle=None,
        historical_unknown_coauthor_clusters=sorted(ledger['exposed_cluster_metadata']),
    )
    for metadata in ledger['exposed_cluster_metadata'].values():
        metadata['editor_coauthor_5yr'] = None
        for roster in metadata['work_rosters'].values():
            roster['editor_coauthor_5yr'] = None
    ledger['historical_migration'] = {
        'kind': 'retrospective-evidence-recovery',
        'recorded_on': '2026-11-04',
        'prior_burned_clusters_sha256': PRIOR_SHA256,
        'evaluation_inventory_sha256': hashlib.sha256(inventory_raw).hexdigest(),
        'evaluation_inventory': json.loads(inventory_raw),
        'limitations': [
            'No historical unseal event or prospective registration is reconstructed.',
            'Unknown coauthor status excludes historical sources from independent-source eligibility.',
            'Empty human-registration and cycle-reset lists record no events in this ledger; they are not proof about events outside it.',
        ],
    }
    return ledger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.resolve() == (ROOT / 'taxonomy/method-study/burned-clusters.json').resolve():
        parser.error('Write a candidate file, inspect it, and apply it separately.')
    args.output.write_text(json.dumps(migrate(), indent=2) + '\n')
