"""Recover evaluated unit identities from historical outputs without inventing an unseal event.

Usage: python3 binder/tools/audit/reconstruct_taxonomy_evaluation.py EVIDENCE_DIR OUTPUT
Only identifiers and hashes are emitted; source transcripts remain at their existing location.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def reconstruct(evidence_dir: Path) -> dict:
    receipts = []
    observed: dict[str, set[str]] = {}
    for task, key in [('cov', 'coverage_judgments'), ('tag', 'indexing_tags')]:
        reference = None
        for reviewer in ['claude', 'codex']:
            for arm in ['Vocab-X', 'Vocab-Y']:
                path = evidence_dir / f'phase6-raw-{reviewer}-{task}-{arm}.md'
                raw = path.read_bytes()
                text = raw.decode()
                blocks = re.findall(r'```json\s*\n(.*?)\n```', text, re.S)
                # Historical outputs used fenced JSON; this reader does not relax
                # the current prospective raw-JSON contract.
                doc = json.loads(blocks[0] if len(blocks) == 1 else text)
                if doc.get('arm') != arm:
                    raise ValueError(f'Wrong arm in {path.name}')
                rows = doc[key]
                ids = [r['unit_id'] for r in rows]
                if any(not isinstance(uid, str) or not uid for uid in ids) or len(set(ids)) != len(ids):
                    raise ValueError(f'Missing or duplicate unit identity in {path.name}')
                if reference is not None and set(ids) != reference:
                    raise ValueError(f'Evaluated unit sets differ in {path.name}')
                reference = set(ids)
                receipts.append({'artifact': path.name, 'sha256': hashlib.sha256(raw).hexdigest(),
                                 'kind': key, 'arm': arm, 'unit_count': len(ids)})
        observed[key] = reference or set()
    slice_path = evidence_dir / 'heldout/phase6-heldout-slice.json'
    raw_slice = slice_path.read_bytes()
    source_slice = json.loads(raw_slice)
    source_ids = {u['unit_id'] for u in source_slice['held_out_test_units']}
    for kind, ids in observed.items():
        if not ids <= source_ids:
            raise ValueError(f'{kind} contains IDs absent from the source slice')
    return {
        'schema_version': 1,
        'record_kind': 'retrospective-evaluation-inventory',
        'evidence_scope': 'Observed unit identities in stored evaluator outputs; no metric recomputation or independent verification.',
        'limitations': [
            'Does not establish when material was unsealed or what reviewers could access.',
            'Does not establish source independence, human participation, or correctness of judgments.',
            'Does not establish a recorded pre-exposure registration.',
            'Source artifacts are maintainer-local; hashes alone do not make the study publicly reproducible.',
        ],
        'source_slice_sha256': hashlib.sha256(raw_slice).hexdigest(),
        'sources': receipts,
        'evaluated_coverage_unit_ids': sorted(observed['coverage_judgments']),
        'evaluated_indexing_unit_ids': sorted(observed['indexing_tags']),
        'counts': {key: len(value) for key, value in observed.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence_dir', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = reconstruct(args.evidence_dir)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['counts'], sort_keys=True))


if __name__ == '__main__':
    main()
