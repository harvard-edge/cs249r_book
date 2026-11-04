"""Exercise recovery failures with constructed artifacts, not study results."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/audit/reconstruct_taxonomy_evaluation.py'
spec = importlib.util.spec_from_file_location('evaluation_reconstruction', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def evidence(tmp_path):
    for task, key in [('cov', 'coverage_judgments'), ('tag', 'indexing_tags')]:
        for reviewer in ['claude', 'codex']:
            for arm in ['Vocab-X', 'Vocab-Y']:
                doc = {'arm': arm, key: [{'unit_id': 'constructed-unit'}]}
                (tmp_path / f'phase6-raw-{reviewer}-{task}-{arm}.md').write_text(json.dumps(doc))
    (tmp_path / 'heldout').mkdir()
    (tmp_path / 'heldout/phase6-heldout-slice.json').write_text(
        json.dumps({'held_out_test_units': [{'unit_id': 'constructed-unit'}]})
    )
    return tmp_path


def test_recovery_distinguishes_observation_from_unseal(evidence):
    record = module.reconstruct(evidence)
    assert record['record_kind'] == 'retrospective-evaluation-inventory'
    assert 'unsealed_coverage_unit_ids' not in record
    assert record['evaluated_coverage_unit_ids'] == ['constructed-unit']


@pytest.mark.parametrize('identities', [
    ['constructed-unit', 'constructed-unit'],
    ['different-constructed-unit'],
])
def test_recovery_rejects_duplicate_or_disagreeing_outputs(evidence, identities):
    path = evidence / 'phase6-raw-codex-cov-Vocab-Y.md'
    path.write_text(json.dumps({'arm': 'Vocab-Y', 'coverage_judgments': [{'unit_id': x} for x in identities]}))
    with pytest.raises(ValueError):
        module.reconstruct(evidence)


def test_recovery_rejects_items_outside_source_slice(evidence):
    (evidence / 'heldout/phase6-heldout-slice.json').write_text('{"held_out_test_units": []}')
    with pytest.raises(ValueError, match='absent from the source slice'):
        module.reconstruct(evidence)
