"""Historical migration must preserve evidence and exclude unresolved independence."""
import gzip
import json
import shutil
from pathlib import Path

import pytest

from core.concept_graph import ConceptGraphError
from core.taxonomy_schema import (
    compute_lineage_components,
    _compute_warrant_lineages,
    validate_burned_clusters_v3_ledger,
    verify_burned_clusters_append_only,
    _load_burned_clusters,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def ledger():
    return json.loads((ROOT / 'taxonomy/method-study/burned-clusters.json').read_text())


def test_migration_preserves_archived_evidence_and_excludes_unknowns(ledger):
    prior = json.loads(gzip.decompress((ROOT / 'taxonomy/method-study/legacy-burned-clusters-v3.json.gz').read_bytes()))
    verify_burned_clusters_append_only(ledger, prior)
    assert ledger['evaluation_attempts'] == []
    closure = compute_lineage_components([], ledger)
    assert set(ledger['historical_unknown_coauthor_clusters']) <= closure['independence_excluded_clusters']
    key = 'pf_sglang_neurips24'
    metadata = ledger['exposed_cluster_metadata'][key]
    all_lineages, independent = _compute_warrant_lineages(
        [dict(metadata, provenance_family_id=key, origin='external')], ledger
    )
    assert all_lineages
    assert not independent
    assert key not in closure['editor_coauthor_5yr_clusters']


@pytest.mark.parametrize('mutation', ['history', 'unknown', 'false_independence', 'receipt', 'unseal'])
def test_migration_rejects_evidence_rewriting(ledger, mutation):
    if mutation == 'history':
        key = next(iter(ledger['historical_phase3_units']))
        ledger['historical_phase3_units'][key]['current_exposure_split'] = 'reserve'
    elif mutation == 'unknown':
        ledger['historical_unknown_coauthor_clusters'].pop()
    elif mutation == 'false_independence':
        key = ledger['historical_unknown_coauthor_clusters'][0]
        ledger['exposed_cluster_metadata'][key]['editor_coauthor_5yr'] = False
    elif mutation == 'receipt':
        ledger['historical_migration']['evaluation_inventory_sha256'] = '0' * 64
    else:
        ledger['historical_migration']['evaluation_inventory']['unsealed_coverage_unit_ids'] = []
    with pytest.raises(ConceptGraphError):
        validate_burned_clusters_v3_ledger(ledger)


@pytest.mark.parametrize('artifact', [
    'legacy-burned-clusters-v3.json.gz', 'evaluation-reconstruction.json',
])
@pytest.mark.parametrize('damage', ['missing', 'changed'])
def test_alternate_checkout_uses_its_own_evidence_even_with_cached_ledger(tmp_path, artifact, damage):
    """A good authoring checkout must not hide broken candidate evidence."""
    target = tmp_path / 'taxonomy/method-study'
    target.mkdir(parents=True)
    for name in ('burned-clusters.json', 'legacy-burned-clusters-v3.json.gz', 'evaluation-reconstruction.json'):
        shutil.copyfile(ROOT / 'taxonomy/method-study' / name, target / name)
    _load_burned_clusters(tmp_path)
    if damage == 'missing':
        (target / artifact).unlink()
    else:
        (target / artifact).write_bytes(b'corrupted evidence')
    with pytest.raises(ConceptGraphError):
        _load_burned_clusters(tmp_path)
