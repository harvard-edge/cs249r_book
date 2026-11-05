"""Missing ledger evidence must fail with actionable field-level diagnostics."""
import gzip
import json
from pathlib import Path

import pytest

from core.concept_graph import ConceptGraphError
from core.taxonomy_schema import validate_burned_clusters_v3_ledger

ROOT = Path(__file__).resolve().parents[2]


def test_ledger_reports_all_missing_migration_fields():
    ledger = json.loads(gzip.decompress((ROOT / 'taxonomy/method-study/legacy-burned-clusters-v3.json.gz').read_bytes()))
    fields = ('editor_coauthor_5yr_clusters', 'stage1_human_registrations', 'refresh_cycle_resets')
    for name in fields:
        ledger.pop(name, None)
    ledger['evaluation_attempts'] = []
    with pytest.raises(ConceptGraphError) as error:
        validate_burned_clusters_v3_ledger(ledger)
    message = str(error.value)
    for name in (*fields, 'evaluation_attempts'):
        assert name in message
    assert 'historical_phase3_units' not in message
    assert 'do not infer completed evaluations' in message


def test_empty_evaluation_history_is_not_accepted_as_migration_default():
    ledger = json.loads(gzip.decompress((ROOT / 'taxonomy/method-study/legacy-burned-clusters-v3.json.gz').read_bytes()))
    ledger.update(editor_coauthor_5yr_clusters=[], stage1_human_registrations=[], refresh_cycle_resets=[], evaluation_attempts=[])
    with pytest.raises(ConceptGraphError, match='fields: evaluation_attempts;'):
        validate_burned_clusters_v3_ledger(ledger)
