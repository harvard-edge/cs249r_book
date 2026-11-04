"""Chapter contracts must retain ownership when a source concept changes."""
import importlib.util
from pathlib import Path

import pytest
import yaml

TOOL = Path(__file__).resolve().parents[1] / 'tools/check_learning_contracts.py'
spec = importlib.util.spec_from_file_location('learning_contract_audit', TOOL)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def course(tmp_path):
    chapter = 'books/vol1/01_example/example_concepts.yml'
    book = tmp_path / chapter
    book.parent.mkdir(parents=True)
    book.write_text(yaml.safe_dump({'concept_map': {'primary_concepts': ['Capacity'], 'secondary_concepts': ['Counterexample']}}))
    notebook = tmp_path / 'labs/vol1/lab_01_example.py'
    notebook.parent.mkdir(parents=True)
    notebook.write_text('# test notebook')
    source = tmp_path / 'labs/vol1/concepts/lab_01_example_concepts.yml'
    source.parent.mkdir()
    contract = {
        'chapter': 1, 'source_notebook': str(notebook.relative_to(tmp_path)),
        'source_chapter_concept_map': chapter, 'chapter_goal': 'Test capacity.',
        'student_goals': ['Compare fit with cost.'],
        'runtime_parts': [{'part': 'A', 'title': 'Fit', 'question': 'Does it fit?', 'goal': 'Compare fit.',
                           'coverage': 'implemented', 'concept_ids': ['v1_01_A']}],
        'core_concepts': [{'id': 'v1_01_A', 'learning_claim': 'Capacity constrains work.',
                           'tradeoff': 'Memory versus admitted work.', 'student_action': 'Change work.',
                           'evidence_to_inspect': 'Capacity verdict.', 'explanation_prompt': 'Why did fit change?',
                           'source_concepts': [{'concept': 'Capacity', 'concept_map': chapter}]}],
        'primary_concept_dispositions': [{'concept': 'Capacity', 'disposition': 'interactive', 'lab_concepts': ['v1_01_A']}],
    }
    def save(): source.write_text(yaml.safe_dump({'lab_concept_map': contract}))
    save()
    return tmp_path, contract, save


def test_declared_route_and_explicit_secondary_disposition_are_valid(course):
    root, contract, save = course
    contract['primary_concept_dispositions'].append({'concept': 'Counterexample', 'disposition': 'reference'})
    save()
    assert audit.validate(root, (1,))['errors'] == []


def test_new_primary_concept_cannot_disappear_from_lab_plan(course):
    root, _, _ = course
    book = root / 'books/vol1/01_example/example_concepts.yml'
    data = yaml.safe_load(book.read_text())
    data['concept_map']['primary_concepts'].append('Bandwidth')
    book.write_text(yaml.safe_dump(data))
    errors = audit.validate(root, (1,))['errors']
    assert any("no disposition 'Bandwidth'" in error for error in errors)


def test_unknown_source_and_unbound_concept_are_reported(course):
    root, contract, save = course
    contract['core_concepts'][0]['source_concepts'][0]['concept'] = 'Invented concept'
    contract['runtime_parts'][0]['concept_ids'] = ['not_a_concept']
    save()
    errors = audit.validate(root, (1,))['errors']
    assert any('unknown source concept' in error for error in errors)
    assert any('unknown concept binding' in error for error in errors)
    assert any('core concept has no investigation' in error for error in errors)


def test_partial_study_requires_its_evidence_boundary(course):
    root, contract, save = course
    contract['runtime_parts'][0]['coverage'] = 'partial'
    save()
    assert any('explicit evidence boundary' in error for error in audit.validate(root, (1,))['errors'])
    contract['runtime_parts'][0]['limitation'] = 'Finite trace, not a hardware measurement.'
    save()
    assert audit.validate(root, (1,))['errors'] == []


def test_disposition_cannot_claim_a_nonexistent_investigation(course):
    root, contract, save = course
    contract['primary_concept_dispositions'][0]['lab_concepts'] = ['v1_01_nonexistent']
    save()
    assert any('unknown investigation binding' in error for error in audit.validate(root, (1,))['errors'])
