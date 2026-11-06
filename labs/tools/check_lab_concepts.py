#!/usr/bin/env python3
"""Validate lab concept ownership and complete paradigm learning bindings.

Checks authoring consistency and provenance, not model fidelity or learning.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONCEPTS = ROOT / 'labs/vol1/concepts'
DATA = ROOT / 'mlsysim/mlsysim/engine/data'


def load_contracts(root: Path = ROOT) -> dict[int, dict]:
    return {data['chapter']: data for path in sorted((root / 'labs/vol1/concepts').glob('*_concepts.yml'))
            for data in [yaml.safe_load(path.read_text())['lab_concept_map']]}


def validate(root: Path = ROOT) -> dict:
    errors: list[str] = []
    contracts = load_contracts(root)
    data = root / 'mlsysim/mlsysim/engine/data'
    catalog = json.loads((data / 'deployment_journeys.json').read_text())
    chapters = {entry['chapter']: entry for entry in catalog['volume_storyboard']['chapters']}
    shared = yaml.safe_load((root / 'labs/vol1/concepts/schema.yml').read_text())
    if not shared.get('north_star') or not shared.get('learning_cycle'):
        errors.append('Shared contract requires a north star and learning cycle')
    narrative = shared.get('narrative_contract', {})
    if [stage.get('stage') for stage in narrative.get('stages', [])] != [
        'Scene', 'Briefing', 'Choice', 'Feedback', 'Explanation', 'Handoff'
    ]:
        errors.append('Narrative contract must connect briefing, choice, consequence feedback, and handoff')
    for field in ('summary', 'before_choice', 'author_fields_not_to_reveal_as_answers',
                  'feedback_rules', 'progression', 'visible_from_start', 'disclosure_rule'):
        if not narrative.get(field):
            errors.append(f'Narrative contract requires {field}')
    if set(contracts) != set(chapters):
        errors.append('Concept maps must cover the complete ordered volume storyboard')
    for n, contract in contracts.items():
        prefix = f'Lab {n:02d}'
        if contract.get('schema_version') != 1 or contract.get('status') != 'proposed learning design':
            errors.append(f'{prefix}: missing schema/status')
        if not (root / contract['source_notebook']).is_file():
            errors.append(f'{prefix}: source notebook missing')
        items = contract['core_concepts']
        expected = [f'v1_{n:02d}_{chr(ord("A") + i)}' for i in range(len(chapters[n]['investigations']))]
        if [item['id'] for item in items] != expected:
            errors.append(f'{prefix}: concepts do not align with intended investigations')
        for item in items:
            for field in ('title', 'investigation_title', 'learning_claim', 'misconception', 'tradeoff', 'student_action',
                          'evidence_to_inspect', 'explanation_prompt', 'assessment', 'scope'):
                if not isinstance(item.get(field), str) or not item[field].strip():
                    errors.append(f'{item["id"]}: missing {field}')
            for ref in item['source_concepts']:
                path = root / ref['concept_map']
                source = yaml.safe_load(path.read_text())['concept_map']
                if ref['concept'] not in source['primary_concepts']:
                    errors.append(f'{item["id"]}: unknown source concept {ref["concept"]}')
                if not path.parent.name.startswith(f'{ref["chapter"]:02d}_'):
                    errors.append(f'{item["id"]}: source chapter mismatch')
                if n not in (0, 16) and ref['chapter'] != n:
                    errors.append(f'{item["id"]}: claims another chapter ownership')
                if n == 16 and ref['chapter'] >= n:
                    errors.append(f'{item["id"]}: conclusion must integrate earlier ownership')
            if n == 0 and item['source_concepts']:
                errors.append(f'{item["id"]}: orientation cannot claim chapter concepts')
        source_path = contract.get('source_chapter_concept_map')
        source_primary = set(yaml.safe_load((root / source_path).read_text())['concept_map']['primary_concepts']) if source_path else set()
        dispositions = {row['concept']: row for row in contract['primary_concept_dispositions']}
        if set(dispositions) != source_primary:
            errors.append(f'{prefix}: every primary concept needs a disposition')
        for name, row in dispositions.items():
            actual = [item['id'] for item in items if any(ref['chapter'] == n and ref['concept'] == name for ref in item['source_concepts'])]
            if row['disposition'] != 'interactive' or row['lab_concepts'] != actual or not actual:
                errors.append(f'{prefix}: disposition for {name} differs from the lab concept mapping')
    tracks = {}
    for story in catalog['scenarios']:
        track = story['track_id'];path = data / 'deployment_learning' / f'{track}.json'
        binding = json.loads(path.read_text())
        if (binding.get('track_id'), binding.get('scenario_id')) != (track, story['scenario_id']):
            errors.append(f'{track}: scenario identity mismatch')
        if binding.get('source_catalog_version') != catalog['version']:
            errors.append(f'{track}: stale narrative catalog version')
        if binding.get('status') != 'proposed learning design' or binding.get('provenance', {}).get('role') != 'authored scenario learning design':
            errors.append(f'{track}: missing design/provenance boundary')
        if [item['chapter'] for item in binding['episodes']] != list(chapters):
            errors.append(f'{track}: missing or unordered episodes')
        for episode in binding['episodes']:
            n = episode['chapter'];expected = [item['id'] for item in contracts[n]['core_concepts']]
            if [item['concept_id'] for item in episode['concepts']] != expected:
                errors.append(f'{track} {n:02d}: concept applications missing, reordered, or renamed')
            if not episode.get('discussion_prompt', '').strip():
                errors.append(f'{track} {n:02d}: integrated discussion prompt missing')
            for item in episode['concepts']:
                for field in ('experiment', 'evidence', 'transfer_question'):
                    if not isinstance(item.get(field), str) or not item[field].strip():
                        errors.append(f'{track} {item["concept_id"]}: missing {field}')
            selected = contracts[n]['scenario_bindings'].get(track, {})
            if selected.get('scenario_id') != story['scenario_id'] or selected.get('episode') != n or root / selected.get('source', '') != path:
                errors.append(f'{track} {n:02d}: concept-file scenario binding disagrees with MLSysIM')
        tracks[track] = len(binding['episodes'])
    return {'errors': errors, 'labs': len(contracts), 'concepts': sum(len(c['core_concepts']) for c in contracts.values()),
            'tracks': tracks, 'applications': sum(len(ep['concepts']) for story in catalog['scenarios']
                for ep in json.loads((data / 'deployment_learning' / f'{story["track_id"]}.json').read_text())['episodes']),
            'scope': 'Source identities, chapter ownership, metadata, coverage, and binding completeness; not learning or numerical validation.'}


if __name__ == '__main__':
    report = validate()
    for error in report['errors']:
        print(error, file=sys.stderr)
    print(f"Validated {report['labs']} lab maps, {report['concepts']} distilled concepts, "
          f"{len(report['tracks'])} paradigms and {report['applications']} concept applications; "
          f"{len(report['errors'])} authoring errors.")
    raise SystemExit(bool(report['errors']))
