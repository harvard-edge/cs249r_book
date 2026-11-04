#!/usr/bin/env python3
"""Audit chapter ownership and complete experiment routes across all lab volumes.

This validates authored contracts, not model fidelity or classroom learning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def validate(root: Path = ROOT, volumes: tuple[int, ...] = (1, 2, 3, 4)) -> dict:
    errors: list[str] = []
    summaries = {}
    claimed_notebooks: set[str] = set()
    source_cache = {}

    def source_names(relative):
        if relative not in source_cache:
            path = root / relative
            if not path.is_file():
                raise ValueError(f'missing chapter concept map {relative}')
            source = yaml.safe_load(path.read_text())['concept_map']
            source_cache[relative] = (set(source.get('primary_concepts', [])),
                                     set(source.get('secondary_concepts', [])))
        return source_cache[relative]

    for volume in volumes:
        entries = sorted((root / f'labs/vol{volume}/concepts').glob('*_concepts.yml'))
        chapters = set()
        part_total = 0
        concept_total = 0
        for path in entries:
            label = str(path.relative_to(root))
            try:
                contract = yaml.safe_load(path.read_text())['lab_concept_map']
                chapter = contract['chapter']
                if chapter in chapters:
                    errors.append(f'{label}: duplicate chapter identity {chapter}')
                chapters.add(chapter)
                notebook = contract['source_notebook']
                if notebook in claimed_notebooks:
                    errors.append(f'{label}: duplicate notebook ownership {notebook}')
                claimed_notebooks.add(notebook)
                if not notebook.startswith(f'labs/vol{volume}/') or not (root / notebook).is_file():
                    errors.append(f'{label}: missing or wrong-volume notebook {notebook}')
                for field in ('chapter_goal', 'student_goals', 'runtime_parts'):
                    if not contract.get(field):
                        errors.append(f'{label}: missing {field}')
                concepts = contract.get('core_concepts', [])
                ids = [c['id'] for c in concepts]
                if len(set(ids)) != len(ids):
                    errors.append(f'{label}: duplicate core concept ids')
                parts = contract['runtime_parts']
                part_ids = [p['part'] for p in parts]
                if len(set(part_ids)) != len(part_ids):
                    errors.append(f'{label}: duplicate investigation part')
                bindings = {identifier for p in parts for identifier in p.get('concept_ids', [])}
                for identifier in bindings - set(ids):
                    errors.append(f'{label}: unknown concept binding {identifier}')
                for identifier in set(ids) - bindings:
                    errors.append(f'{label}: core concept has no investigation {identifier}')
                for part in parts:
                    for field in ('title', 'question', 'goal', 'coverage'):
                        if not part.get(field):
                            errors.append(f'{label}: part {part["part"]} missing {field}')
                    if part.get('coverage') in ('partial', 'reference') and not part.get('limitation'):
                        errors.append(f'{label}: part {part["part"]} needs an explicit evidence boundary')
                for concept in concepts:
                    for field in ('learning_claim', 'tradeoff', 'student_action', 'evidence_to_inspect', 'explanation_prompt'):
                        if not concept.get(field) and not (chapter == 0 and field == 'explanation_prompt'):
                            errors.append(f'{label}: {concept["id"]} missing {field}')
                    for source in concept.get('source_concepts', []):
                        primary, secondary = source_names(source['concept_map'])
                        if source['concept'] not in primary | secondary:
                            errors.append(f'{label}: unknown source concept {source["concept"]!r}')
                chapter_source = contract.get('source_chapter_concept_map')
                if chapter_source:
                    primary, secondary = source_names(chapter_source)
                    dispositions = contract.get('primary_concept_dispositions', [])
                    disposition_names = [d['concept'] for d in dispositions]
                    if len(set(disposition_names)) != len(disposition_names):
                        errors.append(f'{label}: duplicate primary concept disposition')
                    for name in primary - set(disposition_names):
                        errors.append(f'{label}: primary chapter concept has no disposition {name!r}')
                    for name in set(disposition_names) - (primary | secondary):
                        errors.append(f'{label}: disposition names unknown chapter concept {name!r}')
                    for disposition in dispositions:
                        linked_parts = list(disposition.get('parts', []))
                        if disposition.get('part'):
                            linked_parts.append(disposition['part'])
                        linked_concepts = disposition.get('lab_concepts', [])
                        if set(linked_parts) - set(part_ids) or set(linked_concepts) - set(ids):
                            errors.append(f'{label}: disposition has unknown investigation binding for {disposition["concept"]!r}')
                        if disposition.get('disposition') in {'interactive', 'investigated', 'core investigation'} and not (linked_parts or linked_concepts):
                            errors.append(f'{label}: interactive disposition has no investigation for {disposition["concept"]!r}')
                        if not disposition.get('disposition'):
                            errors.append(f'{label}: empty disposition for {disposition["concept"]!r}')
                elif chapter != 0:
                    errors.append(f'{label}: missing chapter concept source')
                part_total += len(parts)
                concept_total += len(concepts)
            except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
                errors.append(f'{label}: {exc}')
        notebooks = {p.relative_to(root).as_posix() for p in (root / f'labs/vol{volume}').glob('lab_*.py')}
        missing = notebooks - claimed_notebooks
        if missing:
            errors.append(f'Volume {volume}: notebooks without concept maps: {sorted(missing)}')
        if not entries:
            errors.append(f'Volume {volume}: no lab concept maps')
        summaries[str(volume)] = {'labs': len(entries), 'parts': part_total, 'concepts': concept_total}
    return {'volumes': summaries, 'errors': errors,
            'scope': 'Source ownership, declared goals, evidence boundaries and concept/part coverage; not numerical fidelity or learning validation.'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--volumes', nargs='+', type=int, choices=(1, 2, 3, 4), default=[1, 2, 3, 4])
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    result = validate(volumes=tuple(args.volumes))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for volume, summary in result['volumes'].items():
            print(f'Volume {volume}: {summary["labs"]} labs, {summary["parts"]} parts, {summary["concepts"]} concepts')
        for error in result['errors']:
            print(error)
        print(f'{len(result["errors"])} authoring errors')
    return bool(result['errors'])


if __name__ == '__main__':
    raise SystemExit(main())
