#!/usr/bin/env python3
"""Compile student-visible learning contracts from chapter-owned lab maps.

Generated package JSON supports native and browser notebooks without embedding
second authored copies of goals or experiment narratives in notebook code.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import tempfile
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'labs'))
sys.path.insert(0, str(ROOT / 'mlsysim'))
from mlsysbook_labs.learning_paths import validate_learning_path


def compile_volume(number: int) -> dict:
    folder = ROOT / f'labs/vol{number}/concepts'
    entries = []
    for source in sorted(folder.glob('*_concepts.yml')):
        document = yaml.safe_load(source.read_text())
        if not isinstance(document, dict) or 'lab_concept_map' not in document:
            raise ValueError(f'{source}: needs lab_concept_map')
        contract = document['lab_concept_map']
        if not contract.get('runtime_parts') or not contract.get('student_goals'):
            raise ValueError(f'{source}: missing student_goals/runtime_parts')
        entries.append((int(contract['chapter']), source, contract))
    entries.sort(key=lambda item: item[0])
    if not entries:
        raise ValueError(f'No concept maps in {folder}')
    if len({n for n, _, _ in entries}) != len(entries):
        raise ValueError(f'Conflicting chapter identities in {folder}')
    compiled = []
    for index, (chapter, source, contract) in enumerate(entries):
        narrative = contract.get('narrative', {})
        parts = []
        for part_index, item in enumerate(contract['runtime_parts']):
            ids = item.get('concept_ids', [])
            if not ids and chapter != 0 and item.get('coverage') not in {'reference', 'extension'}:
                raise ValueError(f'{source}: {item.get("part")} has no concept binding')
            concept_lookup = {c['id']: c for c in contract.get('core_concepts', [])}
            prompts = [concept_lookup[c].get('explanation_prompt', '') for c in ids if c in concept_lookup]
            reversals = [concept_lookup[c].get('reversal_probe', '') or concept_lookup[c].get('reversal_condition', '')
                         for c in ids if c in concept_lookup]
            parts.append(dict(
                id=item['part'], label=item['title'], concept_id=', '.join(ids) or ('Interface orientation; no textbook concept ownership' if chapter == 0 else 'Reference extension; no planned-concept coverage claimed'),
                question=item['question'], goal=item.get('goal') or contract['student_goals'][part_index],
                artifact=item.get('artifact') or 'A saved comparison, causal explanation, and remaining uncertainty.',
                scope=item.get('scope') or 'A chapter concept investigated under the declared study conditions.',
                coverage=item.get('coverage', 'unreviewed'), limitation=item.get('limitation', ''), next=item.get('next', ''),
                explanation_prompt=item.get('explanation_prompt') or ' '.join(dict.fromkeys(p for p in prompts if p)),
                reversal_probe=item.get('reversal_probe') or ' '.join(dict.fromkeys(p for p in reversals if p)),
            ))
        for part_index, part in enumerate(parts):
            if not part.get('next'):
                part['next'] = (parts[part_index + 1]['question'] if part_index + 1 < len(parts)
                                else 'Next, use your saved comparisons in the team discussion.')
            if not part.get('limitation'):
                part.pop('limitation', None)
        following = entries[index + 1][2] if index + 1 < len(entries) else None
        next_lab = ({'title': following['title'],
                     'question': following.get('driving_question') or following['chapter_goal'],
                     'notebook': following['source_notebook']} if following else
                    {'title': 'Review the completed volume',
                     'question': 'Which decisions remain supported, which need new evidence, and what would reverse your recommendation?'})
        result = dict(
            schema_version=1, volume=number, chapter=chapter, title=contract['title'],
            notebook=contract['source_notebook'], goal=contract['chapter_goal'],
            starting_point=narrative.get('previously') or 'Supplied starting case. No compatible earlier student evidence is assumed.',
            prerequisites=contract.get('prerequisites', ['Read the chapter sections associated with the concepts below.']),
            learning_objectives=contract['student_goals'], parts=parts,
            deliverable=narrative.get('submission') or contract.get('discussion_contract') or
                'One integrated discussion supported by comparisons: a decision or hold, causal explanation, rejected alternative, uncertainty, and reversal condition.',
            next_part=parts[0]['question'], next_lab=next_lab,
            chapter_source=contract.get('source_chapter_concept_map', ''),
            concept_map_source=str(source.relative_to(ROOT)), status=contract.get('status', 'under development'),
        )
        scene = narrative.get('situation')
        if scene:
            result['narrative'] = {'scene': scene, 'carry_forward': narrative.get('next') or contract.get('handoff') or next_lab['question'],
                                   'scope': 'Curriculum narrative; student evidence and study validity remain separately identified.'}
        validate_learning_path(result)
        compiled.append(result)
    return dict(schema_version=1, volume=number, generated_from=f'labs/vol{number}/concepts/*_concepts.yml', chapters=compiled)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--volumes', nargs='+', type=int, default=[1, 2, 3, 4])
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    errors = []
    for number in args.volumes:
        try:
            payload = compile_volume(number)
            output = ROOT / f'labs/mlsysbook_labs/data/learning_paths/vol{number}.json'
            text = json.dumps(payload, indent=2, ensure_ascii=False) + '\n'
            if args.check:
                if not output.exists() or output.read_text() != text:
                    errors.append(f'vol{number}: packaged learning contract is missing or stale')
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                if not output.exists() or output.read_text() != text:
                    # Readers must see either complete contract, including when
                    # a preview/test process overlaps source regeneration.
                    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent,
                                                     prefix=f'.{output.name}.', delete=False) as pending:
                        pending.write(text)
                    Path(pending.name).replace(output)
            print(f'Volume {number}: {len(payload["chapters"])} chapter contracts compiled')
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(str(exc))
    for error in errors:
        print(error, file=sys.stderr)
    return int(bool(errors))

if __name__ == '__main__':
    raise SystemExit(main())
