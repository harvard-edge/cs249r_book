#!/usr/bin/env python3
"""Record one role's hash-bound original-figure review, or collect current readiness.

record writes only technical-review.json or learning-review.json beside --entry.
collect is read-only; production readiness requires both production-scoped roles.
Fixture readiness also accepts production-scoped reviews. Evidence must name
individual files, never directories. A verdict records the reviewer's judgment;
this tool verifies freshness and scope, not the truth of the review itself.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[3]
PROTOTYPES = Path('books/_prototypes/vol1-web-animations')
EXTENSION = Path('books/_extensions/mlsysbook-ext/figure-explore')
CATALOG = PROTOTYPES / 'animation-catalog/catalog.json'
DEFAULT_ART_INDEX = Path('/tmp/figures-animation-art-current/index.json')
ROLES = ('technical', 'learning')
SCOPES = ('fixture', 'production')
# Only enrollment immediately following a figure wrapper ID is ignored. Caption
# text, examples, source bytes, and every unrelated attribute remain authoritative.
FIGURE_START = re.compile(r'^:{3,}[ \t]*\{#fig-[^\s}]+', re.M)
ENROLLMENT = re.compile(r'''(?:[ \t]+data-explore(?:-key)?=(?:"[^"\n]*"|'[^'\n]*'))+''')


class ReviewError(ValueError):
    pass


def strict_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ReviewError(f'{path}: duplicate JSON key {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized_source(data):
    text = data.decode('utf-8')
    for match in reversed(list(FIGURE_START.finditer(text))):
        attrs = ENROLLMENT.match(text, match.end())
        if attrs:
            text = text[:attrs.start()] + text[attrs.end():]
    return text.encode('utf-8')


def local_path(value, root):
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def display_path(path, root):
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


def inside(value, root, base=None):
    path = local_path(value, base or root)
    if not path.is_relative_to(root):
        raise ReviewError(f'Artifact escapes repository: {value}')
    if not path.is_file():
        raise ReviewError(f'Missing artifact: {display_path(path, root)}')
    return path


def entry_file(value, root):
    path = local_path(value, root)
    if path.is_dir():
        path = path / 'entry.json'
    path = inside(path, root)
    if path.name != 'entry.json' or not path.parent.is_relative_to(root / PROTOTYPES):
        raise ReviewError('Entry must be entry.json in a book animation prototype folder')
    return path


def extension_asset(value, extension, root):
    if not isinstance(value, str) or not value or Path(value).name != value or '\\' in value:
        raise ReviewError(f'Expected extension-relative asset filename, got {value!r}')
    return inside(extension / value, root)


def current_artifacts(entry_path, art_index, root=ROOT, *, catalog_rows=None, artwork_rows=None):
    """Capture current authoritative files without changing enrollment or metadata."""
    root = root.resolve()
    entry_path = entry_file(entry_path, root)
    entry = strict_json(entry_path)
    key = entry.get('key')
    implementation = entry.get('implementation')
    if not isinstance(key, str) or not isinstance(implementation, dict):
        raise ReviewError('Entry requires key and implementation object')
    for field in ('name', 'script', 'stylesheet'):
        if not isinstance(implementation.get(field), str) or not implementation[field]:
            raise ReviewError(f'{key}: missing implementation {field}')
    catalog_rows = strict_json(root / CATALOG)['figures'] if catalog_rows is None else catalog_rows
    rows = [r for r in catalog_rows if r['key'] == key]
    if len(rows) != 1:
        raise ReviewError(f'{key}: expected exactly one catalog source')
    source = rows[0]['source']
    source_path = inside(source, root)
    source_bytes = source_path.read_bytes()
    figure_id = rows[0]['figure_id']
    if key != f"vol{rows[0]['volume']}:{figure_id}":
        raise ReviewError(f'{key}: catalog key/figure/volume mismatch')
    wrappers = re.findall(r'^:{3,}[ \t]*\{#' + re.escape(figure_id) + r'(?=[\s}])', source_bytes.decode('utf-8'), re.M)
    if len(wrappers) != 1:
        raise ReviewError(f'{key}: expected exactly one current source figure wrapper')
    artwork_rows = strict_json(art_index) if artwork_rows is None else artwork_rows
    artwork = artwork_rows.get(key)
    if not isinstance(artwork, dict) or artwork.get('source') != source:
        raise ReviewError(f'{key}: art index source does not match current owning source')
    # The index locates the canonical rendered artwork; never trust its cached hash.
    asset = local_path(artwork['asset'], root)
    if not asset.is_file():
        raise ReviewError(f'{key}: missing canonical artwork: {asset}')
    art = asset.read_bytes()
    kind = 'png' if art.startswith(b'\x89PNG\r\n\x1a\n') else 'svg' if asset.suffix.lower() == '.svg' else None
    if kind is None:
        raise ReviewError(f'{key}: canonical art must be SVG or PNG')
    extension = root / EXTENSION
    data_scripts = implementation.get('data_scripts', [])
    if not isinstance(data_scripts, list):
        raise ReviewError(f'{key}: data_scripts must be an array')
    modules = [implementation['script'], implementation['stylesheet']]
    # Match fixture loading aliases and bind each effective asset exactly once.
    modules += [implementation[f] for f in ('bundle', 'data_script') if implementation.get(f)]
    modules += data_scripts
    module_hashes = {}
    for value in modules:
        path = extension_asset(value, extension, root)
        module_hashes[display_path(path, root)] = digest(path.read_bytes())
    if 'shared_player' in implementation and type(implementation['shared_player']) is not bool:
        raise ReviewError(f'{key}: shared_player must be boolean')
    runtime = ['figure-explore.js', 'figure-views.js', 'figure-views.css']
    if implementation.get('shared_player'):
        runtime += ['original-player.js', 'original-player.css']
    runtime_hashes = {}
    for value in runtime:
        path = extension_asset(value, extension, root)
        runtime_hashes[display_path(path, root)] = digest(path.read_bytes())
    timing = inside('books/config/shared/html/figure-explore.yml', root)
    artifacts = {
        'entry_semantics_sha256': digest(canonical(entry)),
        'source': {'path': source, 'sha256': digest(normalized_source(source_bytes)),
                   'normalization': 'figure-wrapper enrollment attributes only'},
        'canonical_art': {'source': source, 'kind': kind, 'sha256': digest(art)},
        'module_assets': module_hashes,
        'runtime_assets': runtime_hashes,
        'timing': {'path': display_path(timing, root), 'sha256': digest(timing.read_bytes())},
    }
    return {'key': key, 'entry': display_path(entry_path, root), 'artifacts': artifacts,
            'artifact_fingerprint': digest(canonical(artifacts))}


def evidence_files(values, root):
    result = []
    seen = set()
    for value in values:
        path = local_path(value, root)
        if not path.is_file():
            raise ReviewError(f'Evidence must be an existing individual file: {value}')
        if path in seen:
            continue
        seen.add(path)
        result.append({'path': display_path(path, root), 'sha256': digest(path.read_bytes())})
    if not result:
        raise ReviewError('At least one evidence file is required')
    return result


def record_review(args, root=ROOT):
    root = root.resolve()
    entry = entry_file(args.entry, root)
    if args.role not in ROLES or args.verdict not in ('passed', 'blocked') or args.scope not in SCOPES:
        raise ReviewError('Invalid role, verdict, or review scope')
    if not isinstance(args.basis, str) or not args.basis.strip():
        raise ReviewError('Review basis must be nonempty')
    target = entry.parent / f'{args.role}-review.json'
    if target.is_symlink():
        raise ReviewError(f'Refusing symlink review destination: {target}')
    # Do not accept this record itself, the opposite role, or a self-invalidating
    # entry as evidence. Evidence reports/captures remain independent input files.
    forbidden = {entry, *(entry.parent / f'{role}-review.json' for role in ROLES)}
    evidence = evidence_files(args.evidence, root)
    if any(local_path(e['path'], root) in forbidden for e in evidence):
        raise ReviewError('Evidence must be independent of the entry and role review records')
    current = current_artifacts(entry, args.art_index, root)
    record = {'schema': 1, 'role': args.role, 'verdict': args.verdict,
              'scope': args.scope, 'basis': args.basis.strip(),
              'reviewed_at_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
              **current, 'evidence': evidence}
    # Only the caller's role destination is replaced; no catalog, manifest, source,
    # opposing-role, or readiness state is written.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix=f'.{args.role}-review-', suffix='.tmp', dir=entry.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
        os.replace(temporary, target)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    return {'record': display_path(target, root), **record}


def role_readiness(path, role, current, scope, root):
    reasons = []
    if path.is_symlink():
        return {'ready': False, 'reasons': [{'code': 'invalid-review', 'detail': 'Review record must not be a symlink'}]}
    if not path.is_file():
        return {'ready': False, 'reasons': [{'code': 'missing-review', 'role': role}]}
    try:
        record = strict_json(path)
        required = {'schema', 'role', 'verdict', 'scope', 'basis', 'reviewed_at_utc', 'key', 'entry', 'artifacts', 'artifact_fingerprint', 'evidence'}
        if not isinstance(record, dict) or set(record) != required or type(record['schema']) is not int or record['schema'] != 1:
            raise ReviewError('Expected exact review schema 1')
        if record['role'] != role or record['key'] != current['key'] or record['entry'] != current['entry']:
            raise ReviewError('Review role/key/entry identity mismatch')
        if record['verdict'] not in ('passed', 'blocked') or record['scope'] not in SCOPES or not isinstance(record['basis'], str) or not record['basis'].strip():
            raise ReviewError('Invalid verdict, scope, or basis')
        if record['artifact_fingerprint'] != digest(canonical(record['artifacts'])):
            raise ReviewError('Review artifact fingerprint is internally inconsistent')
        if record['artifact_fingerprint'] != current['artifact_fingerprint']:
            changed = [field for field in current['artifacts'] if record['artifacts'].get(field) != current['artifacts'][field]]
            reasons.append({'code': 'stale-artifacts', 'changed': changed})
        if record['verdict'] != 'passed':
            reasons.append({'code': 'blocked-verdict', 'basis': record['basis']})
        if scope == 'production' and record['scope'] != 'production':
            reasons.append({'code': 'insufficient-scope', 'required': scope, 'actual': record['scope']})
        if not isinstance(record['evidence'], list) or not record['evidence']:
            raise ReviewError('Review requires evidence files')
        for evidence in record['evidence']:
            if not isinstance(evidence, dict) or set(evidence) != {'path', 'sha256'}:
                raise ReviewError('Invalid evidence file binding')
            file = local_path(evidence['path'], root)
            if not file.is_file():
                reasons.append({'code': 'missing-evidence', 'path': evidence['path']})
            elif digest(file.read_bytes()) != evidence['sha256']:
                reasons.append({'code': 'stale-evidence', 'path': evidence['path']})
        return {'ready': not reasons, 'scope': record['scope'], 'verdict': record['verdict'],
                'artifact_fingerprint': record['artifact_fingerprint'], 'reasons': reasons}
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {'ready': False, 'reasons': [{'code': 'invalid-review', 'detail': str(error)}]}


def collect_readiness(entries, art_index, scope='production', root=ROOT):
    """Read-only readiness, with separate stale/missing/blocked reasons per role."""
    if scope not in SCOPES:
        raise ReviewError('Invalid requested scope')
    root = root.resolve()
    paths = entries or sorted((root / PROTOTYPES).glob('original-v*/entry.json'))
    # One read of the shared provenance indexes per collection. Per-entry source
    # and runtime bytes are still read afresh; no cross-invocation hash cache.
    catalog_rows = strict_json(root / CATALOG)['figures']
    artwork_rows = strict_json(art_index)
    results = []
    for value in paths:
        try:
            entry = entry_file(value, root)
            current = current_artifacts(entry, art_index, root, catalog_rows=catalog_rows, artwork_rows=artwork_rows)
            roles = {role: role_readiness(entry.parent / f'{role}-review.json', role, current, scope, root) for role in ROLES}
            results.append({**current, 'ready': all(r['ready'] for r in roles.values()), 'roles': roles, 'reasons': []})
        except (OSError, ValueError, TypeError, KeyError) as error:
            results.append({'entry': str(value), 'ready': False, 'roles': {},
                            'reasons': [{'code': 'invalid-current-artifacts', 'detail': str(error)}]})
    keys = [r['key'] for r in results if r.get('key')]
    duplicates = {key for key in keys if keys.count(key) > 1}
    for result in results:
        if result.get('key') in duplicates:
            result['ready'] = False
            result['reasons'].append({'code': 'duplicate-entry-key'})
    return {'schema': 1, 'required_scope': scope,
            'accepted_keys': sorted(r['key'] for r in results if r['ready']), 'entries': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    record = commands.add_parser('record', help='Write only one role review beside entry.json')
    record.add_argument('--entry', type=Path, required=True)
    record.add_argument('--role', choices=ROLES, required=True)
    record.add_argument('--verdict', choices=('passed', 'blocked'), required=True)
    record.add_argument('--scope', choices=SCOPES, required=True)
    record.add_argument('--basis', required=True)
    record.add_argument('--evidence', action='append', type=Path, required=True)
    record.add_argument('--art-index', type=Path, default=DEFAULT_ART_INDEX)
    collect = commands.add_parser('collect', help='Read-only readiness JSON; defaults to original-v*/entry.json')
    collect.add_argument('--entry', action='append', type=Path)
    collect.add_argument('--scope', choices=SCOPES, default='production')
    collect.add_argument('--art-index', type=Path, default=DEFAULT_ART_INDEX)
    args = parser.parse_args()
    try:
        result = record_review(args) if args.command == 'record' else collect_readiness(args.entry, args.art_index, args.scope)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
