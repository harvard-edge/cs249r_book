#!/usr/bin/env python3
"""Validate reviewed figure lessons and compile deterministic web registries.

No manuscript changes unless --enroll is explicitly supplied. Static and blocked
figures remain unenrolled. Run --require-complete for release coverage checks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
EXT = ROOT / 'books/_extensions/mlsysbook-ext/figure-explore'
CATALOG = ROOT / 'books/_prototypes/vol1-web-animations/animation-catalog/catalog.json'
ATTR = re.compile(r'\s+data-explore(?:-key)?="[^"]*"')


def normalized_source(text):
    return ATTR.sub('', text)


def sha1(data):
    return hashlib.sha1(data).hexdigest()


def structural_svg(data):
    """Ignore metadata and generated identifiers, retain geometry and labels."""
    text = data.decode('utf-8')
    text = re.sub(r'<metadata\b[^>]*>[\s\S]*?</metadata\s*>', '', text)
    identifiers = {}
    def replace_id(match):
        original = match.group(2)
        identifiers.setdefault(original, f'id{len(identifiers)}')
        return 'id=' + match.group(1) + identifiers[original] + match.group(1)
    text = re.sub(r'\bid=(["\'])([^"\']+)\1', replace_id, text)
    text = re.sub(r'#([A-Za-z_][\w:.-]*)', lambda match: '#'+identifiers.get(match.group(1), match.group(1)), text)
    return sha1(text.encode('utf-8'))



def validate(row, original):
    key = row['key']
    for field in ('figure_id', 'source'):
        if row.get(field) != original[field]:
            raise ValueError(f'{key}: {field} does not match catalog')
    if row.get('status') not in ('ready', 'static', 'specialized', 'blocked'):
        raise ValueError(f'{key}: invalid status')
    if row['status'] != 'ready':
        return
    if row.get('artwork_reviewed') is not True:
        raise ValueError(f'{key}: ready without artwork review')
    if not row.get('script'):
        raise ValueError(f'{key}: highlight-only tours are archived; ready lessons require a stateful implementation')
    for field in ('name', 'script', 'stylesheet'):
        if not row.get(field):
            raise ValueError(f'{key}: missing {field}')


def figure_match(text, figure):
    # Match only the wrapper's opening identifier. Captions may contain quoted
    # braces (LaTeX, inline Python); a closing-brace regex is not an attribute parser.
    matches = list(re.finditer(r'^:{3,}\s*\{#' + re.escape(figure) + r'(?=[\s}])', text, re.M))
    if len(matches) != 1:
        raise ValueError(f'{figure}: expected one exact figure wrapper, found {len(matches)}')
    return matches[0]


def enrollment_attributes(text, figure):
    match = figure_match(text, figure)
    # Enrollment always sits immediately after the ID, before original attributes.
    return re.match(r'(?:\s+data-explore(?:-key)?="[^"]*")*', text[match.end():]).group()


def compile_registry(args):
    catalog = {r['key']: r for r in json.loads(CATALOG.read_text())['figures']}
    artwork = json.loads(Path(args.art_index).read_text())
    rows = {}
    invalid = []
    for path in sorted((EXT / 'lessons').glob('vol*-*.json')):
        for row in json.loads(path.read_text()):
            key = row['key']
            if key not in catalog or key in rows:
                raise ValueError(f'{path.name}: unknown or duplicate key {key}')
            try:
                validate(row, catalog[key])
            except ValueError as error:
                if args.require_complete:
                    raise
                invalid.append({'key': key, 'file': path.name, 'error': str(error)})
                continue
            rows[key] = row
    for path in sorted((EXT / 'lessons').glob('specialized-vol*.json')):
        for row in json.loads(path.read_text()):
            key = row['key']
            if key not in catalog:
                raise ValueError(f'{path.name}: unknown key {key}')
            if key in rows and rows[key]['status'] != 'specialized':
                raise ValueError(f'{key}: specialized replacement not reserved')
            validate(row, catalog[key])
            rows[key] = row
    for key, name, script in [('vol1:fig-tensor-memory-layout', 'tensor-memory-layout', 'tensor-memory-layout'), ('vol1:fig-tiling-diagram', 'matrix-tiling', 'matrix-tiling')]:
        rows[key] = {**catalog[key], 'status': 'ready', 'artwork_reviewed': True, 'name': name, 'script': script+'.js', 'stylesheet': script+'.css'}
    missing = sorted(set(catalog) - set(rows))
    unresolved = sorted(k for k, r in rows.items() if r['status'] == 'specialized')
    if args.require_complete and (missing or unresolved):
        raise ValueError(f'Incomplete coverage: missing={missing}, unresolved specialized={unresolved}')
    registry, volumes, source_cache = {}, {n: {} for n in range(1,5)}, {}
    for key, row in sorted(rows.items()):
        if row['status'] != 'ready':
            continue
        original = catalog[key]
        source = original['source']
        path = ROOT / source
        text = source_cache.setdefault(source, path.read_text())
        figure_match(text, original['figure_id'])
        art = artwork.get(key)
        if not art or not Path(art.get('asset', '')).is_file():
            raise ValueError(f'{key}: canonical art missing')
        asset = Path(art['asset'])
        raw = asset.read_bytes()
        from PIL import Image
        with Image.open(art['image']) as im:
            width, height = im.size
        specialized = bool(row.get('script'))
        entry = dict(name=row['name'] if specialized else 'guided-figure', script=row['script'] if specialized else 'guided-figure.js', stylesheet=row['stylesheet'] if specialized else 'guided-figure.css', source=source, source_sha1=sha1(normalized_source(text).encode()), figure_id=original['figure_id'], volume=original['volume'], art_sha1=sha1(raw), art_kind=asset.suffix.lstrip('.').lower(), art_width=width, art_height=height)
        if asset.suffix.lower() == '.svg':
            entry['art_structural_sha1'] = structural_svg(raw)
        for name in ('script', 'stylesheet'):
            if specialized and not (EXT / entry[name]).is_file():
                raise ValueError(f'{key}: missing {entry[name]}')
        registry[key] = entry
        if not specialized:
            lesson = {k: v for k, v in row.items() if k not in ('review_notes',)}
            lesson.update({k: v for k,v in entry.items() if k.startswith('art_')})
            volumes[original['volume']][key] = lesson
    outputs = {EXT/'figure-registry.json': json.dumps(registry, indent=2, ensure_ascii=False)+'\n'}
    for volume, lessons in volumes.items():
        outputs[EXT/f'guided-lessons-vol{volume}.js'] = '// Generated by compile_figure_explore.py; edit lessons/*.json.\nwindow.MLSysGuidedLessons = window.MLSysGuidedLessons || {};\nObject.assign(window.MLSysGuidedLessons, '+json.dumps(lessons, ensure_ascii=False, separators=(',', ':'))+');\n'
    if args.enroll or args.check:
        for original in catalog.values():
            source = original['source']
            source_cache.setdefault(source, (ROOT/source).read_text())
    if args.enroll:
        for source, text in source_cache.items():
            updated = normalized_source(text)
            for key, original in catalog.items():
                if original['source'] != source:
                    continue
                match = figure_match(updated, original['figure_id'])
                wrapper = match.group()
                if key in registry:
                    entry = registry[key]
                    wrapper += f' data-explore="{entry["name"]}" data-explore-key="{key}"'
                updated = updated[:match.start()]+wrapper+updated[match.end():]
            if normalized_source(updated) != normalized_source(text):
                raise AssertionError(f'{source}: enrollment changed static content')
            if updated != text:
                (ROOT/source).write_text(updated)
    for path, content in outputs.items():
        if args.check:
            if not path.is_file() or path.read_text() != content:
                raise ValueError(f'Outdated generated output: {path.relative_to(ROOT)}')
        else:
            path.write_text(content)
    if args.check:
        for key, original in catalog.items():
            wrapper = enrollment_attributes((ROOT/original['source']).read_text(), original['figure_id'])
            if key in registry:
                entry = registry[key]
                if f'data-explore="{entry["name"]}"' not in wrapper or f'data-explore-key="{key}"' not in wrapper:
                    raise ValueError(f'{key}: missing or wrong enrollment')
            elif ATTR.search(wrapper):
                raise ValueError(f'{key}: unready figure has stale enrollment')
    report = {'invalid': invalid, 'catalog_total': len(catalog), 'registered': len(registry), 'status_counts': dict(Counter(r['status'] for r in rows.values())), 'missing': missing, 'unresolved_specialized': unresolved, 'blocked': [k for k,r in rows.items() if r['status']=='blocked'], 'static': [k for k,r in rows.items() if r['status']=='static'], 'by_volume': dict(Counter(str(v['volume']) for v in registry.values()))}
    Path(args.report).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--art-index', default='/tmp/mlsys-figure-art/index.json')
    parser.add_argument('--report', default='/tmp/figure-explore-compile-report.json')
    parser.add_argument('--enroll', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--partial', action='store_true', help='Compile available reviewed lessons (the default).')
    args = parser.parse_args()
    try:
        compile_registry(args)
    except (ValueError, KeyError) as error:
        parser.exit(1, str(error)+'\n')

if __name__ == '__main__':
    main()
