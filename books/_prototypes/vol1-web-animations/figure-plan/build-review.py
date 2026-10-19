"""Render the current chapter review while preserving the original audit inputs.

--build-map accepts a JSON map from volume numbers to HTML output directories,
for example {"1": "books/_build/parallel/RUN/vol1-html/output"}. A value may
instead be {"root": "...", "link_base": "https://..."}. No historical build
is selected automatically. --build-root configures one parallel run for all volumes.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import date
from pathlib import Path
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXT = ROOT / 'books/_extensions/mlsysbook-ext/figure-explore'


def canonical(volume):
    config = (ROOT / f'books/config/_quarto-pdf-vol{volume}.yml').read_text()
    return ['books/' + s for s in re.findall(r'^\s*- (vol\d/\d[^\s#]+\.qmd)', config, re.M)]


def build():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', type=Path, help='Explicit parallel run root for all volumes.')
    parser.add_argument('--build-map', type=Path, help='JSON map of volume to HTML output root; overrides --build-root per volume.')
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    roots = {str(v): str(args.build_root / f'vol{v}-html/output') for v in range(1, 5)} if args.build_root else {}
    if args.build_map:
        roots.update(json.loads(args.build_map.read_text()))
    configured = {}
    for volume, value in roots.items():
        value = {'root': value} if isinstance(value, str) else value
        root = Path(value['root'])
        configured[str(volume)] = {**value, 'root': str(root if root.is_absolute() else ROOT / root)}
    catalog = json.loads((HERE.parent / 'animation-catalog/catalog.json').read_text())
    figures = catalog['figures']
    by_key = {f['key']: f for f in figures}
    selection = json.loads((EXT / 'mechanisms/selection.json').read_text())
    registry = json.loads((EXT / 'figure-registry.json').read_text())
    assert {s['key'] for s in selection} == set(registry), 'Selection/registry mismatch'
    lessons = {lesson['key']: lesson for path in sorted((EXT / 'mechanisms').glob('vol*-*.json')) for lesson in json.loads(path.read_text())}
    static = json.loads((HERE / 'static-delivery.json').read_text())
    delivered = {f['key']: f for f in static['figures']}
    chapters, methods, cache = [], [], {}

    def route(volume, source, figure_id=None, mechanism=False):
        config = configured.get(str(volume))
        if not config:
            return ''
        relative = Path(source).relative_to('books').with_suffix('.html')
        chapter = Path(config['root']) / relative
        if chapter not in cache:
            cache[chapter] = BeautifulSoup(chapter.read_text(), 'html.parser') if chapter.is_file() else None
        soup = cache[chapter]
        if soup is None:
            return ''
        if figure_id:
            node = soup.find(id=figure_id)
            if node is None or (mechanism and 'mechanism-scene' not in str(node.get('data-mlsys-explore', ''))):
                return ''
        url = config['link_base'].rstrip('/') + '/' + relative.as_posix() if config.get('link_base') else os.path.relpath(chapter, output)
        return url + ('#' + figure_id if figure_id else '')

    for volume in range(1, 5):
        audit = json.loads((HERE / f'vol{volume}.json').read_text())
        methods.append({'volume': volume, 'historical_source_review_method': audit['method']})
        expected = canonical(volume)
        assert set(expected) == {c['source'] for c in audit['chapters']}, f'Volume {volume} coverage mismatch'
        for source in expected:
            c = next(c for c in audit['chapters'] if c['source'] == source)
            lines = (ROOT / source).read_text().splitlines()
            c.update(volume=volume, number=int(Path(source).parent.name.split('_')[0]), source_sha256=hashlib.sha256((ROOT / source).read_bytes()).hexdigest())
            c['historical_animation_candidates'] = c.pop('animation_candidates')
            for need in c['needs'] + c.get('reuse_opportunities', []):
                key = f"vol{volume}:{need.get('implemented_figure_id', '')}"
                need['implemented_delivery'] = delivered.get(key)
                need['implementation_status'] = ('Implemented; static source, PNG, actual HTML, and full-volume PDF reviewed.' if key in delivered else 'Implemented in current source; outside the static delivery manifest review scope.' if key in by_key else 'Original audit item; current implementation not recorded.')
                need['audit_line'] = need.get('line', 1)
                need['line'] = need.get('line', 1)
                need['line'] = next((i + 1 for i, line in enumerate(lines) if need.get('implemented_figure_id') and need['implemented_figure_id'] in line), need['line'])
                need['source_excerpt'] = '\n'.join(f'{i + 1}: {lines[i]}' for i in range(max(0, need['line'] - 2), min(len(lines), need['line'] + 2)))[:2200]
            c['built_href'] = route(volume, source)
            c['figures'] = [{**{k: f[k] for k in ['key', 'figure_id', 'title', 'line', 'section', 'assets', 'renderer']}, 'selected': f['key'] in registry, 'status': f['actual_status'], 'review_scope': f.get('review_scope'), 'visual_verification': f['visual_verification'], 'href': route(volume, source, f['figure_id'], f['key'] in registry)} for f in figures if f['source'] == source]
            c['mechanisms'] = []
            for selected in selection:
                if selected['source'] != source:
                    continue
                lesson = lessons[selected['key']]
                current = by_key[selected['key']]
                c['mechanisms'].append({**selected, 'title': lesson['title'], 'scope': lesson['scope'], 'status': current['actual_status'], 'review_scope': current.get('review_scope'), 'runtime_verification': current['runtime_verification'], 'visual_verification': current['visual_verification'], 'steps': [{'title': s['title'], 'explanation': s['explanation'], 'result': s['snapshot'].get('result', '')} for s in lesson['steps']], 'variants': [{'id': v.get('id'), 'title': v.get('title') or v.get('label') or v.get('id')} for v in lesson.get('variants', [])], 'href': route(volume, source, selected['figure_id'], True)})
            c['static_deliveries'] = [f for f in static['figures'] if f['source'] == source]
            chapters.append(c)
    covered = {c['source'] for c in chapters}
    report = {'date': date.today().isoformat(), 'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'scope': 'All numbered chapters in the four canonical PDF configurations. Frontmatter and appendix figures are included in the complete catalog; their visual needs were not audited in this chapter pass.', 'review_basis': 'The original chapter source audits are preserved in vol1.json through vol4.json. Current mechanism enrollment comes from selection.json, with authored lesson data and compiler status. Static delivery review and mechanism approval are separate. No students were recruited and no learning gains were measured.', 'mechanism_review_boundary': catalog['metadata']['review_limit'], 'review_record': catalog['metadata'].get('review_record'), 'configured_builds': {v: {**c, 'root': os.path.relpath(c['root'], ROOT)} for v, c in configured.items()}, 'static_delivery': static, 'methods': methods, 'chapters': chapters, 'counts': {'chapters': len(chapters), 'chapter_figures': sum(len(c['figures']) for c in chapters), 'selected_mechanisms': sum(len(c['mechanisms']) for c in chapters), 'static_deliveries': len(delivered), 'resolved_needs': sum(bool(n['implemented_delivery']) for c in chapters for n in c['needs']), 'resolved_reuse': sum(bool(n['implemented_delivery']) for c in chapters for n in c.get('reuse_opportunities', []))}}
    assert report['counts']['selected_mechanisms'] == len(selection), 'Selected mechanisms outside canonical chapter list'
    (output / 'review.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'index.html').write_text((HERE / 'review-template.html').read_text().replace('REPORT_DATA', json.dumps(report).replace('<', '\\u003c')))
    print(json.dumps(report['counts'], indent=2))


if __name__ == '__main__':
    build()
