"""Build the local figures-first chapter review. Does not edit manuscript sources."""
from pathlib import Path
import hashlib
import html
import json
import os
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INVENTORY = HERE.parent / 'animation-catalog/inventory.json'


def canonical(volume):
    config = (ROOT / f'books/config/_quarto-pdf-vol{volume}.yml').read_text()
    return ['books/' + s for s in re.findall(r'^\s*- (vol\d/\d[^\s#]+\.qmd)', config, re.M)]


def build():
    figures = json.loads(INVENTORY.read_text())
    build_root = json.loads((HERE.parent / 'animation-catalog/catalog.json').read_text())['metadata']['build_root']
    by_key = {f['key']: f for f in figures}
    registry = json.loads((ROOT / 'books/_extensions/mlsysbook-ext/figure-explore/figure-registry.json').read_text())
    chapters = []
    methods = []
    for volume in range(1, 5):
        audit = json.loads((HERE / f'vol{volume}.json').read_text())
        methods.append({'volume': volume, 'method': audit['method']})
        expected = canonical(volume)
        assert set(expected) == {c['source'] for c in audit['chapters']}, f'Volume {volume} chapter coverage mismatch'
        assert len(expected) == len(audit['chapters']), 'Duplicate chapter'
        for source in expected:
            c = next(c for c in audit['chapters'] if c['source'] == source)
            lines = (ROOT / source).read_text().splitlines()
            c.update(volume=volume, number=int(Path(source).parent.name.split('_')[0]), source_sha256=hashlib.sha256((ROOT / source).read_bytes()).hexdigest())
            for n in c['needs']:
                assert 1 <= n['line'] <= len(lines), (source, n)
                assert n['kind'] in ['new figure', 'repair existing']
                assert n['priority'] in ['high', 'medium']
                n['source_excerpt'] = '\n'.join(f'{i + 1}: {lines[i]}' for i in range(max(0, n['line'] - 2), min(len(lines), n['line'] + 2)))[:2200]
            current_ids = set(re.findall(r'^:{3,}\s*\{#(fig-[^\s}]+)', '\n'.join(lines), re.M))
            inventoried_ids = {f['figure_id'] for f in figures if f['source'] == source}
            assert current_ids == inventoried_ids, f'Stale inventory: {source}'
            for reuse in c.get('reuse_opportunities', []):
                assert reuse['source_figure'] in by_key, reuse
            for candidate in c['animation_candidates']:
                key = f"vol{volume}:{candidate['figure_id']}"
                assert key in by_key and by_key[key]['source'] == source, (source, key)
                assert candidate['figure_id'] in lines[candidate['line'] - 1], (source, key, candidate['line'])
                candidate['registered_player'] = key in registry
            for static in c['keep_static']:
                key = f"vol{volume}:{static['figure_id']}"
                assert key in by_key and by_key[key]['source'] == source, (source, key)
            built = ROOT / build_root / f'vol{volume}-html/output' / Path(source).relative_to('books').with_suffix('.html')
            c['built_href'] = os.path.relpath(built, HERE) if built.is_file() else ''
            c['figures'] = [{k: f[k] for k in ['figure_id', 'title', 'line', 'section', 'assets', 'renderer']} for f in figures if f['source'] == source]
            for f in c['figures']:
                assert f['figure_id'] in '\n'.join(lines), (source, f['figure_id'])
            chapters.append(c)
    covered = {c['source'] for c in chapters}
    supplemental = [{k: f[k] for k in ['volume', 'source', 'figure_id', 'title', 'line']} for f in figures if f['source'] not in covered]
    report = {
        'date': '2026-10-18',
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'scope': 'All numbered chapters in the four canonical PDF configurations. Supplementary figure inventory is separate; appendix/frontmatter needs were not audited.',
        'review_basis': 'Structured source review: chapter outlines, all body-figure captions/anchors, and targeted prose at proposed gaps. Not an exhaustive sentence-level audit, rendered-artwork approval, student study, or publication.',
        'static_delivery': json.loads((HERE / 'static-delivery.json').read_text()) if (HERE / 'static-delivery.json').exists() else None,
        'methods': methods, 'chapters': chapters, 'supplemental_figures': supplemental,
        'counts': {'chapters': len(chapters), 'existing_chapter_figures': sum(len(c['figures']) for c in chapters), 'supplemental_figures': len(supplemental), 'proposed_figures': sum(len(c['needs']) for c in chapters), 'reuse_opportunities': sum(len(c.get('reuse_opportunities', [])) for c in chapters), 'animation_candidates': sum(len(c['animation_candidates']) for c in chapters), 'registered_players_in_inventory': sum(f['key'] in registry for f in figures)},
    }
    (HERE / 'review.json').write_text(json.dumps(report, indent=2) + '\n')
    payload = json.dumps(report).replace('<', '\\u003c')
    template = (HERE / 'review-template.html').read_text()
    (HERE / 'index.html').write_text(template.replace('REPORT_DATA', payload))
    print(json.dumps(report['counts']))


if __name__ == '__main__':
    build()
