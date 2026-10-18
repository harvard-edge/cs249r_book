"""Render the compiler-owned catalog without modifying its JSON evidence.

Chapter links are disabled unless an explicit build root contains the figure.
Selected lessons additionally require the mechanism-scene marker. A link base
can map the verified local build to a separately served URL.
"""
import argparse
import copy
import html
import json
import os
from collections import Counter
from pathlib import Path
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--render-only', action='store_true', help='Explicit renderer mode (also the default); never writes catalog.json.')
parser.add_argument('--catalog', type=Path, default=HERE / 'catalog.json')
parser.add_argument('--build-root', type=Path, help='Local parallel HTML build to inspect; no automatic fallback to historical builds.')
parser.add_argument('--link-base', help='URL serving the build root; only locally verified routes are linked.')
parser.add_argument('--output-dir', type=Path, default=HERE)
args = parser.parse_args()
output = args.output_dir.resolve()
output.mkdir(parents=True, exist_ok=True)
data = copy.deepcopy(json.loads(args.catalog.read_text()))
rows = data['figures']
meta = data['metadata']
build = args.build_root
if build and not build.is_absolute():
    build = ROOT / build
if args.link_base and not build:
    parser.error('--link-base requires --build-root so routes can be verified')
meta['render_build_root'] = os.path.relpath(build, ROOT) if build else None
meta['render_link_base'] = args.link_base
meta['render_counts'] = dict(Counter(r['actual_treatment'] for r in rows))
meta['render_chapters'] = len({r['source'] for r in rows})
selection_path = ROOT / 'books/_extensions/mlsysbook-ext/figure-explore/mechanisms/selection.json'
selection = {r['key']: r for r in json.loads(selection_path.read_text())} if selection_path.exists() else {}
cache = {}
for row in rows:
    selected = row['actual_treatment'] != 'static'
    chosen = selection.get(row['key'], {})
    row['catalog_purpose'] = chosen.get('systems_takeaway') or chosen.get('reason') or row.get('systems_goal', '')
    row['catalog_question'] = chosen.get('question') or row.get('actual_question', '')
    row['chapter_url'] = ''
    row['catalog_link_note'] = 'Chapter preview unavailable: no current HTML build has been configured.'
    if not build:
        continue
    route = Path(f"vol{row['volume']}-html/output") / Path(row['source']).relative_to('books').with_suffix('.html')
    chapter = build / route
    if chapter not in cache:
        cache[chapter] = BeautifulSoup(chapter.read_text(), 'html.parser') if chapter.is_file() else None
    soup = cache[chapter]
    node = soup.find(id=row['figure_id']) if soup else None
    if node is None:
        row['catalog_link_note'] = 'Chapter preview unavailable: this figure is absent from the configured build.'
    elif selected and 'mechanism-scene' not in str(node.get('data-mlsys-explore', '')):
        row['catalog_link_note'] = 'Lesson preview unavailable: the configured chapter does not contain the current mechanism scene.'
    else:
        url = args.link_base.rstrip('/') + '/' + route.as_posix() if args.link_base else os.path.relpath(chapter, output)
        row['chapter_url'] = url + '#' + row['figure_id']
        row['catalog_link_note'] = 'Route and figure marker verified in the configured build; visual and behavioral approval are separate.'
links = []
if build:
    base = build / 'vol1-html/output/index.html'
    if base.is_file():
        soup = BeautifulSoup(base.read_text(), 'html.parser')
        for selector, ident in [('#quarto-bootstrap', 'catalog-light'), ('link.quarto-color-alternate[href*="bootstrap-dark"]', 'catalog-dark')]:
            node = soup.select_one(selector)
            if node and (base.parent / node['href']).is_file():
                asset = os.path.relpath((base.parent / node['href']).resolve(), output)
                links.append(f'<link id="{ident}" rel="stylesheet" href="{html.escape(asset)}"' + (' disabled' if ident == 'catalog-dark' else '') + '>')
rendered = (HERE / 'catalog-template.html').read_text().replace('<!-- THEME -->', '\n'.join(links)).replace('/* CATALOG_DATA */', json.dumps(data, ensure_ascii=False).replace('</', '<\\/'))
(output / 'index.html').write_text(rendered)
print(json.dumps({'rendered': str(output / 'index.html'), 'figures': len(rows), 'chapters': meta['render_chapters'], 'treatments': meta['render_counts'], 'enabled_chapter_links': sum(bool(r['chapter_url']) for r in rows), 'catalog_json_modified': False}, indent=2))
