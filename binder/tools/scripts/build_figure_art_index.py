#!/usr/bin/env python3
"""Index canonical rendered figure assets for local visual authoring and QA.

Example: python3 binder/tools/scripts/build_figure_art_index.py --output /tmp/figure-art
No generated artwork or machine-specific index paths are committed by this tool.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup


def atomic_json(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--build-root', type=Path, action='append', help='Repeat for multiple build trees')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    catalog = args.catalog or repo / 'books/_prototypes/vol1-web-animations/animation-catalog/catalog.json'
    figures = json.loads(catalog.read_text())['figures']
    lookup = {f'vol{f["volume"]}:{f["figure_id"]}': f for f in figures}
    output = args.output.resolve()
    assets = output / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    candidates = {}
    roots = args.build_root or [repo / 'books/_build']
    html_paths = set()
    for root in roots:
        if root.exists():
            html_paths.update(p.resolve() for p in root.rglob('*.html'))
    # Prefer the newest actual rendering when chapters exist in several builds.
    for html in sorted(html_paths, key=lambda p: p.stat().st_mtime):
        volumes = re.findall(r'(?:^|[/_-])vol([1-4])(?=[/_.-]|$)', str(html))
        volume = volumes[-1] if volumes else None
        if not volume:
            continue
        try:
            soup = BeautifulSoup(html.read_text(), 'html.parser')
        except (OSError, UnicodeError):
            continue
        for node in soup.find_all(id=lambda value: value and value.startswith('fig-')):
            key = f'vol{volume}:{node["id"]}'
            if key not in lookup:
                continue
            images = []
            for image in node.find_all('img'):
                if image.find_parent('figcaption') is not None:
                    continue
                src = image.get('src', '')
                parsed = urlsplit(src)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    images.append({'src': src, 'error': 'nonlocal image'})
                    continue
                path = (html.parent / unquote(parsed.path)).resolve()
                images.append({'src': src, 'path': str(path)})
            if images:
                candidates[key] = {'html': str(html), 'images': images, 'origin': 'rendered-html'}
    for key, figure in lookup.items():
        if key in candidates:
            continue
        source = repo / figure['source']
        images = []
        for asset in figure.get('assets', []):
            path = (source.parent / asset).resolve()
            if path.is_file() and path.suffix.lower() in {'.svg', '.png', '.jpg', '.jpeg', '.gif', '.webp'}:
                images.append({'src': asset, 'path': str(path)})
        candidates[key] = {'images': images, 'origin': 'source-asset', 'html': None}

    def materialize(item):
        key, candidate = item
        record = {'source': lookup[key]['source'], **candidate}
        if len(candidate['images']) != 1:
            record['unsupported_reason'] = ('multiple figure images; inspect all images' if candidate['images'] else 'no rendered image or source raster/SVG found')
            return key, record
        original = candidate['images'][0]
        if 'path' not in original or not Path(original['path']).is_file():
            record['unsupported_reason'] = 'canonical image is unavailable locally'
            return key, record
        path = Path(original['path'])
        payload = path.read_bytes()
        digest = hashlib.sha1(payload).hexdigest()
        prefix = key.replace(':', '--') + '-' + digest[:12]
        target = assets / (prefix + path.suffix.lower())
        if not target.exists():
            shutil.copyfile(path, target)
        record.update(asset=str(target), asset_sha1=digest)
        if path.suffix.lower() == '.svg':
            raster = assets / (prefix + '.png')
            if not raster.exists():
                temporary = assets / (prefix + '.tmp.png')
                result = subprocess.run(['rsvg-convert', '-w', '1200', str(target), '-o', str(temporary)], capture_output=True, text=True)
                if result.returncode:
                    record['unsupported_reason'] = 'SVG render failed: ' + result.stderr.strip()[:500]
                    return key, record
                os.replace(temporary, raster)
            record['image'] = str(raster)
        elif path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.webp'}:
            record['image'] = str(target)
        else:
            record['unsupported_reason'] = 'unsupported canonical image format: ' + path.suffix
        return key, record

    index = {}
    index_path = output / 'index.json'
    # Preserve a usable previous index while freshly built chapters are arriving.
    if index_path.exists():
        index.update(json.loads(index_path.read_text()))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(materialize, item) for item in candidates.items()]
        for count, future in enumerate(as_completed(futures), 1):
            key, record = future.result()
            index[key] = record
            if count % 20 == 0:
                atomic_json(index_path, index)
    atomic_json(index_path, index)
    ready = sum('image' in row for row in index.values())
    print(f'{ready}/{len(lookup)} figures have canonical review images; index: {index_path}', flush=True)


if __name__ == '__main__':
    main()
