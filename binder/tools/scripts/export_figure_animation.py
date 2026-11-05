#!/usr/bin/env python3
"""Export a finite, captioned GIF of a shared-player figure's committed states.

Run against a rendered chapter or a QA fixture served over HTTP. This captures
step endpoints, not interpolated motion, and never edits the chapter or player.
The GIF has no interactive controls. Raw PNG frames and a provenance manifest
are retained beside it. Requires Playwright Chromium, Pillow, and PyYAML.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[3]
EXT = ROOT / 'books/_extensions/mlsysbook-ext/figure-explore'


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_url(source: str, figure_id: str, site_url: str) -> str:
    """Quarto publishes the books-relative source path under website.site-url."""
    path = Path(source).relative_to('books').with_suffix('.html').as_posix()
    return urljoin(site_url.rstrip('/') + '/', path) + '?view=explore#' + figure_id


def frame_duration(text: str, minimum_ms: int, words_per_minute: int) -> int:
    """Constructed reading allowance, rounded upward to GIF's centisecond tick."""
    import math
    return math.ceil(max(minimum_ms, len(text.split()) * 60000 / words_per_minute) / 10) * 10


def encode_gif(frames: list[Path], durations: list[int], destination: Path) -> None:
    from PIL import Image
    if not frames or len(frames) != len(durations):
        raise ValueError('One duration is required for every frame.')
    images = [Image.open(path).convert('RGB') for path in frames]
    if len({image.size for image in images}) != 1:
        raise ValueError('Export frames must have identical dimensions.')
    # Omitting GIF's loop extension means one playback; loop=0 would loop forever.
    images[0].save(destination, save_all=True, append_images=images[1:],
                   duration=durations, disposal=2, optimize=False)


CAPTURE_CSS = """
.mlsys-original-jump-anchor, .mlsys-original-jump, .mlsys-original-cue,
.mlsys-original-dock, .mlsys-original-controls, .mlsys-original-tooltip,
.mlsys-original-speed-popover, #quarto-header, #quarto-sidebar,
#quarto-margin-sidebar { visibility: hidden !important; pointer-events: none !important; }
"""


def prepare_capture(root, width: int) -> dict:
    """Expand a private browser's stage; reject residual overflow instead of cropping.

    The chapter reading column is often narrower than its figure's minimum SVG
    width. Fix the capture width independently of that column and open only HTML
    scroll containers. Do not change SVG viewBoxes or shrink away source labels.
    """
    return root.evaluate(r"""(root, width) => {
      root.style.setProperty('width', width + 'px', 'important');
      root.style.setProperty('max-width', 'none', 'important');
      const stage = root.querySelector('.mlsys-original-stage');
      // Opaque capture surface prevents the chapter column's border/shadow
      // showing through a diagram expanded beyond that column.
      stage.style.setProperty('background-color', '#fff', 'important');
      stage.style.setProperty('position', 'relative', 'important');
      stage.style.setProperty('z-index', '1', 'important');
      for (let n = stage; n; n = n.parentElement) {
        n.style.setProperty('overflow', 'visible', 'important');
        n.scrollLeft = 0; n.scrollTop = 0;
      }
      stage.style.setProperty('width', width + 'px', 'important');
      stage.style.setProperty('max-width', 'none', 'important');
      // Open inner HTML scrolling panes, including minimum-width source SVGs.
      // SVG overflow keeps its authored setting; validation below catches labels
      // that would require changing source artwork to export correctly.
      for (const node of stage.querySelectorAll('*')) {
        if (node.namespaceURI !== 'http://www.w3.org/1999/xhtml') continue;
        node.scrollLeft = 0; node.scrollTop = 0;
        node.dispatchEvent(new Event('scroll'));
        const css = getComputedStyle(node);
        if (/(auto|scroll|hidden|clip)/.test(css.overflowX + ' ' + css.overflowY)) {
          node.style.setProperty('overflow', 'visible', 'important');
          node.style.setProperty('max-height', 'none', 'important');
        }
      }
      let required = width;
      const box = stage.getBoundingClientRect();
      for (const node of stage.querySelectorAll('*')) {
        if (node.closest('defs,clipPath,mask,symbol')) continue;
        const css = getComputedStyle(node), b = node.getBoundingClientRect();
        if (css.display !== 'none' && css.visibility !== 'hidden' && b.width && b.height)
          required = Math.max(required, b.right - box.left + 8);
      }
      // Avoid iterative growth from the stage's own full-width children.
      if (required > width + 9) {
        root.style.setProperty('width', Math.ceil(required) + 'px', 'important');
        stage.style.setProperty('width', Math.ceil(required) + 'px', 'important');
      }
      return {requested_width: width, capture_width: stage.getBoundingClientRect().width};
    }""", width)


def validate_capture(root) -> dict:
    """Fail when visible diagram content would be cropped by the stage or an SVG."""
    geometry = root.evaluate(r"""root => {
      const stage = root.querySelector('.mlsys-original-stage'), box = stage.getBoundingClientRect();
      const problems = [];
      const outside = (a,b) => a.left < b.left-2 || a.right > b.right+2 || a.top < b.top-2 || a.bottom > b.bottom+2;
      for (const node of stage.querySelectorAll('svg, img, canvas, text, foreignObject, [data-mlsys-keep], [data-mlsys-focus]')) {
        if (node.closest('defs,clipPath,mask,symbol')) continue;
        const css = getComputedStyle(node), b = node.getBoundingClientRect();
        if (!b.width || !b.height || css.display === 'none' || css.visibility === 'hidden' || Number(css.opacity) === 0) continue;
        if (outside(b,box)) problems.push('outside stage: ' + node.tagName + ' ' + (node.textContent || '').slice(0,80));
        const svg = node.ownerSVGElement;
        if (svg && ['hidden','clip','auto','scroll'].includes(getComputedStyle(svg).overflow) && outside(b,svg.getBoundingClientRect()))
          problems.push('outside SVG: ' + node.tagName + ' ' + (node.textContent || '').slice(0,80));
      }
      return {width:box.width,height:box.height,problems};
    }""")
    if geometry['problems']:
        raise ValueError('Figure cannot be exported without clipping: ' + '; '.join(geometry['problems']))
    return geometry


def export_text(locator, remove: str) -> str:
    """Preserve readable block boundaries when removing interactive disclosures."""
    return locator.evaluate(r"""(node, remove) => {
      const copy = node.cloneNode(true);
      copy.querySelectorAll(remove).forEach(e => e.remove());
      copy.querySelectorAll('div,p,section,article,li,dt,dd,tr,h1,h2,h3,h4,br').forEach(e => {
        e.prepend(document.createTextNode('\n')); e.append(document.createTextNode('\n'));
      });
      return copy.textContent.replace(/[ \t]+/g,' ').replace(/\n\s*\n/g,'\n').trim();
    }""", remove)


def export(args) -> dict:
    import base64
    import yaml
    from playwright.sync_api import sync_playwright

    registry = json.loads((EXT / 'figure-registry.json').read_text())
    if args.key not in registry:
        raise ValueError('Unknown registered figure: ' + args.key)
    record = registry[args.key]
    if not record.get('shared_player'):
        raise ValueError('This exporter currently supports shared-player lessons only.')
    config = ROOT / f'books/config/_quarto-html-vol{record["volume"]}.yml'
    site_url = yaml.safe_load(config.read_text())['website']['site-url']
    source_url = canonical_url(record['source'], record['figure_id'], site_url)
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = out.with_suffix('')
    frame_dir.mkdir(exist_ok=True)
    if out.exists() or any(frame_dir.iterdir()):
        raise ValueError('Choose a new output name; exports do not overwrite prior receipts.')
    resources = {}
    errors = []
    states = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': args.width, 'height': 1200},
                                color_scheme='light', reduced_motion='reduce', device_scale_factor=2)
        page.on('pageerror', lambda e: errors.append(str(e)))
        def resource(response):
            if response.request.resource_type in {'document', 'script', 'stylesheet', 'image'} and response.ok:
                try:
                    resources[response.url] = sha256(response.body())
                except Exception:
                    pass
        page.on('response', resource)
        page.goto(args.url, wait_until='networkidle')
        outer = page.locator('[id=' + json.dumps(record['figure_id']) + ']')
        outer.scroll_into_view_if_needed()
        page.wait_for_function('id => { const e=document.getElementById(id); return e && (e.dataset.fixtureState === "ready" || e.dataset.mlsysExploreState === "ready"); }', arg=record['figure_id'])
        explore = outer.get_by_role('tab', name='Explore', exact=True)
        if explore.count():
            explore.click()
        root = outer.locator('.mlsys-original-player')
        root.wait_for(state='visible')
        title = root.locator('.mlsys-original-title').inner_text()
        assumptions = export_text(root.locator('.mlsys-original-assumptions'), 'summary,button,select')
        menu = root.locator('.mlsys-original-sequence select')
        values = menu.locator('option').evaluate_all('(nodes) => nodes.map(n => n.value)')
        page.add_style_tag(content=CAPTURE_CSS)
        # Fix Whole view when offered: a shareable preview must retain the full
        # source composition rather than crop away context as a focus follows.
        for selector in root.locator('select').all():
            if selector.locator('option[value="whole"]').count():
                selector.select_option('whole', force=True)
        capture_width = args.width - 56
        for value in values:
            menu.select_option(value, force=True)
            capture_width = max(capture_width, prepare_capture(root, capture_width)['capture_width'])
        # Keep the capture canvas inside the browser viewport even when the
        # chapter column starts after a sidebar. Otherwise screenshot stitching
        # can carry viewport-edge shadows through a wide SVG. Export-sheet width
        # remains the caller's requested width.
        import math
        page.set_viewport_size({'width': max(args.width, math.ceil(capture_width) + 1024), 'height': 1200})
        for ordinal, value in enumerate(values):
            menu.select_option(value, force=True)
            prepare_capture(root, capture_width)
            page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
            stage = root.locator('.mlsys-original-stage')
            stage.scroll_into_view_if_needed()
            page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
            geometry = validate_capture(root)
            png = stage.screenshot(animations='disabled')
            stage_path = frame_dir / f'stage-{ordinal:03d}.png'
            stage_path.write_bytes(png)
            narration = root.locator('.mlsys-original-narration').inner_text()
            details = export_text(root.locator('.mlsys-original-details'), 'details,button,select')
            # Narrative and details are captured as text from the committed state,
            # not reconstructed from lesson source or inferred from an animation.
            states.append({'value': value, 'narration': narration, 'details': details,
                           'stage': stage_path, 'png': png, 'geometry': geometry})
        if errors:
            raise RuntimeError('Chapter emitted errors: ' + '; '.join(errors))
        from PIL import Image
        import io
        diagram_height = max(Image.open(io.BytesIO(s['png'])).height / Image.open(io.BytesIO(s['png'])).width * (args.width - 56) for s in states)
        sheet = browser.new_page(viewport={'width': args.width, 'height': 1200}, device_scale_factor=1)
        style = '''body{margin:0;background:white;color:#16212b;font:22px/1.4 system-ui,sans-serif}main{box-sizing:border-box;padding:28px;width:100%;background:white}h1{font-size:28px;margin:0 0 8px}p{margin:10px 0}.scope{font-size:17px;color:#44515b}.diagram{width:100%;height:auto;display:block}.diagram-box{overflow:hidden}.narration{font-size:22px;white-space:pre-line}.details{font-size:18px;white-space:pre-line}footer{border-top:1px solid #bcc7cf;margin-top:18px;padding-top:12px;font-size:16px;overflow-wrap:anywhere}.kind{font-size:16px;color:#44515b}'''
        heights = []
        for state in states:
            stage_data = base64.b64encode(state['png']).decode()
            markup = '<!doctype html><meta charset="utf-8"><style>' + style + '</style><main>'
            markup += '<h1>' + html.escape(title) + '</h1><div class="kind">Step-by-step preview · GIF has no interactive controls</div>'
            markup += '<p class="scope">' + html.escape(assumptions) + '</p>'
            markup += '<div class="diagram-box" style="height:' + str(diagram_height) + 'px"><img class="diagram" src="data:image/png;base64,' + stage_data + '"></div>'
            markup += '<p class="narration">' + html.escape(state['narration']) + '</p>'
            if state['details']:
                markup += '<p class="details">' + html.escape(state['details']) + '</p>'
            markup += '<footer><strong>MLSysBook</strong> · Explore the interactive figure:<br>' + html.escape(source_url) + '</footer></main>'
            state['markup'] = markup
            sheet.set_content(markup)
            sheet.locator('.diagram').evaluate('(img) => img.decode()')
            heights.append(sheet.locator('main').bounding_box()['height'])
        import math
        height = math.ceil(max(heights))
        sheet.set_viewport_size({'width': args.width, 'height': height})
        frames = []
        durations = []
        for ordinal, state in enumerate(states):
            sheet.set_content(state['markup'])
            sheet.locator('.diagram').evaluate('(img) => img.decode()')
            sheet.add_style_tag(content=f'main{{min-height:{height}px}}')
            path = frame_dir / f'frame-{ordinal:03d}.png'
            sheet.screenshot(path=str(path))
            frames.append(path)
            durations.append(frame_duration(state['narration'] + ' ' + state['details'], args.minimum_ms, args.words_per_minute))
        browser.close()
    encode_gif(frames, durations, out)
    bound_paths = [ROOT / record['source'], config, Path(__file__).resolve(),
                   ROOT / 'books/config/shared/html/figure-explore.yml']
    for name in ('script', 'stylesheet'):
        if record.get(name):
            bound_paths.append(EXT / record[name])
    bound_paths += [EXT / name for name in ('original-player.js', 'original-player.css', 'figure-explore.js', 'figure-views.js', 'figure-views.css')]
    manifest = {
        'key': args.key, 'kind': 'constructed step-by-step GIF preview',
        'capture_url': args.url, 'source_url': source_url,
        'limitations': ['Committed endpoints only; interpolated motion is not captured.',
                        'No interactive controls; reading durations are constructed display allowances.',
                        'Capture is not evidence of learning outcomes or technical correctness.'],
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
        'command': sys.argv, 'input_sha256': {str(path.relative_to(ROOT)): sha256(path.read_bytes()) for path in bound_paths},
        'fetched_resource_sha256': resources,
        'width': args.width, 'height': height, 'loop': 'once',
        'frames': [{'file': str(path), 'sha256': sha256(path.read_bytes()), 'duration_ms': duration,
                    'state': state['value'], 'capture_geometry': state['geometry'], 'narration': state['narration'], 'details': state['details']}
                   for path, duration, state in zip(frames, durations, states)],
        'output': str(out), 'output_sha256': sha256(out.read_bytes()),
    }
    from datetime import datetime, timezone
    manifest['created_at'] = datetime.now(timezone.utc).isoformat()
    out.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='Rendered chapter or QA fixture URL')
    parser.add_argument('--key', required=True, help='Registered volN:fig-id')
    parser.add_argument('--output', required=True, type=Path, help='New .gif output path')
    parser.add_argument('--width', type=int, default=1100)
    parser.add_argument('--minimum-ms', type=int, default=3500)
    parser.add_argument('--words-per-minute', type=int, default=180)
    args = parser.parse_args()
    if args.width < 320 or args.minimum_ms <= 0 or args.words_per_minute <= 0:
        parser.error('Width must be at least 320; timing values must be positive.')
    if args.output.suffix.lower() != '.gif':
        parser.error('--output must end in .gif')
    manifest = export(args)
    print(json.dumps({k: manifest[k] for k in ('output', 'output_sha256', 'source_url', 'width', 'height')}))


if __name__ == '__main__':
    main()
