#!/usr/bin/env python3
"""Validate original-art paged KV references, reservation, copy ordering, and append isolation."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8882/_build/original-kv-copy-on-write/vol3/05_kv_cache/05_kv_cache.html#fig-vol3-pagedattention-mapping')
parser.add_argument('--engine', choices=['chromium', 'webkit'], default='chromium')
parser.add_argument('--screenshots', type=Path, default=Path('/tmp/original-kv-copy-on-write-review'))
args = parser.parse_args(); args.screenshots.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[4]
original = ET.parse(ROOT / 'books/vol3/05_kv_cache/images/svg/paged_attention_virtual_memory.svg').getroot()
paths = sum(n.tag.endswith('path') for n in original.iter())
report = []
with sync_playwright() as p:
    browser = getattr(p, args.engine).launch()
    for width, theme in [(1440, 'light'), (390, 'light'), (1440, 'dark'), (390, 'dark')]:
        page = browser.new_page(viewport={'width': width, 'height': 1900}, color_scheme=theme)
        errors = []; page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(args.url)
        if theme == 'dark': page.evaluate("document.body.classList.add('quarto-dark');document.body.classList.remove('quarto-light')")
        page.wait_for_function('window.cowPlayer!==undefined')
        root = page.locator('.mlsys-original-kv-cow'); menu = root.get_by_role('combobox')
        assert root.get_attribute('data-cow-step') == '0'
        assert root.get_attribute('data-playing') == 'false'
        assert root.locator('svg path').count() == paths
        assert root.locator('svg').get_attribute('viewBox') == '0 0 720 600'
        for i in range(10):
            menu.select_option(str(i))
            a = [3, 7, 9]
            b = None if i < 4 else [3, 7, 14 if i >= 8 else 9]
            references = {str(k): a.count(k) + (b.count(k) if b else 0) for k in [3, 7, 9, 14]}
            assert json.loads(root.get_attribute('data-cow-references')) == references
            assert root.get_attribute('data-cow-b') == (','.join(map(str, b)) if b else '')
            private = json.loads(root.get_attribute('data-cow-private'))
            assert private == [8 if i >= 6 else None, 9 if i >= 7 else None, 10 if i >= 9 else None, None]
            assert root.get_attribute('data-cow-allocated') == str(i >= 5).lower()
            if any(x is not None for x in private): assert i >= 5
            if b and b[-1] == 14: assert private[:2] == [8, 9]
            if private[2] is not None: assert b[-1] == 14
            assert root.get_attribute('data-cow-source') == '8,9'
            assert root.locator('svg text[x="260"][y="395"]').text_content() == '3'
            assert root.locator('svg text[x="420"][y="395"]').text_content() == '7'
            assert root.locator('svg text[x="580"][y="395"]').text_content() == '9'
            # Full blocks remain drawn with unchanged initialized records throughout.
            full = root.locator('svg text[y="449.5"]').all_text_contents()
            assert full == list(map(str, range(10)))
            if i in [0, 3, 4, 5, 6, 7, 8, 9]:
                page.evaluate("window.scrollTo(0,document.querySelector('.mlsys-original-kv-cow').getBoundingClientRect().top+scrollY-230)")
                root.locator('.okv-canvas').screenshot(path=str(args.screenshots / f'state-{i}-{width}-{theme}.png'))
        # During the first copy, allocation exists but the destination remains unwritten.
        menu.select_option('5'); root.get_by_role('button', name='Next', exact=True).click(); page.wait_for_timeout(250)
        assert root.get_attribute('data-cow-step') == '5'
        assert json.loads(root.get_attribute('data-cow-private')) == [None] * 4
        root.get_by_role('button', name='Pause', exact=True).click()
        packet = root.locator('.okv-packet'); before = packet.get_attribute('transform')
        page.wait_for_timeout(160); assert packet.get_attribute('transform') == before
        root.locator('.okv-canvas').screenshot(path=str(args.screenshots / f'copy-in-flight-{width}-{theme}.png'))
        root.get_by_role('button', name='Play', exact=True).click()
        page.wait_for_function("document.querySelector('.mlsys-original-kv-cow').dataset.cowStep==='6'")
        page.evaluate('window.cowPlayer.pause()')
        assert root.get_attribute('data-cow-b') == '3,7,9'
        assert json.loads(root.get_attribute('data-cow-private')) == [8, None, None, None]
        # A jump or interruption removes transient fork-reference labels.
        menu.select_option('3'); root.get_by_role('button', name='Next', exact=True).click(); page.wait_for_timeout(100)
        assert root.locator('.okv-reference-copy').count() == 3
        menu.select_option('7'); assert root.locator('.okv-reference-copy').count() == 0
        root.focus(); page.keyboard.press('p'); assert root.get_attribute('data-cow-step') == '6'
        page.keyboard.press('r'); assert root.get_attribute('data-cow-step') == '0'
        page.emulate_media(reduced_motion='reduce'); root.focus(); page.keyboard.press('n')
        page.wait_for_function("document.querySelector('.mlsys-original-kv-cow').dataset.cowStep==='1'")
        assert root.get_attribute('data-playing') == 'false'
        menu.select_option('8')
        if width < 600:
            assert root.locator('.okv-pan-cue').is_visible()
            assert root.locator('svg').evaluate('s=>s.getBoundingClientRect().width') >= 960
            assert root.locator('.okv-canvas').evaluate('n=>n.scrollLeft>0&&n.scrollTop>0')
            # Both the candidate mapping and the destination tail fit in the active crop.
            assert root.locator('svg text[x="580"][y="569"]').evaluate('n=>{const a=n.getBoundingClientRect(),b=n.closest(".okv-canvas").getBoundingClientRect();return a.left>=b.left&&a.right<=b.right&&a.top>=b.top&&a.bottom<=b.bottom}')
            assert root.locator('svg rect[x="532.0"][y="492"][width="96"]').evaluate('n=>{const a=n.getBoundingClientRect(),b=n.closest(".okv-canvas").getBoundingClientRect();return a.left>=b.left&&a.right<=b.right&&a.top>=b.top&&a.bottom<=b.bottom}')
            assert root.locator('.okv-inlet').evaluate('n=>{const a=n.getBoundingClientRect(),b=n.closest(".okv-canvas").getBoundingClientRect();return a.left>=b.left&&a.right<=b.right&&a.top>=b.top&&a.bottom<=b.bottom}')
        root.get_by_role('button', name='Expand', exact=True).click()
        assert page.locator('dialog[open]').count() == 1
        assert root.get_attribute('data-cow-step') == '8'
        page.locator('dialog').screenshot(path=str(args.screenshots / f'expanded-{width}-{theme}.png'))
        page.keyboard.press('Escape'); page.wait_for_function("document.activeElement?.textContent==='Expand'")
        menu.select_option('9')
        page.evaluate("window.scrollTo(0,document.querySelector('.mlsys-original-kv-cow').getBoundingClientRect().top+scrollY-230)")
        page.screenshot(path=str(args.screenshots / f'chapter-{width}-{theme}.png'))
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.emulate_media(media='print'); assert not root.is_visible()
        assert page.locator('#fig-vol3-pagedattention-mapping img[data-cow-static]').is_visible()
        assert not errors, errors
        report.append({'width': width, 'theme': theme, 'states': 10, 'copy_before_remap': 'passed', 'a_unchanged': 'passed', 'references': 'passed', 'controls': 'passed', 'static_print': 'passed'})
        page.close()
    page = browser.new_page(java_script_enabled=False); page.goto(args.url)
    assert page.locator('#fig-vol3-pagedattention-mapping img[data-cow-static]').is_visible()
    assert page.locator('.mlsys-original-kv-cow').count() == 0
    page.close(); browser.close()
print(json.dumps(report, indent=2))
