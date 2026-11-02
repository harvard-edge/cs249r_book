#!/usr/bin/env python3
"""Check prefix cutoff, immutable request tokens, historical KV, and endpoint-only recomputation."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8882/_build/original-context-invalidation/vol3/04_context_engineering/04_context_engineering.html#fig-vol3-context-zones')
parser.add_argument('--engine', choices=['chromium', 'webkit'], default='chromium')
parser.add_argument('--screenshots', type=Path, default=Path('/tmp/original-context-review'))
args = parser.parse_args(); args.screenshots.mkdir(parents=True, exist_ok=True)
report = []
with sync_playwright() as p:
    browser = getattr(p, args.engine).launch()
    for width, theme in [(1440, 'light'), (390, 'light'), (1440, 'dark'), (390, 'dark')]:
        page = browser.new_page(viewport={'width': width, 'height': 1900}, color_scheme=theme)
        errors = []; page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(args.url)
        if theme == 'dark': page.evaluate("document.body.classList.add('quarto-dark');document.body.classList.remove('quarto-light')")
        page.wait_for_function('window.contextPlayer!==undefined')
        root = page.locator('.mlsys-original-context'); menu = root.get_by_role('combobox')
        assert root.get_attribute('data-context-step') == '0'
        assert root.get_attribute('data-playing') == 'false'
        assert root.locator('svg').get_attribute('viewBox') == '0 0 680 392'
        reserves = root.locator('svg rect[x="540"][width="100"][height="34"]')
        assert reserves.count() == 3
        reserve_fills = reserves.evaluate_all('nodes=>nodes.map(n=>n.getAttribute("fill"))')
        for i in range(11):
            menu.select_option(str(i))
            expected = [['pending'] * 3 for _ in range(3)]
            if i >= 1: expected[0] = ['new'] * 3
            if i >= 4: expected[1] = ['reuse', 'reuse', 'new' if i >= 5 else 'pending']
            if i >= 8: expected[2] = ['reuse', 'new' if i >= 9 else 'pending', 'new' if i >= 10 else 'pending']
            assert json.loads(root.get_attribute('data-context-states')) == expected
            provenance = ['C0' if i >= 1 else None, 'C1' if i >= 5 else None, 'C2' if i >= 10 else None]
            assert json.loads(root.get_attribute('data-context-t')) == provenance
            assert root.locator('[data-token-row] text').all_text_contents() == ['T'] * 3
            assert root.locator('[data-token-kind="trunk-first"]').evaluate_all('nodes=>nodes.map(n=>n.dataset.tokenId)') == ['u', 'u', 'v']
            assert root.locator('[data-token-kind="leaf-first"]').evaluate_all('nodes=>nodes.map(n=>n.dataset.tokenId)') == ['a', 'b', 'c']
            assert reserves.evaluate_all('nodes=>nodes.map(n=>n.getAttribute("fill"))') == reserve_fills
            assert root.get_attribute('data-context-cutoff') == ('trunk' if i >= 7 else 'leaf' if 3 <= i < 6 else '')
            if i >= 6:
                assert root.locator('[data-kv-row="1"][data-kv-zone="2"] text').text_content() == 'KV(T): C1'
            if i in [0, 3, 4, 5, 7, 8, 9, 10]:
                root.locator('.oci-canvas').screenshot(path=str(args.screenshots / f'state-{i}-{width}-{theme}.png'))
        # A partially illustrated processing pass cannot commit its destination KV.
        menu.select_option('4'); root.get_by_role('button', name='Next', exact=True).click(); page.wait_for_timeout(220)
        assert root.get_attribute('data-context-step') == '4'
        assert json.loads(root.get_attribute('data-context-t'))[1] is None
        assert float(root.locator('.oci-front').get_attribute('x')) + float(root.locator('.oci-front').get_attribute('width')) <= 540
        root.get_by_role('button', name='Pause', exact=True).click()
        narration = root.locator('.mlsys-original-narration').inner_text()
        assert 'When this processing finishes' in narration
        assert 'is now C1' not in narration
        assert root.locator('[data-kv-row="1"][data-kv-zone="2"] text').text_content() == 'KV pending'
        front = root.locator('.oci-front'); before = front.get_attribute('width')
        page.wait_for_timeout(180); assert front.get_attribute('width') == before
        root.locator('.oci-canvas').screenshot(path=str(args.screenshots / f'paused-process-{width}-{theme}.png'))
        root.screenshot(path=str(args.screenshots / f'paused-player-{width}-{theme}.png'))
        root.get_by_role('button', name='Play', exact=True).click()
        page.wait_for_function("document.querySelector('.mlsys-original-context').dataset.contextStep==='5'")
        page.evaluate('window.contextPlayer.pause()')
        assert json.loads(root.get_attribute('data-context-t')) == ['C0', 'C1', None]
        # The same conditional narration contract applies to C2 and the other
        # processing/binding steps, whose destination narration appears in motion.
        for previous, phrase in [(0, 'When this pass finishes'), (3, 'When this binding finishes'), (7, 'When this binding finishes'), (8, 'When this pass finishes'), (9, 'When this processing finishes')]:
            menu.select_option(str(previous)); root.get_by_role('button', name='Next', exact=True).click(); page.wait_for_timeout(80)
            root.get_by_role('button', name='Pause', exact=True).click()
            assert phrase in root.locator('.mlsys-original-narration').inner_text()
            assert root.get_attribute('data-context-step') == str(previous)
            if previous == 9:
                assert json.loads(root.get_attribute('data-context-t'))[2] is None
                assert root.locator('[data-kv-row="2"][data-kv-zone="2"] text').text_content() == 'KV pending'
        # Reuse and comparison overlays cannot survive a menu interruption.
        menu.select_option('3'); root.get_by_role('button', name='Next', exact=True).click(); page.wait_for_timeout(100)
        assert root.locator('.oci-reuse-link').count() == 2
        menu.select_option('8'); assert root.locator('.oci-reuse-link').count() == 0
        root.focus(); page.keyboard.press('p'); assert root.get_attribute('data-context-step') == '7'
        page.keyboard.press('r'); assert root.get_attribute('data-context-step') == '0'
        page.emulate_media(reduced_motion='reduce'); root.focus(); page.keyboard.press('n')
        page.wait_for_function("document.querySelector('.mlsys-original-context').dataset.contextStep==='1'")
        assert root.get_attribute('data-playing') == 'false'
        menu.select_option('10')
        if width < 600:
            assert root.locator('.oci-pan-cue').is_visible()
            assert root.locator('svg').evaluate('n=>n.getBoundingClientRect().width') >= 900
            for selector in ['[data-token-row="2"]', '[data-kv-row="2"][data-kv-zone="2"]']:
                assert root.locator(selector).evaluate('n=>{const a=n.getBoundingClientRect(),b=n.closest(".oci-canvas").getBoundingClientRect();return a.left>=b.left&&a.right<=b.right&&a.top>=b.top&&a.bottom<=b.bottom}')
        root.get_by_role('button', name='Expand', exact=True).click()
        assert page.locator('dialog[open]').count() == 1
        assert root.get_attribute('data-context-step') == '10'
        page.locator('dialog').screenshot(path=str(args.screenshots / f'expanded-{width}-{theme}.png'))
        page.keyboard.press('Escape'); page.wait_for_function("document.activeElement?.textContent==='Expand'")
        page.evaluate("window.scrollTo(0,document.querySelector('.mlsys-original-context').getBoundingClientRect().top+scrollY-230)")
        root.screenshot(path=str(args.screenshots / f'player-{width}-{theme}.png'))
        page.screenshot(path=str(args.screenshots / f'chapter-{width}-{theme}.png'))
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.emulate_media(media='print'); assert not root.is_visible()
        assert page.locator('#fig-vol3-context-zones img[data-context-static]').is_visible()
        assert not errors, errors
        report.append({'width': width, 'theme': theme, 'states': 11, 'token_identity': 'passed', 'prefix_only_reuse': 'passed', 'historical_provenance': 'passed', 'reserve_untouched': 'passed', 'controls': 'passed'})
        page.close()
    page = browser.new_page(java_script_enabled=False); page.goto(args.url)
    assert page.locator('#fig-vol3-context-zones img[data-context-static]').is_visible()
    assert page.locator('.mlsys-original-context').count() == 0
    page.close(); browser.close()
print(json.dumps(report, indent=2))
