#!/usr/bin/env python3
"""Independently check capture accounting, ring wrap, original geometry, and shared controls."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8882/_build/original-stream-ingestion/vol3/07_tool_calling/07_tool_calling.html#fig-vol3-stream-ingestion')
parser.add_argument('--engine', choices=['chromium', 'webkit'], default='chromium')
parser.add_argument('--screenshots', type=Path, default=Path('/tmp/original-stream-ingestion-review'))
args = parser.parse_args()
args.screenshots.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[4]
art = ROOT / 'books/vol3/07_tool_calling/images/svg/vol3-stream-ingestion.svg'
original = ET.parse(art).getroot()
original_paths = sum(n.tag.endswith('path') for n in original.iter())
payload = 'ABCDEFGHIJKLMNOP'
head_capacity, tail_capacity = 4, 8
capacity = head_capacity + tail_capacity
reports = []
with sync_playwright() as p:
    browser = getattr(p, args.engine).launch()
    for width, theme in [(1440, 'light'), (390, 'light'), (1440, 'dark'), (390, 'dark')]:
        page = browser.new_page(viewport={'width': width, 'height': 1900}, color_scheme=theme)
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(args.url)
        if theme == 'dark':
            page.evaluate("document.body.classList.add('quarto-dark');document.body.classList.remove('quarto-light')")
        root = page.locator('.mlsys-original-stream-ingestion')
        root.wait_for()
        page.wait_for_function('window.streamPlayer !== undefined')
        assert root.get_attribute('data-stream-step') == '0'
        assert root.get_attribute('data-playing') == 'false'
        assert root.locator('svg path').count() == original_paths
        assert root.locator('svg').get_attribute('viewBox') == '0 0 854 360'
        menu = root.get_by_role('combobox')
        for n in range(len(payload) + 1):
            menu.select_option(str(n))
            seen, expected = payload[:n], [None] * tail_capacity
            for i, byte in enumerate(seen[head_capacity:]):
                expected[i % tail_capacity] = byte
            actual = root.locator('[data-ring-slot]').evaluate_all("nodes=>nodes.map(n=>n.dataset.byte||null)")
            assert actual == expected, (n, actual, expected)
            head = seen[:head_capacity]
            tail = seen[head_capacity:][-tail_capacity:]
            omitted = len(seen) - len(head) - len(tail)
            assert int(root.get_attribute('data-stream-omitted')) == omitted
            assert root.get_attribute('data-stream-tail') == tail
            assert int(root.get_attribute('data-stream-seen')) == len(head) + len(tail) + omitted
            assert len(head) + len(tail) <= capacity
            assert root.get_attribute('data-stream-staged') == 'false'
        # The final byte overwrites H, while physical slots retain cyclic order.
        assert root.locator('[data-ring-slot]').all_text_contents() == expected
        menu.select_option(str(len(payload)))
        root.get_by_role('button', name='Next', exact=True).click()
        page.wait_for_timeout(120)
        assert root.get_attribute('data-stream-staged') == 'false'
        assert root.locator('.osi-packet text').text_content() == 'copy'
        root.get_by_role('button', name='Next', exact=True).click()
        assert root.get_attribute('data-stream-staged') == 'true'
        assert root.locator('svg').get_by_text(f'{omitted} bytes omitted', exact=True).count() == 1
        assert root.locator('svg').get_by_text(' '.join(tail), exact=True).count() == 1
        assert root.locator('svg').get_by_text('handle: none', exact=True).count() == 1
        # Shared modal retains the very same diagram and restores focus on Escape.
        root.get_by_role('button', name='Expand', exact=True).click()
        assert page.locator('dialog[open]').count() == 1
        assert root.get_attribute('data-stream-step') == '17'
        page.locator('dialog').screenshot(path=str(args.screenshots / f'expanded-{width}-{theme}.png'))
        page.keyboard.press('Escape')
        assert page.locator('dialog[open]').count() == 0
        page.wait_for_function("document.activeElement?.textContent==='Expand'")
        assert root.get_by_role('button', name='Expand', exact=True).evaluate('n=>document.activeElement===n')
        # Store only at endpoint; freezing the shared clock freezes the moving byte.
        menu.select_option('12')
        root.get_by_role('button', name='Next', exact=True).click()
        page.wait_for_timeout(260)
        assert root.get_attribute('data-stream-step') == '12'
        assert root.get_attribute('data-stream-omitted') == '0'
        page.evaluate('window.streamPlayer.pause()')
        packet = root.locator('.osi-packet')
        position = packet.get_attribute('transform')
        page.wait_for_timeout(180)
        assert packet.get_attribute('transform') == position
        page.evaluate('window.streamPlayer.play()')
        page.wait_for_function("document.querySelector('.mlsys-original-stream-ingestion').dataset.streamStep==='13'")
        page.evaluate('window.streamPlayer.pause()')
        assert root.get_attribute('data-stream-omitted') == '1'
        root.focus()
        page.keyboard.press('p')
        assert root.get_attribute('data-stream-step') == '12'
        page.keyboard.press('r')
        assert root.get_attribute('data-stream-step') == '0'
        page.emulate_media(reduced_motion='reduce')
        root.focus(); page.keyboard.press('n')
        page.wait_for_function("document.querySelector('.mlsys-original-stream-ingestion').dataset.streamStep==='1'")
        assert root.get_attribute('data-playing') == 'false'
        # Reinspect wrapped and staged states in the actual chapter, away from fixed nav.
        menu.select_option('13')
        if width < 600:
            assert root.locator('.osi-swipe').is_visible()
            assert root.locator('svg').evaluate('s=>s.getBoundingClientRect().width') >= 1050
            assert root.locator('.osi-canvas').evaluate('n=>n.scrollLeft') > 0
            assert root.locator('.osi-focus').evaluate('n=>{const a=n.getBoundingClientRect(),b=n.closest(".osi-canvas").getBoundingClientRect();return a.left>=b.left&&a.right<=b.right}')
        page.evaluate("window.scrollTo(0,document.querySelector('.mlsys-original-stream-ingestion').getBoundingClientRect().top+scrollY-230)")
        root.screenshot(path=str(args.screenshots / f'wrapped-{width}-{theme}.png'))
        menu.select_option('17')
        root.locator('svg').screenshot(path=str(args.screenshots / f'diagram-{width}-{theme}.png'))
        page.screenshot(path=str(args.screenshots / f'chapter-{width}-{theme}.png'))
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'), (width, theme)
        page.emulate_media(media='print')
        assert not root.is_visible()
        assert page.locator('#fig-vol3-stream-ingestion img[data-stream-static]').is_visible()
        assert not errors, errors
        reports.append({'width': width, 'theme': theme, 'capture_states': len(payload) + 1, 'accounting': 'passed', 'wraparound': 'passed', 'controls': 'passed', 'print_static': 'passed'})
        page.close()
    page = browser.new_page(java_script_enabled=False)
    page.goto(args.url)
    assert page.locator('#fig-vol3-stream-ingestion img[data-stream-static]').is_visible()
    assert page.locator('.mlsys-original-stream-ingestion').count() == 0
    page.close()
    browser.close()
print(json.dumps(reports, indent=2))
