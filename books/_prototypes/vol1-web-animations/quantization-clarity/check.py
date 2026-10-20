#!/usr/bin/env python3
"""Check that moving operands cannot prematurely commit an accumulator result."""
import argparse
import json
from pathlib import Path
import runpy

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[4]
DATA = runpy.run_path(str(ROOT/'books/_extensions/mlsysbook-ext/figure-explore/original-quantization.calc.py'))['calculate']()


def check(url, engine):
    records = []
    with sync_playwright() as playwright:
        browser = getattr(playwright, engine).launch()
        page = browser.new_page(viewport={'width':390, 'height':844})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url, wait_until='networkidle')
        root = page.locator('#fig-weight-activations-quantization .mlsys-original-quantization')
        root.wait_for(state='visible')
        sequence = root.locator('.mlsys-original-sequence select')
        register = root.locator('[data-oq-box="INT32 sum"] .oq-value')
        for source, held, arrived in [(2, 'not started', '0'), (3, '0', str(DATA['products'][0])), (4, str(DATA['products'][0]), str(DATA['accumulator']))]:
            sequence.select_option(str(source))
            root.get_by_role('combobox', name='Playback speed').select_option('0.5')
            # A reader-selected pan must not hide the next transfer.
            canvas = root.locator('.oq-canvas')
            canvas.evaluate('(node)=>node.scrollLeft=node.scrollWidth')
            play = root.get_by_role('button', name='Play', exact=True)
            play.scroll_into_view_if_needed()
            page.wait_for_timeout(200)
            play.click()
            token = root.locator('.oq-token').first
            token.wait_for()
            initial = token.get_attribute('transform')
            page.wait_for_function('(args)=>document.querySelector(args.selector)?.getAttribute("transform")!==args.initial', arg={'selector':'#fig-weight-activations-quantization .oq-token', 'initial':initial})
            root.get_by_role('button', name='Pause', exact=True).click()
            assert root.get_attribute('data-step') == str(source)
            assert register.text_content() == held
            viewport = canvas.bounding_box()
            destination = root.locator('.oq-pending').bounding_box()
            assert destination['x'] >= viewport['x']-2
            assert destination['x']+destination['width'] <= viewport['x']+viewport['width']+2
            assert root.locator('.oq-calculation-label').inner_text().startswith('In transit')
            if source == 2:
                assert 'accumulation not started' in root.locator('.oq-calculation').inner_text()
            else:
                assert 'sum held at '+held in root.locator('.oq-calculation').inner_text()
                pair = source-3
                expected = f"{DATA['qx'][pair]} × {DATA['qw'][pair]}".replace('-', '−')
                assert root.locator('[data-oq-box="Integer matmul"] .oq-value').text_content() == expected
            frozen = token.get_attribute('transform')
            root.get_by_role('combobox', name='Playback speed').select_option('2')
            page.wait_for_timeout(100)
            assert token.get_attribute('transform') == frozen
            assert register.text_content() == held
            root.get_by_role('button', name='Play', exact=True).click()
            page.wait_for_function('(root)=>root.dataset.step===String(Number(root.dataset.oqPending))||!root.dataset.oqPending', arg=root.element_handle())
            if root.get_by_role('button', name='Pause', exact=True).count():
                root.get_by_role('button', name='Pause', exact=True).click()
            assert root.get_attribute('data-step') == str(source+1)
            assert register.text_content() == arrived
            records.append({'source_step':source, 'held':held, 'arrived':arrived, 'passed':True})
        assert not errors, errors
        browser.close()
    return {'engine':engine, 'passed':True, 'checks':records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='Actual chapter URL with Explore selected and the figure anchor.')
    parser.add_argument('--engine', choices=['chromium','webkit'], default='chromium')
    args = parser.parse_args()
    print(json.dumps(check(args.url, args.engine), indent=2))
