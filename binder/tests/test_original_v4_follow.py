"""Horizontal V4 follow targets survive resize/Expand without overriding manual scroll."""
from pathlib import Path

import pytest

playwright = pytest.importorskip('playwright.sync_api')
ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / 'books/_extensions/mlsysbook-ext/figure-explore'
MODULES = [
    '03-request-local-prefill-decode', '05-compounding-error-flywheel',
    '08-rolling-shutter-rows', '10-mug-lease-timeline', '13-loaded-tick',
    '15-hardware-fault-injection', '02-motor-torque-speed-envelope',
    '04-multirate-cadence-buffer', '13-thermal-hysteresis',
]
SCROLLER = """() => [...document.querySelectorAll('#root .mlsys-original-stage, #root .mlsys-original-stage *')]
    .find(n => n.scrollWidth > n.clientWidth + 2 && ['auto','scroll'].includes(getComputedStyle(n).overflowX))"""


@pytest.mark.parametrize('module', MODULES)
def test_horizontal_follow_restores_position_and_leaves_manual_scroll(module):
    # This minimal drawing isolates the viewport contract. Actual source artwork
    # and chapter fidelity are checked separately in the chapter browser sweep.
    name = 'original-v4-' + module
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 390, 'height': 1000}, reduced_motion='reduce')
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.set_content('<div id="outer" style="width:300px" data-mlsys-movement-ms="5000" data-mlsys-reading-pause-ms="10"><div id="root"></div></div>')
        page.add_style_tag(path=str(EXT / 'original-player.css'))
        page.add_style_tag(path=str(EXT / (name + '.css')))
        page.add_script_tag(path=str(EXT / 'original-player.js'))
        page.add_script_tag(path=str(EXT / (name + '.js')))
        page.evaluate("""async name => {
            const svg = document.createElementNS('http://www.w3.org/2000/svg','svg');
            svg.setAttribute('viewBox','0 0 680 500');
            svg.dataset.mlsysArtworkKind='png';
            window.lesson = await MLSysFigureLessons[name](document.querySelector('#root'), {
                outer:document.querySelector('#outer'), verifiedArtwork:svg
            });
        }""", name)
        menu = page.locator('.mlsys-original-sequence select')
        values = menu.locator('option').evaluate_all('(options)=>options.map(o=>o.value)')
        for value in values:
            menu.select_option(value)
            page.wait_for_timeout(60)
            page.evaluate('window.scroller = (' + SCROLLER + ')()')
            before = page.evaluate('scroller.scrollLeft')
            page.get_by_role('button', name='Expand', exact=True).click()
            page.wait_for_timeout(60)
            page.keyboard.press('Escape')
            page.wait_for_timeout(60)
            assert abs(page.evaluate('scroller.scrollLeft') - before) <= 1, (module, value)
        assert before > 0
        page.evaluate('scroller.scrollLeft=0')
        page.wait_for_timeout(100)
        assert page.evaluate('scroller.scrollLeft') == 0, 'paused manual inspection must not recenter'
        page.evaluate("document.querySelector('#outer').style.width='340px'")
        page.wait_for_timeout(100)
        assert page.evaluate('scroller.scrollLeft') > 0, 'a resized frame should recover the active target'
        # Resize an interrupted movement without committing its pending endpoint.
        menu.select_option(values[0])
        page.emulate_media(reduced_motion='no-preference')
        page.get_by_role('button', name='Next', exact=True).click()
        page.wait_for_timeout(40)
        page.evaluate('lesson.pause()')
        evidence = page.evaluate("({step:document.querySelector('#root').dataset.step, progress:document.querySelector('#root').dataset.animationProgress, values:document.querySelector('#root').dataset.values})")
        page.get_by_role('button', name='Expand', exact=True).click()
        page.wait_for_timeout(60)
        page.keyboard.press('Escape')
        page.wait_for_timeout(60)
        assert page.evaluate("({step:document.querySelector('#root').dataset.step, progress:document.querySelector('#root').dataset.animationProgress, values:document.querySelector('#root').dataset.values})") == evidence
        page.evaluate('lesson.destroy()')
        page.evaluate("document.querySelector('#outer').style.width='280px'")
        page.wait_for_timeout(100)
        assert page.locator('#root').text_content() == ''
        assert not errors, errors
        browser.close()
