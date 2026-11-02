"""Verify the short walkthrough, scan completion, and shared detailed states."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', required=True, help='Actual rendered architecture chapter URL')
parser.add_argument('--engine', choices=['chromium', 'webkit'], default='chromium')
parser.add_argument('--screenshots', type=Path, default=Path('/tmp/convolution-guided-review'))
args = parser.parse_args()
args.screenshots.mkdir(parents=True, exist_ok=True)
url = args.url
ROOT = Path(__file__).resolve().parents[3]
expected = json.loads((ROOT/'books/_extensions/mlsysbook-ext/figure-explore/convolution-values.js').read_text().split(' = ',1)[1].rstrip(';\n'))
with sync_playwright() as p:
    browser = getattr(p, args.engine).launch()
    for width, theme in [(1440, 'light'), (390, 'light'), (1440, 'dark'), (390, 'dark')]:
        page = browser.new_page(viewport={'width': width, 'height': 1100}, color_scheme=theme, reduced_motion='reduce')
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function("document.querySelector('#fig-cnn').dataset.mlsysExploreState === 'ready'")
        page.locator('#fig-cnn [role=tab]').filter(has_text='Explore').click()
        panel = page.locator('#fig-cnn .life-panel')
        panel.wait_for(state='visible')
        assert panel.locator('.life-scope').is_visible()
        assert 'before activation' in panel.locator('.life-scope').inner_text()
        scan = panel.locator('[data-control=scan]')
        assert scan.is_disabled()
        # Virtual time exercises autoplay through the deliberate checkpoint.
        page.clock.install()
        panel.locator('[data-control=play]').click()
        for _ in range(6):
            page.clock.run_for(int(page.locator('#fig-cnn').get_attribute('data-mlsys-reading-pause-ms'))+100)
        assert panel.get_attribute('data-step') == '6'
        assert panel.get_attribute('data-running') == 'false'
        assert panel.get_attribute('data-outputs') == '2'
        assert panel.locator('.life-checkpoint').is_visible()
        assert scan.is_visible()
        page.screenshot(path=str(args.screenshots / f'checkpoint-{width}-{theme}.png'))
        scan.click()
        page.clock.run_for(1000)
        assert panel.get_attribute('data-step') == '75'
        assert panel.get_attribute('data-running') == 'false'
        assert panel.get_attribute('data-outputs') == str(len(expected['positions']))
        assert panel.locator('.life-takeaway').is_visible()
        labels = panel.locator('[data-output-index]').all_text_contents()
        assert labels == [str(position['output']) for position in expected['positions']]
        page.screenshot(path=str(args.screenshots / f'complete-{width}-{theme}.png'))
        panel.locator('svg').screenshot(path=str(args.screenshots / f'drawing-{width}-{theme}.png'))
        # Back returns to the actual preceding detailed state, not a scan-only copy.
        panel.locator('[data-control=back]').click()
        assert panel.get_attribute('data-step') == '74'
        assert panel.get_attribute('data-outputs') == '24'
        assert not panel.locator('.life-takeaway').is_visible()
        panel.locator('[data-control=restart]').click()
        assert panel.get_attribute('data-step') == '0'
        assert panel.get_attribute('data-outputs') == '0'
        assert scan.is_disabled()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        assert not errors, errors
        print(json.dumps({'engine': args.engine, 'width': width, 'theme': theme, 'guided_scan': 'passed'}))
        page.close()
    # Also exercise the overview with real interpolation and pause/resume.
    page = browser.new_page(viewport={'width': 1440, 'height': 1100})
    page.goto(url)
    page.wait_for_function("document.querySelector('#fig-cnn').dataset.mlsysExploreState === 'ready'")
    page.locator('#fig-cnn [role=tab]').filter(has_text='Explore').click()
    panel = page.locator('#fig-cnn .life-panel')
    panel.wait_for(state='visible')
    panel.get_by_role('combobox').select_option('6')
    panel.locator('[data-control=scan]').click()
    page.wait_for_timeout(400)
    panel.locator('[data-control=play]').click()
    assert panel.get_attribute('data-moving') == 'true'
    assert panel.get_attribute('data-outputs') == '2'
    frozen = panel.locator('svg').inner_html()
    page.wait_for_timeout(250)
    assert panel.locator('svg').inner_html() == frozen
    page.screenshot(path=str(args.screenshots / 'scan-paused.png'))
    panel.locator('[data-control=play]').click()
    page.wait_for_function("document.querySelector('#fig-cnn .life-panel').dataset.step === '75' && document.querySelector('#fig-cnn .life-panel').dataset.moving === 'false'", timeout=60000)
    assert panel.get_attribute('data-running') == 'false'
    assert panel.locator('[data-output-index]').all_text_contents() == [str(position['output']) for position in expected['positions']]
    print(json.dumps({'engine': args.engine, 'live_scan_pause_resume': 'passed'}))
    page.close()
    browser.close()
