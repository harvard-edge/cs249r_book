"""Authored state regressions; chapter screenshots cover source-art fidelity."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "books/_extensions/mlsysbook-ext/figure-explore"


@pytest.fixture
def page():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content('<div id="root"><div id="stage"></div><div id="details"></div></div>')
        page.evaluate("""() => {
          window.MLSysOriginalPlayer = {create(spec) {
            window.lesson = spec; return {pause(){}, destroy(){}};
          }};
          window.scene = {stage:document.querySelector('#stage'), details:document.querySelector('#details')};
        }""")
        yield page
        browser.close()


def mount(page, name, artwork):
    page.add_script_tag(path=str(EXT / f"{name}.js"))
    page.evaluate("""async ([name, source]) => {
      const root = document.querySelector('#root');
      const art = new DOMParser().parseFromString(source, 'image/svg+xml').documentElement;
      window.mountedPlayer = await MLSysFigureLessons[name](root, {outer:root, verifiedArtwork:art});
    }""", [name, artwork])


def test_curation_records_follow_takeover_intervals(page):
    art = ROOT / "books/vol4/05_data/images/svg/fig05_compounding_error_flywheel.svg"
    mount(page, "original-v4-05-compounding-error-flywheel", art.read_text())
    scene = page.locator(".original-v4-05-compounding-error-flywheel-scene")
    for width in (900, 390):
        page.set_viewport_size({"width":width, "height":800})
        for index in range(4):
            page.evaluate("i => lesson.render(i, scene)", index)
            if width > 500:
                text = scene.text_content()
                assert ("state s: relabel eligible" in text) == (index >= 1)
                assert text.count("state s: excluded") == (2 if index >= 2 else 0)
                assert ("expert action a*: retained" in text) == (index == 3)
            else:
                highlights = scene.locator('rect[fill="#E3F3F5"]').count()
                assert highlights == (0, 1, 3, 4)[index]
        page.evaluate("lesson.render(0, scene)")
        assert "expert action a*: retained" not in scene.text_content()


def test_chunk_arrival_does_not_claim_future_handoff(page):
    # Constructed empty drawing at the canonical geometry: the regression is in
    # the authored handoff overlay, not in the chapter's computed background plot.
    art = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 3606 2881"/>'
    mount(page, "original-v4-03-chunk-supply-horizon", art)
    dot = page.locator('circle[data-motion-object]')
    for index, visible in ((0, False), (1, False), (2, True), (3, False)):
        page.evaluate("i => lesson.render(i, scene)", index)
        assert dot.evaluate("n => n.style.display !== 'none'") == visible
    page.evaluate("lesson.render(1, scene)")
    assert '"consumed":0' in page.locator('#root').get_attribute('data-setpoint-counts')


CAMERA_MODULES = (
    ('03-chunk-supply-horizon', '3606 2881'),
    ('05-action-taps', '680 340'),
    ('07-open-vs-closed-loop', '3658 3901'),
    ('10-latch-force-budget', '3660 2100'),
    ('12-capture-to-onset', '3913 2974'),
    ('14-authority-handshake-fsm', '680 392'),
)


@pytest.mark.parametrize('suffix,geometry', CAMERA_MODULES)
def test_camera_refollows_current_endpoint_on_resize(page, suffix, geometry):
    name = 'original-v4-' + suffix
    page.add_style_tag(path=str(EXT / f'{name}.css'))
    page.add_style_tag(content='#root {width:720px}')
    page.evaluate("""() => {
      const Observer = window.ResizeObserver;
      window.cameraObservers = [];
      window.ResizeObserver = class extends Observer {
        constructor(callback) { super(callback); cameraObservers.push(this); }
        disconnect() { this.disconnected = true; super.disconnect(); }
      };
    }""")
    # Constructed canonical-geometry artwork isolates camera behavior from pixels.
    art = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {geometry}"/>'
    sources = {'05-action-taps':'05_data/images/svg/fig05_four_stream_action_taps.svg',
               '14-authority-handshake-fsm':'14_intervention/images/svg/fig14_authority_handshake_fsm.svg'}
    if suffix in sources:
        art = (ROOT / 'books/vol4' / sources[suffix]).read_text()
    mount(page, name, art)
    for index in range(page.evaluate('lesson.steps.length')):
        page.evaluate('i => lesson.render(i, scene)', index)
        for width in (1200, 390, 720):
            page.evaluate('w => root.style.width = w + "px"', width)
            page.wait_for_timeout(80)
            observed = page.locator('.v4b-canvas').evaluate('(n) => [n.scrollLeft,n.scrollTop]')
            # Re-rendering the same endpoint supplies the canonical focus at the
            # new scale. A resize must already have produced that viewport.
            page.evaluate('i => lesson.render(i, scene)', index)
            expected = page.locator('.v4b-canvas').evaluate('(n) => [n.scrollLeft,n.scrollTop]')
            assert observed == pytest.approx(expected, abs=1)
        page.wait_for_timeout(80)
        page.locator('.v4b-canvas').evaluate('n => {n.scrollTop=0;n.scrollLeft=0;}')
        page.wait_for_timeout(80)
        assert page.locator('.v4b-canvas').evaluate('n => [n.scrollLeft,n.scrollTop]') == [0, 0]
    page.evaluate('mountedPlayer.destroy()')
    assert page.evaluate('cameraObservers.every(o => o.disconnected)')


def test_chunk_tail_stays_visible_in_real_expanded_player(page):
    name = 'original-v4-03-chunk-supply-horizon'
    page.emulate_media(reduced_motion='reduce')
    page.set_viewport_size({'width':1440, 'height':1000})
    for sheet in ('original-player', name):
        page.add_style_tag(path=str(EXT / f'{sheet}.css'))
    page.add_style_tag(content='#root {width:720px;max-width:100%} dialog #root {width:100%}')
    page.add_script_tag(path=str(EXT / 'original-player.js'))
    page.evaluate("Object.assign(root.dataset, {mlsysMovementMs:'1000',mlsysReadingPauseMs:'1000'})")
    mount(page, name, '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 3606 2881"/>')
    page.evaluate('mountedPlayer.go(6)')
    page.wait_for_timeout(80)
    page.evaluate('mountedPlayer.expand()')
    for width in (1440, 390, 1000):
        page.set_viewport_size({'width':width, 'height':1000})
        page.wait_for_timeout(100)
        assert page.locator('.v4b-canvas').evaluate('''canvas => {
          const svg=canvas.querySelector('svg'), point=svg.createSVGPoint();
          point.x=3478; point.y=2290;
          const tail=point.matrixTransform(svg.getScreenCTM()), box=canvas.getBoundingClientRect();
          return tail.x>=box.left && tail.x<=box.right && tail.y>=box.top && tail.y<=box.bottom;
        }''')
    page.evaluate('mountedPlayer.destroy()')
