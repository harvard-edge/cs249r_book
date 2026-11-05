"""Renderer regression: evicted KV blocks retain distinct pool destinations."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "books/_extensions/mlsysbook-ext/figure-explore"
ART = ROOT / "books/vol3/05_kv_cache/images/svg/radix_prefix_tree_architecture.svg"


def test_eviction_appends_to_pool_and_detaches_parent():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content('<div id="root"><div id="stage"></div><div id="details"></div></div>')
        # Exercise authored render/transition endpoints directly. Playback timing
        # is covered by the shared-player tests; no artificial wait is needed here.
        page.evaluate("""() => { window.MLSysOriginalPlayer = {create(spec) {
          window.lesson = spec; return {pause(){}, destroy(){}};
        }}; }""")
        page.add_script_tag(path=str(EXT / "original-v3-radix-tree-cache.js"))
        page.evaluate("""async source => {
          const artwork = new DOMParser().parseFromString(source, 'image/svg+xml').documentElement;
          const root = document.querySelector('#root');
          await MLSysFigureLessons['original-v3-radix-tree-cache'](root, {outer: root, verifiedArtwork: artwork});
          window.scene = {stage: document.querySelector('#stage'), details: document.querySelector('#details')};
          lesson.render(2, scene);
        }""", ART.read_text())
        result = page.evaluate("""() => {
          const art = scene.stage.querySelector('svg'), layer = art.lastElementChild;
          const parent = [...art.querySelectorAll('rect')].find(n => n.getAttribute('x') === '312' && n.getAttribute('y') === '128');
          const beforePulse = parent.getAttribute('stroke-width');
          lesson.transition({from:1, to:2, onFrame: frame => frame(1)});
          lesson.render(2, scene);
          const afterPulse = parent.getAttribute('stroke-width');
          const prior = [...layer.querySelectorAll('rect')].map(n => Number(n.getAttribute('x')));
          lesson.transition({from:2, to:3, onFrame: frame => frame(1)});
          const arriving = [...layer.querySelectorAll('rect[data-motion-object]')].map(n => Number(n.getAttribute('x')));
          lesson.render(3, scene);
          const committed = [...art.lastElementChild.querySelectorAll('rect')].map(n => Number(n.getAttribute('x')));
          const edge = [...art.querySelectorAll('path')].find(n => n.getAttribute('d') === 'M266.5,94 V111 H376 V128');
          const evictedOpacity = Number(edge.style.opacity);
          lesson.render(0, scene);
          return {prior, arriving, committed, beforePulse, afterPulse, evictedOpacity, restoredOpacity:Number(edge.style.opacity)};
        }""")
        assert result["beforePulse"] == result["afterPulse"]
        assert result["prior"] and result["arriving"]
        assert set(result["prior"]).isdisjoint(result["arriving"])
        assert sorted(result["prior"] + result["arriving"]) == result["committed"]
        assert result["evictedOpacity"] < result["restoredOpacity"] == 1
        browser.close()
