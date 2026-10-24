"""Separate inline drawings must keep their captured SVG fragment IDs local."""
from pathlib import Path
import importlib.util
import pytest

playwright=pytest.importorskip('playwright.sync_api')
ROOT=Path(__file__).resolve().parents[2]
EXT=ROOT/'books/_extensions/mlsysbook-ext/figure-explore'
spec=importlib.util.spec_from_file_location('figure_compiler_namespace',ROOT/'binder/tools/scripts/compile_figure_explore.py')
compiler=importlib.util.module_from_spec(spec);spec.loader.exec_module(compiler)


def test_inline_drawings_keep_captured_glyph_clip_and_style_references_local():
    source='''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 20 20" aria-labelledby="title0">
      <title id="title0">Local drawing</title>
      <style>#glyph0 {fill: rgb(0, 128, 0)} .clipped {clip-path: url('#clip0')}</style>
      <defs><clipPath id="clip0"><rect width="20" height="20"/></clipPath><path id="glyph0" d="M0 0H10V10Z"/></defs>
      <use href="#glyph0" clip-path="url(#clip0)"/>
      <use xlink:href="#glyph0" class="clipped"/>
    </svg>'''
    digest=compiler.structural_svg(source.encode())
    body=''.join(f'<div id="fig-{name}" data-mlsys-explore="fixture" data-mlsys-art-kind="svg" data-mlsys-art-sha1="{digest}"><figure><img src="/art.svg" alt="Original {name}"><figcaption>{name} caption</figcaption></figure></div>' for name in ('one','two'))
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch()
        page=browser.new_page()
        page.route('https://figure.test/**',lambda route:route.fulfill(content_type='image/svg+xml' if route.request.url.endswith('art.svg') else 'text/html',body=source if route.request.url.endswith('art.svg') else body))
        page.goto('https://figure.test/chapter')
        page.evaluate('''() => {
          window.MLSysFigureLessons={fixture(root,{verifiedArtwork}){
            const svg=verifiedArtwork.cloneNode(true);
            const glyph=svg.querySelector('path').id,clip=svg.querySelector('clipPath').id;
            root.append(svg);
            const next=document.createElement('button');next.textContent='Next';
            next.onclick=()=>{
              // Real factories capture references once, then reuse them later.
              svg.querySelector('use').setAttribute('href','#'+glyph);
              svg.querySelector('use').setAttribute('clip-path','url(#'+clip+')');
            };
            root.append(next);return {pause(){}};
          }};
        }''')
        page.add_script_tag(path=str(EXT/'figure-views.js'))
        page.add_script_tag(path=str(EXT/'figure-explore.js'))
        page.wait_for_function("[...document.querySelectorAll('[data-mlsys-explore]')].every(n=>n.dataset.mlsysExploreState==='ready')")
        for outer in ('#fig-one','#fig-two'):
            page.locator(outer+' [role=tab]').filter(has_text='Explore').click()
            page.locator(outer+' .mlsys-explore-panel button').click()
        results=page.evaluate('''() => [...document.querySelectorAll('.mlsys-explore-panel svg')].map(svg=>{
          const target=svg.querySelector('path');
          return {ids:[...svg.querySelectorAll('[id]')].map(n=>n.id),
            references:[...svg.querySelectorAll('use')].map(n=>n.getAttribute('href')||n.getAttributeNS('http://www.w3.org/1999/xlink','href')),
            glyph:target.id,color:getComputedStyle(target).fill,title:svg.getAttribute('aria-labelledby'),
            clip:svg.querySelector('use').getAttribute('clip-path'),style:svg.querySelector('style').textContent};
        })''')
        assert len(results)==2
        assert len({identity for drawing in results for identity in drawing['ids']})==6
        for drawing in results:
            assert drawing['references']==['#'+drawing['glyph']]*2
            assert drawing['title'] in drawing['ids']
            assert drawing['color']=='rgb(0, 128, 0)'
            assert '#'+drawing['glyph'] in drawing['style']
            assert any('#'+identity in drawing['clip'] and '#'+identity in drawing['style'] for identity in drawing['ids'] if 'clip0' in identity)
        assert page.locator('figcaption').all_text_contents()==['one caption','two caption']
        assert page.locator('.mlsys-diagram-panel img').count()==2
        browser.close()
