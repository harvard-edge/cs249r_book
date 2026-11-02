"""QA fixtures load the production data bundle before their candidate factory."""
from argparse import Namespace
import importlib.util
import json
import mimetypes
from pathlib import Path
import shutil
from urllib.parse import urlparse

import pytest

playwright = pytest.importorskip('playwright.sync_api')
ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / 'books/_extensions/mlsysbook-ext/figure-explore'


def test_bundle_precedes_alias_data_and_factory_and_is_loaded_once(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('fixture_builder', ROOT / 'binder/tools/scripts/build_original_figure_fixture.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    books = tmp_path / 'books'
    extension = books / '_extensions/mlsysbook-ext/figure-explore'
    extension.mkdir(parents=True)
    for name in ['figure-explore.js', 'figure-views.js', 'figure-views.css', 'original-player.js', 'original-player.css']:
        shutil.copyfile(EXT / name, extension / name)
    (extension / 'fixture.css').write_text('')
    (extension / 'bundle.js').write_text('window.loadOrder=["bundle"];window.fixtureConstant=42;')
    (extension / 'alias.js').write_text('if(window.fixtureConstant!==42)throw Error("Bundle missing");window.loadOrder.push("alias");')
    (extension / 'factory.js').write_text('''if(window.fixtureConstant!==42)throw Error('Bundle must precede factory module');
    window.loadOrder.push('module');window.MLSysFigureLessons={fixture(root,{verifiedArtwork}){
      if(window.fixtureConstant!==42)throw Error('Bundle must load before factory');
      window.loadOrder.push('factory');root.append(verifiedArtwork);return {pause(){}};
    }};''')
    chapter = books / '_build/source/output/vol1/chapter/chapter.html'
    chapter.parent.mkdir(parents=True)
    chapter.write_text('''<!doctype html><html><head><title>Fixture</title></head><body>
      <div id="fig-fixture" data-mlsys-explore="old"><figure><img src="original.svg" alt="Original"><figcaption>Original caption</figcaption></figure></div>
    </body></html>''')
    asset = chapter.parent / 'original.svg'
    asset.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10"/></svg>')
    timing = books / 'config/shared/html/figure-explore.yml'
    timing.parent.mkdir(parents=True)
    timing.write_text('figure-explore:\n  movement-ms: 1800\n  reading-pause-ms: 5000\n')
    index = tmp_path / 'index.json'
    index.write_text(json.dumps({'vol1:fig-fixture': {'html':str(chapter), 'asset':str(asset)}}))
    entry = tmp_path / 'candidate/entry.json'
    entry.parent.mkdir()
    entry.write_text(json.dumps({'key':'vol1:fig-fixture', 'implementation': {
        'name':'fixture', 'script':'factory.js', 'stylesheet':'fixture.css',
        'bundle':'bundle.js', 'data_script':'bundle.js', 'data_scripts':['alias.js', 'bundle.js']}}))
    monkeypatch.setattr(builder, 'ROOT', tmp_path)
    monkeypatch.setattr(builder, 'BOOKS', books)
    monkeypatch.setattr(builder, 'EXT', extension)
    result = builder.build(Namespace(entry=entry, art_index=index, output=None,
                                     base_url='http://fixture.test', data_script=['alias.js']))
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        def serve(route):
            path = books / urlparse(route.request.url).path.lstrip('/')
            if not path.is_file():
                route.fulfill(status=404, body='Not found')
            else:
                route.fulfill(body=path.read_bytes(), content_type=mimetypes.guess_type(path)[0] or 'application/octet-stream')
        page.route('http://fixture.test/**', serve)
        page.goto(result['url'])
        page.wait_for_function("document.querySelector('[data-fixture-qa-only]').dataset.fixtureState==='ready'")
        assert page.evaluate('window.loadOrder') == ['bundle', 'alias', 'module', 'factory']
        assert not errors
        assert page.locator('[data-mlsys-explore]').count() == 0
        assert page.locator('figcaption').all_text_contents() == ['Original caption']
        assert page.locator('.mlsys-explore-panel svg').is_visible()
        browser.close()
