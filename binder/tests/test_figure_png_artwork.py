"""Original PNG factories receive only hash-verified, decoded embedded bytes."""
from hashlib import sha1
from pathlib import Path
import struct
import zlib

import pytest

playwright = pytest.importorskip('playwright.sync_api')
ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / 'books/_extensions/mlsysbook-ext/figure-explore'


def png_bytes():
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    header = struct.pack('>IIBBBBB', 2, 3, 8, 2, 0, 0, 0)
    pixels = b'\x00\xff\x00\x00\x00\x00\xff' * 3
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b'')


def run_fixture(page, *, failure=None, registered=True):
    artwork = png_bytes()
    if failure == 'decode':
        artwork = artwork[:33] + artwork[-12:]
    elif failure == 'header':
        artwork = b'not a PNG'
    digest = '0' * 40 if failure == 'hash' else sha1(artwork).hexdigest()
    dimensions = 'data-mlsys-art-width="9"' if failure == 'dimensions' else 'data-mlsys-art-width="2" data-mlsys-art-height="3"'
    if not registered:
        dimensions = ''
    body = f'''<style>.mlsys-explore-panel svg {{width:240px;height:auto;image-rendering:pixelated}}</style>
    <div id="fig-png" data-mlsys-explore="fixture" data-mlsys-art-kind="png" data-mlsys-art-sha1="{digest}" {dimensions}>
      <figure><img src="/art.png" alt="Original raster"><figcaption>Original caption</figcaption></figure>
    </div>'''
    requests = []
    def route(request):
        if request.request.url.endswith('art.png'):
            requests.append(request.request.resource_type)
            request.fulfill(status=503 if failure == 'fetch' else 200, content_type='image/png', body=artwork)
        else:
            request.fulfill(content_type='text/html', body=body)
    page.route('https://figure.test/**', route)
    page.goto('https://figure.test/chapter')
    page.evaluate('''failure => {
      window.originalNodes=[...document.querySelector('figure').childNodes];
      window.factoryCalls=0;
      if(failure==='source'){
        const decode=Image.prototype.decode;
        Image.prototype.decode=async function(){await decode.call(this);document.querySelector('figure img').src='/changed.png';};
      }
      window.MLSysFigureLessons={fixture(root,{verifiedArtwork}){
        window.factoryCalls++;
        window.verified=verifiedArtwork;
        root.append(verifiedArtwork.cloneNode(true));
        if(failure==='factory')throw Error('Factory failed');
        return {pause(){}};
      }};
    }''', failure)
    page.add_script_tag(path=str(EXT / 'figure-views.js'))
    page.add_script_tag(path=str(EXT / 'figure-explore.js'))
    page.wait_for_function("['ready','failed'].includes(document.querySelector('#fig-png').dataset.mlsysExploreState)")
    return artwork, requests


@pytest.mark.parametrize('registered', [False, True])
def test_verified_png_has_intrinsic_geometry_exact_embedded_bytes_and_one_fetch(registered):
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        artwork, requests = run_fixture(page, registered=registered)
        result = page.evaluate('''() => ({
          state:document.querySelector('#fig-png').dataset.mlsysExploreState,
          tag:verified.localName,viewBox:verified.getAttribute('viewBox'),
          width:verified.getAttribute('width'),height:verified.getAttribute('height'),
          metadata:{...verified.dataset},href:verified.querySelector('image').getAttribute('href'),
          imageAttributes:[...verified.querySelector('image').attributes].map(a=>[a.name,a.value]),
          children:verified.children.length,calls:factoryCalls
        })''')
        import base64
        assert result['state'] == 'ready' and result['calls'] == 1
        assert result['tag'] == 'svg' and result['viewBox'] == '0 0 2 3'
        assert (result['width'], result['height']) == ('2', '3')
        assert result['metadata'] == {'mlsysArtworkKind': 'png', 'mlsysArtworkWidth': '2', 'mlsysArtworkHeight': '3'}
        assert base64.b64decode(result['href'].split(',', 1)[1]) == artwork
        assert result['children'] == 1
        assert dict(result['imageAttributes']) == {'x':'0','y':'0','width':'2','height':'3','href':result['href']}
        assert requests.count('fetch') == 1
        assert all(kind in ('image', 'fetch') for kind in requests)
        assert page.locator('.mlsys-diagram-panel img').get_attribute('src') == '/art.png'
        assert page.locator('figcaption').all_text_contents() == ['Original caption']
        page.get_by_role('tab', name='Explore', exact=True).click()
        page.locator('.mlsys-explore-panel svg').screenshot(path='/tmp/figure-png-wrapper-verified.png')
        browser.close()


@pytest.mark.parametrize('failure', ['hash', 'fetch', 'dimensions', 'header', 'decode', 'source', 'factory'])
def test_png_failure_retains_exact_static_nodes_and_caption(failure):
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        run_fixture(page, failure=failure)
        assert page.locator('#fig-png').get_attribute('data-mlsys-explore-state') == 'failed'
        assert page.evaluate("document.querySelector('figure').childNodes.length===originalNodes.length && [...document.querySelector('figure').childNodes].every((n,i)=>n===originalNodes[i])")
        assert page.locator('figcaption').all_text_contents() == ['Original caption']
        assert page.locator('[role=tab]').count() == 0
        assert page.locator('figure svg').count() == 0
        assert page.evaluate('factoryCalls') == (1 if failure == 'factory' else 0)
        browser.close()
