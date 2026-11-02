#!/usr/bin/env python3
"""Exercise the factory inside the full rendered owning chapter; no production enrollment."""
import argparse,json,functools,http.server,threading
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[4]
EXT=ROOT/'books/_extensions/mlsysbook-ext/figure-explore'
parser=argparse.ArgumentParser();parser.add_argument('--screenshots',default='/tmp/original-kv-paging-review');args=parser.parse_args();shots=Path(args.screenshots);shots.mkdir(exist_ok=True)
art=json.load(open('/tmp/figures-animation-art-current/index.json'))['vol1:fig-serving-paged-kv-mapping'];chapter=Path(art['html'])
class QuietHandler(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(QuietHandler,directory=str(chapter.parents[2])))
threading.Thread(target=server.serve_forever,daemon=True).start()
chapter_url=f'http://127.0.0.1:{server.server_port}/'+chapter.relative_to(chapter.parents[2]).as_posix()
with sync_playwright() as pw:
 for engine in ['chromium','webkit']:
  browser=getattr(pw,engine).launch()
  for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
   page=browser.new_page(viewport={'width':width,'height':1000},color_scheme=theme,reduced_motion='reduce');errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   page.route('**/release-manifest.json',lambda route:route.fulfill(status=200,body='{}',content_type='application/json'));page.goto(chapter_url);page.wait_for_load_state('networkidle');page.add_style_tag(content=(EXT/'original-player.css').read_text()+(EXT/'original-kv-paging.css').read_text())
   page.add_script_tag(content=(EXT/'original-player.js').read_text());page.add_script_tag(content=(EXT/'original-kv-paging.js').read_text())
   page.evaluate('''async art=>{const outer=document.getElementById('fig-serving-paged-kv-mapping');outer.querySelectorAll('.mlsys-figure-views').forEach(n=>n.remove());const staticFigure=outer.querySelector('figure');if(staticFigure)staticFigure.style.display='none';const root=document.createElement('div');outer.append(root);outer.dataset.mlsysMovementMs='1800';outer.dataset.mlsysReadingPauseMs='5000';const svg=new DOMParser().parseFromString(art,'image/svg+xml').documentElement;window.kvController=await window.MLSysFigureLessons['original-kv-paging'](root,{outer,verifiedArtwork:svg});window.kvRoot=root;}''',Path(art['asset']).read_text())
   root=page.locator('.mlsys-original-kv-paging');root.scroll_into_view_if_needed();assert root.is_visible()
   select=root.locator('.mlsys-original-sequence select');assert select.locator('option').count()==15
   for i in range(15):
    select.select_option(str(i));assert root.get_attribute('data-step')==str(i)
    assert root.get_attribute('data-kv-released')==str(i==14).lower()
    assert root.get_attribute('data-kv-resident')==('C C D D' if i==14 else 'E E C C E E D D')
    if i in [0,2,3,11,12,13,14]:root.screenshot(path=str(shots/f'{engine}-{width}-{theme}-{i}.png'))
   root.focus();page.keyboard.press('p');assert root.get_attribute('data-step')=='13';page.keyboard.press('n');page.wait_for_function("window.kvRoot.dataset.step==='14'")
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   page.emulate_media(reduced_motion='no-preference');select.select_option('3');root.get_by_role('button',name='Expand',exact=True).click();root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(350);assert root.get_attribute('data-step')=='3';root.get_by_role('button',name='Pause',exact=True).click();before=root.locator('[aria-hidden="true"] g').last.get_attribute('transform');page.wait_for_timeout(250);assert root.get_attribute('data-step')=='3';assert root.locator('[aria-hidden="true"] g').last.get_attribute('transform')==before;root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function("window.kvRoot.dataset.step==='4'",timeout=7000);root.get_by_role('button',name='Pause',exact=True).click()
   page.keyboard.press('Escape');page.emulate_media(reduced_motion='reduce');select.select_option('11');root.get_by_role('button',name='Expand',exact=True).click();assert page.locator('dialog[open]').count()==1;page.keyboard.press('Escape');assert page.locator('dialog[open]').count()==0
   page.emulate_media(media='print');assert not root.is_visible();page.emulate_media(media='screen');assert not errors,errors
   print(engine,width,theme,'passed');page.close()
  browser.close()
