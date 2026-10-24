#!/usr/bin/env python3
"""Exercise the factory inside the full rendered owning chapter; no production enrollment."""
import argparse,json,math,functools,http.server,threading
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[4]
EXT=ROOT/'books/_extensions/mlsysbook-ext/figure-explore'
parser=argparse.ArgumentParser();parser.add_argument('--screenshots',default='/tmp/original-activation-checkpointing-review');parser.add_argument('--engine',choices=['chromium','webkit']);parser.add_argument('--width',type=int);args=parser.parse_args();shots=Path(args.screenshots);shots.mkdir(exist_ok=True)
art=json.load(open('/tmp/figures-animation-art-current/index.json'))['vol1:fig-activation-checkpointing'];chapter=Path(art['html'])
class QuietHandler(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(QuietHandler,directory=str(chapter.parents[2])))
threading.Thread(target=server.serve_forever,daemon=True).start()
chapter_url=f'http://127.0.0.1:{server.server_port}/'+chapter.relative_to(chapter.parents[2]).as_posix()
with sync_playwright() as pw:
 for engine in ['chromium','webkit']:
  if args.engine and engine!=args.engine:continue
  browser=getattr(pw,engine).launch()
  for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
   if args.width and width!=args.width:continue
   page=browser.new_page(viewport={'width':width,'height':1000},color_scheme=theme,reduced_motion='reduce');errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   page.route('**/release-manifest.json',lambda route:route.fulfill(status=200,body='{}',content_type='application/json'));page.goto(chapter_url);page.wait_for_load_state('networkidle');page.add_style_tag(content=(EXT/'original-player.css').read_text()+(EXT/'original-activation-checkpointing.css').read_text())
   page.add_script_tag(content=(EXT/'original-player.js').read_text());page.add_script_tag(content=(EXT/'original-activation-checkpointing.js').read_text())
   page.evaluate('''async art=>{const outer=document.getElementById('fig-activation-checkpointing');outer.querySelectorAll('.mlsys-figure-views').forEach(n=>n.remove());const staticFigure=outer.querySelector('figure');if(staticFigure)staticFigure.style.display='none';const root=document.createElement('div');outer.append(root);outer.dataset.mlsysMovementMs='1800';outer.dataset.mlsysReadingPauseMs='5000';const svg=new DOMParser().parseFromString(art,'image/svg+xml').documentElement;window.checkpointController=await window.MLSysFigureLessons['original-activation-checkpointing'](root,{outer,verifiedArtwork:svg});window.checkpointRoot=root;}''',Path(art['asset']).read_text())
   root=page.locator('.mlsys-original-activation-checkpointing');root.scroll_into_view_if_needed();assert root.is_visible()
   select=root.locator('.mlsys-original-sequence select');total=select.locator('option').count();assert total==22,total
   vals=[.5]
   for _ in range(4):vals.append(math.tanh(vals[-1]))
   grads=[None]*4;adjoint=1
   for i in range(4,0,-1):
    dz=adjoint*(1-vals[i]**2);grads[i-1]=dz*vals[i-1];adjoint=dz
   for i in range(total):
    select.select_option(str(i));assert root.get_attribute('data-step')==str(i)
    label=select.locator('option:checked').inner_text()
    if 'Recompute a2' in label:assert root.get_attribute('data-checkpoint-live')=='0,1,2,3'
    if 'Backward layer 3' in label:assert root.get_attribute('data-checkpoint-live')=='0,1,2,3'
    if 'Release a3' in label:assert root.get_attribute('data-checkpoint-live')=='0,1,2'
    if 'Release a2' in label:assert root.get_attribute('data-checkpoint-live')=='0,1'
    if 'Release a1' in label:assert root.get_attribute('data-checkpoint-live')=='0'
    if i in [0,2,3,8,9,12,13,14,15,20,21]:root.screenshot(path=str(shots/f'{engine}-{width}-{theme}-{i}.png'))
   assert root.get_attribute('data-checkpoint-live')==''
   assert root.get_attribute('data-checkpoint-recomputed')=='3'
   assert root.get_attribute('data-checkpoint-gradients')=='4'
   for i in range(4):
    actual=float(root.locator(f'[data-gradient="{i+1}"]').inner_text().split(':')[1]);assert abs(actual-grads[i])<.0006
   root.focus();page.keyboard.press('p');assert root.get_attribute('data-step')=='20';page.keyboard.press('n');page.wait_for_function("window.checkpointRoot.dataset.step==='21'")
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   page.emulate_media(reduced_motion='no-preference');select.select_option('1');root.get_by_role('button',name='Expand',exact=True).click();root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(350);assert root.get_attribute('data-step')=='1';root.get_by_role('button',name='Pause',exact=True).click();before=root.locator('[aria-hidden="true"] g').last.get_attribute('transform');page.wait_for_timeout(250);assert root.get_attribute('data-step')=='1';assert root.locator('[aria-hidden="true"] g').last.get_attribute('transform')==before;root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function("window.checkpointRoot.dataset.step==='2'",timeout=7000);root.get_by_role('button',name='Pause',exact=True).click()
   select.select_option('8');root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(250)
   assert root.get_attribute('data-step')=='8';assert root.get_attribute('data-checkpoint-recomputed')=='0'
   assert root.locator('[data-activation-node="8"] text').last.text_content()==f'{vals[3]:.3f}'
   assert root.locator('[data-activation-node="9"] text').last.text_content()=='—'
   root.get_by_role('button',name='Next',exact=True).click();assert root.get_attribute('data-step')=='9';assert root.get_attribute('data-checkpoint-recomputed')=='1'
   page.keyboard.press('Escape');page.emulate_media(reduced_motion='reduce');select.select_option('11');root.get_by_role('button',name='Expand',exact=True).click();assert page.locator('dialog[open]').count()==1;page.keyboard.press('Escape');assert page.locator('dialog[open]').count()==0
   page.emulate_media(media='print');assert not root.is_visible();page.emulate_media(media='screen');assert not errors,errors
   print(engine,width,theme,'passed');page.close()
  browser.close()
