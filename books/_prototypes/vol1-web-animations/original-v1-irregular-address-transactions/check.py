#!/usr/bin/env python3
"""Full owning-chapter fixture; production enrollment is checked separately."""
import argparse,functools,http.server,json,threading
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[4];EXT=ROOT/'books/_extensions/mlsysbook-ext/figure-explore';ENTRY=json.loads((Path(__file__).parent/'entry.json').read_text());KEY=ENTRY['key'];NAME=ENTRY['implementation']['name'];FID=KEY.split(':')[1]
p=argparse.ArgumentParser();p.add_argument('--engine',choices=['chromium','webkit']);p.add_argument('--width',type=int);args=p.parse_args();art=json.load(open('/tmp/figures-animation-art-current/index.json'))[KEY];chapter=Path(art['html']);shots=Path('/tmp/'+NAME+'-review');shots.mkdir(exist_ok=True)
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(chapter.parents[2])));threading.Thread(target=server.serve_forever,daemon=True).start();url=f'http://127.0.0.1:{server.server_port}/'+chapter.relative_to(chapter.parents[2]).as_posix()
with sync_playwright() as pw:
 for engine in ['chromium','webkit']:
  if args.engine and args.engine!=engine:continue
  browser=getattr(pw,engine).launch()
  for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
   if args.width and args.width!=width:continue
   page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme,reduced_motion='reduce');errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.route('**/release-manifest.json',lambda route:route.fulfill(status=200,body='{}',content_type='application/json'));page.goto(url);page.wait_for_load_state('networkidle');page.add_style_tag(content=(EXT/'original-player.css').read_text()+(EXT/ENTRY['implementation']['stylesheet']).read_text());page.add_script_tag(content=(EXT/'original-player.js').read_text());page.add_script_tag(content=(EXT/ENTRY['implementation']['script']).read_text())
   page.evaluate('''async({fid,name,art})=>{const outer=document.getElementById(fid);outer.querySelectorAll('.mlsys-figure-views').forEach(n=>n.remove());const stat=outer.querySelector('figure');if(stat)stat.style.display='none';const root=document.createElement('div');outer.append(root);outer.dataset.mlsysMovementMs='1800';outer.dataset.mlsysReadingPauseMs='5000';const svg=new DOMParser().parseFromString(art,'image/svg+xml').documentElement;window.testController=await window.MLSysFigureLessons[name](root,{outer,verifiedArtwork:svg});window.testRoot=root;}''',{'fid':FID,'name':NAME,'art':Path(art['asset']).read_text()})
   root=page.locator('.'+NAME);root.scroll_into_view_if_needed();menu=root.locator('.mlsys-original-sequence select');count=menu.locator('option').count();assert count>=4
   endpoints=[]
   for i in range(count):
    menu.select_option(str(i));assert root.get_attribute('data-step')==str(i);ledger=json.loads(root.get_attribute('data-ledger'));endpoints.append(ledger)
    if i in {0,count//2,count-1}:root.screenshot(path=str(shots/f'{engine}-{width}-{theme}-{i}.png'))
   expected=ENTRY.get('expected_final',{})
   for key,value in expected.items():assert endpoints[-1].get(key)==value,(key,endpoints[-1],value)
   menu.select_option('1');a=root.get_attribute('data-ledger');menu.select_option(str(count-1));menu.select_option('1');assert root.get_attribute('data-ledger')==a
   root.focus();page.keyboard.press('p');assert root.get_attribute('data-step')=='0';page.keyboard.press('n');page.wait_for_function("window.testRoot.dataset.step==='1'")
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   page.emulate_media(reduced_motion='no-preference');menu.select_option('0');root.get_by_role('button',name='Expand',exact=True).click();root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(300);assert root.get_attribute('data-step')=='0';before=root.get_attribute('data-ledger');root.get_by_role('button',name='Pause',exact=True).click();packet=root.locator('[aria-hidden="true"] g').first;transform=packet.get_attribute('transform') if packet.count() else None;paint_before=root.locator('[data-mechanism-values]').inner_html();page.wait_for_timeout(250);assert root.get_attribute('data-ledger')==before;assert root.locator('[data-mechanism-values]').inner_html()==paint_before
   if transform:assert packet.get_attribute('transform')==transform
   root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function("window.testRoot.dataset.step==='1'",timeout=7000);root.get_by_role('button',name='Pause',exact=True).click();page.keyboard.press('Escape');assert page.locator('dialog[open]').count()==0
   page.emulate_media(media='print');assert not root.is_visible();page.emulate_media(media='screen');assert not errors,errors;print(engine,width,theme,'PASS',count,'states');page.close()
  browser.close()
