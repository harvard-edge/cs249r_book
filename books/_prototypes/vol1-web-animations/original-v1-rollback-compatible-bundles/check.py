#!/usr/bin/env python3
"""Full owning-chapter fixture for the rollback-bundle Explore view; production enrollment is checked separately."""
import argparse,functools,http.server,json,threading
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[4];EXT=ROOT/'books/_extensions/mlsysbook-ext/figure-explore';ENTRY=json.loads((Path(__file__).parent/'entry.json').read_text());KEY=ENTRY['key'];NAME=ENTRY['implementation']['name'];FID=KEY.split(':')[1]
p=argparse.ArgumentParser();p.add_argument('--engine',choices=['chromium','webkit']);p.add_argument('--width',type=int);args=p.parse_args();art=json.load(open('/tmp/figures-animation-art-current/index.json'))[KEY];chapter=Path(art['html']);shots=Path('/tmp/'+NAME+'-review');shots.mkdir(exist_ok=True)
THEME='(t)=>{document.body.classList.toggle("quarto-dark",t==="dark");document.body.classList.toggle("quarto-light",t!=="dark");document.documentElement.setAttribute("data-bs-theme",t);for(const l of document.querySelectorAll("link.quarto-color-scheme")){l.disabled=t==="dark"?!l.classList.contains("quarto-color-alternate"):l.classList.contains("quarto-color-alternate")}}'
LIVE={'in-flight','cancel-requested'}
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(chapter.parents[2])));threading.Thread(target=server.serve_forever,daemon=True).start();url=f'http://127.0.0.1:{server.server_port}/'+chapter.relative_to(chapter.parents[2]).as_posix()
def invariants(i,led,prev):
 # Counters must follow the per-request ledger; release only after the last pinned consumer.
 b2=sum(1 for r in ['R1','R2','R3','R4'] if led['requests'][r] in LIVE);assert led['b2_in_flight']==b2,(i,led)
 assert led['b1_in_flight']==sum(1 for r in ['R5','R6'] if led['requests'][r] in LIVE),(i,led)
 assert led['log_records']==4 and led['log_flagged'] is True
 if led['b2_state'] in ('releasable','released'):assert b2==0
 if led['b2_state']=='released':assert prev and prev['b2_state']=='releasable'
 assert led['target']==(2 if i==0 else 1)
 assert led['refused']==('R3' if i==4 else None)
 if i==5:assert led['requests']['R4']=='cancel-requested' and led['b2_in_flight']==prev['b2_in_flight']
 if i==6:assert led['requests']['R4']=='cancel-confirmed' and led['b2_in_flight']==prev['b2_in_flight']-1
with sync_playwright() as pw:
 for engine in ['chromium','webkit']:
  if args.engine and args.engine!=engine:continue
  browser=getattr(pw,engine).launch()
  for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
   if args.width and args.width!=width:continue
   page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme,reduced_motion='reduce');errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.route('**/release-manifest.json',lambda route:route.fulfill(status=200,body='{}',content_type='application/json'));page.goto(url);page.wait_for_load_state('networkidle');page.evaluate(THEME,theme);page.add_style_tag(content=(EXT/'original-player.css').read_text()+(EXT/ENTRY['implementation']['stylesheet']).read_text());page.add_script_tag(content=(EXT/'original-player.js').read_text());page.add_script_tag(content=(EXT/ENTRY['implementation']['script']).read_text())
   page.evaluate('''async({fid,name,art})=>{const outer=document.getElementById(fid);outer.querySelectorAll('.mlsys-figure-views').forEach(n=>n.remove());const stat=outer.querySelector('figure')||outer.firstElementChild;const root=document.createElement('div');outer.insertBefore(root,outer.firstChild);[...outer.children].forEach(n=>{if(n!==root&&!n.matches('figcaption,.figure-caption,[id$=caption]')&&!n.querySelector('figcaption'))n.style.display='none';});outer.dataset.mlsysMovementMs='1800';outer.dataset.mlsysReadingPauseMs='5000';const svg=new DOMParser().parseFromString(art,'image/svg+xml').documentElement;window.testController=await window.MLSysFigureLessons[name](root,{outer,verifiedArtwork:svg});window.testRoot=root;}''',{'fid':FID,'name':NAME,'art':Path(art['asset']).read_text()})
   root=page.locator('.'+NAME);root.scroll_into_view_if_needed();assert root.get_attribute('data-step')=='0' and root.get_attribute('data-playing')=='false','no autoplay'
   menu=root.locator('.mlsys-original-sequence select');count=menu.locator('option').count();assert count==9,count
   endpoints=[];prev=None
   for i in range(count):
    menu.select_option(str(i));assert root.get_attribute('data-step')==str(i);led=json.loads(root.get_attribute('data-ledger'));invariants(i,led,prev);endpoints.append(led);prev=led
    # Visible overlay text must agree with the ledger in whichever layout is displayed.
    visible=root.locator('svg.rb-mobile' if width<=700 else 'svg.rb-original');assert visible.is_visible()
    labels=visible.locator('[data-rb-overlay] text').evaluate_all('n=>n.map(e=>e.textContent)')
    assert 'target: Bundle %d'%led['target'] in labels,(labels,led)
    b2='released' if led['b2_state']=='released' else '0 · releasable' if led['b2_in_flight']==0 else 'in flight %d'%led['b2_in_flight'];assert b2 in labels,(labels,b2)
    root.screenshot(path=str(shots/f'{engine}-{width}-{theme}-{i}.png'))
   for key,value in ENTRY['expected_final'].items():assert endpoints[-1].get(key)==value,(key,endpoints[-1],value)
   # Menu jumps are exact and repeatable, independent of path.
   menu.select_option('3');a=root.get_attribute('data-ledger');menu.select_option(str(count-1));menu.select_option('0');menu.select_option('3');assert root.get_attribute('data-ledger')==a
   root.focus();page.keyboard.press('n');page.wait_for_function("window.testRoot.dataset.step==='4'");page.keyboard.press('p');assert root.get_attribute('data-step')=='3';page.keyboard.press('r');assert root.get_attribute('data-step')=='0';page.keyboard.press('k');page.wait_for_timeout(50);page.keyboard.press('k')
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'chapter overflow'
   over=root.evaluate("r=>{const b=r.getBoundingClientRect();return [...r.querySelectorAll('*')].filter(n=>n.getClientRects().length&&!n.closest('svg')&&n.getBoundingClientRect().right>b.right+1).map(n=>n.className||n.tagName).slice(0,5)}");assert not over,over
   # Real motion: pausing freezes the drawing and leaves the committed ledger unchanged.
   page.emulate_media(reduced_motion='no-preference');menu.select_option('1');root.get_by_role('button',name='Expand',exact=True).click();assert page.locator('dialog[open]').count()==1
   root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(500);root.get_by_role('button',name='Pause',exact=True).click()
   assert root.get_attribute('data-step')=='1' and root.get_attribute('data-pending')=='2';before=root.get_attribute('data-ledger');assert json.loads(before)['b1_in_flight']==0
   layer=root.locator('[data-rb-overlay="mobile"]' if width<=700 else '[data-rb-overlay="desktop"]');frozen=layer.inner_html();tok=layer.locator('[data-token="R5"]');assert tok.count()==1;tr=tok.get_attribute('transform')
   page.wait_for_timeout(300);assert layer.inner_html()==frozen and root.get_attribute('data-ledger')==before and tok.get_attribute('transform')==tr
   assert 'On completion' in root.locator('.mlsys-original-narration').inner_text()
   page.screenshot(path=str(shots/f'{engine}-{width}-{theme}-paused-admission.png'))
   root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function("window.testRoot.dataset.step==='2'",timeout=8000);root.get_by_role('button',name='Pause',exact=True).click();assert json.loads(root.get_attribute('data-ledger'))['b1_in_flight']==2
   page.keyboard.press('Escape');assert page.locator('dialog[open]').count()==0
   # Reduced motion commits immediately with no pending endpoint.
   page.emulate_media(reduced_motion='reduce');menu.select_option('6');root.get_by_role('button',name='Next',exact=True).click();assert root.get_attribute('data-step')=='7' and root.get_attribute('data-pending') is None
   assert not errors,errors
   # WebKit emulated print relayout can emit the shared, documented ResizeObserver delivery warning; nothing else is tolerated.
   page.emulate_media(media='print');assert not root.is_visible();page.emulate_media(media='screen');assert all(engine=='webkit' and e.startswith('ResizeObserver loop') for e in errors),errors;print(engine,width,theme,'PASS',count,'states',flush=True);page.close()
  browser.close()
