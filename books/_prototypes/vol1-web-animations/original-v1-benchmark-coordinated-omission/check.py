#!/usr/bin/env python3
"""Full owning-chapter fixture; production enrollment is checked separately.

Mounts the factory on the verified artwork inside the actual rendered chapter,
then exercises every endpoint against the calc script's ledgers in Chromium and
WebKit, desktop and phone, light and dark. Screenshots: /tmp/<name>-review/.
"""
import argparse,functools,http.server,json,subprocess,threading
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[4];EXT=ROOT/'books/_extensions/mlsysbook-ext/figure-explore';ENTRY=json.loads((Path(__file__).parent/'entry.json').read_text());KEY=ENTRY['key'];NAME=ENTRY['implementation']['name'];FID=KEY.split(':')[1]
p=argparse.ArgumentParser();p.add_argument('--engine',choices=['chromium','webkit']);p.add_argument('--width',type=int);args=p.parse_args();art=json.load(open('/tmp/figures-animation-art-current/index.json'))[KEY];chapter=Path(art['html']);shots=Path('/tmp/'+NAME+'-review');shots.mkdir(exist_ok=True)
calc=json.loads(subprocess.check_output(['python3',str(EXT/(NAME+'.calc.py'))],text=True));LEDGERS=calc['ledgers']
assert calc['expected_final']==ENTRY['expected_final'],'entry.json expected_final is stale'
THEME='(t)=>{document.body.classList.toggle("quarto-dark",t==="dark");document.body.classList.toggle("quarto-light",t!=="dark");document.documentElement.setAttribute("data-bs-theme",t);for(const l of document.querySelectorAll("link.quarto-color-scheme")){l.disabled=t==="dark"?!l.classList.contains("quarto-color-alternate"):l.classList.contains("quarto-color-alternate")}}'
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(chapter.parents[2])));threading.Thread(target=server.serve_forever,daemon=True).start();url=f'http://127.0.0.1:{server.server_port}/'+chapter.relative_to(chapter.parents[2]).as_posix()
cursor_x=lambda root:root.locator('[data-co-cursor]').get_attribute('d').split(',')[0][1:]
with sync_playwright() as pw:
 for engine in ['chromium','webkit']:
  if args.engine and args.engine!=engine:continue
  browser=getattr(pw,engine).launch()
  for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
   if args.width and args.width!=width:continue
   page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme,reduced_motion='reduce');errors=[];phase=['load'];page.on('pageerror',lambda e:errors.append((phase[0],str(e))));page.route('**/release-manifest.json',lambda route:route.fulfill(status=200,body='{}',content_type='application/json'));page.goto(url);page.wait_for_load_state('networkidle');page.evaluate(THEME,theme)
   static_html=page.locator('#'+FID+' figure').first.inner_html()
   page.add_style_tag(content=(EXT/'original-player.css').read_text()+(EXT/ENTRY['implementation']['stylesheet']).read_text());page.add_script_tag(content=(EXT/'original-player.js').read_text());page.add_script_tag(content=(EXT/ENTRY['implementation']['script']).read_text())
   page.evaluate('''async({fid,name,art})=>{const outer=document.getElementById(fid);outer.querySelectorAll('.mlsys-figure-views').forEach(n=>n.remove());const stat=outer.querySelector('figure');if(stat)stat.style.display='none';const root=document.createElement('div');outer.append(root);outer.dataset.mlsysMovementMs='1800';outer.dataset.mlsysReadingPauseMs='5000';const svg=new DOMParser().parseFromString(art,'image/svg+xml').documentElement;window.testController=await window.MLSysFigureLessons[name](root,{outer,verifiedArtwork:svg});window.testRoot=root;}''',{'fid':FID,'name':NAME,'art':Path(art['asset']).read_text()})
   root=page.locator('.'+NAME);root.scroll_into_view_if_needed()
   # No autoplay: mounted paused at the first endpoint; the static figure is untouched.
   assert root.get_attribute('data-step')=='0' and root.get_attribute('data-playing')=='false'
   assert page.locator('#'+FID+' figure').first.inner_html()==static_html
   phase[0]='endpoints';menu=root.locator('.mlsys-original-sequence select');count=menu.locator('option').count();assert count==len(LEDGERS)==10,count
   for i in range(count):
    menu.select_option(str(i));assert root.get_attribute('data-step')==str(i);ledger=json.loads(root.get_attribute('data-ledger'))
    assert ledger==LEDGERS[i],(i,ledger,LEDGERS[i])
    # Figure counters and the text ledger show the same committed counts.
    shown=[int(t) for t in root.locator('[data-mechanism-values] tspan').all_text_contents()]
    assert shown==[ledger['outstanding'][0],ledger['recorded'][0],ledger['outstanding'][1],ledger['recorded'][1]],shown
    cells=root.locator('.co-ledger tbody td').all_inner_texts();assert [int(c.split()[0]) for c in cells]==[ledger['sentInPause'][0],ledger['sentInPause'][1],ledger['outstanding'][0],ledger['outstanding'][1],ledger['recorded'][0],ledger['recorded'][1],ledger['recordedFromPause'][0],ledger['recordedFromPause'][1]],cells
    box,stage=root.locator('.co-ledger').bounding_box(),root.locator('.mlsys-original-stage').bounding_box();assert box['x']>=stage['x']-.5 and box['x']+box['width']<=stage['x']+stage['width']+.5,(box,stage)
    assert root.locator('.co-ledger').evaluate('t=>t.scrollWidth<=t.clientWidth+1'),root.locator('.co-ledger').evaluate('t=>[t.scrollWidth,t.clientWidth,t.offsetWidth,getComputedStyle(t).display,getComputedStyle(t).tableLayout]')
    root.screenshot(path=str(shots/f'{engine}-{width}-{theme}-{i}.png'))
   expected=ENTRY['expected_final']
   for key,value in expected.items():assert ledger.get(key)==value,(key,ledger,value)
   phase[0]='jumps-keys'
   # Menu jumps are state-independent.
   menu.select_option('3');a=root.get_attribute('data-ledger');menu.select_option(str(count-1));menu.select_option('3');assert root.get_attribute('data-ledger')==a
   # Keyboard: P/N/R; reduced motion commits N immediately with no pending interpolation.
   root.focus();page.keyboard.press('p');assert root.get_attribute('data-step')=='2';page.keyboard.press('n');assert root.get_attribute('data-step')=='3' and root.get_attribute('data-pending') is None;page.keyboard.press('r');assert root.get_attribute('data-step')=='0'
   page.evaluate('window.scrollTo(0,0)');page.wait_for_timeout(300);root.evaluate('r=>r.scrollIntoView({block:"start"})');page.wait_for_timeout(400);root.focus();page.keyboard.press('k');page.wait_for_timeout(100);assert root.get_attribute('data-playing')=='true',page.evaluate('window.testController.state');page.keyboard.press('k');assert root.get_attribute('data-playing')=='false'
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'chapter overflow'
   phase[0]='expanded-motion'
   # Normal motion in the expanded view: pausing freezes cursor, ledger, counts, and narration.
   page.emulate_media(reduced_motion='no-preference');menu.select_option('2');root.get_by_role('button',name='Expand',exact=True).click();assert page.locator('dialog[open]').count()==1
   root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(600);root.get_by_role('button',name='Pause',exact=True).click()
   assert root.get_attribute('data-step')=='2' and root.get_attribute('data-pending')=='3';before=(cursor_x(root),root.get_attribute('data-ledger'),root.locator('[data-mechanism-values]').inner_html(),root.locator('.mlsys-original-narration').inner_text())
   x=float(before[0]);assert 121.9<x<223.6,x  # strictly between the two endpoints
   assert json.loads(before[1])==LEDGERS[2],'committed ledger must not advance while moving'
   page.screenshot(path=str(shots/f'{engine}-{width}-{theme}-paused-motion.png'));page.wait_for_timeout(400)
   assert before==(cursor_x(root),root.get_attribute('data-ledger'),root.locator('[data-mechanism-values]').inner_html(),root.locator('.mlsys-original-narration').inner_text())
   root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function("window.testRoot.dataset.step==='3'",timeout=8000);root.get_by_role('button',name='Pause',exact=True).click()
   assert json.loads(root.get_attribute('data-ledger'))==LEDGERS[3]
   page.screenshot(path=str(shots/f'{engine}-{width}-{theme}-expanded.png'));page.keyboard.press('Escape');assert page.locator('dialog[open]').count()==0;phase[0]='after-expand'
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'chapter overflow after expand'
   assert not errors,errors;phase[0]='print'
   # WebKit emits a ResizeObserver delivery warning under print-media emulation; the accepted
   # async-timing module reproduces it identically, so only that warning is tolerated, only here.
   page.emulate_media(media='print');assert not root.is_visible();page.emulate_media(media='screen');page.wait_for_timeout(200)
   assert all(engine=='webkit' and e==('print','ResizeObserver loop completed with undelivered notifications.') for e in errors),errors;print(engine,width,theme,'PASS',count,'states',flush=True);page.close()
  browser.close()
