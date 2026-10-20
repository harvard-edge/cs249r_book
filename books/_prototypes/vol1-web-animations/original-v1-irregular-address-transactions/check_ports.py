#!/usr/bin/env python3
"""Full owning-chapter fixture; production enrollment is checked separately."""
import argparse,functools,http.server,json,threading
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[4];EXT=ROOT/'books/_extensions/mlsysbook-ext/figure-explore';ENTRY=json.loads((Path(__file__).parent/'entry.json').read_text());KEY=ENTRY['key'];NAME=ENTRY['implementation']['name'];FID=KEY.split(':')[1]
p=argparse.ArgumentParser();p.add_argument('--engine',choices=['chromium','webkit']);p.add_argument('--width',type=int);args=p.parse_args();art=json.load(open('/tmp/figures-animation-art-current/index.json'))[KEY];chapter=Path(art['html']);shots=Path('/tmp/'+NAME+'-ports-review');shots.mkdir(exist_ok=True)
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
   page.evaluate('''async({fid,name,art})=>{const outer=document.getElementById(fid);outer.querySelectorAll('.mlsys-figure-views').forEach(n=>n.remove());const stat=outer.querySelector('figure');if(stat)stat.style.display='none';const root=document.createElement('div');outer.append(root);outer.dataset.mlsysMovementMs='1800';outer.dataset.mlsysReadingPauseMs='5000';const svg=new DOMParser().parseFromString(art,'image/svg+xml').documentElement;const actualCreate=window.MLSysOriginalPlayer.create;window.MLSysOriginalPlayer={...window.MLSysOriginalPlayer,create:opts=>{window.testSpec=opts;return actualCreate(opts)}};window.testController=await window.MLSysFigureLessons[name](root,{outer,verifiedArtwork:svg});window.testRoot=root;}''',{'fid':FID,'name':NAME,'art':Path(art['asset']).read_text()})
   root=page.locator('.'+NAME);root.scroll_into_view_if_needed();menu=root.locator('.mlsys-original-sequence select');count=menu.locator('option').count();assert count>=4
   checks=0
   for target in [1,2,3,4,5,8,9,10]:
    for fraction in [0,.15,.35,.5,.65,.85,1]:
     menu.select_option(str(target-1));before=root.get_attribute('data-ledger')
     result=page.evaluate("""({target,fraction})=>{window.testSpec.transition({from:target-1,to:target,onFrame:f=>f(fraction)});const svg=window.testRoot.querySelector('svg');const motion=svg.querySelector('[aria-hidden=true]');const packet=motion.querySelector('rect');const p=packet.getBoundingClientRect();const v=svg.getBoundingClientRect();const overlap=(a,b)=>Math.min(a.right,b.right)-Math.max(a.left,b.left)>.1&&Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>.1;const hits=[...svg.querySelectorAll('use,text')].filter(n=>!motion.contains(n)&&!n.closest('defs')).map(n=>({tag:n.tagName,text:n.textContent,box:n.getBoundingClientRect()})).filter(n=>n.box.width>0&&n.box.height>0&&overlap(p,n.box));return{hits:hits.map(n=>({tag:n.tag,text:n.text,box:{x:n.box.x,y:n.box.y,w:n.box.width,h:n.box.height}})),inside:p.left>=v.left-.2&&p.right<=v.right+.2&&p.top>=v.top-.2&&p.bottom<=v.bottom+.2};}""",{'target':target,'fraction':fraction})
     assert not result['hits'],(engine,width,theme,target,fraction,result)
     assert result['inside'],(engine,width,theme,target,fraction,'outside viewport')
     assert root.get_attribute('data-ledger')==before
     if fraction==.35:root.locator('.original-mechanism-viewport').screenshot(path=str(shots/f'{engine}-{width}-{theme}-edge{target}-paused35.png'))
     checks+=1
   assert not errors,errors
   print(engine,width,theme,'PASS',checks,'packet frames; glyph bounds/viewport/ledger')
   page.close()
  browser.close()
