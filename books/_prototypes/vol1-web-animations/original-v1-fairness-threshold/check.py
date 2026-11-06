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
calc=json.loads(subprocess.check_output(['python3',str(EXT/(NAME+'.calc.py'))],text=True));LEDGERS=calc['ledgers'];D=calc['D']
assert calc['expected_final']==ENTRY['expected_final'],'entry.json expected_final is stale'
sx=lambda pos:D['tx']+D['k']*(D['u0']+pos*(D['u1']-D['u0']))
THEME='(t)=>{document.body.classList.toggle("quarto-dark",t==="dark");document.body.classList.toggle("quarto-light",t!=="dark");document.documentElement.setAttribute("data-bs-theme",t);for(const l of document.querySelectorAll("link.quarto-color-scheme")){l.disabled=t==="dark"?!l.classList.contains("quarto-color-alternate"):l.classList.contains("quarto-color-alternate")}}'
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(chapter.parents[2])));threading.Thread(target=server.serve_forever,daemon=True).start();url=f'http://127.0.0.1:{server.server_port}/'+chapter.relative_to(chapter.parents[2]).as_posix()
# Once the chapter is enrolled, its build carries the production Explore attributes on this figure.
# The fixture mounts its own copy, so it serves the chapter with only those attributes removed from
# this one figure div (an unenrolled host); production enrollment is checked by check_original_figures.py.
import re
def unenrolled(route):
 html=chapter.read_text();m=re.search(r'<div id="'+FID+r'"[^>]*>',html);tag=re.sub(r' data-(?:mlsys-)?explore(?:-key)?="[^"]*"| data-mlsys-(?:movement|reading-pause)-ms="[^"]*"','',m.group(0))
 route.fulfill(status=200,body=html[:m.start()]+tag+html[m.end():],content_type='text/html; charset=utf-8')
boundary_x=lambda root:tuple(float(root.locator(f'[data-ft-boundary="{g}"]').get_attribute('x1')) for g in (0,1))
def figure_counts(root,L):
 """In-figure counters and the text ledger show the same committed values as data-ledger."""
 vals=root.locator('[data-mechanism-values] > text').all_text_contents();acc,a,b=vals[0],vals[1],vals[2]
 if L['phase']=='none':assert (acc,a,b)==('','',''),vals
 elif L['phase']=='decide':assert acc=='Accuracy: not yet checked' and a==f"{L['approved'][0]} of 8 approved" and b==f"{L['approved'][1]} of 8 approved",vals
 elif L['phase']=='compare':assert acc==f"Accuracy {L['before']['accuracyPct']}% → {L['accuracyPct']}%" and a.startswith(f"{L['before']['errors'][0]} → {L['errors'][0]} error") and b.startswith(f"{L['before']['errors'][1]} → {L['errors'][1]} error"),vals
 else:assert acc==f"Accuracy {L['correct']}/16 = {L['accuracyPct']}%" and a.startswith(f"{L['errors'][0]} error") and b.startswith(f"{L['errors'][1]} error"),vals
 wrong=[int(v) for v in root.locator('[data-ft-wrong]').evaluate_all('ns=>ns.map(n=>n.dataset.ftWrong)')];assert wrong==L.get('wrong',[]),(wrong,L)
 cells=root.locator('.ft-ledger:not(.ft-history) tbody td').all_inner_texts();scored='errors' in L
 want=['–']*8 if L['phase']=='none' else [f"{L['approved'][0]} of 8",f"{L['approved'][1]} of 8"]+['–']*6 if not scored else [f"{L['approved'][0]} of 8",f"{L['approved'][1]} of 8",f"{L['repayersApproved'][0]} of 4 ({L['tpr'][0]})",f"{L['repayersApproved'][1]} of 4 ({L['tpr'][1]})",f"{L['defaultersApproved'][0]} of 4 ({L['fpr'][0]})",f"{L['defaultersApproved'][1]} of 4 ({L['fpr'][1]})",str(L['errors'][0]),str(L['errors'][1])]
 assert cells==want,(cells,want)
 hist=root.locator('.ft-history tbody tr').all_inner_texts();assert len(hist)==(0 if L['step']<2 else L['step']-1-(L['step']>=8)),hist
 if L['boundary'] is None:assert not root.locator('[data-ft-boundary]').first.is_visible()
 else:assert all(abs(x-sx(b))<1e-6 for x,b in zip(boundary_x(root),L['boundary'])),(boundary_x(root),L['boundary'])
 # The sweep result appears at the compare step and stays through the split steps.
 assert root.locator('.ft-sweep').is_visible()==(L['step']>=8)
# The approve/deny tags never touch a dashed candidate line, a drawn marker, a legend glyph, or an error ring.
TAGS_CLEAR='''r=>{const svg=r.querySelector('.ft-source svg'),R=e=>e.getBoundingClientRect(),ov=(a,b)=>a.left<b.right&&b.left<a.right&&a.top<b.bottom&&b.top<a.bottom;
 const tags=[...svg.querySelectorAll('[data-ft-overlay] text')].filter(t=>t.textContent&&getComputedStyle(t).display!=='none'&&t.style.visibility!=='hidden');
 const obs=[...svg.querySelectorAll('path')].filter(p=>p.hasAttribute('stroke-dasharray')||/69\\.999695%|100%, 0%, 0%/.test(p.getAttribute('stroke')||'')).concat([...svg.querySelectorAll('[data-ft-rings] circle')]);
 return {shown:tags.map(t=>t.textContent),hits:tags.flatMap(t=>obs.filter(o=>ov(R(t),R(o))).map(o=>t.textContent+' x '+(o.localName==='circle'?'ring':o.hasAttribute('stroke-dasharray')?'dashed line':'marker')))};}'''
def strip_matches(root,L):
 """The sweep strip lists the calc's correct count at every shared position, from the compare step on."""
 box=root.locator('.ft-strip-box');assert box.is_visible()==(L['step']>=8)
 S=LEDGERS[8]['sweep'];vals=root.locator('.ft-strip-cell .ft-val').all_text_contents()
 assert [int(c) for c in vals]==S['correctByPosition']==[round(16*float(w['accuracyPct'])/100) for w in calc['sweep']],(vals,S)
 assert root.locator('.ft-strip-cell.ft-best').count()==S['bestCount']
 if box.is_visible():
  clip=root.locator('.ft-strip span').evaluate_all('ns=>ns.filter(n=>n.offsetParent&&n.scrollWidth>n.clientWidth+1).map(n=>n.textContent)');assert not clip,clip
  assert box.evaluate('b=>b.scrollWidth<=b.clientWidth+1'),'strip overflow'
tag_log=[]
with sync_playwright() as pw:
 for engine in ['chromium','webkit']:
  if args.engine and args.engine!=engine:continue
  browser=getattr(pw,engine).launch()
  for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
   if args.width and args.width!=width:continue
   page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme,reduced_motion='reduce');errors=[];phase=['load'];page.on('pageerror',lambda e:errors.append((phase[0],str(e))));page.route('**/release-manifest.json',lambda route:route.fulfill(status=200,body='{}',content_type='application/json'));page.route(url,unenrolled);page.goto(url);page.wait_for_load_state('networkidle');page.evaluate(THEME,theme)
   static_html=page.locator('#'+FID+' figure').first.inner_html()
   page.add_style_tag(content=(EXT/'original-player.css').read_text()+(EXT/ENTRY['implementation']['stylesheet']).read_text());page.add_script_tag(content=(EXT/'original-player.js').read_text());page.add_script_tag(content=(EXT/ENTRY['implementation']['script']).read_text())
   page.evaluate('''async({fid,name,art})=>{const outer=document.getElementById(fid);outer.querySelectorAll('.mlsys-figure-views').forEach(n=>n.remove());const stat=outer.querySelector('figure');if(stat)stat.style.display='none';const root=document.createElement('div');outer.append(root);outer.dataset.mlsysMovementMs='1800';outer.dataset.mlsysReadingPauseMs='5000';const svg=new DOMParser().parseFromString(art,'image/svg+xml').documentElement;window.testController=await window.MLSysFigureLessons[name](root,{outer,verifiedArtwork:svg});window.testRoot=root;}''',{'fid':FID,'name':NAME,'art':Path(art['asset']).read_text()})
   root=page.locator('.'+NAME);root.scroll_into_view_if_needed()
   # No autoplay: mounted paused at the first endpoint; the static figure is untouched.
   assert root.get_attribute('data-step')=='0' and root.get_attribute('data-playing')=='false'
   assert page.locator('#'+FID+' figure').first.inner_html()==static_html
   phase[0]='endpoints';menu=root.locator('.mlsys-original-sequence select');count=menu.locator('option').count();assert count==len(LEDGERS)==12,count
   for i in range(count):
    menu.select_option(str(i));assert root.get_attribute('data-step')==str(i);ledger=json.loads(root.get_attribute('data-ledger'))
    assert ledger==LEDGERS[i],(i,ledger,LEDGERS[i]);figure_counts(root,ledger);strip_matches(root,ledger)
    tags=root.evaluate(TAGS_CLEAR);assert not tags['hits'],(i,tags)
    if width>700 and ledger['boundary'] is not None:assert tags['shown'] and all(tags['shown']),(i,tags)
    tag_log.append((engine,width,theme,i,tags['shown']))
    for sel in ['.ft-ledger:not(.ft-history)','.ft-history','.ft-source']:
     loc=root.locator(sel)
     if not loc.is_visible():continue
     box,stage=loc.bounding_box(),root.locator('.mlsys-original-stage').bounding_box();assert box['x']>=stage['x']-.5 and box['x']+box['width']<=stage['x']+stage['width']+.5,(sel,box,stage)
     assert loc.evaluate('t=>t.scrollWidth<=t.clientWidth+1'),sel
    root.screenshot(path=str(shots/f'{engine}-{width}-{theme}-{i}.png'))
   for key,value in ENTRY['expected_final'].items():assert ledger.get(key)==value,(key,ledger,value)
   narr=root.locator('.mlsys-original-narration').inner_text();F=ENTRY['expected_final']
   # Final: TPR equalized, FPR cost, error kind changed, legal caveat, accuracy does not settle it.
   assert f"TPR is {F['tpr'][1]}, equal to A's {F['tpr'][0]}" in narr and f"FPR is {F['fpr'][1]} against A's {F['fpr'][0]}" in narr and 'an approved defaulter' in narr and '3 denied repayers' in narr and 'prohibited in US employment selection' in narr and F['accuracyPct']+'%' in narr,narr
   # B cutoff 1 already beats every shared boundary with no false positive; the cost comes with equal TPR.
   B1=LEDGERS[9];menu.select_option('9');narr9=root.locator('.mlsys-original-narration').inner_text()
   assert f"FPR stays at {B1['fpr'][1]}" in narr9 and f"already {B1['accuracyPct']}%" in narr9 and f"best shared boundary's {LEDGERS[8]['sweep']['bestPct']}%" in narr9,narr9
   menu.select_option('11');narr11=root.locator('.mlsys-original-narration').inner_text()
   assert f"the same as at {B1['name']}" in narr11 and f"FPR was {B1['fpr'][1]} but its TPR only {B1['tpr'][1]}" in narr11 and 'price of equal TPR' in narr11,narr11
   menu.select_option('0');assert 'markers further left have higher scores' in root.locator('.mlsys-original-narration').inner_text()
   # Compare: every shared position checked; 81.25% unique; B keeps three denied repayers.
   S=LEDGERS[8]['sweep'];menu.select_option('8');narr8=root.locator('.mlsys-original-narration').inner_text()
   assert f"all {S['cutoffs']} shared positions" in narr8 and f"{S['bestPct']}% is reached only here" in narr8 and f"at most {S['nextPct']}%" in narr8 and 'all denied repayers' in narr8 and f"(TPR {LEDGERS[8]['tpr'][1]})" in narr8,narr8
   menu.select_option(str(count-1))
   phase[0]='jumps-keys'
   # Menu jumps are state-independent.
   menu.select_option('4');a=(root.get_attribute('data-ledger'),root.locator('[data-mechanism-values]').inner_html());menu.select_option(str(count-1));menu.select_option('0');menu.select_option('4');assert (root.get_attribute('data-ledger'),root.locator('[data-mechanism-values]').inner_html())==a
   # Keyboard: P/N/R; reduced motion commits N immediately with no pending interpolation.
   root.focus();page.keyboard.press('p');assert root.get_attribute('data-step')=='3';page.keyboard.press('n');assert root.get_attribute('data-step')=='4' and root.get_attribute('data-pending') is None;page.keyboard.press('r');assert root.get_attribute('data-step')=='0'
   page.evaluate('window.scrollTo(0,0)');page.wait_for_timeout(300);root.evaluate('r=>r.scrollIntoView({block:"start"})');page.wait_for_timeout(400);root.focus();page.keyboard.press('k');page.wait_for_timeout(100);assert root.get_attribute('data-playing')=='true',page.evaluate('window.testController.state');page.keyboard.press('k');assert root.get_attribute('data-playing')=='false'
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'chapter overflow'
   phase[0]='expanded-motion'
   # Normal motion in the expanded view: pausing freezes boundary, ledger, rings, counts, and narration.
   page.emulate_media(reduced_motion='no-preference');menu.select_option('3');root.get_by_role('button',name='Expand',exact=True).click();assert page.locator('dialog[open]').count()==1
   root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(600);root.get_by_role('button',name='Pause',exact=True).click()
   assert root.get_attribute('data-step')=='3' and root.get_attribute('data-pending')=='4';before=(boundary_x(root),root.get_attribute('data-ledger'),root.locator('[data-mechanism-values]').inner_html(),root.locator('.mlsys-original-narration').inner_text())
   x=before[0];assert all(sx(D['stops'][4][g])<x[g]<sx(D['stops'][3][g]) for g in (0,1)),x  # strictly between the two endpoints (moving left)
   assert json.loads(before[1])==LEDGERS[3],'committed ledger must not advance while moving'
   page.screenshot(path=str(shots/f'{engine}-{width}-{theme}-paused-motion.png'));page.wait_for_timeout(400)
   assert before==(boundary_x(root),root.get_attribute('data-ledger'),root.locator('[data-mechanism-values]').inner_html(),root.locator('.mlsys-original-narration').inner_text())
   root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function("window.testRoot.dataset.step==='4'",timeout=8000);root.get_by_role('button',name='Pause',exact=True).click()
   assert json.loads(root.get_attribute('data-ledger'))==LEDGERS[4]
   page.screenshot(path=str(shots/f'{engine}-{width}-{theme}-expanded.png'));page.keyboard.press('Escape');assert page.locator('dialog[open]').count()==0;phase[0]='after-expand'
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'chapter overflow after expand'
   # Split motion in the page: A's segment stays on the 81.25% line while B's moves right; pausing freezes it.
   phase[0]='split-motion';menu.select_option('9');root.scroll_into_view_if_needed();root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(600);root.get_by_role('button',name='Pause',exact=True).click()
   assert root.get_attribute('data-step')=='9' and root.get_attribute('data-pending')=='10' and json.loads(root.get_attribute('data-ledger'))==LEDGERS[9]
   xa,xb=boundary_x(root);assert abs(xa-sx(D['stops'][9][0]))<1e-6 and sx(D['stops'][9][1])<xb<sx(D['stops'][10][1]),(xa,xb)
   page.wait_for_timeout(400);assert boundary_x(root)==(xa,xb);root.screenshot(path=str(shots/f'{engine}-{width}-{theme}-paused-split.png'))
   root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function("window.testRoot.dataset.step==='10'",timeout=8000);root.get_by_role('button',name='Pause',exact=True).click()
   assert json.loads(root.get_attribute('data-ledger'))==LEDGERS[10]
   assert not errors,errors;phase[0]='print'
   # WebKit emits a ResizeObserver delivery warning under print-media emulation; the accepted
   # async-timing module reproduces it identically, so only that warning is tolerated, only here.
   page.emulate_media(media='print');assert not root.is_visible();page.emulate_media(media='screen');page.wait_for_timeout(200)
   assert all(engine=='webkit' and e==('print','ResizeObserver loop completed with undelivered notifications.') for e in errors),errors;print(engine,width,theme,'PASS',count,'states',flush=True);page.close()
  browser.close()
for row in tag_log:
 if row[1]>700:print('tags',*row)
