"""Verify this original-art mechanism in a disposable copy of the actual themed chapter."""
import argparse,base64,hashlib,json,re,struct
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--engine',default='chromium',choices=['chromium','webkit']);parser.add_argument('--screenshots',type=Path);a=parser.parse_args()
here=Path(__file__).resolve().parent;repo=here.parents[3];entry=json.loads((here/'entry.json').read_text());name=entry['implementation']['name'];key=entry['key'];ext=repo/'books/_extensions/mlsysbook-ext/figure-explore';out=a.screenshots or Path('/tmp')/(name+'-review');out.mkdir(exist_ok=True,parents=True)
artentry=json.loads(Path('/tmp/figures-animation-art-current/index.json').read_text())[key];chapter=Path(artentry['html']);asset=Path(artentry['asset']);raw=asset.read_bytes();assert hashlib.sha1(raw).hexdigest()==artentry['asset_sha1']
soup=BeautifulSoup(chapter.read_text(),'html.parser')
for script in soup.find_all('script'):script.decompose()
soup.head.insert(0,soup.new_tag('base',href=chapter.parent.as_uri()+'/'))
for n in ['original-player',name]:
 soup.head.append(soup.new_tag('link',rel='stylesheet',href=(ext/(n+'.css')).as_uri()));soup.body.append(soup.new_tag('script',src=(ext/(n+'.js')).as_uri()))
fid=key.split(':')[1];outer=soup.select_one('#'+fid)
for k in list(outer.attrs):
 if k.startswith('data-'):del outer[k]
outer['data-mlsys-movement-ms']='600';outer['data-mlsys-reading-pause-ms']='180'
figure=outer.find('figure');static=figure.find('div',attrs={'aria-describedby':True});assert static;static['hidden']=''
style=soup.new_tag('style');style.string=f'@media print {{#{fid} figure>div[aria-describedby]{{display:block!important}}}}';soup.head.append(style)
root=soup.new_tag('section',id='v1c-review');figure.insert(0,root)
if asset.suffix=='.png':
 w,h=struct.unpack('>II',raw[16:24]);artcode=f'''const art=document.createElementNS('http://www.w3.org/2000/svg','svg');art.setAttribute('viewBox','0 0 {w} {h}');art.dataset.mlsysArtworkKind='png';const im=document.createElementNS(art.namespaceURI,'image');im.setAttribute('width','{w}');im.setAttribute('height','{h}');im.setAttribute('href','data:image/png;base64,{base64.b64encode(raw).decode()}');art.append(im);'''
else:
 artcode='const art=new DOMParser().parseFromString('+json.dumps(raw.decode())+',"image/svg+xml").documentElement;const ids=new Map();art.querySelectorAll("[id]").forEach((n,i)=>{ids.set(n.id,"v1c-art-"+i);n.id="v1c-art-"+i;});art.querySelectorAll("*").forEach(n=>{for(const attr of [...n.attributes]){let v=attr.value;ids.forEach((id,old)=>{v=v.replaceAll("url(#"+old+")","url(#"+id+")");if(v==="#"+old)v="#"+id;});n.setAttributeNS(attr.namespaceURI,attr.name,v);}});'
script=soup.new_tag('script');script.string='(async()=>{'+artcode+'window.reviewPlayer=await window.MLSysFigureLessons['+json.dumps(name)+'](document.querySelector("#v1c-review"),{outer:document.querySelector('+json.dumps('#'+fid)+'),verifiedArtwork:art});})().catch(e=>{window.fixtureError=String(e);console.error(e);});';soup.body.append(script)
fixture=out/'chapter.html';fixture.write_text(str(soup));reports=[]
def check_evidence(i,e):
 k=key.split('fig-',1)[1]
 if k=='tensor-memory-layout' and i:
  j=(i-1)%3;r=(i-1)//3;assert e==dict(i=r,j=j,rowOffset=r*3+j,columnOffset=r+j*2,value=i)
 elif k=='im2col-diagram' and i:
  assert e['rows']==min(i,4) and e['copies']==min(i,4)*8
  if i>4:assert e['outputs']==i-4
 elif k=='block-sparse-gemm':
  coords=[(0,0),(0,1),(1,0),(1,2),(2,1),(2,2)];v=[0,0,0]
  for r,c in coords[:min(i,6)]:v[r]+=sum(range(c*3+1,c*3+4))
  assert e['partial']==v and e['tiles']==min(i,6)
 elif k=='depthwise-pointwise-channels':
  if i in [10,11]:assert e['middle']==[9,18,27]
  if i>=11:assert e['outputs']==[54,-18]
  if i==12:assert e['middle']==[None]*3
 elif k=='serving-chunked-prefill':
  if i<=5:assert e['a']==min(i,4) and e['ta']==int(i==5)
  else:assert e['a']==4 and e['ta']==1 and e['b']==min((i-5)//2,4) and e['tb']==max((i-6)//2,0)
 elif k=='ml-ops-dlrm-node-day':
  assert e['lanes'][:4]==['serving']*2+['standby']*2
  assert e['lanes'].count('retraining')==(4 if i in [1,2] else 0)
  assert e['lanes'].count('canary')==(2 if i>=4 else 0)
 elif k=='training-step-memory':
  dims=[784,128,64,10];params=[a*b+b for a,b in zip(dims,dims[1:])]
  for b,s in zip([32,1024],e['cases']):
   live=list(range(i+1)) if i<=3 else list(range(7-i)) if i<7 else []
   grads=[] if i<=3 else list(range(2,5-i,-1)) if i<7 else [2,1,0]
   assert s['live']==live and s['grads']==grads,(i,s,live,grads)
   assert abs(s['actKB']-sum(b*dims[k]*4 for k in live)/1000)<1e-9
   assert abs(s['gradKB']-sum(params[k]*4 for k in grads)/1000)<1e-9
with sync_playwright() as p:
 browser=getattr(p,a.engine).launch()
 def fresh(width,theme,reduce=False):
  page=browser.new_page(viewport={'width':width,'height':1000},color_scheme=theme,reduced_motion='reduce' if reduce else 'no-preference');page.goto(fixture.as_uri());page.wait_for_function('window.reviewPlayer || window.fixtureError');assert not page.evaluate('window.fixtureError'),page.evaluate('window.fixtureError')
  page.evaluate("theme=>{document.documentElement.setAttribute('data-bs-theme',theme);document.body.classList.toggle('quarto-dark',theme==='dark');document.body.classList.toggle('quarto-light',theme==='light');for(const l of document.querySelectorAll('link.quarto-color-scheme'))l.disabled=l.classList.contains('quarto-color-alternate')?theme!=='dark':theme==='dark';}",theme)
  r=page.locator('#v1c-review');r.scroll_into_view_if_needed();return page,r
 for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
  page,r=fresh(width,theme);count=r.locator('.mlsys-original-sequence option').count();snap=[]
  for i in range(count):
   r.locator('.mlsys-original-sequence select').select_option(str(i));assert r.get_attribute('data-lesson-state')==str(i)
   e=json.loads(r.get_attribute('data-state-evidence'));check_evidence(i,e);snap.append(e)
   assert r.locator('.v1c-source-anchor svg').count()==1
   assert not r.locator('.v1c-diagram').is_visible()
   if key.endswith('block-sparse-gemm'):
    assert r.locator('text[data-source-weight]').count()==108
    assert r.locator('text[data-source-output-row]').all_text_contents()==[str(e['partial'][j//3]) for j in range(9)]
   elif key.endswith('depthwise-pointwise-channels'):
    for group,values in [('standard',e['standard']),('middle',e['middle']),('output',e['outputs'])]:
     for j,v in enumerate(values):assert r.locator(f'text[data-source-node="{group}-{j}"]').text_content()==('—' if v is None else str(v))
   elif key.endswith('serving-chunked-prefill'):
    for j in range(7):
     visible=j in [0,6] or (e['ta']>0 if j==1 else j-2<e['tb'])
     assert r.locator(f'path[data-source-token="{j}"]').evaluate('(n)=>getComputedStyle(n).opacity')==('1' if visible else '0')

   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   assert r.locator('.v1c-flight').count()==0
   r.screenshot(path=str(out/f'state-{width}-{theme}-{i}.png'))
  camera=r.get_by_role('combobox',name='Inspect source geometry');endstate=r.get_attribute('data-state-evidence')
  for option in camera.locator('option').evaluate_all('(ns)=>ns.map(n=>n.value)'):
   camera.select_option(option);assert r.get_attribute('data-state-evidence')==endstate
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   r.locator('.v1c-source-anchor').screenshot(path=str(out/f'camera-{width}-{theme}-{option}.png'))
  (out/f'states-{width}-{theme}.json').write_text(json.dumps(snap,indent=2));page.close()
  # Fresh normal-motion viewport; never use a tall element capture before playback.
  page,r=fresh(width,theme);r.focus();page.keyboard.press('n');page.wait_for_timeout(110);page.evaluate('window.reviewPlayer.pause()');assert r.get_attribute('data-lesson-state')=='0';assert 'Pending:' in r.locator('.v1c-status').inner_text()
  before=r.locator('.mlsys-original-stage').inner_html();page.wait_for_timeout(130);assert before==r.locator('.mlsys-original-stage').inner_html();world=r.locator('[data-source-motion]').inner_html();viewbox=r.locator('.v1c-source-anchor svg').get_attribute('viewBox');visual=(r.get_attribute('data-source-camera-phase'),r.get_attribute('data-source-camera-progress'));sourcecue=r.locator('.v1c-source-cue').inner_text();page.set_viewport_size({'width':840 if width==390 else 390,'height':1000});page.wait_for_timeout(80);assert world==r.locator('[data-source-motion]').inner_html();assert viewbox==r.locator('.v1c-source-anchor svg').get_attribute('viewBox');assert visual==(r.get_attribute('data-source-camera-phase'),r.get_attribute('data-source-camera-progress'));assert sourcecue==r.locator('.v1c-source-cue').inner_text();page.set_viewport_size({'width':width,'height':1000});page.wait_for_timeout(40);before=r.locator('.mlsys-original-stage').inner_html();r.locator('.v1c-source-anchor').scroll_into_view_if_needed();page.wait_for_timeout(60);page.screenshot(path=str(out/f'paused-{width}-{theme}.png'));page.evaluate('window.reviewPlayer.play()');page.wait_for_timeout(90);page.evaluate('window.reviewPlayer.pause()');assert before!=r.locator('.mlsys-original-stage').inner_html()
  page.evaluate('window.reviewPlayer.go(1)');r.focus();page.keyboard.press('p');assert r.get_attribute('data-lesson-state')=='0';page.keyboard.press('r');assert r.get_attribute('data-lesson-state')=='0'
  page.emulate_media(reduced_motion='reduce');r.get_by_role('button',name='Next',exact=True).click();page.wait_for_function('window.reviewPlayer.state.index===1');r.focus();page.keyboard.press('k');page.wait_for_timeout(20);page.keyboard.press('k');page.evaluate('window.reviewPlayer.pause()');r.get_by_role('button',name='Expand',exact=True).click();assert page.locator('dialog').is_visible();page.keyboard.press('Escape');assert page.locator('dialog').count()==0;page.wait_for_timeout(150)
  page.evaluate('window.reviewPlayer.expand()');page.evaluate('i=>window.reviewPlayer.go(i)',count-2);page.wait_for_timeout(100);page.evaluate('window.reviewPlayer.play()');page.wait_for_function('window.reviewPlayer.state.index==='+str(count-1)+' && !window.reviewPlayer.state.playing');replay=page.evaluate("()=>{[...document.querySelectorAll('#v1c-review button')].find(b=>b.textContent==='Replay').click();window.reviewPlayer.pause();return window.reviewPlayer.state.index;}");assert replay<=1;page.evaluate('window.reviewPlayer.closeExpanded()');page.emulate_media(media='print');assert not r.is_visible();assert page.locator('#'+fid+' figure>div[aria-describedby]').is_visible();page.emulate_media(media='screen');r.scroll_into_view_if_needed();page.screenshot(path=str(out/f'chapter-{width}-{theme}.png'));page.close();reports.append({'width':width,'theme':theme,'states':count,'invariants':'passed','controls_motion_freeze_expand_print':'passed'})
 browser.close()
hashes={'source_art_sha1':artentry['asset_sha1'],'script_sha256':hashlib.sha256((ext/(name+'.js')).read_bytes()).hexdigest(),'stylesheet_sha256':hashlib.sha256((ext/(name+'.css')).read_bytes()).hexdigest()};hashes['shared_player_sha256']=hashlib.sha256((ext/'original-player.js').read_bytes()).hexdigest();report={'key':key,'fixture':fixture.as_uri(),'hashes':hashes,'profiles':reports};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
