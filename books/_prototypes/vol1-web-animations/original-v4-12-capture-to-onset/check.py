#!/usr/bin/env python3
"""Full owning-chapter fixture and original-artwork mechanism validation."""
import argparse,base64,hashlib,json,re,runpy
from pathlib import Path
from bs4 import BeautifulSoup
from PIL import Image
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[4];EXT=ROOT/'books/_extensions/mlsysbook-ext/figure-explore';HERE=Path(__file__).resolve().parent
ENTRY=json.loads((HERE/'entry.json').read_text());KEY=ENTRY['key'];NAME=ENTRY['implementation']['name'];FIG=KEY.split(':')[1];INFO=json.loads(Path('/tmp/figures-animation-art-current/index.json').read_text())[KEY]
def fixture(out):
 source=Path(INFO['html']);soup=BeautifulSoup(source.read_text(),'html.parser')
 for node in soup.find_all('script'):node.decompose()
 soup.head.insert(0,soup.new_tag('base',href=source.parent.as_uri()+'/'))
 style=soup.new_tag('style');style.string=(EXT/'original-player.css').read_text()+'\n'+(EXT/(NAME+'.css')).read_text()+'\n@media screen{.original-fixture-static{display:none!important}}';soup.head.append(style)
 asset=Path(INFO['asset']);raw=asset.read_bytes();assert hashlib.sha1(raw).hexdigest()==INFO['asset_sha1']
 if asset.suffix=='.png':
  w,h=Image.open(asset).size;art=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" data-mlsys-artwork-kind="png"><image width="{w}" height="{h}" href="data:image/png;base64,{base64.b64encode(raw).decode()}"/></svg>'
 else:art=raw.decode()
 script=soup.new_tag('script');script.string=(EXT/'original-player.js').read_text()+'\n'+(EXT/(NAME+'.js')).read_text()+'\n'+'''(async()=>{try{
 const theme=new URL(location.href).searchParams.get('theme')||'light';document.documentElement.setAttribute('data-bs-theme',theme);document.body.classList.toggle('quarto-dark',theme==='dark');document.body.classList.toggle('quarto-light',theme==='light');for(const link of document.querySelectorAll('link.quarto-color-scheme'))link.disabled=link.classList.contains('quarto-color-alternate')?theme!=='dark':theme==='dark';for(const link of document.querySelectorAll('link.quarto-color-scheme-extra'))link.disabled=true;
 const outer=document.getElementById(FIG);outer.dataset.mlsysMovementMs='900';outer.dataset.mlsysReadingPauseMs='70';const root=document.createElement('section');root.id='original-fixture';outer.prepend(root);const art=new DOMParser().parseFromString(ART,'image/svg+xml').documentElement;window.fixtureLesson=await window.MLSysFigureLessons[NAME](root,{outer,verifiedArtwork:art});for(const n of outer.querySelectorAll('img'))n.classList.add('original-fixture-static');document.body.dataset.fixtureReady='true';
 }catch(error){document.body.dataset.fixtureError=error.stack;console.error(error)}})();'''.replace('FIG',json.dumps(FIG)).replace('ART',json.dumps(art)).replace('NAME',json.dumps(NAME))
 soup.body.append(script);path=out/'fixture.html';path.write_text(str(soup));return path

def verify(page,i):
 root=page.locator('#original-fixture');assert root.get_attribute('data-step')==str(i)
 key=KEY.split(':')[1]
 if key=='fig-05-action-taps':
  assert root.get_attribute('data-logged')==','.join(str(j+1) for j,k in enumerate([1,2,3,5]) if i>=k)
  assert root.get_attribute('data-measured')==('0' if i>=5 else 'pending')
 if key=='fig-14-authority-handshake-fsm':
  assert root.get_attribute('data-authority')==['AUTO','PEND','PEND','BLEND','BLEND','MAN','FALLBACK','FALLBACK','AUTO','PROTECTIVE'][i]
  assert root.get_attribute('data-policy-token')==str(i<=2 or i==8).lower()
 if key=='fig-07-open-vs-closed-loop':
  import math
  x=float(root.get_attribute('data-position'));assert abs(float(root.get_attribute('data-offset'))-x*math.tan(math.radians(1)))<1e-10
 if key=='fig-10-latch-force-budget':
  f=float(root.get_attribute('data-force'));assert f<=100.00000001
  if i==2:assert abs(f-39)<1e-9
  if i==5:assert abs(f-100)<1e-9
 if key=='fig-12-capture-to-onset':
  t=float(root.get_attribute('data-time'));assert abs(float(root.get_attribute('data-travel'))-t*1.3)<1e-9
  if i==1:assert root.get_attribute('data-lease-started')=='false'
 if key=='fig-03-chunk-supply-horizon':
  if i==3:assert root.get_attribute('data-first-chunk-state')=='replaced'
  if i>=5:assert root.get_attribute('data-first-evidence-valid')=='false'
  counts=json.loads(root.get_attribute('data-setpoint-counts'));assert sum(counts.values())==16
  if i>=3:assert counts['consumed']==2

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--screenshots',default='/tmp/'+NAME+'-review');a=parser.parse_args();out=Path(a.screenshots);out.mkdir(parents=True,exist_ok=True);path=fixture(out)
 if (HERE/'calc.py').exists():
  expected=runpy.run_path(str(HERE/'calc.py'))['D'];source=(EXT/(NAME+'.js')).read_text();actual=json.loads(re.search(r'const D=(\{.*?\});',source).group(1));assert actual==expected,(actual,expected)
 with sync_playwright() as p:
  browser=p.chromium.launch()
  for width in [1440,390]:
   for theme in ['light','dark']:
    page=browser.new_page(viewport={'width':width,'height':900 if width==390 else 1100},color_scheme=theme,reduced_motion='reduce');errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto(path.as_uri()+'?theme='+theme,wait_until='load');page.wait_for_function("document.body.dataset.fixtureReady==='true'||document.body.dataset.fixtureError");assert not page.locator('body').get_attribute('data-fixture-error'),page.locator('body').get_attribute('data-fixture-error')
    root=page.locator('#original-fixture');root.evaluate('(e)=>window.scrollTo(0,window.scrollY+e.getBoundingClientRect().top-90)');count=root.locator('select option').count()
    for i in range(count):
     root.locator('select').select_option(str(i));verify(page,i);root.locator('.mlsys-original-stage').evaluate('(e)=>window.scrollTo(0,window.scrollY+e.getBoundingClientRect().top-280)');page.screenshot(path=str(out/f'{width}-{theme}-step{i}.png'))
    root.locator('select').select_option('0');root.focus();page.keyboard.press('n');verify(page,1);page.keyboard.press('p');verify(page,0);page.keyboard.press('r');verify(page,0)
    root.get_by_role('button',name='Expand',exact=True).click();assert page.locator('dialog[open]').count()==1;page.screenshot(path=str(out/f'{width}-{theme}-expanded.png'));page.keyboard.press('Escape');assert page.locator('dialog[open]').count()==0
    assert root.evaluate('(e)=>{const r=e.getBoundingClientRect();return r.left>=0&&r.right<=innerWidth+1}')
    page.emulate_media(media='print');assert not root.is_visible();assert page.locator('#'+FIG+' img').first.is_visible();page.close()
    # Fresh viewport-only context for pausable motion, avoiding tall component IO artifacts.
    page=browser.new_page(viewport={'width':width,'height':900 if width==390 else 1100},color_scheme=theme,reduced_motion='no-preference');page.goto(path.as_uri()+'?theme='+theme,wait_until='load');page.wait_for_function("document.body.dataset.fixtureReady==='true'");root=page.locator('#original-fixture');root.evaluate('(e)=>window.scrollTo(0,window.scrollY+e.getBoundingClientRect().top-90)');root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(180);page.evaluate('window.fixtureLesson.pause()')
    frozen=root.locator('.mlsys-original-stage, .mlsys-original-details, .mlsys-original-narration, .mlsys-original-current-cue').evaluate_all('(els)=>els.map(e=>e.innerHTML)');progress=root.get_attribute('data-animation-progress');assert progress is not None and 0<float(progress)<1,progress;assert 'not yet completed' in root.locator('.mlsys-original-narration').inner_text();page.wait_for_timeout(140);assert frozen==root.locator('.mlsys-original-stage, .mlsys-original-details, .mlsys-original-narration, .mlsys-original-current-cue').evaluate_all('(els)=>els.map(e=>e.innerHTML)');assert progress==root.get_attribute('data-animation-progress');page.screenshot(path=str(out/f'{width}-{theme}-paused.png'));root.evaluate('(e)=>e.focus({preventScroll:true})');page.keyboard.press('k');page.wait_for_timeout(80);assert float(root.get_attribute('data-animation-progress'))>float(progress);page.keyboard.press('k');assert root.get_attribute('data-playing')=='false';root.locator('select').select_option('0');page.emulate_media(reduced_motion='reduce');root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function("n=>document.querySelector('#original-fixture').dataset.step===String(n)",arg=count-1,timeout=8000);verify(page,count-1);root.get_by_role('button',name='Restart',exact=True).click();verify(page,0);assert not errors,errors;page.close();print('PASS',NAME,width,theme)
  nojs=browser.new_page(java_script_enabled=False);nojs.goto(path.as_uri(),wait_until='load');assert nojs.locator('#'+FIG+' img').first.is_visible();nojs.close();browser.close()
 assert hashlib.sha1(Path(INFO['asset']).read_bytes()).hexdigest()==INFO['asset_sha1'];print('FIXTURE',path.as_uri(),'SHA1',INFO['asset_sha1'])
if __name__=='__main__':main()
