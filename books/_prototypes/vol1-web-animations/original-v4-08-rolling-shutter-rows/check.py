#!/usr/bin/env python3
"""Source-chapter review fixture. Copied into each independently owned prototype."""
import argparse,hashlib,json,base64
from PIL import Image
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[4];EXT=ROOT/'books/_extensions/mlsysbook-ext/figure-explore'
HERE=Path(__file__).resolve().parent;ENTRY=json.loads((HERE/'entry.json').read_text());KEY=ENTRY['key'];NAME=ENTRY['implementation']['name'];FIG=KEY.split(':')[1]
INDEX=json.loads(Path('/tmp/figures-animation-art-current/index.json').read_text())[KEY]

def fixture(out):
 source=Path(INDEX['html']);soup=BeautifulSoup(source.read_text(),'html.parser')
 for node in soup.find_all('script'):node.decompose()
 base=soup.new_tag('base',href=source.parent.as_uri()+'/');soup.head.insert(0,base)
 style=soup.new_tag('style');style.string=(EXT/'original-player.css').read_text()+'\n'+(EXT/ENTRY['implementation']['stylesheet']).read_text()+'\n@media screen{.original-fixture-static{display:none!important}}';soup.head.append(style)
 asset=Path(INDEX['asset'])
 if asset.suffix=='.png':
  w,h=Image.open(asset).size;svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" data-mlsys-artwork-kind="png" data-mlsys-artwork-width="{w}" data-mlsys-artwork-height="{h}"><image width="{w}" height="{h}" href="data:image/png;base64,{base64.b64encode(asset.read_bytes()).decode()}"/></svg>'
 else:svg=asset.read_text()
 player=(EXT/'original-player.js').read_text();module=(EXT/ENTRY['implementation']['script']).read_text()
 script=soup.new_tag('script');script.string=player+'\n'+module+'\n'+'''(async()=>{try{
 const theme=new URL(location.href).searchParams.get('theme')||'light';document.documentElement.setAttribute('data-bs-theme',theme);document.body.classList.toggle('quarto-dark',theme==='dark');document.body.classList.toggle('quarto-light',theme==='light');for(const link of document.querySelectorAll('link.quarto-color-scheme'))link.disabled=link.classList.contains('quarto-color-alternate')?theme!=='dark':theme==='dark';for(const link of document.querySelectorAll('link.quarto-color-scheme-extra'))link.disabled=true;
 const outer=document.getElementById(FIG);outer.dataset.mlsysMovementMs='900';outer.dataset.mlsysReadingPauseMs='70';const root=document.createElement('section');root.id='original-fixture';outer.prepend(root);const art=new DOMParser().parseFromString(ART,'image/svg+xml').documentElement;window.fixtureLesson=await window.MLSysFigureLessons[NAME](root,{outer,verifiedArtwork:art});outer.querySelector('img').classList.add('original-fixture-static');document.body.dataset.fixtureReady='true';
 }catch(error){document.body.dataset.fixtureError=error.stack;console.error(error)}})();'''.replace('FIG',json.dumps(FIG)).replace('ART',json.dumps(svg)).replace('NAME',json.dumps(NAME))
 soup.body.append(script);path=out/'fixture.html';path.write_text(str(soup));return path

def verify(page,index):
 root=page.locator('#original-fixture');assert root.get_attribute('data-step')==str(index)
 state=json.loads(root.get_attribute('data-original-state'));assert state['index']==index
 assert root.locator('svg').count()>=1
 assert root.locator('p[class$="-proof"]').inner_text()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--engine',choices=['chromium','webkit'],default='chromium');parser.add_argument('--screenshots',default='/tmp/'+NAME+'-review');a=parser.parse_args();out=Path(a.screenshots);out.mkdir(parents=True,exist_ok=True);path=fixture(out)
 original_hash=hashlib.sha256(Path(INDEX['asset']).read_bytes()).hexdigest()
 with sync_playwright() as p:
  browser=getattr(p,a.engine).launch()
  for width in [1440,390]:
   for theme in ['light','dark']:
    page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme,reduced_motion='reduce');page.goto(path.as_uri()+'?theme='+theme,wait_until='load');page.wait_for_function("document.body.dataset.fixtureReady==='true'||document.body.dataset.fixtureError");assert not page.locator('body').get_attribute('data-fixture-error'),page.locator('body').get_attribute('data-fixture-error')
    root=page.locator('#original-fixture');root.scroll_into_view_if_needed();count=root.locator('select option').count();assert count==3
    for index in range(count):
     root.locator('select').select_option(str(index));verify(page,index);root.screenshot(path=str(out/f'{a.engine}-{width}-{theme}-step{index}.png'))
    root.locator('select').select_option('0');root.focus();page.keyboard.press('n');page.wait_for_function("document.querySelector('#original-fixture').dataset.step==='1'");verify(page,1);assert root.get_attribute('data-playing')=='false';page.keyboard.press('p');verify(page,0)
    page.emulate_media(reduced_motion='no-preference');root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(150);assert root.get_attribute('data-step')=='0';assert 0<float(root.get_attribute('data-animation-progress'))<1
    page.evaluate('window.fixtureLesson.pause()');frozen=root.get_attribute('data-animation-progress');objects=root.locator('[data-motion-object]').evaluate_all('(els)=>els.map(el=>[el.getAttribute("transform"),el.getAttribute("style")])');page.wait_for_timeout(100);assert root.get_attribute('data-animation-progress')==frozen;assert root.locator('[data-motion-object]').evaluate_all('(els)=>els.map(el=>[el.getAttribute("transform"),el.getAttribute("style")])')==objects;root.screenshot(path=str(out/f'{a.engine}-{width}-{theme}-moving.png'))
    root.locator('select').select_option('0');verify(page,0)
    page.emulate_media(reduced_motion='reduce');root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function("i=>document.querySelector('#original-fixture').dataset.step===String(i)",arg=count-1,timeout=8000);verify(page,count-1)
    root.get_by_role('button',name='Expand',exact=True).click();page.wait_for_timeout(100);assert page.locator('dialog[open]').count()==1;verify(page,count-1);page.screenshot(path=str(out/f'{a.engine}-{width}-{theme}-expanded.png'));page.keyboard.press('Escape');page.wait_for_timeout(100);assert page.locator('dialog[open]').count()==0;verify(page,count-1)
    assert root.evaluate('(el)=>{const r=el.getBoundingClientRect();return r.left>=0&&r.right<=innerWidth+1}')
    page.screenshot(path=str(out/f'{a.engine}-{width}-{theme}-chapter.png'));page.emulate_media(media='print');assert not root.is_visible();assert page.locator('#'+FIG+' img').first.is_visible();page.close();print('PASS',NAME,a.engine,width,theme)
  nojs=browser.new_page(java_script_enabled=False);nojs.goto(path.as_uri(),wait_until='load');assert nojs.locator('#'+FIG+' img').first.is_visible();nojs.close();browser.close()
 assert hashlib.sha256(Path(INDEX['asset']).read_bytes()).hexdigest()==original_hash
 print('FIXTURE',path.as_uri())
if __name__=='__main__':main()


def shared_check():
 # Exercise shared Diagram/Explore controller in real HTTP chapter fixtures.
 import subprocess
 fixture=json.loads(subprocess.check_output(['python3',str(ROOT/'binder/tools/scripts/build_original_figure_fixture.py'),'--entry',str(HERE/'entry.json')],text=True))
 out=Path('/tmp/'+NAME+'-shared-review');out.mkdir(exist_ok=True)
 with sync_playwright() as p:
  browser=p.chromium.launch()
  for width in [1440,390]:
   for theme in ['light','dark']:
    context=browser.new_context(viewport={'width':width,'height':1100},reduced_motion='reduce',color_scheme=theme)
    page=context.new_page();page.goto(fixture['url'],wait_until='load');page.wait_for_function('window.fixturePlayer||window.fixtureError');assert not page.evaluate('window.fixtureError')
    page.evaluate('''theme=>{document.documentElement.setAttribute('data-bs-theme',theme);document.body.classList.toggle('quarto-dark',theme==='dark');document.body.classList.toggle('quarto-light',theme==='light');for(const link of document.querySelectorAll('link.quarto-color-scheme'))link.disabled=link.classList.contains('quarto-color-alternate')?theme!=='dark':theme==='dark';for(const link of document.querySelectorAll('link.quarto-color-scheme-extra'))link.disabled=true;}''',theme)
    root=page.locator('#'+FIG+' .mlsys-original-player');root.scroll_into_view_if_needed();count=root.locator('select option').count()
    for index in range(count):
     root.locator('select').select_option(str(index));assert root.get_attribute('data-step')==str(index)
     assert json.loads(root.get_attribute('data-original-state'))['index']==index
    root.locator('select').select_option('0');root.focus();page.keyboard.press('n');assert root.get_attribute('data-step')=='1';page.keyboard.press('p');assert root.get_attribute('data-step')=='0';page.keyboard.press('r');assert root.get_attribute('data-step')=='0'
    # Fresh normal-motion sample keeps the relevant stage in the viewport.
    page.emulate_media(reduced_motion='no-preference');root.locator('select').select_option('0');root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(120);page.evaluate('window.fixturePlayer.pause()');frozen=root.get_attribute('data-animation-progress');assert 0<float(frozen)<1
    snapshot=root.locator('[data-motion-object]').evaluate_all('(els)=>els.map(el=>el.outerHTML)');page.wait_for_timeout(100);assert root.get_attribute('data-animation-progress')==frozen;assert root.locator('[data-motion-object]').evaluate_all('(els)=>els.map(el=>el.outerHTML)')==snapshot
    page.screenshot(path=str(out/f'{width}-{theme}-paused-viewport.png'))
    outer=page.locator('#'+FIG);outer.get_by_role('tab',name='Diagram',exact=True).click();page.wait_for_timeout(100);assert not root.is_visible();assert root.get_attribute('data-animation-progress')==frozen
    outer.get_by_role('tab',name='Explore',exact=True).click();assert root.is_visible();assert root.get_attribute('data-animation-progress')==frozen
    page.emulate_media(reduced_motion='reduce');root.locator('select').select_option(str(count-1));root.get_by_role('button',name='Expand',exact=True).click();assert page.locator('dialog[open]').count()==1;page.keyboard.press('Escape');assert page.locator('dialog[open]').count()==0
    if width==1440 and theme=='light':
     # Current stage is captured independently of the sticky chapter header.
     root.locator('.mlsys-original-stage').screenshot(path=str(out/'final-mechanism.png'))
    assert root.evaluate('(el)=>el.getBoundingClientRect().right<=innerWidth+1')
    context.close();print('SHARED PASS',NAME,width,theme)
  browser.close()
 print('SHARED FIXTURE',fixture['url'])

if __name__=='__main__':shared_check()
