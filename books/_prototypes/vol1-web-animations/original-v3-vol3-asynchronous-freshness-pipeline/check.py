#!/usr/bin/env python3
"""Source-chapter review fixture. Copied into each independently owned prototype."""
import argparse,hashlib,json
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
 svg=Path(INDEX['asset']).read_text();player=(EXT/'original-player.js').read_text();module=(EXT/ENTRY['implementation']['script']).read_text()
 script=soup.new_tag('script');script.string=player+'\n'+module+'\n'+'''(async()=>{try{
 const theme=new URL(location.href).searchParams.get('theme')||'light';document.documentElement.setAttribute('data-bs-theme',theme);document.body.classList.toggle('quarto-dark',theme==='dark');document.body.classList.toggle('quarto-light',theme==='light');for(const link of document.querySelectorAll('link.quarto-color-scheme'))link.disabled=link.classList.contains('quarto-color-alternate')?theme!=='dark':theme==='dark';for(const link of document.querySelectorAll('link.quarto-color-scheme-extra'))link.disabled=true;
 const outer=document.getElementById(FIG);outer.dataset.mlsysMovementMs='900';outer.dataset.mlsysReadingPauseMs='70';const root=document.createElement('section');root.id='original-fixture';outer.prepend(root);const art=new DOMParser().parseFromString(ART,'image/svg+xml').documentElement;window.fixtureLesson=await window.MLSysFigureLessons[NAME](root,{outer,verifiedArtwork:art});outer.querySelector('img').classList.add('original-fixture-static');document.body.dataset.fixtureReady='true';
 }catch(error){document.body.dataset.fixtureError=error.stack;console.error(error)}})();'''.replace('FIG',json.dumps(FIG)).replace('ART',json.dumps(svg)).replace('NAME',json.dumps(NAME))
 soup.body.append(script);path=out/'fixture.html';path.write_text(str(soup));return path

def verify(page,index):
 root=page.locator('#original-fixture');assert root.get_attribute('data-step')==str(index)
 assert root.get_attribute('data-mechanism-state')==str(index)
 facts=json.loads(root.get_attribute('data-facts'))
 assert root.get_attribute('data-invariant')
 # Original paths and line geometry are preserved verbatim; live overlays add
 # identity and progress without replacing the source architecture.
 from lxml import etree as ET
 original=ET.parse(INDEX['asset']).getroot()
 paths=[n.get('d')for n in original.iter()if isinstance(n.tag,str)and ET.QName(n).localname=='path']
 assert root.locator('svg').evaluate('(s)=>[...s.querySelectorAll("path")].filter(n=>!n.closest("[data-live-overlay]")).map(n=>n.getAttribute("d"))')==paths
 assert int(root.get_attribute('data-original-child-count'))==sum(isinstance(n.tag,str)for n in original)
 logical={b['id']:b['label']for b in facts['boxes']}
 if index>=4:
  assert 'check v1' in logical['A'] and 'lag '+str(1-0) in logical['A']
  assert 'check v3' in logical['B'] and 'lag '+str(3-1) in logical['B']
  assert 'v0' in root.locator('[data-object="A"]').text_content() and 'lag 1' in root.locator('[data-object="A"]').text_content()
 if index==6: assert 'lag '+str(4-0)+' > 2' in logical['C']
 assert root.locator('[data-object="C"]').count()==(index>=5)
 txt=root.locator('svg').text_content();assert 'Synchronous group' in txt and 'Asynchronous collection' in txt and 'Admission path for one record' in txt

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--engine',choices=['chromium','webkit'],default='chromium');parser.add_argument('--screenshots',default='/tmp/'+NAME+'-review');a=parser.parse_args();out=Path(a.screenshots);out.mkdir(parents=True,exist_ok=True);path=fixture(out)
 original_hash=hashlib.sha256(Path(INDEX['asset']).read_bytes()).hexdigest()
 with sync_playwright() as p:
  browser=getattr(p,a.engine).launch()
  for width in [1440,390]:
   for theme in ['light','dark']:
    page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme,reduced_motion='reduce');page.goto(path.as_uri()+'?theme='+theme,wait_until='load');page.wait_for_function("document.body.dataset.fixtureReady==='true'||document.body.dataset.fixtureError");assert not page.locator('body').get_attribute('data-fixture-error'),page.locator('body').get_attribute('data-fixture-error')
    root=page.locator('#original-fixture');root.scroll_into_view_if_needed();count=root.locator('select option').count();assert count==7
    for index in range(count):
     root.locator('select').select_option(str(index));verify(page,index);root.screenshot(path=str(out/f'{a.engine}-{width}-{theme}-step{index}.png'))
    root.locator('select').select_option('0');root.focus();page.keyboard.press('n');page.wait_for_function("document.querySelector('#original-fixture').dataset.step==='1'");verify(page,1);assert root.get_attribute('data-playing')=='false';page.keyboard.press('p');verify(page,0)
    page.close();page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme,reduced_motion='no-preference');page.goto(path.as_uri()+'?theme='+theme,wait_until='load');page.wait_for_function("document.body.dataset.fixtureReady==='true'");root=page.locator('#original-fixture');root.scroll_into_view_if_needed();committed=root.get_attribute('data-facts');root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(150);assert root.get_attribute('data-step')=='0';assert 0<float(root.get_attribute('data-animation-progress'))<1
    page.evaluate('window.fixtureLesson.pause()');frozen=root.get_attribute('data-animation-progress');objects=root.locator('[data-motion-object]').evaluate_all('(els)=>els.map(el=>[el.getAttribute("transform"),el.getAttribute("opacity"),el.querySelector("rect")?.getAttribute("width"),el.getAttribute("style")])');page.wait_for_timeout(100);assert root.get_attribute('data-animation-progress')==frozen;assert root.locator('[data-motion-object]').evaluate_all('(els)=>els.map(el=>[el.getAttribute("transform"),el.getAttribute("opacity"),el.querySelector("rect")?.getAttribute("width"),el.getAttribute("style")])')==objects;assert root.get_attribute('data-mechanism-state')=='0';assert root.get_attribute('data-facts')==committed;page.screenshot(path=str(out/f'{a.engine}-{width}-{theme}-moving.png'))
    root.focus();page.keyboard.press('k');page.wait_for_timeout(50);root.get_by_role('button',name='Next',exact=True).click();verify(page,1);assert root.get_attribute('data-playing')=='false'
    page.emulate_media(reduced_motion='reduce');root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function("i=>document.querySelector('#original-fixture').dataset.step===String(i)",arg=count-1,timeout=8000);verify(page,count-1)
    root.get_by_role('button',name='Expand',exact=True).click();page.wait_for_timeout(100);assert page.locator('dialog[open]').count()==1;verify(page,count-1);page.screenshot(path=str(out/f'{a.engine}-{width}-{theme}-expanded.png'));page.keyboard.press('Escape');page.wait_for_timeout(100);assert page.locator('dialog[open]').count()==0;verify(page,count-1)
    assert root.evaluate('(el)=>{const r=el.getBoundingClientRect();return r.left>=0&&r.right<=innerWidth+1}')
    if width==1440: assert root.locator('.v3b-canvas').evaluate('(el)=>el.scrollWidth<=el.clientWidth+1')
    root.focus();page.keyboard.press('r');verify(page,0);root.locator('select').select_option(str(count-1));verify(page,count-1)
    page.screenshot(path=str(out/f'{a.engine}-{width}-{theme}-chapter.png'));page.emulate_media(media='print');assert not root.is_visible();assert page.locator('#'+FIG+' img').first.is_visible();page.close();print('PASS',NAME,a.engine,width,theme)
  nojs=browser.new_page(java_script_enabled=False);nojs.goto(path.as_uri(),wait_until='load');assert nojs.locator('#'+FIG+' img').first.is_visible();nojs.close();browser.close()
 assert hashlib.sha256(Path(INDEX['asset']).read_bytes()).hexdigest()==original_hash
 print('FIXTURE',path.as_uri())
if __name__=='__main__':main()
