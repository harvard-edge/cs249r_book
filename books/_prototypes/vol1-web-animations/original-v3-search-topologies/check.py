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
 import subprocess
 record=json.loads(subprocess.check_output(['python3',str(ROOT/'binder/tools/scripts/build_original_figure_fixture.py'),'--entry',str(HERE/'entry.json')],text=True))
 return record['url']

def theme_page(page,theme):
 page.evaluate("""theme=>{document.documentElement.setAttribute('data-bs-theme',theme);document.body.classList.toggle('quarto-dark',theme==='dark');document.body.classList.toggle('quarto-light',theme==='light');for(const link of document.querySelectorAll('link.quarto-color-scheme'))link.disabled=link.classList.contains('quarto-color-alternate')?theme!=='dark':theme==='dark';for(const link of document.querySelectorAll('link.quarto-color-scheme-extra'))link.disabled=true;}""",theme)

def verify(page,index):
 root=page.locator('[id$="-fixture-explore"]');assert root.get_attribute('data-step')==str(index)
 s=json.loads(root.get_attribute('data-original-state'))
 if NAME.endswith('block-diagonal-mask'):
  assert s['masked']==(index>0)
  if index==2: assert s['probabilities'][:5]==[0]*5 and sum(s['probabilities'])==1
 elif NAME.endswith('hierarchical-budget'):
  assert sum(s['mills'].values())==s['budget']
  if index==3: assert s['mills']['r']==2500 and not s['remoteSettled']
 elif NAME.endswith('intent-before-dispatch'):
  assert s['durable']==(index>=1) and s['established']==(index==3)
 elif NAME.endswith('collection-pipeline'):
  assert s['creditHeld']==(index<4) and s['terminalRecord']==(index==4)
 elif NAME.endswith('workspace-overlay-visibility'):
  assert s['lower']=={'code':'old','data':'base'} and s['taskB']=={'code':'old','data':'base'}
 elif NAME.endswith('radix-tree-cache'):
  assert s['residentOther']==(index<3) and s['residentFinal']==(index==0)
 elif NAME.endswith('search-topologies'):
  assert s['bestCompleted']==(4 if index>=3 else 0) and s['treePathUpdated']==(index==4)
  if index==3:
   colors=root.locator('svg').evaluate("svg=>[...svg.children].filter(n=>n.tagName==='g').slice(0,2).flatMap((p,i)=>[...p.querySelectorAll('circle')].filter(c=>+c.getAttribute('cx')===[180,160][i]).map(c=>[c.getAttribute('fill'),c.getAttribute('stroke')]))")
   assert colors and all(c==['#e0f2fe','#0284c7'] for c in colors)

def moving_edges(browser,path,width,theme,out,engine):
 for source in range(len(ENTRY['storyboard'])-1):
  page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme,reduced_motion='no-preference')
  page.goto(path,wait_until='load');page.wait_for_function("window.fixturePlayer && document.querySelector('[data-fixture-state]')?.dataset.fixtureState==='ready'");theme_page(page,theme)
  root=page.locator('[id$="-fixture-explore"]');root.locator('select').select_option(str(source));root.get_by_role('button',name='Next',exact=True).scroll_into_view_if_needed();page.wait_for_timeout(120);root.get_by_role('button',name='Next',exact=True).evaluate("async b=>{b.click();await new Promise(r=>setTimeout(r,120));window.fixturePlayer.pause()}")
  assert root.get_attribute('data-step')==str(source);assert 0<float(root.get_attribute('data-animation-progress'))<1
  page.evaluate('window.fixturePlayer.pause()');frozen=root.get_attribute('data-animation-progress');committed=root.get_attribute('data-original-state');page.wait_for_timeout(100);assert root.get_attribute('data-animation-progress')==frozen and root.get_attribute('data-original-state')==committed
  root.locator('.mlsys-original-stage').scroll_into_view_if_needed();page.wait_for_timeout(60);page.screenshot(path=str(out/f'{engine}-{width}-{theme}-motion{source}-{source+1}.png'))
  page.emulate_media(reduced_motion='reduce');page.wait_for_function('i=>window.fixtureRoot.dataset.step===String(i)',arg=source+1);verify(page,source+1);page.close()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--engine',choices=['chromium','webkit'],default='chromium');parser.add_argument('--screenshots',default='/tmp/'+NAME+'-review');a=parser.parse_args();out=Path(a.screenshots);out.mkdir(parents=True,exist_ok=True);path=fixture(out)
 original_hash=hashlib.sha256(Path(INDEX['asset']).read_bytes()).hexdigest()
 with sync_playwright() as p:
  browser=getattr(p,a.engine).launch()
  for width in [1440,390]:
   for theme in ['light','dark']:
    page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme,reduced_motion='reduce');page.goto(path+'?theme='+theme,wait_until='load');page.wait_for_function("document.querySelector('[data-fixture-state]')?.dataset.fixtureState==='ready'||window.fixtureError");assert not page.evaluate('window.fixtureError'),page.evaluate('window.fixtureError');theme_page(page,theme)
    root=page.locator('[id$="-fixture-explore"]');root.scroll_into_view_if_needed();count=root.locator('select option').count();assert count==len(ENTRY["storyboard"])
    for index in range(count):
     root.locator('select').select_option(str(index));verify(page,index);root.screenshot(path=str(out/f'{a.engine}-{width}-{theme}-step{index}.png'))
    root.locator('select').select_option('0');root.focus();page.keyboard.press('n');page.wait_for_function("window.fixtureRoot.dataset.step==='1'");verify(page,1);assert root.get_attribute('data-playing')=='false';page.keyboard.press('p');verify(page,0)
    page.close();page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme,reduced_motion='no-preference');page.goto(path+'?theme='+theme,wait_until='load');page.wait_for_function("document.querySelector('[data-fixture-state]')?.dataset.fixtureState==='ready'");theme_page(page,theme);root=page.locator('[id$="-fixture-explore"]');root.scroll_into_view_if_needed();root.get_by_role('button',name='Next',exact=True).scroll_into_view_if_needed();page.wait_for_timeout(120);root.get_by_role('button',name='Next',exact=True).evaluate("async b=>{b.click();await new Promise(r=>setTimeout(r,120));window.fixturePlayer.pause()}");assert root.get_attribute('data-step')=='0';assert 0<float(root.get_attribute('data-animation-progress'))<1
    page.evaluate('window.fixturePlayer.pause()');frozen=root.get_attribute('data-animation-progress');objects=root.locator('[data-motion-object]').evaluate_all('(els)=>els.map(el=>[el.getAttribute("transform"),el.getAttribute("style")])');page.wait_for_timeout(100);assert root.get_attribute('data-animation-progress')==frozen;assert root.locator('[data-motion-object]').evaluate_all('(els)=>els.map(el=>[el.getAttribute("transform"),el.getAttribute("style")])')==objects;page.screenshot(path=str(out/f'{a.engine}-{width}-{theme}-moving.png'))
    root.focus();root.evaluate("r=>{r.dispatchEvent(new KeyboardEvent('keydown',{key:'k',bubbles:true}));[...r.querySelectorAll('button')].find(b=>b.textContent==='Next').click()}");verify(page,1);assert root.get_attribute('data-playing')=='false'
    page.emulate_media(reduced_motion='reduce');root.get_by_role('button',name='Play',exact=True).scroll_into_view_if_needed();page.wait_for_timeout(150);root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function("i=>window.fixtureRoot.dataset.step===String(i)",arg=count-1,timeout=count*6000+1000);verify(page,count-1)
    root.get_by_role('button',name='Expand',exact=True).click();page.wait_for_timeout(100);assert page.locator('dialog[open]').count()==1;verify(page,count-1);page.screenshot(path=str(out/f'{a.engine}-{width}-{theme}-expanded.png'));page.keyboard.press('Escape');page.wait_for_timeout(100);assert page.locator('dialog[open]').count()==0;verify(page,count-1)
    assert root.evaluate('(el)=>{const r=el.getBoundingClientRect();return r.left>=0&&r.right<=innerWidth+1}')
    page.screenshot(path=str(out/f'{a.engine}-{width}-{theme}-chapter.png'));page.get_by_role('tab',name='Diagram',exact=True).click();assert not root.is_visible();assert page.locator('#'+FIG+' img').first.is_visible();page.get_by_role('tab',name='Explore',exact=True).click();verify(page,count-1);page.emulate_media(media='print');assert not root.is_visible();assert page.locator('#'+FIG+' img').first.is_visible();page.close();moving_edges(browser,path,width,theme,out,a.engine);print('PASS',NAME,a.engine,width,theme)
  nojs=browser.new_page(java_script_enabled=False);nojs.goto(path,wait_until='load');assert nojs.locator('#'+FIG+' img').first.is_visible();nojs.close();browser.close()
 assert hashlib.sha256(Path(INDEX['asset']).read_bytes()).hexdigest()==original_hash
 print('FIXTURE',path)
if __name__=='__main__':main()
