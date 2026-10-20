#!/usr/bin/env python3
"""Inspect the lesson on the actual chapter, using its original SVG and theme."""
import argparse, json
from pathlib import Path
from playwright.sync_api import sync_playwright
BASE=Path(__file__).resolve().parents[3]/'_extensions/mlsysbook-ext/figure-explore'
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8882/_build/original-figure-life/vol1/06_nn_architectures/06_nn_architectures.html');p.add_argument('--engine',default='chromium');p.add_argument('--screenshots',default='/tmp/original-prefill-review');a=p.parse_args()
art=Path(json.load(open('/tmp/figures-animation-art-current/index.json'))['vol1:fig-prefill-decode']['asset']).read_text()
out=Path(a.screenshots);out.mkdir(exist_ok=True)
with sync_playwright() as pw:
 browser=getattr(pw,a.engine).launch()
 for width in [1440,390]:
  for dark in [False,True]:
   context=browser.new_context(viewport={'width':width,'height':1050},color_scheme='dark' if dark else 'light',reduced_motion='reduce')
   page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto(a.url);page.wait_for_load_state('networkidle')
   page.evaluate('(dark)=>{document.body.classList.toggle("quarto-dark",dark);document.body.classList.toggle("quarto-light",!dark);}',dark)
   for name in ['original-player','original-prefill']:
    page.add_script_tag(path=str(BASE/(name+'.js')));page.add_style_tag(path=str(BASE/(name+'.css')))
   page.evaluate('''async art=>{const outer=document.querySelector('#fig-prefill-decode'); outer.dataset.mlsysMovementMs='900';outer.dataset.mlsysReadingPauseMs='100';const figure=outer.querySelector('figure')||outer; const caption=figure.querySelector('figcaption');const root=document.createElement('div');root.id='prefill-test';figure.querySelectorAll(':scope > :not(figcaption)').forEach(n=>n.hidden=true);figure.insertBefore(root,caption);const svg=new DOMParser().parseFromString(art,'image/svg+xml').documentElement;window.testPrefill=await window.MLSysFigureLessons['original-prefill'](root,{outer,verifiedArtwork:svg});}''',art)
   root=page.locator('#prefill-test');root.scroll_into_view_if_needed();page.wait_for_timeout(100)
   assert root.get_attribute('data-cache-count')=='0'
   for step,count in [(2,4),(3,4),(4,4),(5,5),(6,5),(7,5),(8,5),(9,6),(10,6),(11,6),(12,6),(13,7),(14,7),(15,7),(16,0)]:
    page.evaluate('(i)=>window.testPrefill.go(i)',step)
    assert root.get_attribute('data-cache-count')==str(count)
    assert root.locator('[data-cache-token]').evaluate_all('(els)=>els.filter(e=>getComputedStyle(e).visibility!=="hidden").length')==count
    if step in [2,4,6,8,13,14,15,16]:
     root.screenshot(path=str(out/f'{a.engine}-{width}-{ "dark" if dark else "light"}-step{step}.png'))
   page.evaluate('window.testPrefill.go(6)');assert root.locator('[data-cache-token="mat"]').evaluate('(e)=>getComputedStyle(e).visibility')=='hidden'
   page.evaluate('window.testPrefill.go(14)');assert root.locator('[data-token="[end]"] text').evaluate('(e)=>getComputedStyle(e).visibility')=='visible';assert root.locator('[data-cache-token="[end]"]').count()==0
   page.evaluate('window.testPrefill.restart()');root.scroll_into_view_if_needed();page.wait_for_timeout(100);root.focus();page.keyboard.press('n');page.wait_for_function("document.querySelector('#prefill-test').dataset.prefillState==='1'");page.keyboard.press('p');assert root.get_attribute('data-prefill-state')=='0'
   # Normal-motion writes expose the committed cache only after the movement.
   page.emulate_media(reduced_motion='no-preference');page.wait_for_timeout(150);assert page.evaluate("()=>{window.testPrefill.go(1);window.testPrefill.next();return document.querySelector('#prefill-test').dataset.cacheCount}")=='0';page.wait_for_function("document.querySelector('#prefill-test').dataset.cacheCount==='4'")
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
   assert not errors,errors
   print(a.engine,width,'dark' if dark else 'light','PASS')
   context.close()
 browser.close()
