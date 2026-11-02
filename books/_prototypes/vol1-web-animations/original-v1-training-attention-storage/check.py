#!/usr/bin/env python3
"""Build and exercise an actual owning chapter QA fixture, preserving static view."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[4];EXT=ROOT/'books/_extensions/mlsysbook-ext/figure-explore';NAME=Path(__file__).parent.name
p=argparse.ArgumentParser();p.add_argument('--screenshots',default='/tmp/'+NAME+'-review');p.add_argument('--engine',default='chromium');args=p.parse_args();shots=Path(args.screenshots);shots.mkdir(exist_ok=True)
entry=Path(__file__).parent/'entry.json';fixture=json.loads(subprocess.check_output(['/tmp/venv/bin/python',str(ROOT/'binder/tools/scripts/build_original_figure_fixture.py'),'--entry',str(entry)],text=True));print(fixture['url'],flush=True)
with sync_playwright() as pw:
 browser=getattr(pw,args.engine).launch()
 for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
  page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme,reduced_motion='reduce');errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto(fixture['url']);page.wait_for_function("window.fixtureRoot && document.querySelector('[data-fixture-state=ready]')");page.evaluate('(t)=>{document.body.classList.toggle("quarto-dark",t==="dark");document.body.classList.toggle("quarto-light",t!=="dark");document.documentElement.setAttribute("data-bs-theme",t);for(const l of document.querySelectorAll("link.quarto-color-scheme")){l.disabled=t==="dark"?!l.classList.contains("quarto-color-alternate"):l.classList.contains("quarto-color-alternate")}}',theme)
  root=page.locator('.'+NAME);page.get_by_role('tab',name='Explore',exact=True).click();root.scroll_into_view_if_needed();sel=root.locator('.mlsys-original-sequence select');count=sel.locator('option').count()
  for i in range(count):
   sel.select_option(str(i));assert root.get_attribute('data-endpoint')==str(i)

   if NAME.endswith('allocator-live-cached-blocks'):
    assert int(root.get_attribute('data-live'))+int(root.get_attribute('data-reusable'))==int(root.get_attribute('data-reserved'))==12
    assert int(root.get_attribute('data-largest'))<6 if i==4 else True
   elif NAME.endswith('grad-accumulation'):
    assert int(root.get_attribute('data-contributions'))==(0 if i==8 else min(3,i//2))
    assert int(root.get_attribute('data-activations'))==((i+1)//2 if i in [1,3,5] else 0)
    assert int(root.get_attribute('data-updates'))==int(i>=7)
   elif NAME.endswith('kv-cache-growth'):
    assert float(root.get_attribute('data-kv-gb'))<=50.599345920000005
    assert root.get_attribute('data-rejected')==str(i==2).lower()
    if i==2:assert root.get_attribute('data-tokens')=='4800'
   elif NAME.endswith('python-tax'):
    assert int(root.get_attribute('data-operations'))==(i//2 if i<=10 else 5 if i>=13 else 0)
   elif NAME.endswith('systolic-array'):
    assert root.get_attribute('data-outputs')==('5 5' if i>=4 else '5 _' if i>=3 else '_ _')
   elif NAME.endswith('training-attention-storage'):
    proof=root.locator('.vb-readout').inner_text()
    assert proof.startswith('One query, four keys · ')
    if i==5:assert 'initialized; no block merged yet' in proof
    if i in [3,9]:assert abs(float(root.get_attribute('data-output-value'))-sum([2,4,6*2,8*2])/sum([1,1,2,2]))<1e-12
    if i>=4:assert root.get_attribute('data-full-intermediate')=='none'
   elif NAME.endswith('data-selection-resnet-data-path'):
    assert int(root.get_attribute('data-reads'))==(4 if i>=6 else 2 if 1<=i<5 else 0)
   root.screenshot(path=str(shots/f'{args.engine}-{width}-{theme}-{i}.png'))
  root.focus();page.keyboard.press('p');assert root.get_attribute('data-step')==str(count-2);page.keyboard.press('n');assert root.get_attribute('data-step')==str(count-1);page.keyboard.press('r');assert root.get_attribute('data-step')=='0';assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'chapter overflow'
  root.get_by_role('button',name='Expand',exact=True).click();assert page.locator('dialog[open]').count()==1;page.screenshot(path=str(shots/f'{args.engine}-{width}-{theme}-expanded.png'));page.keyboard.press('Escape');assert page.locator('dialog[open]').count()==0
  if width==1440 and theme=='light':
   root.get_by_role('button',name='Expand',exact=True).click();page.emulate_media(reduced_motion='no-preference');root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(250);root.get_by_role('button',name='Pause',exact=True).click();before=root.locator('.mlsys-original-stage').inner_html();narr=root.locator('.mlsys-original-narration').inner_text();page.wait_for_timeout(250);assert before==root.locator('.mlsys-original-stage').inner_html();assert narr==root.locator('.mlsys-original-narration').inner_text();page.screenshot(path=str(shots/'paused-motion.png'));root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function("window.fixtureRoot.dataset.step==='1'",timeout=12000);root.get_by_role('button',name='Pause',exact=True).click();page.keyboard.press('Escape')
  page.get_by_role('tab',name='Diagram',exact=True).click();assert not root.is_visible();page.screenshot(path=str(shots/f'{args.engine}-{width}-{theme}-static.png'));page.get_by_role('tab',name='Explore',exact=True).click();page.emulate_media(media='print');assert not root.is_visible();assert not errors,errors;page.close();print(NAME,width,theme,'PASS',flush=True)
 # JavaScript disabled must preserve original diagram.
 page=browser.new_page(java_script_enabled=False);page.goto(fixture['url']);assert page.locator('#'+json.loads(entry.read_text())['key'].split(':')[1]+' img').count()>0;page.close();browser.close()
for suffix in ['js','css']:print(suffix,hashlib.sha256((EXT/(NAME+'.'+suffix)).read_bytes()).hexdigest())
