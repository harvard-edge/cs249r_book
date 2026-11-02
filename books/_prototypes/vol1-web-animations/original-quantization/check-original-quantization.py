"""Focused original-art lesson check in the existing, fully themed chapter render.

This review fixture mounts the lesson in browser memory; it does not modify a
rendered chapter on disk or stand in for the production Quarto build.
"""
import argparse
import importlib.util
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
import yaml

REPO=Path(__file__).resolve().parents[4]
ASSETS=REPO/'books/_extensions/mlsysbook-ext/figure-explore'
parser=argparse.ArgumentParser()
parser.add_argument('--engine',default='chromium',choices=['chromium','webkit'])
parser.add_argument('--screenshots',default='/tmp/original-quantization-review')
args=parser.parse_args()
out=Path(args.screenshots);out.mkdir(parents=True,exist_ok=True)
entry=json.loads(Path('/tmp/figures-animation-art-current/index.json').read_text())['vol1:fig-weight-activations-quantization']
timing=yaml.safe_load((REPO/'books/config/shared/html/figure-explore.yml').read_text())['figure-explore']
spec=importlib.util.spec_from_file_location('quant_calcs',ASSETS/'original-quantization.calc.py')
calc=importlib.util.module_from_spec(spec);spec.loader.exec_module(calc)
expected=calc.calculate()
with sync_playwright() as p:
 browser=getattr(p,args.engine).launch()
 for width,theme in [(1440,'light'),(1440,'dark'),(390,'light'),(390,'dark')]:
  page=browser.new_page(viewport={'width':width,'height':1800 if width>700 else 1100},color_scheme=theme)
  page.goto(Path(entry['html']).as_uri(),wait_until='load')
  page.evaluate("""theme=>{document.body.classList.toggle('quarto-dark',theme==='dark');document.body.classList.toggle('quarto-light',theme==='light');document.documentElement.setAttribute('data-bs-theme',theme);document.querySelectorAll('link.quarto-color-scheme').forEach(l=>{l.disabled=theme==='dark'? !l.classList.contains('quarto-color-alternate'):l.classList.contains('quarto-color-alternate')});}""",theme)
  for name in ['original-player.css','original-quantization.css']:page.add_style_tag(path=str(ASSETS/name))
  for name in ['original-player.js','original-quantization.data.js','original-quantization.js']:page.add_script_tag(path=str(ASSETS/name))
  assert page.evaluate('MLSysOriginalQuantizationData')==expected, 'regenerate the exported numerical data'
  page.evaluate("""async config=>{const outer=document.getElementById('fig-weight-activations-quantization');const caption=outer.querySelector('figcaption')?.cloneNode(true);outer.replaceChildren();const root=document.createElement('div');outer.append(root);if(caption)outer.append(caption);outer.dataset.mlsysMovementMs=String(config.movement);outer.dataset.mlsysReadingPauseMs=String(config.reading);const art=new DOMParser().parseFromString(config.svg,'image/svg+xml').documentElement;window.quantRoot=root;window.quantPlayer=await MLSysFigureLessons['original-quantization'](root,{outer,verifiedArtwork:document.importNode(art,true)});outer.scrollIntoView({block:'center'});}""",{'svg':Path(entry['asset']).read_text(),'movement':timing['movement-ms'],'reading':timing['reading-pause-ms']})
  page.wait_for_timeout(150)
  assert page.evaluate('quantPlayer.state.index')==0
  assert page.locator('.oq-overlay').count()==1
  assert page.locator('.mlsys-original-shortcuts').is_visible()
  for step in range(9):
   page.evaluate('(i)=>quantPlayer.go(i)',step)
   assert page.locator('[data-oq-box="INT32 sum"] .oq-value').text_content()==('empty' if step<4 else str(expected['products'][0] if step==4 else expected['accumulator']))
   assert page.locator('[data-oq-box="INT8 output"] .oq-value').text_content()==(str(expected['output']) if step==8 else '—')
   assert page.evaluate('quantRoot.scrollWidth <= quantRoot.clientWidth + 1'), page.evaluate('Array.from(quantRoot.querySelectorAll("*")).filter(e=>e.scrollWidth>e.clientWidth+1).map(e=>[e.className.baseVal||e.className,e.clientWidth,e.scrollWidth])')
   assert page.evaluate("""Array.from(quantRoot.querySelectorAll('[data-oq-box]')).every(g=>{const r=g.querySelector('rect').getBBox();return Array.from(g.querySelectorAll('text')).every(t=>{const b=t.getBBox();return b.x>=r.x-0.1&&b.x+b.width<=r.x+r.width+0.1&&b.y>=r.y&&b.y+b.height<=r.y+r.height;});})"""), 'embedded text escaped its original box'
   if step in [0,4,5,6,8]:
    page.locator('#fig-weight-activations-quantization').screenshot(path=str(out/f'{args.engine}-{width}-{theme}-step-{step}.png'))
  page.emulate_media(reduced_motion='reduce');page.evaluate('quantPlayer.restart();quantPlayer.next()');page.wait_for_timeout(100)
  assert page.evaluate('quantPlayer.state.index')==1
  page.emulate_media(reduced_motion='no-preference');page.evaluate('quantPlayer.restart();quantPlayer.next()');page.wait_for_timeout(200);page.evaluate('quantPlayer.pause()')
  a=page.locator('.oq-token').get_attribute('transform');page.wait_for_timeout(150);assert page.locator('.oq-token').get_attribute('transform')==a
  page.evaluate('quantPlayer.go(2);quantPlayer.next()');page.wait_for_timeout(200);page.evaluate('quantPlayer.pause()')
  assert page.locator('.oq-token').count()==2, 'both integer branches must flow into matmul'
  assert set(page.locator('.oq-token text').all_text_contents())=={'xq','wq'}
  page.locator('#fig-weight-activations-quantization').screenshot(path=str(out/f'{args.engine}-{width}-{theme}-paired-flow.png'))
  page.evaluate('quantPlayer.go(5);quantPlayer.back()');assert page.evaluate('quantPlayer.state.index')==4
  page.locator('.mlsys-original-sequence select').select_option('8');assert page.evaluate('quantPlayer.state.index')==8
  page.locator('.mlsys-original-player').focus();page.keyboard.press('r');assert page.evaluate('quantPlayer.state.index')==0
  page.emulate_media(media='print');assert not page.locator('.mlsys-original-player').is_visible();page.emulate_media(media='screen')
  page.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-chapter.png'))
  print(args.engine,width,theme,'PASS: original SVG, nine exact states, numeric endpoints, controls, reduced motion, pause, containment')
  page.close()
 browser.close()
