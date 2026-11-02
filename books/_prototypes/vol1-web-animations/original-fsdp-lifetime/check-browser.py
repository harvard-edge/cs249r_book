#!/usr/bin/env python3
"""Review the original lifetime SVG in its actual rendered chapter."""
import argparse,json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[4];EXT=ROOT/'books/_extensions/mlsysbook-ext/figure-explore'
entry=json.loads(Path('/tmp/figures-animation-art-current/index.json').read_text())['vol2:fig-fsdp-parameter-lifetime']
a=argparse.ArgumentParser();a.add_argument('--engine',choices=['chromium','webkit'],default='chromium');a.add_argument('--screenshots',default='/tmp/original-fsdp-review');args=a.parse_args()
out=Path(args.screenshots);out.mkdir(parents=True,exist_ok=True)
# Independent declared expected current state: (full parameters, saved acts, gradient).
expected=[('future','future','future'),('live','empty','empty'),('live','live','empty'),('empty','live','empty'),('live','live','empty'),('live','live','full'),('empty','empty','full'),('empty','empty','shard')]
with sync_playwright() as p:
 browser=getattr(p,args.engine).launch()
 for width in [1440,390]:
  for theme in ['light','dark']:
   page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme,reduced_motion='reduce')
   page.route('**/*.js',lambda route:route.abort());page.goto(Path(entry['html']).as_uri(),wait_until='load')
   page.evaluate("""theme=>{document.documentElement.setAttribute('data-bs-theme',theme);document.body.classList.toggle('quarto-dark',theme==='dark');document.body.classList.toggle('quarto-light',theme==='light');for(const link of document.querySelectorAll('link.quarto-color-scheme'))link.disabled=link.classList.contains('quarto-color-alternate')?theme!=='dark':theme==='dark';for(const link of document.querySelectorAll('link.quarto-color-scheme-extra'))link.disabled=true;}""",theme)
   baseline=page.evaluate('document.documentElement.scrollWidth')
   for script in ['original-player.js','original-fsdp-lifetime.js']:page.add_script_tag(content=(EXT/script).read_text())
   for style in ['original-player.css','original-fsdp-lifetime.css']:page.add_style_tag(content=(EXT/style).read_text())
   page.evaluate("""async artwork=>{const outer=document.querySelector('#fig-fsdp-parameter-lifetime');outer.dataset.mlsysMovementMs='800';outer.dataset.mlsysReadingPauseMs='120';const root=document.createElement('section');root.id='fsdp-test-root';for(const old of outer.querySelectorAll('.mlsys-figure-explore,.mlsys-figure-tabs,.mlsys-figure-tablist'))old.remove();const image=outer.querySelector('img');if(image)image.style.display='none';outer.prepend(root);const svg=new DOMParser().parseFromString(artwork,'image/svg+xml').documentElement;window.fsdpLesson=await window.MLSysFigureLessons['original-fsdp-lifetime'](root,{outer,verifiedArtwork:svg});}""",Path(entry['asset']).read_text())
   root=page.locator('#fsdp-test-root');root.scroll_into_view_if_needed()
   assert root.locator('svg').count()==1;assert root.locator('select option').count()==8
   def check_state(index):
    column=min(5,max(0,index-1))
    expected_live=['Parameter shard','Optimizer shard','Full parameters' if index in [1,2,4,5] else '', 'Saved activations' if index in [2,3,4,5] else '', 'Full gradient' if index in [5,6] else 'Reduced shard' if index==7 else '', '']
    for row,value in enumerate(expected_live):
     slot=root.locator(f'[data-live-row="{row}"]');obj=slot.locator('.original-fsdp-live-object');assert slot.get_attribute('data-resident')==str(bool(value)).lower();assert obj.text_content()==value;assert obj.is_visible()==bool(value)
    for c in range(6):
     for row in [0,1]:
      cell=root.locator(f'g[data-fsdp-column="{c}"][data-fsdp-row="{row}"]');assert cell.get_attribute('data-allocation-state')=='present';assert cell.evaluate('(el)=>+el.style.opacity')==1
    for row,state in zip([2,3,4],expected[index]):
     cell=root.locator(f'g[data-fsdp-column="{column}"][data-fsdp-row="{row}"]');assert cell.get_attribute('data-allocation-state')==state,(index,row,state)
     if width<700:
      assert cell.evaluate('(el)=>{const a=el.getBoundingClientRect(),b=el.closest(".original-fsdp-stage").getBoundingClientRect();return a.left>=b.left+160&&a.right<=b.right}'),(index,'activephaseoffscreen')
    for c in [1,4]:
     optional=root.locator(f'g[data-fsdp-column="{c}"][data-fsdp-row="5"]');assert optional.get_attribute('data-allocation-state')=='optional-not-executed';assert optional.locator('rect[stroke-dasharray]').count()==1
    if index in [6,7]:assert root.locator('g[data-fsdp-column="5"][data-fsdp-row="4"] text').text_content()==('full' if index==6 else 'shard')
   check_state(0);root.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-step0.png'))
   for index in range(1,8):
    root.get_by_role('button',name='Next',exact=True).click();page.wait_for_function("i=>document.querySelector('#fsdp-test-root').dataset.step===String(i)",arg=index);check_state(index);assert root.get_attribute('data-playing')=='false'
    root.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-step{index}.png'))
   assert page.evaluate('document.documentElement.scrollWidth')<=baseline
   # Direct jumps and backwards reconstruct the original source state exactly.
   for index in [3,6,2,7,0]:root.locator('select').select_option(str(index));check_state(index)
   root.focus();page.keyboard.press('n');page.wait_for_function("document.querySelector('#fsdp-test-root').dataset.step==='1'");page.keyboard.press('p');assert root.get_attribute('data-step')=='0'
   page.emulate_media(reduced_motion='no-preference');root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(150)
   assert root.get_attribute('data-step')=='0'
   cell=root.locator('g[data-fsdp-column="0"][data-fsdp-row="2"]');opacity=cell.evaluate('(el)=>+el.style.opacity');assert .12<opacity<1
   entering=root.locator('[data-live-row="2"] .original-fsdp-live-object');assert 0<entering.evaluate('(el)=>+el.style.opacity')<1
   root.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-moving.png'))
   page.evaluate('window.fsdpLesson.pause()');frozen=cell.evaluate('(el)=>+el.style.opacity');page.wait_for_timeout(100);assert cell.evaluate('(el)=>+el.style.opacity')==frozen
   live_frozen=entering.evaluate('(el)=>+el.style.opacity');page.wait_for_timeout(60);assert entering.evaluate('(el)=>+el.style.opacity')==live_frozen
   root.focus();page.keyboard.press('k');page.wait_for_timeout(60);root.get_by_role('button',name='Next',exact=True).click();check_state(1);assert root.get_attribute('data-playing')=='false'
   page.emulate_media(reduced_motion='reduce');root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function("document.querySelector('#fsdp-test-root').dataset.step==='7'",timeout=6000);check_state(7)
   root.get_by_role('button',name='Expand',exact=True).click();assert page.locator('dialog[open]').count()==1;page.wait_for_timeout(75);check_state(7);root.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-expanded.png'));page.keyboard.press('Escape');assert page.locator('dialog[open]').count()==0;page.wait_for_timeout(75);check_state(7)
   if width<700:
    root.locator('.original-fsdp-stage').evaluate('(el)=>el.scrollLeft=0');page.wait_for_timeout(100);assert root.locator('.original-fsdp-stage').evaluate('(el)=>el.scrollLeft')==0
    root.locator('select').select_option('7');check_state(7)
   page.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-chapter.png'));print('PASS',args.engine,width,theme);page.close()
 browser.close()
