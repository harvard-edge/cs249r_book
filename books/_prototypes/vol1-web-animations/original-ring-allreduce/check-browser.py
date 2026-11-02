#!/usr/bin/env python3
"""Mount the new factory in the actual rendered chapter; verify original cells."""
import argparse, json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[4]
EXT=ROOT/'books/_extensions/mlsysbook-ext/figure-explore'
entry=json.loads(Path('/tmp/figures-animation-art-current/index.json').read_text())['vol2:fig-ring-allreduce']
parser=argparse.ArgumentParser(); parser.add_argument('--engine',default='chromium',choices=['chromium','webkit']); parser.add_argument('--screenshots',default='/tmp/original-ring-review'); args=parser.parse_args()
out=Path(args.screenshots);out.mkdir(exist_ok=True,parents=True)
with sync_playwright() as p:
 browser=getattr(p,args.engine).launch()
 for width in [1440,390]:
  for theme in ['light','dark']:
   page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme,reduced_motion='reduce')
   page.route('**/*.js',lambda route:route.abort())
   page.goto(Path(entry['html']).as_uri(),wait_until='load')
   page.evaluate("""theme=>{document.documentElement.setAttribute('data-bs-theme',theme);document.body.classList.toggle('quarto-dark',theme==='dark');document.body.classList.toggle('quarto-light',theme==='light');for(const link of document.querySelectorAll('link.quarto-color-scheme'))link.disabled=link.classList.contains('quarto-color-alternate')?theme!=='dark':theme==='dark';for(const link of document.querySelectorAll('link.quarto-color-scheme-extra'))link.disabled=true;}""",theme)
   baseline_width=page.evaluate('document.documentElement.scrollWidth')
   for script in ['original-player.js','original-ring-allreduce.js']:page.add_script_tag(content=(EXT/script).read_text())
   for style in ['original-player.css','original-ring-allreduce.css']:page.add_style_tag(content=(EXT/style).read_text())
   page.evaluate("""async artwork=>{const outer=document.querySelector('#fig-ring-allreduce');outer.dataset.mlsysMovementMs='800';outer.dataset.mlsysReadingPauseMs='250';const root=document.createElement('section');root.id='ring-test-root';for(const old of outer.querySelectorAll('.mlsys-figure-explore,.mlsys-figure-tabs,.mlsys-figure-tablist'))old.remove();const image=outer.querySelector('img');if(image)image.style.display='none';outer.prepend(root);const svg=new DOMParser().parseFromString(artwork,'image/svg+xml').documentElement;window.ringLesson=await window.MLSysFigureLessons['original-ring-allreduce'](root,{outer,verifiedArtwork:svg});}""",Path(entry['asset']).read_text())
   root=page.locator('#ring-test-root'); root.scroll_into_view_if_needed()
   assert page.locator('#ring-test-root svg').count()==1
   assert page.locator('#ring-test-root select option').count()==7
   # Local data starts at its source rank; no finished sums exist initially.
   for rank in range(4):
    for chunk in 'ABCD': assert page.locator(f'text[data-panel="0"][data-rank="{rank}"][data-chunk="{chunk}"]').get_attribute('data-contributors')==str(rank)
   root.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-start.png'))
   for index in range(1,7):
    root.get_by_role('button',name='Next',exact=True).click();page.wait_for_function("i=>document.querySelector('#ring-test-root').dataset.step===String(i)",arg=index)
    assert root.get_attribute('data-playing')=='false'
    if index<=3:
     got=page.locator(f'text[data-panel="0"][data-rank="{index}"][data-chunk="A"]').get_attribute('data-contributors')
     assert got==','.join(map(str,range(index+1)))
    if index>=3:
     expected_holders={3,*range(index-3)}
     for rank in range(4):
      got=page.locator(f'text[data-panel="1"][data-rank="{rank}"][data-chunk="A"]').get_attribute('data-contributors')
      assert got==('0,1,2,3' if rank in expected_holders else '')
    if width<700:
     panel=0 if index<=3 else 1;rank=index if index<=3 else index-4
     assert page.locator(f'text[data-panel="{panel}"][data-rank="{rank}"][data-chunk="A"]').evaluate('(el)=>{const a=el.getBoundingClientRect(),b=el.closest(".original-ring-stage").getBoundingClientRect();return a.left>=b.left&&a.right<=b.right}'), f"Active A offscreen at step {index}"
    if index in [3,4,6]:root.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-step{index}.png'))
   for rank in range(4):
    for chunk in 'ABCD':assert page.locator(f'text[data-panel="1"][data-rank="{rank}"][data-chunk="{chunk}"]').get_attribute('data-contributors')=='0,1,2,3'
   if width<700:
    root.locator('.original-ring-stage').evaluate('(el)=>el.scrollLeft=0');page.wait_for_timeout(100);assert root.locator('.original-ring-stage').evaluate('(el)=>el.scrollLeft')==0
    root.locator('select').select_option('6')
   assert page.evaluate('document.documentElement.scrollWidth')<=baseline_width, 'New chapter overflow'
   assert page.evaluate('document.querySelector("#ring-test-root").getBoundingClientRect().right<=window.innerWidth+1'), page.evaluate('({width:innerWidth,root:document.querySelector("#ring-test-root").getBoundingClientRect().toJSON()})')
   root.get_by_role('button',name='Back',exact=True).click();assert root.get_attribute('data-step')=='5'
   root.locator('select').select_option('2');assert root.get_attribute('data-step')=='2'
   root.focus();page.keyboard.press('r');assert root.get_attribute('data-step')=='0'
   page.emulate_media(reduced_motion='no-preference');root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(70)
   assert root.locator('.original-ring-moving g').count()==4
   if width<700:assert page.locator('text[data-panel="0"][data-rank="1"][data-chunk="A"]').evaluate('(el)=>{const a=el.getBoundingClientRect(),b=el.closest(".original-ring-stage").getBoundingClientRect();return a.left>=b.left&&a.right<=b.right}')
   root.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-moving.png'))
   page.evaluate('window.ringLesson.pause()');before=root.locator('.original-ring-moving g').first.get_attribute('transform');page.wait_for_timeout(100);assert root.locator('.original-ring-moving g').first.get_attribute('transform')==before
   root.focus();page.keyboard.press('k');page.wait_for_timeout(50);root.get_by_role('button',name='Next',exact=True).click();assert root.get_attribute('data-step')=='1';assert root.get_attribute('data-playing')=='false'
   page.emulate_media(reduced_motion='reduce');root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function("document.querySelector('#ring-test-root').dataset.step==='6'",timeout=6000)
   assert root.get_attribute('data-playing')=='false'
   root.get_by_role('button',name='Expand',exact=True).click();assert page.locator('dialog[open]').count()==1;assert root.get_attribute('data-step')=='6'
   root.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-expanded.png'))
   page.keyboard.press('Escape');assert page.locator('dialog[open]').count()==0;assert root.get_attribute('data-step')=='6'
   if width<700:
    root.locator('.original-ring-stage').evaluate('(el)=>el.scrollLeft=el.scrollWidth');root.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-scrolled.png'))
   page.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-chapter.png'))
   print('PASS',args.engine,width,theme);page.close()
 browser.close()
