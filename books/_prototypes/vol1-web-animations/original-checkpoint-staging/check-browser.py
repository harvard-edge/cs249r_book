#!/usr/bin/env python3
"""Verify the source four-lane timeline with immutable checkpoint identity."""
import argparse,hashlib,json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[4];EXT=ROOT/'books/_extensions/mlsysbook-ext/figure-explore'
entry=json.loads(Path('/tmp/figures-animation-art-current/index.json').read_text())['vol2:fig-storage-checkpoint-staging']
a=argparse.ArgumentParser();a.add_argument('--engine',choices=['chromium','webkit'],default='chromium');a.add_argument('--screenshots',default='/tmp/original-checkpoint-review');args=a.parse_args()
out=Path(args.screenshots);out.mkdir(parents=True,exist_ok=True)
asset=Path(entry['asset']);canonical=hashlib.sha1(asset.read_bytes()).hexdigest()
with sync_playwright() as p:
 browser=getattr(p,args.engine).launch()
 for width in [1440,390]:
  for theme in ['light','dark']:
   page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme,reduced_motion='reduce')
   page.route('**/*.js',lambda route:route.abort());page.goto(Path(entry['html']).as_uri(),wait_until='load')
   page.evaluate("""theme=>{document.documentElement.setAttribute('data-bs-theme',theme);document.body.classList.toggle('quarto-dark',theme==='dark');document.body.classList.toggle('quarto-light',theme==='light');for(const link of document.querySelectorAll('link.quarto-color-scheme'))link.disabled=link.classList.contains('quarto-color-alternate')?theme!=='dark':theme==='dark';for(const link of document.querySelectorAll('link.quarto-color-scheme-extra'))link.disabled=true;}""",theme)
   baseline=page.evaluate('document.documentElement.scrollWidth')
   for script in ['original-player.js','original-checkpoint-staging.js']:page.add_script_tag(content=(EXT/script).read_text())
   for style in ['original-player.css','original-checkpoint-staging.css']:page.add_style_tag(content=(EXT/style).read_text())
   page.add_style_tag(content='@media screen{.checkpoint-static-fixture{display:none!important}}')
   page.evaluate("""async artwork=>{const outer=document.querySelector('#fig-storage-checkpoint-staging');outer.dataset.mlsysMovementMs='800';outer.dataset.mlsysReadingPauseMs='120';const root=document.createElement('section');root.id='checkpoint-test-root';for(const old of outer.querySelectorAll('.mlsys-figure-explore,.mlsys-figure-tabs,.mlsys-figure-tablist'))old.remove();const image=outer.querySelector('img');if(image)image.classList.add('checkpoint-static-fixture');outer.prepend(root);const svg=new DOMParser().parseFromString(artwork,'image/svg+xml').documentElement;window.checkpointLesson=await window.MLSysFigureLessons['original-checkpoint-staging'](root,{outer,verifiedArtwork:svg});}""",asset.read_text())
   root=page.locator('#checkpoint-test-root');root.scroll_into_view_if_needed()
   assert root.locator('select option').count()==9;assert root.locator('[data-staging-group]').count()==11
   def check(index):
    assert root.locator('.original-checkpoint-restore').get_attribute('data-restore-generation')==('g1' if index>=7 else 'g0')
    live=root.locator('[data-staging-group="live"]');assert live.get_attribute('data-live-step')==('i+1' if index>=2 else 'i');assert live.get_attribute('data-running')==str(index>=2).lower()
    for rank in ['A','B']:
     card=root.locator(f'[data-rank-status="{rank}"]');assert card.get_attribute('data-snapshot')=='i';assert card.locator('.original-checkpoint-rank-receipt').get_attribute('data-durable')==str(index>=(4 if rank=='A' else 5)).lower();assert card.locator('.original-checkpoint-rank-checks').get_attribute('data-verified')==str(index>=6).lower()
     if width<700:assert card.evaluate('(el)=>{const a=el.getBoundingClientRect();return a.left>=0&&a.right<=innerWidth}'), 'Both rank cards must fit'
     local=root.locator(f'[data-staging-kind="local"][data-staging-shard="{rank}"]');assert local.get_attribute('data-resident')==str(1<=index<8).lower();assert local.get_attribute('data-snapshot')=='i';assert 'i+1' not in local.text_content();assert local.is_visible()==(1<=index<8)
     durable=root.locator(f'[data-staging-kind="durable"][data-staging-shard="{rank}"]');assert durable.get_attribute('data-resident')==str(index>=5).lower();assert durable.is_visible()==(index>=5);assert durable.get_attribute('data-snapshot')=='i'
     receipt=root.locator(f'[data-staging-kind="receipt"][data-staging-shard="{rank}"]');assert receipt.get_attribute('data-durable')==str(index>=(4 if rank=='A' else 5)).lower()
    assert root.locator('[data-staging-group="verify"]').get_attribute('data-verified')==str(index>=6).lower();assert root.locator('[data-staging-group="publish"]').get_attribute('data-committed')==str(index>=7).lower()
    if index==4:assert 'A: durable' in root.locator('[data-staging-kind="receipt"][data-staging-shard="A"]').text_content();assert 'B: pending' in root.locator('[data-staging-kind="receipt"][data-staging-shard="B"]').text_content()
    if width<700:
     selector='[data-staging-group="pause"]' if index==0 else '[data-staging-kind="local"][data-staging-shard="A"]' if index in [1,2] else '[data-staging-kind="receipt"][data-staging-shard="A"]' if index in [3,4] else '[data-staging-kind="durable"][data-staging-shard="A"]' if index==5 else '[data-staging-group="verify"]' if index==6 else '[data-staging-group="publish"]'
     assert root.locator(selector).evaluate('(el)=>{const a=el.getBoundingClientRect(),b=el.closest(".original-checkpoint-stage").getBoundingClientRect();return a.left>=b.left+125&&a.right<=b.right}'),(index,'active stage offscreen')
     if index>=2:assert root.locator('[data-live-badge]').evaluate('(el)=>{const a=el.getBoundingClientRect(),b=el.closest(".original-checkpoint-stage").getBoundingClientRect();return a.left>=b.left+125&&a.right<=b.right}')
   check(0);root.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-step0.png'))
   for index in range(1,9):
    root.get_by_role('button',name='Next',exact=True).click();page.wait_for_function("i=>document.querySelector('#checkpoint-test-root').dataset.step===String(i)",arg=index);check(index);assert root.get_attribute('data-playing')=='false';root.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-step{index}.png'))
   assert page.evaluate('document.documentElement.scrollWidth')<=baseline
   for index in [4,2,7,1,6,0]:root.locator('select').select_option(str(index));check(index)
   root.focus();page.keyboard.press('n');page.wait_for_function("document.querySelector('#checkpoint-test-root').dataset.step==='1'");page.keyboard.press('p');check(0)
   # True interpolation freezes; no local write is committed mid-action.
   page.emulate_media(reduced_motion='no-preference');root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(150)
   shard=root.locator('[data-staging-kind="local"][data-staging-shard="A"]');assert 0<shard.evaluate('(el)=>+el.style.opacity')<1;assert shard.get_attribute('data-resident')=='false'
   page.evaluate('window.checkpointLesson.pause()');frozen=shard.evaluate('(el)=>+el.style.opacity');page.wait_for_timeout(100);assert shard.evaluate('(el)=>+el.style.opacity')==frozen
   root.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-moving.png'));root.focus();page.keyboard.press('k');page.wait_for_timeout(50);root.get_by_role('button',name='Next',exact=True).click();check(1)
   # Copy movement, per-shard acknowledgments, verification and publication
   # must visibly progress while committed endpoints remain unchanged.
   root.locator('select').select_option('2');root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(150);assert root.locator('[data-copy-packets] > g').count()==1;root.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-copy-moving.png'));root.get_by_role('button',name='Next',exact=True).click();check(3)
   for start,rank in [(3,'A'),(4,'B')]:
    root.locator('select').select_option(str(start));root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(150)
    receipt=root.locator(f'[data-staging-kind="receipt"][data-staging-shard="{rank}"]');assert receipt.get_attribute('data-durable')=='false';assert 'pending' in receipt.text_content();assert root.locator('.original-checkpoint-restore').get_attribute('data-restore-generation')=='g0'
    bar=root.locator(f'[data-rank-status="{rank}"] .original-checkpoint-receipt-progress');assert 0<float(bar.evaluate('(el)=>parseFloat(el.style.width)'))<100
    root.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-write{rank}-moving.png'));page.evaluate('window.checkpointLesson.pause()');frozenbar=bar.evaluate('(el)=>el.style.width');page.wait_for_timeout(70);assert bar.evaluate('(el)=>el.style.width')==frozenbar;root.get_by_role('button',name='Next',exact=True).click();check(start+1)
   root.locator('select').select_option('5');root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(150);assert root.locator('[data-staging-group="verify"]').get_attribute('data-verified')=='false';assert all('checking' in text for text in root.locator('.original-checkpoint-rank-checks').all_text_contents());root.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-verify-moving.png'));root.get_by_role('button',name='Next',exact=True).click();check(6)
   root.locator('select').select_option('6');root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(150);assert root.locator('.original-checkpoint-restore').get_attribute('data-restore-generation')=='g0';assert root.locator('[data-staging-group="publish"]').get_attribute('data-committed')=='false';assert 'A@i' in root.locator('[data-staging-group="publish"]').text_content();root.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-manifest-moving.png'));root.get_by_role('button',name='Next',exact=True).click();check(7)
   root.locator('select').select_option('1');page.emulate_media(reduced_motion='reduce');root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function("document.querySelector('#checkpoint-test-root').dataset.step==='8'",timeout=6000);check(8)
   root.get_by_role('button',name='Expand',exact=True).click();page.wait_for_timeout(75);assert page.locator('dialog[open]').count()==1;check(8);page.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-expanded.png'));page.keyboard.press('Escape');page.wait_for_timeout(75);assert page.locator('dialog[open]').count()==0;check(8)
   if width<700:root.locator('.original-checkpoint-stage').evaluate('(el)=>el.scrollLeft=0');page.wait_for_timeout(75);assert root.locator('.original-checkpoint-stage').evaluate('(el)=>el.scrollLeft')==0
   root.locator('select').select_option('8');page.screenshot(path=str(out/f'{args.engine}-{width}-{theme}-chapter.png'))
   page.emulate_media(media='print');assert not root.is_visible();assert page.locator('#fig-storage-checkpoint-staging img').first.is_visible()
   print('PASS',args.engine,width,theme);page.close()
 # No-JavaScript uses the chapter's untouched complete static figure.
 nojs=browser.new_page(java_script_enabled=False);nojs.goto(Path(entry['html']).as_uri(),wait_until='load');assert nojs.locator('#fig-storage-checkpoint-staging img').first.is_visible();nojs.close();browser.close()
assert hashlib.sha1(asset.read_bytes()).hexdigest()==canonical
