#!/usr/bin/env python3
"""Inspect original-art endpoint invariants and real shared-player interactions."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
from playwright.sync_api import sync_playwright
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];spec=json.loads((HERE/'lesson.json').read_text());name=spec['name'];entry=json.loads((HERE/'entry.json').read_text());art=json.loads(Path('/tmp/figures-animation-art-current/index.json').read_text())[entry['key']]
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--engine',choices=['chromium','webkit'],default='chromium');parser.add_argument('--screenshots',type=Path,default=Path('/tmp')/(name+'-review'));parser.add_argument('--url',default='http://127.0.0.1:8882/_build/'+name+'/'+str(Path(art['source']).relative_to('books')).replace('.qmd','.html')+'#'+entry['key'].split(':')[1]);args=parser.parse_args();args.screenshots.mkdir(exist_ok=True,parents=True)

def invariants(f,key):
 if key.endswith('event-sourcing'):
  assert f['log_intact'] and f['durable_reservation'] and f['durable_intent'];assert not f['model_reexecuted'] and not f['tool_reexecuted'] and not f['result_recorded']
  if f['projection_intent']:assert f['projection_reserved'] and f['authorized']
 if key.endswith('idempotency-flow'):
  assert f['effect_committed']==f['result_recorded'];assert f['operation']=='op_A';assert not f['duplicate_effect']
 if key.endswith('streaming-cancellation'):
  assert not f['dispatched'];
  if f['kv_released']:assert f['honored'] and f['remote_stopped']
  if not f['honored']:assert not f['kv_released']
 if key.endswith('checkpoint-consistent-cut'):
  assert f['c0_available'];
  if f['c1_published']:assert f['payload_durable'] and f['compatible']
  if f['restored']:assert f['pending_test'] and f['reservation_held'] and not f['tool_reexecuted']
 if key.endswith('saga-flow'):
  if f['provision_dispatched']:assert f['registered']
  if f['release_dispatched']:assert f['bound'] and f['test_stopped'] and f['artifacts_preserved']
  if f['closed']:assert f['release_confirmed']
  if not f['release_confirmed'] and f['registered']:assert f['owned']
 if key.endswith('occ-worktrees'):
  if f['published']:assert f['checked_candidate']==f['candidate'] and f['expected_head']==f['target_head'] and f['guard']
  if f['head_moved']:assert not f['published']
 if key.endswith('access-prefilter'):
  assert not f['masked_traversed'];assert f['permission_scope_unchanged']

FOCUS=[['intent'], ['attempt', 'request'], ['effect', 'result'], ['timeout'], ['attempt', 'request'], ['retained'], ['attempt', 'request'], ['conflict']]

reports=[]
with sync_playwright() as p:
 browser=getattr(p,args.engine).launch()
 for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
  page=browser.new_page(viewport={'width':width,'height':2000},color_scheme=theme);errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto(args.url)
  if theme=='dark':page.evaluate("document.body.classList.add('quarto-dark');document.body.classList.remove('quarto-light')")
  page.wait_for_function('window.wavePlayer!==undefined');root=page.locator('.'+name);menu=root.locator('.mlsys-original-sequence select');assert root.get_attribute('data-original-step')=='0';assert root.get_attribute('data-playing')=='false'
  for variant in spec.get('variants') or [{'id':'','states':spec['states']}]:
   if spec.get('variants'):root.get_by_label(spec['variantLabel'],exact=True).select_option(variant['id'])
   frames=variant['states'];assert all(len(frame['description'])>=50 for frame in frames),'Narration must be a complete explanation'
   for i,frame in enumerate(frames):
    menu.select_option(str(i));actual=json.loads(root.get_attribute('data-original-facts'));assert actual==frame['facts'];invariants(actual,entry['key']);assert root.locator('.v3-original-readout strong').all_text_contents()==frame['values']
    for k,v in dict(spec.get('defaults',{}),**frame.get('text',{})).items():
     b=spec['slots'][k];n=root.locator(f'svg text[x="{b["x"]}"][y="{b["y"]}"]');assert n.text_content()==str(v)
    # Inspect the painted active-object text inside its local pan viewport.
    targets=FOCUS[i]
    if entry['key'].endswith('checkpoint-consistent-cut') and variant['id']=='torn':targets=[['previous','fallback'],['tornrecord'],['tornwork','tornstatus'],['reject'],['reject'],['previous','fallback'],['suffix'],['pending']][i]
    if entry['key'].endswith('access-prefilter') and variant['id']=='exact' and i>=3:targets=['score',[410,334]]
    for target in targets:
     binding=spec['slots'][target] if isinstance(target,str) else dict(x=target[0],y=target[1])
     visible=root.locator('.v3-original-canvas').evaluate("(canvas,b)=>{const svg=canvas.querySelector('svg');const n=[...svg.querySelectorAll('text')].find(n=>+n.getAttribute('x')===b.x&&+n.getAttribute('y')===b.y);const a=n.getBoundingClientRect(),c=canvas.getBoundingClientRect();return {ok:a.left>=c.left-1&&a.right<=c.left+canvas.clientWidth+1&&a.top>=c.top-1&&a.bottom<=c.top+canvas.clientHeight+1,text:n.textContent,paint:[a.left-c.left,a.right-c.left,a.top-c.top,a.bottom-c.top],viewport:[canvas.clientWidth,canvas.clientHeight]}}",binding)
     assert visible['ok'],(variant['id'],i,target,visible)
    root.locator('.v3-original-canvas').screenshot(path=str(args.screenshots/f'{variant["id"] or "default"}-state-{i}-{width}-{theme}.png'))
   # A visible packet freezes along with the shared clock; destination facts remain uncommitted.
   target=next((i for i,s in enumerate(frames) if i and s.get('motion')),None)
   if target is not None:
    menu.select_option(str(target-1));root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(160);root.get_by_role('button',name='Pause',exact=True).click();assert root.get_attribute('data-original-step')==str(target-1);packet=root.locator('.v3-original-packet');assert packet.get_attribute('visibility')=='visible';position=packet.get_attribute('transform');page.wait_for_timeout(120);assert packet.get_attribute('transform')==position
    root.screenshot(path=str(args.screenshots/f'{variant["id"] or "default"}-paused-{width}-{theme}.png'));root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function('(index)=>document.querySelector(".'+name+'").dataset.originalStep===String(index)',arg=target);page.evaluate('window.wavePlayer.pause()')
   # Pause every causal action on the narrow dark profile, before endpoint commit.
   if width==390 and theme=='dark':
    for destination in range(1,len(frames)):
     menu.select_option(str(destination-1));root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(100);root.get_by_role('button',name='Pause',exact=True).click()
     assert root.get_attribute('data-original-step')==str(destination-1)
     assert json.loads(root.get_attribute('data-original-facts'))==frames[destination-1]['facts']
     assert frames[destination]['description'] in root.locator('.mlsys-original-narration').inner_text()
     packet=root.locator('.v3-original-packet');position=packet.get_attribute('transform');page.wait_for_timeout(60);assert packet.get_attribute('transform')==position
     root.screenshot(path=str(args.screenshots/f'{variant["id"] or "default"}-paused-action-{destination}-{width}-{theme}.png'))
     menu.select_option(str(destination))
   root.focus();page.keyboard.press('r');assert root.get_attribute('data-original-step')=='0';page.emulate_media(reduced_motion='reduce');root.focus();page.keyboard.press('n');page.wait_for_function('document.querySelector(".'+name+'").dataset.originalStep==="1"');assert root.get_attribute('data-playing')=='false';root.focus();page.keyboard.press('p');assert root.get_attribute('data-original-step')=='0';page.emulate_media(reduced_motion='no-preference')
   menu.select_option(str(len(frames)-1));root.get_by_role('button',name='Expand',exact=True).click();assert root.get_attribute('data-original-step')==str(len(frames)-1);page.keyboard.press('Escape');page.wait_for_function("document.activeElement?.getAttribute('aria-label')==='Expand' || document.activeElement?.textContent==='Expand'");assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
  page.evaluate('window.scrollTo(0,document.querySelector(".'+name+'").getBoundingClientRect().top+scrollY-230)');page.screenshot(path=str(args.screenshots/f'chapter-{width}-{theme}.png'));page.emulate_media(media='print');assert not root.is_visible();assert page.locator('#'+entry['key'].split(':')[1]+' img[data-wave-static]').is_visible();assert not errors,errors;reports.append({'width':width,'theme':theme,'all_states':'passed','invariants':'passed','controls':'passed'});page.close()
 page=browser.new_page(java_script_enabled=False);page.goto(args.url);assert page.locator('#'+entry['key'].split(':')[1]+' img[data-wave-static]').is_visible();assert page.locator('.'+name).count()==0;page.close();browser.close()
print(json.dumps({'key':entry['key'],'engine':args.engine,'profiles':reports},indent=2))
