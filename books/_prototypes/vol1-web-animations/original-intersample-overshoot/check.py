"""Render the original capture-buffer player in a complete existing Volume IV chapter.

The disposable HTML fixture retains the chapter theme, prose, caption, and art.
Production enrollment and full-volume builds are checked separately.
"""
import argparse,json,re
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--engine',choices=['chromium','webkit'],default='chromium')
parser.add_argument('--art-index',type=Path,default=Path('/tmp/figures-animation-art-current/index.json'))
parser.add_argument('--screenshots',type=Path,default=Path('/tmp/original-intersample-overshoot-review'))
a=parser.parse_args();a.screenshots.mkdir(parents=True,exist_ok=True)
repo=Path(__file__).resolve().parents[4]
ext=repo/'books/_extensions/mlsysbook-ext/figure-explore'
entry=json.loads(a.art_index.read_text())['vol4:fig-11-intersample-overshoot']
chapter=Path(entry['html']); soup=BeautifulSoup(chapter.read_text(),'html.parser')
for script in soup.find_all('script'):script.decompose()
base=soup.new_tag('base',href=chapter.parent.as_uri()+'/');soup.head.insert(0,base)
for name in ['original-player','original-intersample-overshoot']:
 soup.head.append(soup.new_tag('link',rel='stylesheet',href=(ext/f'{name}.css').as_uri()))
 if name=='original-intersample-overshoot':soup.body.append(soup.new_tag('script',src=(ext/'original-intersample-overshoot-data.js').as_uri()))
 soup.body.append(soup.new_tag('script',src=(ext/f'{name}.js').as_uri()))
outer=soup.select_one('#fig-11-intersample-overshoot');figure=outer.find('figure')
for key in list(outer.attrs):
 if key.startswith('data-'):del outer[key]
# Explicit test settings, not runtime defaults. The production caller gets YAML metadata.
outer['data-mlsys-movement-ms']='500';outer['data-mlsys-reading-pause-ms']='200'
static=figure.find('div',attrs={'aria-describedby':True});static['hidden']=''
print_style=soup.new_tag('style');print_style.string='@media print { #fig-11-intersample-overshoot figure > div[aria-describedby] { display:block!important; } }';soup.head.append(print_style)
root=soup.new_tag('section',id='intersample-review-player');figure.insert(0,root)
script=soup.new_tag('script');script.string='''(async()=>{const root=document.querySelector('#intersample-review-player'),outer=document.querySelector('#fig-11-intersample-overshoot');const art=new DOMParser().parseFromString('''+json.dumps(Path(entry['asset']).read_text())+''','image/svg+xml').documentElement;const originalCreate=window.MLSysOriginalPlayer.create;window.MLSysOriginalPlayer={create:options=>{const transition=options.transition;options.transition=ctx=>transition({...ctx,onFrame(callback){window.intersampleTestFrame=callback;ctx.onFrame(callback)}});return originalCreate(options)}};window.intersamplePlayer=await window.MLSysFigureLessons['original-intersample-overshoot'](root,{outer,verifiedArtwork:art});})().catch(e=>{window.fixtureError=String(e);console.error(e)});''';soup.body.append(script)
fixture=a.screenshots/'chapter.html';fixture.write_text(str(soup))
reports=[]
with sync_playwright() as p:
 browser=getattr(p,a.engine).launch()
 for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
  page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme)
  page.goto(fixture.as_uri());page.wait_for_function('window.intersamplePlayer || window.fixtureError');assert not page.evaluate('window.fixtureError'),page.evaluate('window.fixtureError')
  page.evaluate("theme=>{document.documentElement.setAttribute('data-bs-theme',theme);document.body.classList.toggle('quarto-dark',theme==='dark');document.body.classList.toggle('quarto-light',theme==='light');for(const l of document.querySelectorAll('link.quarto-color-scheme')){l.disabled=l.classList.contains('quarto-color-alternate')?theme!=='dark':theme==='dark';}}",theme)
  root=page.locator('#intersample-review-player');root.scroll_into_view_if_needed();static_path=page.locator('#fig-11-intersample-overshoot figure > div[aria-describedby]')
  assert root.locator('svg text').evaluate_all("xs=>xs.filter(n=>n.getAttribute('fill')==='#1A202C').every(n=>getComputedStyle(n).fill==='rgb(26, 32, 44)')")
  expected=[(0,2.5,False),(.25,3,False),(.5,3.5,False),(.75,3,False),(1,2.5,False),(0,2,True),(.5,3,True),(1,2,True)]
  for step,(time,velocity,compare) in enumerate(expected):
   page.evaluate('i=>window.intersamplePlayer.go(i)',step)
   snapshot=root.evaluate("""r=>({time:+r.dataset.time,velocity:+r.dataset.currentVelocity,rise:+r.dataset.rise,compare:r.dataset.compare,samples:r.dataset.endpointChecks,violation:r.dataset.observedOverspeed,positions:[...r.querySelectorAll('[data-intersample-marker]')].map(n=>[+n.getAttribute('cx'),+n.getAttribute('cy')]),sampleCenters:[...r.querySelectorAll('[data-sample-check]')].map(n=>[+n.getAttribute('cx'),+n.getAttribute('cy')])})""")
   assert snapshot['time']==time and snapshot['velocity']==velocity and snapshot['compare']==str(compare).lower(),snapshot
   assert snapshot['rise']==2*min(time,1-time),snapshot
   assert snapshot['samples']=='passed' and snapshot['sampleCenters']==[[140,330],[600,330],[140,360],[600,360]],snapshot
   assert snapshot['positions'][1 if compare else 0]==[140+460*time,480-60*velocity],snapshot
   assert snapshot['violation']==str(step>=2).lower(),snapshot
   assert root.locator('path[d="M140 70 H370 V180 H600"]').count()==2
   root.locator('.oio-canvas').screenshot(path=str(a.screenshots/f'diagram-{width}-{theme}-{step}.png'))
   root.screenshot(path=str(a.screenshots/f'player-{width}-{theme}-{step}.png'))
  # Normal motion freezes the cursor, velocity markers and signed-area shading.
  page.evaluate('window.intersamplePlayer.go(1);window.intersamplePlayer.next()');page.wait_for_timeout(140);page.evaluate('window.intersamplePlayer.pause()')
  def live_readout():
   return root.evaluate('''r=>({time:r.dataset.time,velocity:r.dataset.currentVelocity,relation:r.dataset.currentRelation,status:r.querySelector('.oio-status').textContent,equation:r.querySelector('.oio-equation').textContent,statusVelocity:r.querySelector('.oio-status').dataset.velocity,equationVelocity:r.querySelector('.oio-equation').dataset.velocity,markerY:+r.querySelector('[data-intersample-marker="bronze"]').getAttribute('cy'),markerX:r.querySelector('[data-intersample-marker="bronze"]').getAttribute('cx')})''')
  paused=live_readout();page.wait_for_timeout(180);assert live_readout()==paused
  assert float(paused['time'])>.25 and float(paused['velocity'])>3 and paused['relation']=='above',paused
  assert paused['velocity']==paused['statusVelocity']==paused['equationVelocity'],paused
  assert abs((480-paused['markerY'])/60-float(paused['velocity']))<1e-12,paused
  assert 'ABOVE ceiling' in paused['status'] and '(at ceiling)' not in paused['status'],paused
  root.screenshot(path=str(a.screenshots/f'paused-live-{width}-{theme}.png'))
  # Deterministic registered-frame replay checks rounding next to the ceiling.
  # This fixture hook calls the real transition's onFrame, never a replacement renderer.
  page.evaluate('window.intersamplePlayer.go(1);window.intersamplePlayer.next();window.intersamplePlayer.pause();window.intersampleTestFrame(0.000001)')
  tiny=live_readout();assert float(tiny['velocity'])>3 and tiny['relation']=='above',tiny
  assert 'ABOVE ceiling' in tiny['status'] and '(3 +' in tiny['status'],tiny
  assert tiny['velocity']==tiny['statusVelocity']==tiny['equationVelocity'],tiny
  assert '(3 +' in tiny['equation'] and 'Δv' in tiny['equation'],tiny
  root.screenshot(path=str(a.screenshots/f'paused-tiny-overspeed-{width}-{theme}.png'))
  page.evaluate('window.intersamplePlayer.go(1);window.intersamplePlayer.play()');page.wait_for_timeout(80);page.evaluate('window.intersamplePlayer.pause()')
  page.evaluate('window.intersamplePlayer.go(7)');root.focus();page.keyboard.press('p');assert root.get_attribute('data-intersample-step')=='6'
  page.emulate_media(reduced_motion='reduce');page.keyboard.press('n');page.wait_for_function("document.querySelector('#intersample-review-player').dataset.intersampleStep==='7'")
  root.get_by_role('button',name='Expand',exact=True).click();assert page.locator('dialog').is_visible();assert root.get_attribute('data-intersample-step')=='7';page.keyboard.press('Escape');page.wait_for_timeout(70);assert page.locator('dialog').count()==0
  page.emulate_media(media='print');assert not root.is_visible();assert static_path.is_visible();page.emulate_media(media='screen')
  root.get_by_role('button',name='Restart',exact=True).click();assert root.get_attribute('data-intersample-step')=='0'
  root.get_by_role('button',name='Play',exact=True).click()
  page.wait_for_function("document.querySelector('#intersample-review-player').dataset.intersampleStep==='7' && document.querySelector('#intersample-review-player').dataset.playing==='false'",timeout=10000)
  root.get_by_role('button',name='Replay',exact=True).click();page.evaluate('window.intersamplePlayer.pause()');assert root.get_attribute('data-playing')=='false'
  root.focus();page.keyboard.press('r');assert root.get_attribute('data-intersample-step')=='0';page.keyboard.press('k');page.keyboard.press('k');assert root.get_attribute('data-playing')=='false'
  assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
  root.scroll_into_view_if_needed();page.screenshot(path=str(a.screenshots/f'chapter-{width}-{theme}.png'))
  reports.append({'width':width,'theme':theme,'states':8,'signed_area_endpoints_peak_reserve':'passed','pause_keyboard_reduced_motion_expand_print':'passed'})
  page.close()
 browser.close()
print(json.dumps(reports,indent=2))
