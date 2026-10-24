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
parser.add_argument('--screenshots',type=Path,default=Path('/tmp/original-capture-buffer-review'))
a=parser.parse_args();a.screenshots.mkdir(parents=True,exist_ok=True)
repo=Path(__file__).resolve().parents[4]
ext=repo/'books/_extensions/mlsysbook-ext/figure-explore'
entry=json.loads(a.art_index.read_text())['vol4:fig-08-shared-capture-buffer']
chapter=Path(entry['html']); soup=BeautifulSoup(chapter.read_text(),'html.parser')
for script in soup.find_all('script'):script.decompose()
base=soup.new_tag('base',href=chapter.parent.as_uri()+'/');soup.head.insert(0,base)
for name in ['original-player','original-capture-buffer']:
 soup.head.append(soup.new_tag('link',rel='stylesheet',href=(ext/f'{name}.css').as_uri()))
 soup.body.append(soup.new_tag('script',src=(ext/f'{name}.js').as_uri()))
outer=soup.select_one('#fig-08-shared-capture-buffer');figure=outer.find('figure')
for key in list(outer.attrs):
 if key.startswith('data-'):del outer[key]
# Explicit test settings, not runtime defaults. The production caller gets YAML metadata.
outer['data-mlsys-movement-ms']='500';outer['data-mlsys-reading-pause-ms']='200'
static=figure.find('div',attrs={'aria-describedby':True});static['hidden']=''
print_style=soup.new_tag('style');print_style.string='@media print { #fig-08-shared-capture-buffer figure > div[aria-describedby] { display:block!important; } }';soup.head.append(print_style)
root=soup.new_tag('section',id='capture-review-player');figure.insert(0,root)
script=soup.new_tag('script');script.string='''(async()=>{const root=document.querySelector('#capture-review-player'),outer=document.querySelector('#fig-08-shared-capture-buffer');const art=new DOMParser().parseFromString('''+json.dumps(Path(entry['asset']).read_text())+''','image/svg+xml').documentElement;window.capturePlayer=await window.MLSysFigureLessons['original-capture-buffer'](root,{outer,verifiedArtwork:art});})().catch(e=>{window.fixtureError=String(e);console.error(e)});''';soup.body.append(script)
fixture=a.screenshots/'chapter.html';fixture.write_text(str(soup))
reports=[]
with sync_playwright() as p:
 browser=getattr(p,a.engine).launch()
 for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
  page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme)
  page.goto(fixture.as_uri());page.wait_for_function('window.capturePlayer || window.fixtureError');assert not page.evaluate('window.fixtureError'),page.evaluate('window.fixtureError')
  page.evaluate("theme=>{document.documentElement.setAttribute('data-bs-theme',theme);document.body.classList.toggle('quarto-dark',theme==='dark');document.body.classList.toggle('quarto-light',theme==='light');for(const l of document.querySelectorAll('link.quarto-color-scheme')){l.disabled=l.classList.contains('quarto-color-alternate')?theme!=='dark':theme==='dark';}}",theme)
  root=page.locator('#capture-review-player');root.scroll_into_view_if_needed();static_path=page.locator('#fig-08-shared-capture-buffer figure > div[aria-describedby]')
  for step in range(13):
   page.evaluate('i=>window.capturePlayer.go(i)',step)
   snapshot=root.evaluate("""r=>({refs:+r.dataset.references,eligible:r.dataset.requeueEligible,requeued:r.dataset.requeued,complete:r.dataset.dmaComplete,b:r.dataset.converted,page:[...r.querySelectorAll('rect[x="220"][y="190"]')].map(n=>[n.dataset.frame,+n.dataset.references,n.dataset.requeueEligible]),mappings:[...r.querySelectorAll('rect[y="290"]')].map(n=>[n.dataset.storage,+n.dataset.references]),converted:[...r.querySelectorAll('rect[x="230"][y="360"]')].map(n=>[n.dataset.frame,n.dataset.storage,n.dataset.allocated]),copies:[...r.querySelectorAll('rect[y="50"]')].map(n=>n.dataset.frame)})""")
   expected=sum(step>=7 and step<9+i for i in range(3))
   assert snapshot['refs']==expected,snapshot
   assert snapshot['eligible']==str(step>=11).lower(),snapshot
   assert snapshot['requeued']==str(step==12).lower(),snapshot
   assert snapshot['complete']==str(step>=6).lower(),snapshot
   assert snapshot['b']==str(step>=8).lower(),snapshot
   assert all(storage=='A' for storage,refs in snapshot['mappings']),snapshot
   assert sum(refs for storage,refs in snapshot['mappings'])==expected,snapshot
   assert snapshot['page']==[['F' if step>=5 else '',expected,str(step>=11).lower()]],snapshot
   assert snapshot['converted']==[['F' if step>=8 else '', 'B', str(step>=8).lower()]],snapshot
   assert snapshot['copies']==['F' if step>=i+1 else '' for i in range(4)],snapshot
   if expected:assert snapshot['eligible']=='false' and snapshot['requeued']=='false'
   if step in [0,4,5,7,8,10,11,12]:
    root.screenshot(path=str(a.screenshots/f'player-{width}-{theme}-{step}.png'))
  # The retained-output inspection changes the viewport, never ownership state.
  inspect=root.get_by_role('button',name='Inspect retained B',exact=True);inspect.click()
  assert root.get_attribute('data-capture-step')=='12' and root.get_attribute('data-requeued')=='true'
  root.screenshot(path=str(a.screenshots/f'retained-B-{width}-{theme}.png'))
  root.get_by_role('button',name='Return to requeue',exact=True).click()
  # Pause while a copy travels: destination must not commit early.
  page.evaluate('window.capturePlayer.go(1);window.capturePlayer.next()');page.wait_for_timeout(120);page.evaluate('window.capturePlayer.pause()')
  token=root.locator('.ocb-transient');x=token.get_attribute('cx');page.wait_for_timeout(180);assert token.get_attribute('cx')==x
  assert root.get_attribute('data-capture-step')=='1'
  page.evaluate('window.capturePlayer.play()');page.wait_for_timeout(100);page.evaluate('window.capturePlayer.pause()')
  page.evaluate('window.capturePlayer.go(10)');root.focus();page.keyboard.press('p');assert root.get_attribute('data-capture-step')=='9'
  page.emulate_media(reduced_motion='reduce');root.get_by_role('button',name='Next',exact=True).click();page.wait_for_function("document.querySelector('#capture-review-player').dataset.captureStep==='10'")
  # Expanded inspection preserves state, Escape returns keyboard focus.
  root.get_by_role('button',name='Expand',exact=True).click();assert page.locator('dialog').is_visible();assert root.get_attribute('data-capture-step')=='10';page.keyboard.press('Escape');page.wait_for_timeout(70);assert page.locator('dialog').count()==0
  page.emulate_media(media='print');assert not root.is_visible();assert static_path.is_visible();page.emulate_media(media='screen')
  assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
  root.scroll_into_view_if_needed();page.screenshot(path=str(a.screenshots/f'chapter-{width}-{theme}.png'))
  reports.append({'width':width,'theme':theme,'states':13,'identity_refcounts_last_consumer':'passed','pause_keyboard_reduced_motion_expand_print':'passed'})
  page.close()
 browser.close()
print(json.dumps(reports,indent=2))
