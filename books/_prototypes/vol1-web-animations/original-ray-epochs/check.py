"""Render the original ray player in a complete existing Volume IV chapter.

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
parser.add_argument('--screenshots',type=Path,default=Path('/tmp/original-ray-epochs-review'))
a=parser.parse_args();a.screenshots.mkdir(parents=True,exist_ok=True)
repo=Path(__file__).resolve().parents[4]
ext=repo/'books/_extensions/mlsysbook-ext/figure-explore'
entry=json.loads(a.art_index.read_text())['vol4:fig-09-ray-update-epochs']
chapter=Path(entry['html']); soup=BeautifulSoup(chapter.read_text(),'html.parser')
for script in soup.find_all('script'):script.decompose()
base=soup.new_tag('base',href=chapter.parent.as_uri()+'/');soup.head.insert(0,base)
for name in ['original-player','original-ray-epochs']:
 soup.head.append(soup.new_tag('link',rel='stylesheet',href=(ext/f'{name}.css').as_uri()))
 soup.body.append(soup.new_tag('script',src=(ext/f'{name}.js').as_uri()))
outer=soup.select_one('#fig-09-ray-update-epochs');figure=outer.find('figure')
for key in list(outer.attrs):
 if key.startswith('data-'):del outer[key]
# Explicit test settings, not runtime defaults. The production caller gets YAML metadata.
outer['data-mlsys-movement-ms']='500';outer['data-mlsys-reading-pause-ms']='200'
static=figure.find('div',attrs={'aria-describedby':True});static['hidden']=''
root=soup.new_tag('section',id='ray-review-player');figure.insert(0,root)
script=soup.new_tag('script');script.string='''(async()=>{const root=document.querySelector('#ray-review-player'),outer=document.querySelector('#fig-09-ray-update-epochs');const art=new DOMParser().parseFromString('''+json.dumps(Path(entry['asset']).read_text())+''','image/svg+xml').documentElement;window.rayPlayer=await window.MLSysFigureLessons['original-ray-epochs'](root,{outer,verifiedArtwork:art});})().catch(e=>{window.fixtureError=String(e);console.error(e)});''';soup.body.append(script)
fixture=a.screenshots/'chapter.html';fixture.write_text(str(soup))
reports=[]
with sync_playwright() as p:
 browser=getattr(p,a.engine).launch()
 for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
  page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme)
  page.goto(fixture.as_uri());page.wait_for_function('window.rayPlayer || window.fixtureError');assert not page.evaluate('window.fixtureError'),page.evaluate('window.fixtureError')
  page.evaluate("theme=>{document.documentElement.setAttribute('data-bs-theme',theme);document.body.classList.toggle('quarto-dark',theme==='dark');document.body.classList.toggle('quarto-light',theme==='light');for(const l of document.querySelectorAll('link.quarto-color-scheme')){l.disabled=l.classList.contains('quarto-color-alternate')?theme!=='dark':theme==='dark';}}",theme)
  root=page.locator('#ray-review-player');root.scroll_into_view_if_needed()
  assert root.locator('svg rect[data-evidence-epoch]').count()==70
  original=page.locator('#fig-09-ray-update-epochs svg')
  assert original.locator('rect[width="34"][height="34"]').count()==70
  for step in range(13):
   page.evaluate('i=>window.rayPlayer.go(i)',step)
   snapshot=root.evaluate('''r=>({free:r.dataset.updatedFree,hit:r.dataset.hitRecorded,published:r.dataset.mapPublished,behind:[...r.querySelectorAll('rect[y="102"][width="34"]')].slice(10).map(c=>[c.dataset.evidenceEpoch,c.getAttribute('fill')]),off:[...r.querySelectorAll('rect[width="34"][height="34"]')].filter(c=>c.getAttribute('y')!=='102').map(c=>[c.dataset.evidenceEpoch,c.dataset.occupancy]),ray:+r.querySelector('line[x1="74"]').getAttribute('x2')})''')
   assert int(snapshot['free'])==max(0,min(9,step-1)),snapshot
   assert snapshot['hit']==str(step>=11).lower()
   assert all(epoch=='0' and 'url(' in fill for epoch,fill in snapshot['behind']),snapshot
   assert all(epoch=='unchanged' and state=='unchanged' for epoch,state in snapshot['off']),snapshot
   assert snapshot['ray']<=415,snapshot
   if step in [0,7,11,12]:
    root.screenshot(path=str(a.screenshots/f'player-{width}-{theme}-{step}.png'))
  # A manual transition freezes both ray motion and semantic state on pause.
  page.evaluate('window.rayPlayer.go(1);window.rayPlayer.next()');page.wait_for_timeout(120);page.evaluate('window.rayPlayer.pause()')
  x=root.locator('line[x1="74"]').get_attribute('x2');page.wait_for_timeout(180);assert root.locator('line[x1="74"]').get_attribute('x2')==x
  assert root.get_attribute('data-updated-free')=='0'
  page.evaluate('window.rayPlayer.play()');page.wait_for_timeout(100);page.evaluate('window.rayPlayer.pause()')
  page.evaluate('window.rayPlayer.go(7)');root.focus();page.keyboard.press('p');assert root.get_attribute('data-ray-step')=='6'
  page.emulate_media(reduced_motion='reduce');root.get_by_role('button',name='Next',exact=True).click();page.wait_for_function("document.querySelector('#ray-review-player').dataset.rayStep==='7'")
  assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
  root.scroll_into_view_if_needed();page.screenshot(path=str(a.screenshots/f'chapter-{width}-{theme}.png'))
  reports.append({'width':width,'theme':theme,'states':13,'behind_return_unchanged':True,'off_ray_unchanged':True,'pause_keyboard_reduced_motion':'passed'})
  page.close()
 browser.close()
print(json.dumps(reports,indent=2))
