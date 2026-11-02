"""Inspect one original-art mechanism inside its actual chapter."""
import argparse,json,math,base64
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from PIL import Image
A=argparse.ArgumentParser(description=__doc__);A.add_argument('--engine',choices=['chromium','webkit'],default='chromium');A.add_argument('--screenshots',type=Path);a=A.parse_args()
here=Path(__file__).resolve().parent;repo=here.parents[3];entry=json.loads((here/'entry.json').read_text());name=entry['implementation']['name'];out=a.screenshots or Path('/tmp')/(name+'-'+a.engine);out.mkdir(parents=True,exist_ok=True)
ext=repo/'books/_extensions/mlsysbook-ext/figure-explore';art=json.load(open('/tmp/figures-animation-art-current/index.json'))[entry['key']];chapter=Path(art['html']);soup=BeautifulSoup(chapter.read_text(),'html.parser')
for n in soup.find_all('script'):n.decompose()
soup.head.insert(0,soup.new_tag('base',href=chapter.parent.as_uri()+'/'))
for n in ['original-player',name]:
 soup.head.append(soup.new_tag('link',rel='stylesheet',href=(ext/(n+'.css')).as_uri()));soup.body.append(soup.new_tag('script',src=(ext/(n+'.js')).as_uri()))
outer=soup.find(id=entry['key'].split(':')[1]);figure=outer.find('figure')
for k in list(outer.attrs):
 if k.startswith('data-'):del outer[k]
outer['data-mlsys-movement-ms']='500';outer['data-mlsys-reading-pause-ms']='100'
static=figure.find('div',attrs={'aria-describedby':True})
if not static:raise RuntimeError('Missing canonical static holder')
static['hidden']='';static['data-fixture-static']='true';panel=soup.new_tag('section',id='v4-review');figure.insert(0,panel)
style=soup.new_tag('style');style.string='@media print{[data-fixture-static]{display:block!important}}';soup.head.append(style)
asset=Path(art['asset'])
if asset.suffix=='.svg':raw=asset.read_text()
else:
 w,h=Image.open(asset).size;raw=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" data-mlsys-artwork-kind="png"><image width="{w}" height="{h}" href="data:image/png;base64,{base64.b64encode(asset.read_bytes()).decode()}"/></svg>'
script=soup.new_tag('script');script.string='(async()=>{window.reviewPlayer=await window.MLSysFigureLessons['+json.dumps(name)+'](document.querySelector("#v4-review"),{outer:document.getElementById('+json.dumps(outer['id'])+'),verifiedArtwork:new DOMParser().parseFromString('+json.dumps(raw)+',"image/svg+xml").documentElement});})().catch(e=>window.fixtureError=String(e));';soup.body.append(script);fixture=out/'chapter.html';fixture.write_text(str(soup))
verify=None
if (here/'invariants.py').exists():
 d={};exec((here/'invariants.py').read_text(),d);verify=d['verify']
reports=[]
with sync_playwright() as p:
 b=getattr(p,a.engine).launch()
 for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
  page=b.new_page(viewport={'width':width,'height':1100},color_scheme=theme);errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto(fixture.as_uri());page.wait_for_function('window.reviewPlayer||window.fixtureError');assert not page.evaluate('window.fixtureError'),page.evaluate('window.fixtureError')
  page.evaluate("theme=>{document.documentElement.setAttribute('data-bs-theme',theme);document.body.classList.toggle('quarto-dark',theme==='dark');document.body.classList.toggle('quarto-light',theme==='light');for(const l of document.querySelectorAll('link.quarto-color-scheme'))l.disabled=l.classList.contains('quarto-color-alternate')?theme!=='dark':theme==='dark';}",theme)
  root=page.locator('#v4-review');root.scroll_into_view_if_needed();count=root.locator('select option').count();snapshots=[]
  for i in range(count):
   root.locator('select').select_option(str(i));v=json.loads(root.get_attribute('data-values'));assert all(not isinstance(x,float)or math.isfinite(x)for x in v.values());snapshots.append(v)
   if verify:verify(i,v)
   root.screenshot(path=str(out/f'player-{width}-{theme}-{i}.png'));root.locator('svg').screenshot(path=str(out/f'diagram-{width}-{theme}-{i}.png'))
  page.evaluate('window.reviewPlayer.go(0);window.reviewPlayer.next()');page.wait_for_timeout(160);page.evaluate('window.reviewPlayer.pause()');assert 'Next endpoint, not yet completed:' in root.locator('.mlsys-original-narration').inner_text();paused=root.get_attribute('data-values');readout=root.locator('.v4life-live').inner_text();page.wait_for_timeout(150);assert root.get_attribute('data-values')==paused and root.locator('.v4life-live').inner_text()==readout
  root.screenshot(path=str(out/f'paused-{width}-{theme}.png'))
  page.evaluate('n=>window.reviewPlayer.go(n)',count-1);root.focus();page.keyboard.press('p');assert root.get_attribute('data-scene-step')==str(count-2);page.emulate_media(reduced_motion='reduce');page.wait_for_function("matchMedia('(prefers-reduced-motion: reduce)').matches");root.focus();page.keyboard.press('n');assert root.get_attribute('data-scene-step')==str(count-1)
  root.get_by_role('button',name='Expand',exact=True).click();assert page.locator('dialog').is_visible();page.keyboard.press('Escape');page.wait_for_timeout(80);assert not page.locator('dialog').count()
  page.emulate_media(media='print');assert not root.is_visible();assert page.locator('[data-fixture-static]').is_visible();page.emulate_media(media='screen');assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');assert not errors,errors
  reports.append(dict(width=width,theme=theme,states=count,invariants='passed',controls='passed'));page.close()
 b.close()
(out/'report.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports))
