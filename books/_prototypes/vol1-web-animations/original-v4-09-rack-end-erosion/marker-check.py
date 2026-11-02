"""Focused rack-marker calibration on the verified original PNG; no model changes."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];EXT=REPO/'books/_extensions/mlsysbook-ext/figure-explore';NAME='original-v4-09-rack-end-erosion'
parser=argparse.ArgumentParser();parser.add_argument('--engine',choices=['chromium','webkit'],default='chromium');parser.add_argument('--fixture',type=Path,default=Path('/tmp')/(NAME+'-chromium/chapter.html'));parser.add_argument('--screenshots',type=Path);args=parser.parse_args();out=args.screenshots or Path('/tmp')/(NAME+'-marker-'+args.engine);out.mkdir(parents=True,exist_ok=True)
info=json.loads(Path('/tmp/figures-animation-art-current/index.json').read_text())['vol4:fig-09-rack-end-erosion'];art=Path(info['asset']);assert hashlib.sha1(art.read_bytes()).hexdigest()=='0708a4e98d1225bdedf88a63208d36eeced21b89'
pixels=np.array(Image.open(art).convert('RGB'));assert pixels.shape[:2]==(3981,3400)
# Independently measured centers: source1000mm grid2425; source0mm blue tail3630.5.
zero,thousand=3630.5,2425.0
fixture=out/'chapter.html';shutil.copyfile(args.fixture,fixture);records=[]
with sync_playwright() as p:
 browser=getattr(p,args.engine).launch()
 for width in [1440,390]:
  for theme in ['light','dark']:
   page=browser.new_page(viewport={'width':width,'height':1100 if width==1440 else 900},color_scheme=theme,reduced_motion='reduce');page.goto(fixture.as_uri());page.wait_for_function('window.reviewPlayer||window.fixtureError');assert not page.evaluate('window.fixtureError')
   page.evaluate("theme=>{document.documentElement.setAttribute('data-bs-theme',theme);document.body.classList.toggle('quarto-dark',theme==='dark');document.body.classList.toggle('quarto-light',theme==='light');for(const l of document.querySelectorAll('link.quarto-color-scheme'))l.disabled=l.classList.contains('quarto-color-alternate')?theme!=='dark':theme==='dark';}",theme)
   root=page.locator('#v4-review');select=root.locator('select')
   for i in range(5):
    select.select_option(str(i));values=json.loads(root.get_attribute('data-values'));marker=root.locator('[data-original-state-overlay] > circle:not([data-motion-object])');x=float(marker.get_attribute('cx'));y=float(marker.get_attribute('cy'));expected=zero+(thousand-zero)*values['certified_ahead_mm']/1000;assert abs(y-expected)<1e-8,(i,x,y,expected)
    # At the630mm case, the mark must lie inside the actual blue source stroke.
    if i==2:
     column=pixels[:,round(x)];blue=(column[:,0]<40)&(column[:,1]>50)&(column[:,1]<150)&(column[:,2]>100)&(column[:,2]>column[:,1]);ys=np.where(blue & (np.arange(len(column))>2300))[0];assert ys.min()<=y<=ys.max(),(y,ys.min(),ys.max())
    root.locator('.mlsys-original-stage').evaluate('(e)=>window.scrollTo(0,scrollY+e.getBoundingClientRect().top-160)');page.screenshot(path=str(out/f'{width}-{theme}-state{i}-scene.png'))
    # Phone normally focuses the reach geometry. Expand reveals the unchanged original plot.
    root.get_by_role('button',name='Expand',exact=True).click();page.wait_for_timeout(100);marker.evaluate('(e)=>e.scrollIntoView({block:"center",inline:"center"})');page.screenshot(path=str(out/f'{width}-{theme}-state{i}-plot.png'));page.keyboard.press('Escape');page.wait_for_timeout(50)
    records.append(dict(engine=args.engine,width=width,theme=theme,index=i,age_ms=values['age_ms'],depth_mm=values['certified_ahead_mm'],marker_x=x,marker_y=y,expected_y=expected))
   page.close()
 browser.close()
(out/'report.json').write_text(json.dumps(records,indent=2)+'\n');print('PASS',args.engine,'all states/desktop/phone/light/dark',hashlib.sha256((EXT/(NAME+'.js')).read_bytes()).hexdigest());print('PNG',out)
