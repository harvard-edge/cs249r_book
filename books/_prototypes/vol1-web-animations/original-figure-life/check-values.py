from playwright.sync_api import sync_playwright
import json
import argparse
from pathlib import Path
parser=argparse.ArgumentParser(description='Check the optional chapter-derived convolution values against every window.')
parser.add_argument('--url',default='http://127.0.0.1:8882/_build/original-figure-life/vol1/06_nn_architectures/06_nn_architectures.html?view=explore#fig-cnn')
parser.add_argument('--screenshots',type=Path,default=Path('/tmp/original-convolution-review'))
args=parser.parse_args();args.screenshots.mkdir(parents=True,exist_ok=True)
url=args.url
with sync_playwright() as p:
 b=p.chromium.launch();report=[]
 for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
  page=b.new_page(viewport={'width':width,'height':1100},color_scheme=theme,reduced_motion='reduce')
  errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto(url)
  panel=page.locator('#fig-cnn .life-panel');panel.wait_for(state='visible')
  inset=panel.locator('.life-values');assert not inset.is_visible()
  diagram_width=panel.locator('.life-art svg').bounding_box()['width']
  paths=panel.locator('svg path').count()
  panel.get_by_role('button',name='Show values',exact=True).click()
  assert inset.is_visible();assert panel.get_attribute('data-step')=='0'
  box=panel.locator('.life-art svg').bounding_box()
  assert box['width']>200 and box['height']>150
  assert box['width']/diagram_width>.65
  assert 'Advance to select' in inset.inner_text()
  menu=panel.get_by_role('combobox')
  for n in range(25):
   # Independently compute the expected dot product from the declared inputs.
   row,column=divmod(n,5)
   expected=sum(int(column+c==2)*[-1,0,1][c] for r in range(3) for c in range(3))
   menu.select_option(str(n*3+1));assert inset.locator('.life-value-result').count()==0
   assert inset.locator('.life-patch td').count()==9
   assert inset.locator('.life-filter td').all_text_contents()==['-1','0','1']*3
   menu.select_option(str(n*3+2));assert inset.get_attribute('data-output')==str(expected)
   assert inset.locator('.life-value-result').inner_text()==f'Sum ready: {expected}'
   menu.select_option(str(n*3+3));assert inset.locator('.life-value-result').inner_text()==f'Output[{row}, {column}] = {expected}'
   assert panel.get_attribute('data-outputs')==str(n+1)
  if width==1440 and theme=='light':
   first=sum(int(c==2)*[-1,0,1][c] for r in range(3) for c in range(3))
   page.emulate_media(reduced_motion='no-preference');menu.select_option('0')
   panel.get_by_role('button',name='Next',exact=True).click()
   assert 'Moving the window' in inset.inner_text()
   page.wait_for_function("document.querySelector('#fig-cnn .life-panel').dataset.moving === 'false'")
   panel.get_by_role('button',name='Next',exact=True).click()
   assert inset.locator('.life-value-result').count()==0
   page.wait_for_function("document.querySelector('#fig-cnn .life-panel').dataset.moving === 'false'")
   assert inset.locator('.life-value-result').inner_text()==f'Sum ready: {first}'
   panel.get_by_role('button',name='Next',exact=True).click()
   assert panel.get_attribute('data-outputs')=='0'
   assert inset.locator('.life-value-result').inner_text()==f'Sum ready: {first}'
   page.wait_for_function("document.querySelector('#fig-cnn .life-panel').dataset.moving === 'false'")
   assert inset.locator('.life-value-result').inner_text()==f'Output[0, 0] = {first}'
   page.emulate_media(reduced_motion='reduce')
  assert panel.locator('svg path').count()==paths
  table=panel.locator('.life-patch table');cells=table.locator('tr').first.locator('td')
  assert sum(cells.nth(i).bounding_box()['width'] for i in range(3)) > table.bounding_box()['width']*.9
  menu.select_option('3')
  page.evaluate("document.querySelector('#fig-cnn').scrollIntoView({block:'start'})")
  page.screenshot(path=str(args.screenshots/f'values-{width}-{theme}.png'))
  inset.screenshot(path=str(args.screenshots/f'values-inset-{width}-{theme}.png'))
  before=panel.get_attribute('data-step');panel.get_by_role('button',name='Hide values').click()
  assert not inset.is_visible();assert panel.get_attribute('data-step')==before
  assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
  assert not errors,errors
  report.append({'width':width,'theme':theme,'windows_checked':25,'passed':True});page.close()
 print(json.dumps(report,indent=2));b.close()
