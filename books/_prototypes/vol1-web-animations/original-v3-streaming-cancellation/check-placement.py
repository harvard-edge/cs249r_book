#!/usr/bin/env python3
"""Focused native case-menu placement and state consistency under shared layout C."""
import argparse,json
from pathlib import Path
from playwright.sync_api import sync_playwright
HERE=Path(__file__).resolve().parent
spec=json.loads((HERE/'lesson.json').read_text());name=spec['name'];entry=json.loads((HERE/'entry.json').read_text());art=json.loads(Path('/tmp/figures-animation-art-current/index.json').read_text())[entry['key']]
a=argparse.ArgumentParser(description=__doc__);a.add_argument('--engine',choices=['chromium','webkit'],default='chromium');a.add_argument('--screenshots',type=Path,default=Path('/tmp/v3-case-placement'));args=a.parse_args();args.screenshots.mkdir(exist_ok=True,parents=True)
url='http://127.0.0.1:8882/_build/'+name+'/'+str(Path(art['source']).relative_to('books')).replace('.qmd','.html')+'#'+entry['key'].split(':')[1]

def placement(root):
 assert root.evaluate("r=>{const ns=[...r.children],n=r.querySelector('.mlsys-original-narration'),c=r.querySelector('.v3-original-choice'),t=r.querySelector('.mlsys-original-controls'),s=r.querySelector('.mlsys-original-stage');return c.parentNode===r&&ns.indexOf(n)<ns.indexOf(c)&&c.nextElementSibling===t&&ns.indexOf(t)<ns.indexOf(s)}"),'Case menu must follow narration and precede toolbar'
 assert root.evaluate("r=>{const a=r.querySelector('.v3-original-choice').getBoundingClientRect(),b=r.querySelector('.mlsys-original-controls').getBoundingClientRect(),s=r.querySelector('.mlsys-original-stage').getBoundingClientRect();return a.bottom<=b.top+1&&b.bottom<=s.top+1}"),'Painted menu, toolbar, stage order'

with sync_playwright() as p:
 browser=getattr(p,args.engine).launch()
 for theme in ['light','dark']:
  page=browser.new_page(viewport={'width':390,'height':2100},color_scheme=theme);errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto(url);page.wait_for_function('window.wavePlayer!==undefined')
  if theme=='dark':page.evaluate("document.body.classList.add('quarto-dark');document.body.classList.remove('quarto-light')")
  root=page.locator('.'+name);choice=root.get_by_label(spec['variantLabel'],exact=True);menu=root.locator('.mlsys-original-sequence select');placement(root)
  for variant in spec['variants']:
   choice.select_option(variant['id']);assert root.get_attribute('data-original-step')=='0';assert menu.locator('option').all_text_contents()==[str(i+1)+'. '+s['label'] for i,s in enumerate(variant['states'])]
   for i,frame in enumerate(variant['states']):
    menu.select_option(str(i));assert json.loads(root.get_attribute('data-original-facts'))==frame['facts'];assert frame['description'] in root.locator('.mlsys-original-narration').inner_text();placement(root)
   menu.select_option('0');root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(100);root.get_by_role('button',name='Pause',exact=True).click();assert json.loads(root.get_attribute('data-original-facts'))==variant['states'][0]['facts'];placement(root)
   # Switching cases interrupts movement and resets the exact new case, without moving menus.
   choice.select_option(variant['id']);assert root.get_attribute('data-original-step')=='0';assert json.loads(root.get_attribute('data-original-facts'))==variant['states'][0]['facts']
   page.emulate_media(reduced_motion='reduce');root.focus();page.keyboard.press('n');page.wait_for_function('(name)=>document.querySelector("."+name).dataset.originalStep==="1"',arg=name);assert json.loads(root.get_attribute('data-original-facts'))==variant['states'][1]['facts'];page.emulate_media(reduced_motion='no-preference')
   menu.select_option(str(min(5,len(variant['states'])-1)));root.screenshot(path=str(args.screenshots/(name+'-'+args.engine+'-'+theme+'-'+variant['id']+'.png')));root.get_by_role('button',name='Expand',exact=True).click();placement(root);page.keyboard.press('Escape');page.wait_for_function("document.activeElement?.getAttribute('aria-label')==='Expand' || document.activeElement?.textContent==='Expand'");placement(root)
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');assert not errors,errors;page.close()
 browser.close()
print(json.dumps({'key':entry['key'],'engine':args.engine,'phone_light_dark':'passed','case_before_toolbar':'passed','every_case_every_state_facts_menu_narration':'passed','pause_reset_keyboard_reduced_motion_expand':'passed'}))
