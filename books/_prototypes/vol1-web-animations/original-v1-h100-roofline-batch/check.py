#!/usr/bin/env python3
"""Build the owning-chapter QA fixture and exercise every endpoint, control, and view.

Run from the repository root with a Python that has Playwright:
  python3 books/_prototypes/vol1-web-animations/original-v1-h100-roofline-batch/check.py [--engine chromium|webkit] [--width 1440|390]
calc.py needs MLSysIM's dependencies; set RB_CALC_PYTHON if ../MLSysBook/.venv is absent.
"""
import argparse,functools,http.server,json,os,subprocess,sys,threading
from pathlib import Path
from playwright.sync_api import sync_playwright
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BOOKS=ROOT/'books';EXT=BOOKS/'_extensions/mlsysbook-ext/figure-explore'
ENTRY=json.loads((HERE/'entry.json').read_text());KEY=ENTRY['key'];NAME=ENTRY['implementation']['name'];FID=KEY.split(':')[1]
p=argparse.ArgumentParser();p.add_argument('--engine',choices=['chromium','webkit']);p.add_argument('--width',type=int,choices=[1440,390]);p.add_argument('--screenshots',default='/tmp/'+NAME+'-review');args=p.parse_args()
shots=Path(args.screenshots);shots.mkdir(exist_ok=True)

# 1. The factory's embedded data must equal the MLSysIM calculation (asserted endpoints run inside calc.py).
calc_python=os.environ.get('RB_CALC_PYTHON') or str(ROOT.parent/'MLSysBook/.venv/bin/python')
data=json.loads(subprocess.check_output([calc_python,str(EXT/(NAME+'.calc.py')),'--check'],text=True,env={**os.environ,'PYTHONPATH':str(ROOT/'mlsysim')}))
LEDGERS,STATES=data['ledgers'],data['states']
assert data['expected_final']==ENTRY['expected_final'],'entry.json expected_final is stale';print('calc.py --check PASS',flush=True)

# 2. Serve books/ on a private port so a concurrent fixture server is never disturbed.
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*a):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(BOOKS)));threading.Thread(target=server.serve_forever,daemon=True).start()
base=f'http://127.0.0.1:{server.server_port}'
fixture=json.loads(subprocess.check_output([sys.executable,str(ROOT/'binder/tools/scripts/build_original_figure_fixture.py'),'--entry',str(HERE/'entry.json'),'--base-url',base],text=True));print(fixture['url'],flush=True)

THEME='(t)=>{document.body.classList.toggle("quarto-dark",t==="dark");document.body.classList.toggle("quarto-light",t!=="dark");document.documentElement.setAttribute("data-bs-theme",t);for(const l of document.querySelectorAll("link.quarto-color-scheme")){l.disabled=t==="dark"?!l.classList.contains("quarto-color-alternate"):l.classList.contains("quarto-color-alternate")}}'
# Plate text stays inside its plate; nothing in the stage overflows; readout cells do not clip.
GEOMETRY='''(root)=>{const bad=[];const svg=root.querySelector('.rb-plot svg');
 const panel=svg.querySelector('[data-rb-panel]');if(panel.getClientRects().length){const r=panel.querySelector('rect').getBBox();for(const t of panel.querySelectorAll('text')){const b=t.getBBox();if(b.x<r.x||b.x+b.width>r.x+r.width||b.y<r.y||b.y+b.height>r.y+r.height)bad.push('panel: '+t.textContent);}}
 const stage=root.querySelector('.mlsys-original-stage').getBoundingClientRect();for(const n of root.querySelectorAll('.mlsys-original-stage *')){if(n instanceof SVGElement&&n.localName!=='svg')continue;const b=n.getBoundingClientRect();if(b.width&&(b.right>stage.right+1||b.left<stage.left-1))bad.push('stage overflow: '+(n.getAttribute('class')||n.localName));}
 for(const c of root.querySelectorAll('.rb-cell, .rb-traffic p')){if(c.scrollWidth>c.clientWidth+1)bad.push('clipped '+c.textContent);}
 return bad;}'''

def invariants(i,root,width):
 ledger=json.loads(root.get_attribute('data-ledger'));assert ledger==LEDGERS[i],(i,ledger,LEDGERS[i])
 s=STATES[i];dot=root.locator('[data-motion-object]')
 assert abs(float(dot.get_attribute('cx'))-s['xy'][0])<1e-6 and abs(float(dot.get_attribute('cy'))-s['xy'][1])<1e-6,'marker off its computed endpoint'
 # Exactly one binding segment is highlighted, and it matches the committed bound.
 vis=root.evaluate("r=>[...r.querySelectorAll('[data-rb-binding]')].filter(n=>n.style.visibility!=='hidden').map(n=>n.dataset.rbBinding)")
 assert vis==[s['bound']],(i,vis);assert root.get_attribute('data-bound')==s['bound']
 # Below the ridge the bound is bandwidth x intensity; after it, the FP16 peak. Weight bytes never change.
 assert (s['ai']<data['h100']['ridge'])==(s['bound']=='bandwidth') and s['attainable']<=data['h100']['peak']
 assert ledger['weight_mb']==LEDGERS[0]['weight_mb']
 # The printed batch-32 dot is faded and ringed only when the marker has left it.
 ghost=root.evaluate("r=>r.querySelector('[data-rb-overlay] circle[stroke-dasharray]').style.visibility");assert ghost==('hidden' if i==0 else 'visible'),(i,ghost)
 # Readable text carries the same committed values at every width.
 txt=root.locator('.rb-readout').inner_text();assert s['text']['ai'] in txt and s['text']['attainable'] in txt and s['text']['batch'] in txt,txt
 if width>700:assert root.locator('[data-rb-panel] text').nth(1).text_content().startswith(s['text']['ai'])
 else:assert not root.locator('[data-rb-panel]').is_visible(),'plate hidden on phones'

with sync_playwright() as pw:
 for engine in ['chromium','webkit']:
  if args.engine and args.engine!=engine:continue
  browser=getattr(pw,engine).launch()
  for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
   if args.width and args.width!=width:continue
   page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme,reduced_motion='reduce');errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   page.route('**/release-manifest.json',lambda route:route.fulfill(status=200,body='{}',content_type='application/json'))
   page.goto(fixture['url']);page.wait_for_function("document.querySelector('[data-fixture-state]')?.dataset.fixtureState==='ready' || window.fixtureError",timeout=30000)
   assert not page.evaluate('window.fixtureError||null'),page.evaluate('window.fixtureError')
   page.evaluate(THEME,theme)
   root=page.locator('.'+NAME);page.get_by_role('tab',name='Explore',exact=True).click();root.scroll_into_view_if_needed()
   assert root.get_attribute('data-step')=='0' and root.get_attribute('data-playing')=='false','no autoplay on open'
   menu=root.locator('.mlsys-original-sequence select');count=menu.locator('option').count();assert count==len(LEDGERS)==6,count
   for i in range(count):
    menu.select_option(str(i));assert root.get_attribute('data-step')==str(i)
    invariants(i,root,width);bad=root.evaluate(GEOMETRY);assert not bad,(i,bad)
    root.screenshot(path=str(shots/f'{engine}-{width}-{theme}-{i}.png'))
   final=json.loads(root.get_attribute('data-ledger'))
   for k,v in ENTRY['expected_final'].items():assert final.get(k)==v,(k,final.get(k),v)
   # Menu jumps reproduce the same endpoint regardless of history.
   menu.select_option('3');a=(root.get_attribute('data-ledger'),root.locator('[data-motion-object]').get_attribute('cx'));menu.select_option(str(count-1));menu.select_option('1');menu.select_option('3')
   assert (root.get_attribute('data-ledger'),root.locator('[data-motion-object]').get_attribute('cx'))==a
   # Keyboard stepping under reduced motion commits immediately.
   menu.select_option(str(count-1));root.focus();page.keyboard.press('p');assert root.get_attribute('data-step')==str(count-2);page.keyboard.press('n');assert root.get_attribute('data-step')==str(count-1);page.keyboard.press('r');assert root.get_attribute('data-step')=='0'
   page.keyboard.press('n');assert root.get_attribute('data-step')=='1' and root.get_attribute('data-playback')=='idle' and root.get_attribute('data-pending') is None,'reduced motion has no intermediate state'
   page.keyboard.press('r');page.evaluate('window.scrollTo(0,0)');page.wait_for_timeout(300);root.evaluate('r=>r.scrollIntoView({block:"start"})');page.wait_for_timeout(400);root.focus();page.keyboard.press('k');page.wait_for_timeout(100);assert root.get_attribute('data-playing')=='true',page.evaluate('window.fixturePlayer.state');page.keyboard.press('k');assert root.get_attribute('data-playing')=='false';page.keyboard.press('r')
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'chapter overflow'
   root.get_by_role('button',name='Expand',exact=True).click();assert page.locator('dialog[open]').count()==1;page.screenshot(path=str(shots/f'{engine}-{width}-{theme}-expanded.png'));page.keyboard.press('Escape');assert page.locator('dialog[open]').count()==0
   if theme=='light':
    # Normal motion inside Expand (always in view): pause freezes marker, ledger, readout, and narration.
    page.emulate_media(reduced_motion='no-preference');menu.select_option('0');root.get_by_role('button',name='Expand',exact=True).click()
    root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(450);root.get_by_role('button',name='Pause',exact=True).click()
    assert root.get_attribute('data-step')=='0' and root.get_attribute('data-pending')=='1'
    dot=root.locator('[data-motion-object]');snap=lambda:(dot.get_attribute('cx'),dot.get_attribute('cy'),root.get_attribute('data-ledger'),root.locator('.rb-readout').inner_text(),root.locator('.mlsys-original-narration').inner_text())
    before=snap();x=float(before[0]);assert STATES[1]['xy'][0]<x<STATES[0]['xy'][0],x
    assert json.loads(before[2])==LEDGERS[0],'committed ledger must not advance while moving';assert 'On arrival' in before[4],before[4]
    page.screenshot(path=str(shots/f'{engine}-{width}-paused-motion.png'));page.wait_for_timeout(450);assert snap()==before,'pause must freeze the state'
    root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function("window.fixtureRoot.dataset.step==='1'",timeout=12000);root.get_by_role('button',name='Pause',exact=True).click()
    assert json.loads(root.get_attribute('data-ledger'))==LEDGERS[1];page.keyboard.press('Escape');page.emulate_media(reduced_motion='reduce')
   page.get_by_role('tab',name='Diagram',exact=True).click();assert not root.is_visible();page.screenshot(path=str(shots/f'{engine}-{width}-{theme}-static.png'))
   page.get_by_role('tab',name='Explore',exact=True).click();assert not errors,errors
   # WebKit's print reflow makes the chapter's own Quarto sidebar ResizeObserver report a benign loop notice
   # (the accepted sibling modules tolerate the same notice); only that notice is tolerated, and only here.
   page.emulate_media(media='print');assert not root.is_visible(),'print shows Diagram';page.emulate_media(media='screen');page.wait_for_timeout(200)
   errors=[e for e in errors if not e.startswith('ResizeObserver loop completed')];assert not errors,errors;page.close();print(engine,width,theme,'PASS',count,'states',flush=True)
  if not args.width:
   # JavaScript disabled keeps the original static figure and caption.
   page=browser.new_page(java_script_enabled=False);page.goto(fixture['url']);assert page.locator('#'+FID+' img').count()>0 and page.locator('#'+FID+' figcaption').count()==1;page.close()
  browser.close()
server.shutdown();print('screenshots',shots)
