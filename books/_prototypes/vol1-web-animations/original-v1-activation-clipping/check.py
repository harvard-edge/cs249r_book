#!/usr/bin/env python3
"""Build the owning-chapter QA fixture and exercise every endpoint, control, and view.

Run from the repository root with a Python that has Playwright:
  python3 books/_prototypes/vol1-web-animations/original-v1-activation-clipping/check.py [--engine chromium|webkit] [--width 1440|390]
The calculation check needs a Python with MLSysIM's dependencies (default ../MLSysBook/.venv/bin/python;
override with AC_CALC_PYTHON).
"""
import argparse,functools,http.server,json,os,subprocess,sys,threading
from pathlib import Path
from playwright.sync_api import sync_playwright
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BOOKS=ROOT/'books';EXT=BOOKS/'_extensions/mlsysbook-ext/figure-explore'
ENTRY=json.loads((HERE/'entry.json').read_text());KEY=ENTRY['key'];NAME=ENTRY['implementation']['name'];FID=KEY.split(':')[1]
p=argparse.ArgumentParser();p.add_argument('--engine',choices=['chromium','webkit']);p.add_argument('--width',type=int,choices=[1440,390]);p.add_argument('--screenshots',default='/tmp/'+NAME+'-review');args=p.parse_args()
shots=Path(args.screenshots);shots.mkdir(exist_ok=True)

# 1. The factory's embedded data must equal calc.py (TikZ tables + MLSysIM UINT8 grid).
calc_python=os.environ.get('AC_CALC_PYTHON') or next((c for c in [str(ROOT.parent/'MLSysBook/.venv/bin/python'),sys.executable] if Path(c).exists() and subprocess.run([c,'-c','import pint'],capture_output=True).returncode==0),None)
assert calc_python,'Set AC_CALC_PYTHON to a Python with MLSysIM dependencies'
env={**os.environ,'PYTHONPATH':str(ROOT/'mlsysim')}
data=json.loads(subprocess.check_output([calc_python,str(EXT/(NAME+'.calc.py')),'--check'],text=True,env=env))
ROWS={r['betaText']:r for r in data['story_rows']};print('calc.py --check PASS',flush=True)
STORY=[None,'2.1','1.6','1.1','0.5','1.1']

# 2. Serve books/ on a private port so a concurrent fixture server is never disturbed.
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*a):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(BOOKS)));threading.Thread(target=server.serve_forever,daemon=True).start()
base=f'http://127.0.0.1:{server.server_port}'
fixture=json.loads(subprocess.check_output([sys.executable,str(ROOT/'binder/tools/scripts/build_original_figure_fixture.py'),'--entry',str(HERE/'entry.json'),'--base-url',base],text=True));print(fixture['url'],flush=True)

THEME='(t)=>{document.body.classList.toggle("quarto-dark",t==="dark");document.body.classList.toggle("quarto-light",t!=="dark");document.documentElement.setAttribute("data-bs-theme",t);for(const l of document.querySelectorAll("link.quarto-color-scheme")){l.disabled=t==="dark"?!l.classList.contains("quarto-color-alternate"):l.classList.contains("quarto-color-alternate")}}'
# Added ruler labels must stay inside the canvas and never overlap each other; HTML must not overflow the stage.
GEOMETRY='''(root)=>{const bad=[];const svg=root.querySelector('.ac-source svg');const vb=svg.viewBox.baseVal;
 const texts=[...svg.querySelectorAll('.ac-ruler text')].filter(t=>t.getClientRects().length&&t.textContent);
 const box=t=>t.getBBox();for(const t of texts){const b=box(t);if(b.x<vb.x-.1||b.x+b.width>vb.x+vb.width+.1||b.y<vb.y||b.y+b.height>vb.y+vb.height+.1)bad.push('outside: '+t.textContent);}
 for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++){const a=box(texts[i]),b=box(texts[j]);if(a.x<b.x+b.width-.2&&b.x<a.x+a.width-.2&&a.y<b.y+b.height-.2&&b.y<a.y+a.height-.2)bad.push('overlap: '+texts[i].textContent+' / '+texts[j].textContent);}
 const stage=root.querySelector('.mlsys-original-stage').getBoundingClientRect();for(const n of root.querySelectorAll('.mlsys-original-stage *')){if(n instanceof SVGElement&&n.localName!=='svg')continue;const b=n.getBoundingClientRect();if(b.width&&(b.right>stage.right+1||b.left<stage.left-1))bad.push('stage overflow: '+(n.getAttribute('class')||n.localName)+' '+[b.left,b.right,stage.left,stage.right].map(Math.round));}
 for(const c of root.querySelectorAll('.ac-card'))if(c.scrollWidth>c.clientWidth+1)bad.push('card overflow '+c.dataset.kind);
 return bad;}'''
CLIPPED="r=>[...r.querySelectorAll('[data-clipped=true]')].filter(n=>n.getClientRects().length).length"
LINE_X="r=>r.querySelector('.ac-source svg [data-clipping-overlay] path').dataset.x"

def invariants(i,ledger,root):
 b=STORY[i]
 if b is None:
  assert ledger=={'beta':None} and root.evaluate(CLIPPED)==0;return
 row=ROWS[b]
 for k in ('beta','step','halfStep','clippedBins','clippedShare','maxClipError'):assert ledger[k]==row[k],(i,k,ledger[k],row[k])
 assert root.evaluate(CLIPPED)==row['clippedBins'],(i,'hatched bars differ from ledger')
 assert root.locator('.ac-explore').is_visible()==(i==5),(i,'slider only in the final state')

with sync_playwright() as pw:
 for engine in ['chromium','webkit']:
  if args.engine and args.engine!=engine:continue
  browser=getattr(pw,engine).launch()
  for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
   if args.width and args.width!=width:continue
   page=browser.new_page(viewport={'width':width,'height':1100},device_scale_factor=2,color_scheme=theme,reduced_motion='reduce');errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   page.route('**/release-manifest.json',lambda route:route.fulfill(status=200,body='{}',content_type='application/json'))
   page.goto(fixture['url']);page.wait_for_function("document.querySelector('[data-fixture-state]')?.dataset.fixtureState==='ready' || window.fixtureError",timeout=30000)
   assert not page.evaluate('window.fixtureError||null'),page.evaluate('window.fixtureError')
   page.evaluate(THEME,theme)
   root=page.locator('.'+NAME);page.get_by_role('tab',name='Explore',exact=True).click();root.scroll_into_view_if_needed()
   assert root.get_attribute('data-step')=='0' and root.get_attribute('data-playing')=='false','no autoplay on open'
   menu=root.locator('.mlsys-original-sequence select');count=menu.locator('option').count();assert count==len(STORY),count
   ledgers=[]
   for i in range(count):
    menu.select_option(str(i));assert root.get_attribute('data-step')==str(i)==root.get_attribute('data-endpoint')
    ledger=json.loads(root.get_attribute('data-ledger'));invariants(i,ledger,root);ledgers.append(ledger)
    bad=root.evaluate(GEOMETRY);assert not bad,(i,bad)
    root.screenshot(path=str(shots/f'{engine}-{width}-{theme}-{i:02d}.png'))
   for k,v in ENTRY['expected_final'].items():assert ledgers[-1].get(k)==v,(k,ledgers[-1].get(k),v)
   # The trade-off is monotone along the story: the step falls while the clipped share rises.
   s=[l['step'] for l in ledgers[1:5]];c=[l['clippedShare'] for l in ledgers[1:5]];assert s==sorted(s,reverse=True) and c==sorted(c),(s,c)
   # Reader-controlled threshold in the final state: both ends of the slider, then a menu jump resets it.
   slider=root.locator('.ac-explore input[type=range]')
   for value,tag in [('25','wide'),('1','narrow')]:
    slider.fill(value);led=json.loads(root.get_attribute('data-ledger'));bad=root.evaluate(GEOMETRY);assert not bad,(tag,bad)
    if value=='25':assert led['beta']==2.5 and led['clippedBins']==0 and root.evaluate(CLIPPED)==0
    else:assert led['beta']==0.1 and led['clippedBins']==root.evaluate(CLIPPED)==20
    root.screenshot(path=str(shots/f'{engine}-{width}-{theme}-slider-{tag}.png'))
   menu.select_option('0');menu.select_option(str(count-1));assert json.loads(root.get_attribute('data-ledger'))==ledgers[-1],'final state resets to the printed candidate'
   # Menu jumps reproduce the same endpoint regardless of history.
   menu.select_option('2');a=root.get_attribute('data-ledger');menu.select_option(str(count-1));menu.select_option('0');menu.select_option('2');assert root.get_attribute('data-ledger')==a
   # Keyboard stepping under reduced motion commits immediately.
   menu.select_option(str(count-1));root.focus();page.keyboard.press('p');assert root.get_attribute('data-step')==str(count-2);page.keyboard.press('n');assert root.get_attribute('data-step')==str(count-1);page.keyboard.press('r');assert root.get_attribute('data-step')=='0'
   page.keyboard.press('n');assert root.get_attribute('data-step')=='1' and root.get_attribute('data-playback')=='idle','reduced motion has no intermediate state'
   page.keyboard.press('r')
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'chapter overflow'
   root.get_by_role('button',name='Expand',exact=True).click();assert page.locator('dialog[open]').count()==1;page.screenshot(path=str(shots/f'{engine}-{width}-{theme}-expanded.png'));page.keyboard.press('Escape');assert page.locator('dialog[open]').count()==0
   if theme=='light':
    # Normal motion inside Expand (always in view): pause freezes the line, hatching, ledger, and narration.
    page.emulate_media(reduced_motion='no-preference');menu.select_option('2');root.get_by_role('button',name='Expand',exact=True).click()
    root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(500);assert root.get_attribute('data-step')=='2' and root.get_attribute('data-pending')=='3'
    root.get_by_role('button',name='Pause',exact=True).click();x=root.evaluate(LINE_X);led=root.get_attribute('data-ledger');nar=root.locator('.mlsys-original-narration').inner_text();clipped=root.evaluate(CLIPPED)
    assert 'Paused' in nar and 'In progress' in nar,nar;assert json.loads(led)['beta']==1.6,'readout keeps the committed threshold while moving'
    page.wait_for_timeout(500);assert root.evaluate(LINE_X)==x and root.get_attribute('data-ledger')==led and root.locator('.mlsys-original-narration').inner_text()==nar and root.evaluate(CLIPPED)==clipped
    page.screenshot(path=str(shots/f'{engine}-{width}-paused-motion.png'))
    root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function("window.fixtureRoot.dataset.step==='3'",timeout=12000);root.get_by_role('button',name='Pause',exact=True).click()
    assert json.loads(root.get_attribute('data-ledger'))['beta']==1.1;page.keyboard.press('Escape');page.emulate_media(reduced_motion='reduce')
   page.get_by_role('tab',name='Diagram',exact=True).click();assert not root.is_visible();page.screenshot(path=str(shots/f'{engine}-{width}-{theme}-static.png'))
   page.get_by_role('tab',name='Explore',exact=True).click();assert not errors,errors
   # WebKit's print reflow can make the chapter's own Quarto sidebar ResizeObserver report a benign loop
   # notice; only that notice is tolerated, and only here.
   page.emulate_media(media='print');assert not root.is_visible(),'print shows Diagram';page.emulate_media(media='screen');page.wait_for_timeout(200)
   errors=[e for e in errors if not e.startswith('ResizeObserver loop completed')];assert not errors,errors;page.close();print(engine,width,theme,'PASS',count,'states',flush=True)
  if not args.width:
   # JavaScript disabled keeps the original static figure and caption.
   page=browser.new_page(java_script_enabled=False);page.goto(fixture['url']);assert page.locator('#'+FID+' img, #'+FID+' svg').count()>0 and page.locator('#'+FID+' figcaption').count()==1;page.close()
  browser.close()
server.shutdown();print('screenshots',shots)
