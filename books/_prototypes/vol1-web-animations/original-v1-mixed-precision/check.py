#!/usr/bin/env python3
"""Build the owning-chapter QA fixture and exercise every endpoint, control, and view.

Run from the repository root with a Python that has Playwright:
  python3 books/_prototypes/vol1-web-animations/original-v1-mixed-precision/check.py [--engine chromium|webkit] [--width 1440|390]
"""
import argparse,functools,http.server,json,os,subprocess,sys,threading
from pathlib import Path
from playwright.sync_api import sync_playwright
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BOOKS=ROOT/'books';EXT=BOOKS/'_extensions/mlsysbook-ext/figure-explore'
ENTRY=json.loads((HERE/'entry.json').read_text());KEY=ENTRY['key'];NAME=ENTRY['implementation']['name'];FID=KEY.split(':')[1]
p=argparse.ArgumentParser();p.add_argument('--engine',choices=['chromium','webkit']);p.add_argument('--width',type=int,choices=[1440,390]);p.add_argument('--screenshots',default='/tmp/'+NAME+'-review');args=p.parse_args()
shots=Path(args.screenshots);shots.mkdir(exist_ok=True)

# 1. The factory's embedded data must equal the binary16 calculation (needs numpy).
calc_python=os.environ.get('MP_CALC_PYTHON') or next((c for c in [sys.executable,str(ROOT.parent/'MLSysBook/.venv/bin/python')] if Path(c).exists() and subprocess.run([c,'-c','import numpy'],capture_output=True).returncode==0),None)
assert calc_python,'Set MP_CALC_PYTHON to a Python with numpy'
data=json.loads(subprocess.check_output([calc_python,str(EXT/(NAME+'.calc.py')),'--check'],text=True))
T=data['trials'];print('calc.py --check PASS',flush=True)

# 2. Serve books/ on a private port so a concurrent fixture server is never disturbed.
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*a):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(BOOKS)));threading.Thread(target=server.serve_forever,daemon=True).start()
base=f'http://127.0.0.1:{server.server_port}'
fixture=json.loads(subprocess.check_output([sys.executable,str(ROOT/'binder/tools/scripts/build_original_figure_fixture.py'),'--entry',str(HERE/'entry.json'),'--base-url',base],text=True));print(fixture['url'],flush=True)

THEME='(t)=>{document.body.classList.toggle("quarto-dark",t==="dark");document.body.classList.toggle("quarto-light",t!=="dark");document.documentElement.setAttribute("data-bs-theme",t);for(const l of document.querySelectorAll("link.quarto-color-scheme")){l.disabled=t==="dark"?!l.classList.contains("quarto-color-alternate"):l.classList.contains("quarto-color-alternate")}}'
# Tags and the center panel must stay inside their plates; the narrow grid must not overflow the stage.
GEOMETRY='''(root)=>{const bad=[];const svg=root.querySelector('.mp-source svg');
 for(const g of svg.querySelectorAll('.mp-tags g')){if(!g.getClientRects().length)continue;const r=g.querySelector('rect').getBBox();for(const t of g.querySelectorAll('text')){if(!t.textContent)continue;const b=t.getBBox();if(b.x<r.x-.1||b.x+b.width>r.x+r.width+.1||b.y<r.y-.1||b.y+b.height>r.y+r.height+.1)bad.push('tag '+g.dataset.node+': '+t.textContent);}}
 const panel=svg.querySelector('.mp-panel');if(panel.getClientRects().length){const r=panel.querySelector('rect').getBBox();for(const t of panel.querySelectorAll('text')){const b=t.getBBox();if(b.x<r.x-.1||b.x+b.width>r.x+r.width+.1)bad.push('panel: '+t.textContent);}}
 const stage=root.querySelector('.mlsys-original-stage').getBoundingClientRect();for(const n of root.querySelectorAll('.mlsys-original-stage *')){if(n instanceof SVGElement&&n.localName!=='svg')continue;const b=n.getBoundingClientRect();if(b.width&&(b.right>stage.right+1||b.left<stage.left-1))bad.push('stage overflow: '+(n.getAttribute('class')||n.localName));}
 for(const c of root.querySelectorAll('.mp-card')){if(c.getClientRects().length&&c.scrollWidth>c.clientWidth+1)bad.push('card overflow '+c.dataset.node);}
 return bad;}'''

def invariants(i,ledger,count):
 # Mechanism rules: values are committed per state; overflow never reaches the master weight.
 w0=data['toy']['w32']
 if i==0:assert ledger['w16'] is None and ledger['S'] is None
 if i>=1:assert ledger['w16']==data['cast']['w16']
 if i==4:assert ledger['scaledSmall']==0 and ledger['smallClass']=='underflow' and ledger['scaledLarge']==3
 if i==6:assert ledger['scaledSmall']==T[1]['small'] and ledger['smallClass']=='subnormal'
 if i==8:assert ledger['scaledSmall']==2**-11 and ledger['scaledLarge']=='inf' and ledger['largeClass']=='overflow'
 # The nonfinite entry is found in the unscaled FP32 gradients (step 5); only the update (step 6) is skipped.
 if i==9:assert ledger['grad32Small']==data['toy']['g_small'] and ledger['grad32Large']=='inf' and ledger['update']=='none' and ledger['S']==T[2]['S']
 if i==10:assert ledger['update']=='skipped' and ledger['w32']==w0 and ledger['grad32Large']=='inf' and ledger['S']==T[3]['S']
 if i==11:assert ledger['grad32Small'] is None and ledger['grad32Large'] is None and ledger['update']=='none'
 if i in (5,7,11):assert ledger['scaledSmall'] is None,'a new scale discards the previous trial'
 if i<14:assert ledger['w32']==w0,'the master weight changes only at the update'
 if i==13:assert ledger['grad32Small']==data['toy']['g_small'] and ledger['grad32Large']==data['toy']['g_large']
 if i>=14:assert ledger['w32']==data['update']['w_new'] and ledger['update']=='applied'
 if i==15:assert ledger['bf16W']==data['bf16']['w'] and ledger['bf16Small']==data['toy']['g_small']
MARKS={10:['6'],15:['3']}  # skip marks only the update; BF16 drops only the scale step

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
   menu=root.locator('.mlsys-original-sequence select');count=menu.locator('option').count();assert count==16,count
   ledgers=[]
   for i in range(count):
    menu.select_option(str(i));assert root.get_attribute('data-step')==str(i)==root.get_attribute('data-endpoint')
    ledger=json.loads(root.get_attribute('data-ledger'));invariants(i,ledger,count);ledgers.append(ledger)
    assert root.evaluate("r=>[...r.querySelectorAll('[data-mark]')].map(n=>n.dataset.mark)")==MARKS.get(i,[]),(i,'marks')
    bad=root.evaluate(GEOMETRY);assert not bad,(i,bad)
    root.screenshot(path=str(shots/f'{engine}-{width}-{theme}-{i:02d}.png'))
   for k,v in ENTRY['expected_final'].items():assert ledgers[-1].get(k)==v,(k,ledgers[-1].get(k),v)
   # Menu jumps reproduce the same endpoint regardless of history.
   menu.select_option('8');a=root.get_attribute('data-ledger');menu.select_option(str(count-1));menu.select_option('2');menu.select_option('8');assert root.get_attribute('data-ledger')==a
   # Keyboard stepping under reduced motion commits immediately.
   menu.select_option(str(count-1));root.focus();page.keyboard.press('p');assert root.get_attribute('data-step')==str(count-2);page.keyboard.press('n');assert root.get_attribute('data-step')==str(count-1);page.keyboard.press('r');assert root.get_attribute('data-step')=='0'
   page.keyboard.press('n');assert root.get_attribute('data-step')=='1' and root.get_attribute('data-playback')=='idle','reduced motion has no intermediate state'
   page.keyboard.press('r')
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'chapter overflow'
   root.get_by_role('button',name='Expand',exact=True).click();assert page.locator('dialog[open]').count()==1;page.screenshot(path=str(shots/f'{engine}-{width}-{theme}-expanded.png'));page.keyboard.press('Escape');assert page.locator('dialog[open]').count()==0
   if theme=='light':
    # Normal motion inside Expand (always in view): pause freezes the token, ledger, and narration.
    page.emulate_media(reduced_motion='no-preference');menu.select_option('2');root.get_by_role('button',name='Expand',exact=True).click()
    root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(400);assert root.get_attribute('data-step')=='2' and root.get_attribute('data-pending')=='3'
    root.get_by_role('button',name='Pause',exact=True).click();tok=root.locator('.mp-source svg g[aria-hidden=true] g').first;tr=tok.get_attribute('transform');led=root.get_attribute('data-ledger');nar=root.locator('.mlsys-original-narration').inner_text()
    assert 'Next · Scale by S = 1, no scaling (stage 3)' in nar and 'Pending' not in nar,nar;page.wait_for_timeout(400);assert tok.get_attribute('transform')==tr and root.get_attribute('data-ledger')==led and root.locator('.mlsys-original-narration').inner_text()==nar
    page.screenshot(path=str(shots/f'{engine}-{width}-paused-motion.png'))
    # The paused token must not cover the "3. Scale" step-label plate (drawing units: x 98.63, y 116.04, 28.37 x 8.82).
    clear=root.evaluate('''r=>{const t=r.querySelector('.mp-source svg g[aria-hidden=true] g');const m=t.getAttribute('transform').match(/translate\\(([-\\d.e]+) ([-\\d.e]+)\\)/);const w=+t.querySelector('rect').getAttribute('width'),x=+m[1],y=+m[2];
      const P=[98.63,116.04,28.37,8.82];return x+w/2<P[0]||x-w/2>P[0]+P[2]||y+4.5<P[1]||y-4.5>P[1]+P[3];}''');assert clear,'token covers the step label'
    root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function("window.fixtureRoot.dataset.step==='3'",timeout=12000);root.get_by_role('button',name='Pause',exact=True).click()
    # Round four (2026-10-30): one Escape does one thing; a hovered tooltip takes the first Escape. Move the pointer off the bar so this Escape closes Expand.
    assert json.loads(root.get_attribute('data-ledger'))['scaledLoss']==T[0]['scaled_loss'];page.mouse.move(2,2);page.keyboard.press('Escape');page.emulate_media(reduced_motion='reduce')
   page.get_by_role('tab',name='Diagram',exact=True).click();assert not root.is_visible();page.screenshot(path=str(shots/f'{engine}-{width}-{theme}-static.png'))
   page.get_by_role('tab',name='Explore',exact=True).click();assert not errors,errors
   # WebKit's print reflow makes the chapter's own Quarto sidebar ResizeObserver report a benign loop notice
   # (reproduced with the figure left in Diagram view); only that notice is tolerated, and only here.
   page.emulate_media(media='print');assert not root.is_visible(),'print shows Diagram';page.emulate_media(media='screen');page.wait_for_timeout(200)
   errors=[e for e in errors if not e.startswith('ResizeObserver loop completed')];assert not errors,errors;page.close();print(engine,width,theme,'PASS',count,'states',flush=True)
  if not args.width:
   # JavaScript disabled keeps the original static figure and caption.
   page=browser.new_page(java_script_enabled=False);page.goto(fixture['url']);assert page.locator('#'+FID+' img, #'+FID+' svg').count()>0 and page.locator('#'+FID+' figcaption').count()==1;page.close()
  browser.close()
server.shutdown();print('screenshots',shots)
