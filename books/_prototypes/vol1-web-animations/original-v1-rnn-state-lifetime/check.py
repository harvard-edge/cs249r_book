#!/usr/bin/env python3
"""Build the owning-chapter QA fixture and exercise every endpoint, control, and view.

Run from the repository root with a Python that has Playwright:
  python3 books/_prototypes/vol1-web-animations/original-v1-rnn-state-lifetime/check.py [--engine chromium|webkit] [--width 1440|390]
"""
import argparse,functools,http.server,json,os,subprocess,sys,threading
from pathlib import Path
from playwright.sync_api import sync_playwright
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BOOKS=ROOT/'books';EXT=BOOKS/'_extensions/mlsysbook-ext/figure-explore'
ENTRY=json.loads((HERE/'entry.json').read_text());KEY=ENTRY['key'];NAME=ENTRY['implementation']['name'];FID=KEY.split(':')[1]
p=argparse.ArgumentParser();p.add_argument('--engine',choices=['chromium','webkit']);p.add_argument('--width',type=int,choices=[1440,390]);p.add_argument('--screenshots',default='/tmp/'+NAME+'-review');args=p.parse_args()
shots=Path(args.screenshots);shots.mkdir(exist_ok=True)

# 1. The factory's embedded data must equal the MLSysIM calculation (MLSysIM needs numpy).
calc_python=os.environ.get('RNN_CALC_PYTHON') or next((c for c in [sys.executable,str(ROOT.parent/'MLSysBook/.venv/bin/python')] if Path(c).exists() and subprocess.run([c,'-c','import numpy'],capture_output=True).returncode==0),None)
assert calc_python,'Set RNN_CALC_PYTHON to a Python with numpy'
data=json.loads(subprocess.check_output([calc_python,str(EXT/(NAME+'.calc.py')),'--check'],text=True,env={**os.environ,'PYTHONPATH':str(ROOT/'mlsysim')}))
S=data['states'];H=data['hidden']
# The entry's scope and storyboard restate calc values; they must not drift from them.
text=ENTRY['implementation']['scope']+' '.join(ENTRY['storyboard'])
for k in ['recurrent_text','input_macs_text','per_step_text','input_ahead_text','total_text','training_peak_text']:assert data[k] in text,k
assert data['chapter']['training_values_text'] in text and data['chapter']['training_bytes_text'] in text
print('calc.py --check PASS',flush=True)

# 2. Serve books/ on a private port so a concurrent fixture server is never disturbed.
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*a):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(BOOKS)));threading.Thread(target=server.serve_forever,daemon=True).start()
base=f'http://127.0.0.1:{server.server_port}'
fixture=json.loads(subprocess.check_output([sys.executable,str(ROOT/'binder/tools/scripts/build_original_figure_fixture.py'),'--entry',str(HERE/'entry.json'),'--base-url',base],text=True));print(fixture['url'],flush=True)

THEME='(t)=>{document.body.classList.toggle("quarto-dark",t==="dark");document.body.classList.toggle("quarto-light",t!=="dark");document.documentElement.setAttribute("data-bs-theme",t);for(const l of document.querySelectorAll("link.quarto-color-scheme")){l.disabled=t==="dark"?!l.classList.contains("quarto-color-alternate"):l.classList.contains("quarto-color-alternate")}}'
# Badge text stays inside its plate and clear of the drawing's label plates, chips stay clear of plates,
# visible badges never overlap each other, and nothing in the stage or panel overflows horizontally.
GEOMETRY='''(root)=>{const bad=[];const svg=root.querySelector('.rnn-source svg');
 const plates=[...svg.querySelectorAll(':scope > path[fill="rgb(100%, 100%, 100%)"]')].map(p=>p.getBBox());
 const hit=(a,b)=>a.x<b.x+b.width-.2&&b.x<a.x+a.width-.2&&a.y<b.y+b.height-.2&&b.y<a.y+a.height-.2;
 const boxes=[];
 const shown=getComputedStyle(svg.querySelector('.rnn-badges')).display!=='none';
 for(const g of shown?svg.querySelectorAll('.rnn-badges > g'):[]){const r=g.querySelector('rect').getBBox(),t=g.querySelector('text').getBBox();
  if(t.x<r.x-.1||t.x+t.width>r.x+r.width+.1)bad.push('text outside badge: '+g.textContent);
  for(const p of plates)if(hit(r,p))bad.push('badge covers a label plate: '+g.textContent);
  for(const o of boxes)if(hit(r,o))bad.push('badges overlap: '+g.textContent);boxes.push(r);
  // The badge ends clear of its column's vertical W_yh/W_hx arrow stroke (stroke half-width under 1 unit).
  const HX=[235.551,394.16,552.766][+g.dataset.badge];if(g.dataset.badge!=null&&r.x+r.width>HX-1.5)bad.push('badge touches arrow: '+g.textContent);
  const vb=svg.viewBox.baseVal;if(r.x<vb.x||r.x+r.width>vb.x+vb.width)bad.push('badge outside view: '+g.textContent);}
 const stage=root.querySelector('.mlsys-original-stage').getBoundingClientRect();
 for(const n of root.querySelectorAll('.mlsys-original-stage *')){if(n instanceof SVGElement&&n.localName!=='svg')continue;const b=n.getBoundingClientRect();if(b.width&&(b.right>stage.right+1||b.left<stage.left-1))bad.push('stage overflow: '+(n.getAttribute('class')||n.localName));}
 for(const c of root.querySelectorAll('.rnn-slot, .rnn-compare-row')){if(c.getClientRects().length&&c.scrollWidth>c.clientWidth+1)bad.push('panel overflow '+c.className);}
 return bad;}'''

def invariants(i,ledger):
 s=S[i]
 assert ledger['nodes']==s['nodes'] and ledger['retainedValues']==s['retainedValues']==s['nodes'].count('live')*H
 assert ledger['totalMacs']==ledger['recurrentMacs']+ledger['inputMacs']
 assert ledger['recurrentMacs']==ledger['whhReads']*data['recurrent_macs'],'every recurrent MAC reads the one W_hh copy'
 if ledger['mode']=='inference':
  assert ledger['retainedValues']==(H if i else 0),'inference keeps exactly one committed state'
  # A state is freed only once its successor (its only reader) exists.
  for k,st in enumerate(ledger['nodes']):
   if st=='freed':assert ledger['nodes'][k+1] in ('live','freed')
 if ledger['mode']=='training':
  assert ledger['inputMacs']==3*data['input_macs'],'input projections run ahead'
  if ledger['phase']=='backward':
   k=s['active'];assert all(x=='freed' for x in ledger['nodes'][k:]) and all(x=='live' for x in ledger['nodes'][:k]),'release right to left after own backward step'
 # A node waits only while its predecessor has not been written.
 for k,st in enumerate(ledger['nodes']):
  if st=='blocked':assert k>0 and ledger['nodes'][k-1] in ('next','blocked')
EXPECT_RETAINED=[0,H,H,H,0,H,2*H,3*H,2*H,H,0,0]

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
   menu=root.locator('.mlsys-original-sequence select');count=menu.locator('option').count();assert count==len(S)==12,count
   ledgers=[]
   for i in range(count):
    menu.select_option(str(i));assert root.get_attribute('data-step')==str(i)==root.get_attribute('data-endpoint')
    ledger=json.loads(root.get_attribute('data-ledger'));invariants(i,ledger);ledgers.append(ledger)
    assert ledger['retainedValues']==EXPECT_RETAINED[i],(i,ledger['retainedValues'])
    # The readable panel states the same retained count as the ledger.
    panel=root.locator('.rnn-panel').inner_text()
    if i<count-1:assert S[i]['retainedText'] in panel,(i,panel)
    else:assert data['chapter']['training_values_text'] in panel and data['inference_peak_text'] in panel
    if width==1440:assert root.locator('.rnn-badges > g[data-badge]').count()==sum(st!='idle' for st in S[i]['nodes'])
    bad=root.evaluate(GEOMETRY);assert not bad,(i,bad)
    root.screenshot(path=str(shots/f'{engine}-{width}-{theme}-{i:02d}.png'))
   for k,v in ENTRY['expected_final'].items():assert ledgers[-1].get(k)==v,(k,ledgers[-1].get(k),v)
   # Menu jumps reproduce the same endpoint regardless of history.
   menu.select_option('7');a=root.get_attribute('data-ledger');menu.select_option(str(count-1));menu.select_option('2');menu.select_option('7');assert root.get_attribute('data-ledger')==a
   # Keyboard stepping under reduced motion commits immediately.
   menu.select_option(str(count-1));root.focus();page.keyboard.press('p');assert root.get_attribute('data-step')==str(count-2);page.keyboard.press('n');assert root.get_attribute('data-step')==str(count-1);page.keyboard.press('r');assert root.get_attribute('data-step')=='0'
   page.keyboard.press('n');assert root.get_attribute('data-step')=='1' and root.get_attribute('data-playback')=='idle','reduced motion has no intermediate state'
   page.keyboard.press('r')
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'chapter overflow'
   root.get_by_role('button',name='Expand',exact=True).click();assert page.locator('dialog[open]').count()==1;page.screenshot(path=str(shots/f'{engine}-{width}-{theme}-expanded.png'));page.keyboard.press('Escape');assert page.locator('dialog[open]').count()==0
   if theme=='light':
    # Normal motion inside Expand (always in view): pause freezes the drawing, ledger, and narration.
    page.emulate_media(reduced_motion='no-preference');menu.select_option('5');root.get_by_role('button',name='Expand',exact=True).click()
    root.get_by_role('button',name='Next',exact=True).click();page.wait_for_timeout(400);assert root.get_attribute('data-step')=='5' and root.get_attribute('data-pending')=='6'
    root.get_by_role('button',name='Pause',exact=True).click();mot=root.locator('.rnn-motion').inner_html();led=root.get_attribute('data-ledger');nar=root.locator('.mlsys-original-narration').inner_text()
    assert 'Pending' in nar,nar;assert json.loads(led)['retainedValues']==H,'committed count while moving';assert mot.strip(),'motion layer drawn'
    page.wait_for_timeout(400);assert root.locator('.rnn-motion').inner_html()==mot and root.get_attribute('data-ledger')==led and root.locator('.mlsys-original-narration').inner_text()==nar
    page.screenshot(path=str(shots/f'{engine}-{width}-paused-motion.png'))
    root.get_by_role('button',name='Play',exact=True).click();page.wait_for_function("window.fixtureRoot.dataset.step==='6'",timeout=12000);root.get_by_role('button',name='Pause',exact=True).click()
    assert json.loads(root.get_attribute('data-ledger'))['retainedValues']==2*H;page.keyboard.press('Escape');page.emulate_media(reduced_motion='reduce')
   page.get_by_role('tab',name='Diagram',exact=True).click();assert not root.is_visible();page.screenshot(path=str(shots/f'{engine}-{width}-{theme}-static.png'))
   page.get_by_role('tab',name='Explore',exact=True).click();assert not errors,errors
   # WebKit's print reflow makes the chapter's own Quarto sidebar ResizeObserver report a benign loop notice
   # (also with the figure left in Diagram view); only that notice is tolerated, and only here.
   page.emulate_media(media='print');assert not root.is_visible(),'print shows Diagram';page.emulate_media(media='screen');page.wait_for_timeout(200)
   errors=[e for e in errors if not e.startswith('ResizeObserver loop completed')];assert not errors,errors;page.close();print(engine,width,theme,'PASS',count,'states',flush=True)
  if not args.width:
   # JavaScript disabled keeps the original static figure and caption.
   page=browser.new_page(java_script_enabled=False);page.goto(fixture['url']);assert page.locator('#'+FID+' img, #'+FID+' svg').count()>0 and page.locator('#'+FID+' figcaption').count()==1;page.close()
  browser.close()
server.shutdown();print('screenshots',shots)
