"""Synthetic playback-contract checks; fixture timings are not book defaults."""
from playwright.sync_api import sync_playwright
from pathlib import Path
base=Path(__file__).resolve().parents[3] / 'books/_extensions/mlsysbook-ext/figure-explore'
with sync_playwright() as p:
 for engine in [p.chromium,p.webkit]:
  browser=engine.launch()
  for width,theme in [(1440,'light'),(390,'dark')]:
   page=browser.new_page(viewport={'width':width,'height':1000})
   errors=[]
   page.on('pageerror',lambda error:errors.append(str(error)))
   page.set_content('<style>body{font:18px system-ui;margin:20px;max-width:850px}body.dark{--bs-body-bg:#202429;--bs-body-color:#eee;--bs-border-color:#63717c;--bs-secondary-color:#b0bcc7;--bs-tertiary-bg:#2b3138;background:#202429;color:#eee}svg{background:#fff}</style><div id=outer data-mlsys-movement-ms=300 data-mlsys-reading-pause-ms=100><div id=root></div></div><input id=editing>')
   page.add_style_tag(path=str(base/'original-player.css'))
   page.add_script_tag(path=str(base/'original-player.js'))
   page.evaluate('''theme=>{document.body.classList.toggle('dark',theme==='dark');const root=document.querySelector('#root');const outer=document.querySelector('#outer');window.player=MLSysOriginalPlayer.create({root,outer,layout:'classic',title:'Values moving through a diagram',scope:'Illustrative contributions; motion does not represent measured latency.',steps:[{label:'Ready',description:'The value remains at the source.'},{label:'Read',description:'Read the source value.'},{label:'Compute',description:'The compute stage receives the value.'},{label:'Write',description:'Commit the result to the destination.'}],render(index,{stage,details}){if(!stage.firstChild)stage.innerHTML='<svg viewBox="0 0 440 140" role="img" aria-label="Value flow"><path d="M50 70 H370" stroke="#667" stroke-width="2"/><circle cx="50" cy="70" r="18" fill="#3d7b9e"/></svg>';const c=stage.querySelector('circle');c.setAttribute('cx',50+100*index);details.textContent='Committed state: '+index;},transition({from,to,stage,animate,onFrame}){const c=stage.querySelector('circle');animate(c,[{opacity:1},{opacity:.3},{opacity:1}]);onFrame(t=>{c.setAttribute('cx',50+100*(from+(to-from)*t));root.dataset.progress=t;});}});}''',theme)
   page.evaluate('player.next()');page.wait_for_timeout(100);assert page.locator('.mlsys-original-narration').inner_text().startswith('In motion · 2 / 4 · Read');assert page.locator('#root').get_attribute('data-step')=='0';assert page.locator('.mlsys-original-controls button').first.inner_text()=='Pause';page.evaluate('player.pause()');assert page.locator('.mlsys-original-narration').inner_text().startswith('Paused · 2 / 4 · Read')
   before=page.locator('circle').get_attribute('cx');page.wait_for_timeout(150)
   assert page.locator('circle').get_attribute('cx')==before
   assert page.evaluate('player.state.index')==0
   page.evaluate('player.play()');page.wait_for_function('player.state.index===3 && !player.state.playing')
   assert page.evaluate('player.state.phase')=='idle'
   page.evaluate('player.go(1)');assert page.locator('circle').get_attribute('cx')=='150'
   page.evaluate('player.next()');page.wait_for_timeout(30);page.evaluate('player.next()')
   assert page.evaluate('player.state.index')==2
   page.wait_for_timeout(450);assert page.evaluate('player.state.index')==2
   page.locator('#root').focus();page.keyboard.press('p');assert page.evaluate('player.state.index')==1
   page.locator('#editing').focus();page.keyboard.press('n');assert page.evaluate('player.state.index')==1
   page.evaluate('player.restart()');page.locator('#root').focus();page.keyboard.press('n');page.wait_for_function('player.state.index===1')
   page.evaluate('player.expand()');assert page.locator('dialog').is_visible()
   page.keyboard.press('Tab');assert page.evaluate("!!document.activeElement.closest('dialog')")
   page.screenshot(path=f'/tmp/original-player-{engine.name}-{width}-{theme}.png')
   page.keyboard.press('Escape');assert page.locator('dialog').count()==0
   assert page.locator('#outer #root').count()==1
   page.evaluate('player.play()');page.wait_for_timeout(40)
   page.evaluate("document.querySelector('#outer').hidden=true")
   page.wait_for_timeout(50);assert not page.evaluate('player.state.playing')
   x=page.locator('circle').get_attribute('cx');page.wait_for_timeout(100);assert page.locator('circle').get_attribute('cx')==x
   page.evaluate("document.querySelector('#outer').hidden=false; player.go(2)")
   page.emulate_media(reduced_motion='reduce');page.evaluate('player.next()');page.wait_for_function('player.state.index===3')
   assert not page.evaluate('player.state.playing')
   # Reduced-motion navigation commits atomically, even when a visibility or
   # explicit pause arrives before the next animation frame.
   page.evaluate('player.go(0); player.next(); player.pause()')
   assert page.evaluate('player.state.index===1 && player.state.target===null && player.state.phase==="idle"')
   page.evaluate('player.play()');page.wait_for_function('player.state.index===3 && !player.state.playing')
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   page.evaluate('player.go(1); player.next()');page.wait_for_timeout(20);page.evaluate('player.back()');assert page.evaluate('player.state.index')==1;page.evaluate('player.expand(); player.play()');page.wait_for_timeout(30);page.evaluate('player.destroy()');page.wait_for_timeout(400);assert page.locator('#root').text_content()=='';assert page.locator('dialog').count()==0
   assert not errors,errors
   print(engine.name,width,theme,'PASS')
  browser.close()


# ---------------------------------------------------------------------------
# Opt-in compact layout (layout:'compact'). The default-layout checks above are
# unchanged; everything below covers only the opt-in path.
# ---------------------------------------------------------------------------
import re, sys, json

# Intro lint (2026-10-29, compact pilot round two): one imperative sentence of at
# most 30 words, no contractions, no second person, and every number it states
# must appear in that figure's step menu, step labels, or committed narration.
INTRO_MAX_WORDS = 30
_CONTRACTION = re.compile(r"\b\w+(?:n['’]t|['’](?:ll|re|ve|d|m))\b|\b(?:it|that|what|there|here|let|who|he|she)['’]s\b", re.I)
_YOU = re.compile(r"\byou(?:r|rs|rself|selves)?\b|\byou['’]\w+", re.I)
_NUM = re.compile(r"(?<![\w.⁰¹²³⁴⁵⁶⁷⁸⁹])\d{1,3}(?:,\d{3})+(?:\.\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹]*|(?<![\w.,⁰¹²³⁴⁵⁶⁷⁸⁹])\d+(?:\.\d+)?[⁰¹²³⁴⁵⁶⁷⁸⁹]*(?![\d⁰¹²³⁴⁵⁶⁷⁸⁹])")
_NOT_IMPERATIVE = {'you', 'we', 'the', 'this', 'these', 'a', 'an', 'in', 'here', 'it', 'there', 'when', 'as', 'each'}


def numbers(text):
    return {m.group(0) for m in _NUM.finditer(text)}


def lint_intro(intro, corpus):
    """Return a list of problems; corpus = menu option texts + step labels + committed narration."""
    problems = []
    intro = (intro or '').strip()
    if not intro:
        return ['missing intro']
    words = intro.split()
    if len(words) > INTRO_MAX_WORDS:
        problems.append(f'{len(words)} words > {INTRO_MAX_WORDS}')
    if not intro.endswith('.') or re.search(r'[.!?]\s', intro):
        problems.append('not exactly one sentence')
    if _CONTRACTION.search(intro):
        problems.append('contraction: ' + _CONTRACTION.search(intro).group(0))
    if _YOU.search(intro):
        problems.append('second person: ' + _YOU.search(intro).group(0))
    if words[0].lower().strip('“"') in _NOT_IMPERATIVE or words[0].lower().endswith('ing'):
        problems.append('does not open with an imperative verb: ' + words[0])
    missing = numbers(intro) - numbers(' \n '.join(corpus))
    if missing:
        problems.append('numbers absent from steps: ' + ', '.join(sorted(missing)))
    return problems


# The lint itself: known-good and known-bad cases.
_CORPUS = ['1 / 6 · Printed point: batch 32', '2 / 6 · Drop to batch 1', '6 / 6 · Batch 1,024: intensity rises', 'S = 2¹⁵ overflows']
assert lint_intro('Move the marker from batch 32 to batch 1, then to batch 1,024, and watch S = 2¹⁵ overflow.', _CORPUS) == []
assert any('numbers' in x for x in lint_intro('Move the marker from batch 1 to batch 2,048.', _CORPUS))
assert any('numbers' in x for x in lint_intro('Move the marker to batch 512.', _CORPUS))
assert any('second person' in x for x in lint_intro('Watch how your batch moves.', _CORPUS))
assert any('contraction' in x for x in lint_intro('Watch the batch, which won’t move.', _CORPUS))
assert any('contraction' in x or 'second person' in x for x in lint_intro("You'll watch the batch move.", _CORPUS))
assert any('words' in x for x in lint_intro('Watch ' + 'the batch ' * 16 + 'move.', _CORPUS))
assert any('sentence' in x for x in lint_intro('Watch the batch. It moves.', _CORPUS))
assert any('imperative' in x for x in lint_intro('The batch moves.', _CORPUS))
assert numbers('FP16 and FP32 at batch-32, B1, 1,024 and 2¹⁵') == {'32', '1,024', '2¹⁵'}
print('compact intro lint self-test PASS')

# Static: every module that opts into the compact layout must pass an intro.
for module in sorted(base.glob('*.js')):
    text = module.read_text()
    if re.search(r"layout\s*:\s*['\"]compact['\"]", text):
        assert re.search(r"\bintro\s*:", text), (module.name, 'compact layout without intro')

# Label lint (2026-10-29, round three): the menu counter is the only step number, so a step label
# must not begin with a bare number ("4. Backprop"); write "Backprop (stage 4)".
_NUMBERED = re.compile(r'^\s*\d+[.)]\s')


def lint_label(text):
    """Problem string for a menu option ('9 / 16 · label') or a raw label; '' if fine."""
    label = re.sub(r'^\d+(?:\s*→\s*\d+)?\s*/\s*\d+\s*·\s*', '', text)
    return 'label starts with a bare number: ' + label if _NUMBERED.match(label) else ''


assert lint_label('9 / 16 · 4. Backprop') and lint_label('4. Backprop') and not lint_label('9 / 16 · Backprop (stage 4)') and not lint_label('2 / 4 · 2:1 ratio')
print('compact label lint self-test PASS')

# Coverage of the visible diagram by the pinned dock (+ cue) or the jump pill, and whether some
# step cue (bar, cue, pill, or narration heading) is on screen. Shared with the scroll sweeps.
COVER = '''()=>{const r=document.querySelector('#root'),q=s=>r.querySelector(s),dlg=r.closest('dialog');
const body=dlg?dlg.querySelector('.mlsys-original-dialog-body').getBoundingClientRect():null;
const vt=body?body.top:parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--mlsys-original-header-offset'))||0,vb=body?body.bottom:innerHeight;
const s=q('.mlsys-original-stage').getBoundingClientRect(),d=q('.mlsys-original-dock'),k=d.getBoundingClientRect(),c=q('.mlsys-original-cue'),j=q('.mlsys-original-jump');
const shown=n=>getComputedStyle(n).visibility==='visible',vis=Math.max(0,Math.min(s.bottom,vb)-Math.max(s.top,vt));
const area=b=>Math.max(0,Math.min(b.bottom,s.bottom,vb)-Math.max(b.top,s.top,vt))*Math.max(0,Math.min(b.right,s.right)-Math.max(b.left,s.left));
let covered=0;const parts=[];if(d.dataset.stuck==='bottom')parts.push(k);if(shown(c))parts.push(c.getBoundingClientRect());if(shown(j))parts.push(j.getBoundingClientRect());parts.forEach(b=>covered+=area(b));
const h=q('.mlsys-original-narration strong').getBoundingClientRect(),barIn=k.top>=vt-1&&k.bottom<=vb+1;
return {vis,stageH:s.height,frac:vis?covered/(vis*s.width):0,barIn,cue:shown(c),pill:shown(j),head:h.top>=Math.max(vt,k.bottom)-1&&h.bottom<=vb+1,explained:barIn||shown(c)||shown(j)||(h.top>=vt&&h.bottom<=vb)}}'''

COMPACT = '''([theme, intro])=>{document.body.classList.toggle('dark',theme==='dark');const root=document.querySelector('#root');const outer=document.querySelector('#outer');
const spec={root,outer,layout:'compact',intro,title:'Values moving through a diagram',scope:'Illustrative contributions; motion does not represent measured latency.',
viewControls:(()=>{const l=document.createElement('label');l.textContent='Diagram view ';const s=document.createElement('select');s.setAttribute('aria-label','Diagram view');s.append(new Option('Whole diagram','whole'));l.append(s);return l;})(),
steps:[{label:'Ready',description:'The value remains at the source.'},{label:'2. Read the source',description:'Read the source value.'},{label:'3. Compute',description:'The compute stage receives the value.'},{label:'Write',description:'Commit the result to the destination.'}],
render(index,{stage,details,assumptions}){if(!stage.firstChild)stage.innerHTML='<svg viewBox="0 0 440 300" role="img" aria-label="Value flow"><path d="M50 150 H370" stroke="#667" stroke-width="2"/><circle cx="50" cy="150" r="18" fill="#3d7b9e"/></svg>';const c=stage.querySelector('circle');c.setAttribute('cx',50+100*index);details.textContent='Committed state: '+index;window.sawAssumptions=!!assumptions;},
transition({from,to,stage,onFrame}){const c=stage.querySelector('circle');onFrame(t=>c.setAttribute('cx',50+100*(from+(to-from)*t)));}};
window.player=MLSysOriginalPlayer.create(spec);}'''
INTRO = 'Follow the value from the source to the destination and watch the circle move.'
BOX = '(s)=>{const r=document.querySelector(s).getBoundingClientRect();return {top:r.top,bottom:r.bottom,left:r.left,right:r.right,width:r.width,height:r.height}}'
with sync_playwright() as p:
 for engine in [p.chromium,p.webkit]:
  browser=engine.launch()
  for width,theme in [(1440,'light'),(390,'dark')]:
   page=browser.new_page(viewport={'width':width,'height':800})
   errors=[]
   page.on('pageerror',lambda error:errors.append(str(error)))
   page.set_content('<style>body{font:18px system-ui;margin:20px;max-width:850px}body.dark{--bs-body-bg:#202429;--bs-body-color:#eee;--bs-border-color:#63717c;--bs-secondary-color:#b0bcc7;--bs-tertiary-bg:#2b3138;background:#202429;color:#eee}svg{background:#fff}.spacer{height:1400px}</style><div class=spacer></div><div id=outer data-mlsys-movement-ms=400 data-mlsys-reading-pause-ms=120><div id=root></div></div><div class=spacer></div>')
   page.add_style_tag(path=str(base/'original-player.css'))
   page.add_script_tag(path=str(base/'original-player.js'))
   # Unmodified lessons inherit the diagram-first layout without invented prose.
   page.evaluate('''()=>{const root=document.createElement('div');document.body.append(root);const player=MLSysOriginalPlayer.create({root,outer:document.querySelector('#outer'),title:'Existing lesson',scope:'Existing assumptions',steps:[{label:'Start',description:'Existing explanation.'}],render(){}});window.defaultLayout=root.dataset.layout;window.defaultIntroHidden=root.querySelector('.mlsys-original-intro').hidden;player.destroy();root.remove();}''')
   assert page.evaluate("window.defaultLayout==='compact' && window.defaultIntroHidden"),'default layout must be compact without an empty intro'
   # A compact player without an intro is a configuration error.
   for bad in ['', '   ', None]:
    page.evaluate('''(i)=>{try{MLSysOriginalPlayer.create({root:document.createElement('div'),outer:document.querySelector('#outer'),layout:'compact',intro:i,title:'t',scope:'s',steps:[{label:'a'}],render(){}});window.threw=false}catch(e){window.threw=/intro/.test(e.message)}}''',bad)
    assert page.evaluate('window.threw'),('compact without intro must throw',bad)
   page.evaluate(COMPACT,[theme,INTRO])
   root=page.locator('#root')
   order=page.evaluate("[...document.querySelector('#root').children].map(n=>n.className)")
   assert order==['mlsys-original-title','mlsys-original-intro','mlsys-original-assumptions','mlsys-original-view-controls','mlsys-original-stage','mlsys-original-jump-anchor','mlsys-original-dock','mlsys-original-narration','mlsys-original-details'],order
   assert page.locator('.mlsys-original-view-controls select[aria-label="Diagram view"]').count()==1,'viewControls sit beside the stage'
   assert root.get_attribute('data-numbered-labels')=='2','lint hook counts labels that start with a stage number'
   assert page.evaluate('window.sawAssumptions'),'compact render receives the Assumptions disclosure'
   assert root.get_attribute('data-layout')=='compact' and root.get_attribute('aria-describedby') is None,'root does not re-read the intro'
   assert page.locator('.mlsys-original-assumptions-summary').inner_text()=='Assumptions'
   assert page.locator('.mlsys-original-shortcuts,.mlsys-original-current-cue,.mlsys-original-speed').count()==0
   group=root.get_by_role('group',name='Animation controls')
   assert group.get_attribute('title') is None,'no group tooltip'
   desc=page.locator('#'+group.get_attribute('aria-describedby'))
   assert desc.count()==1 and desc.evaluate('n=>n.hidden') and 'N next' in desc.text_content() and 'minus slower' in desc.text_content(),'shortcut description is referenced, not rendered twice'
   assert root.get_by_role('combobox',name='Playback speed').count()==0,'no speed dropdown in the bar'
   phone = page.evaluate("document.querySelector('#root').getBoundingClientRect().width")<=660
   SPEED=re.compile(r'^Speed \d(\.\d)?×$')
   visible=['Play','Back','Next','Expand']+([] if phone else ['Restart'])
   for name in visible:assert group.get_by_role('button',name=name,exact=True).count()==1,name
   assert group.get_by_role('button',name=SPEED).count()==1 and group.get_by_role('button',name=SPEED).inner_text().strip().endswith('1×'),'Speed button names and shows the rate'
   # Back/Next are single chevrons (one stroke path), not skip-to-start/end bars.
   assert page.evaluate("[...document.querySelectorAll('.mlsys-original-btn-back svg, .mlsys-original-btn-next svg')].every(g=>g.querySelectorAll('path').length===1&&!g.querySelector('[fill=\"currentColor\"]'))")
   if phone:assert group.get_by_role('button',name='Restart',exact=True).count()==0,'phone Restart lives in the popover'
   bars=page.evaluate('''()=>[...document.querySelectorAll('.mlsys-original-controls > :is(button,label)')].filter(n=>n.getClientRects().length).map(n=>{const r=n.getBoundingClientRect();return [Math.round(r.top+r.height/2),Math.round(r.width),Math.round(r.height)]})''')
   assert max(c for c,_,_ in bars)-min(c for c,_,_ in bars)<=4,('one row',bars)
   # Phones: 44 px for every target. Desktop WebKit draws a native menu at its own height (>= 24 px).
   assert all(w>=44 and h>=44 for _,w,h in bars) if phone else all(h>=(24 if w>200 else 40) for _,w,h in bars),('targets',bars)
   assert page.evaluate('''()=>[...document.querySelectorAll('.mlsys-original-controls button')].every(b=>!b.title&&b.querySelector(':scope > .mlsys-original-tip[aria-hidden="true"]')?.textContent)'''),'visible tooltips replace title'
   menu=group.get_by_role('combobox',name='Sequence',exact=True)
   assert menu.locator('option').evaluate_all('xs=>xs.map(x=>x.textContent)')==['1 / 4 · Ready','2 / 4 · Read the source (stage 2)','3 / 4 · Compute (stage 3)','4 / 4 · Write'],'one counter; stage numbers become (stage N)'
   assert all(not lint_label(x) for x in menu.locator('option').evaluate_all('xs=>xs.map(x=>x.textContent)'))
   st=page.evaluate(BOX,'.mlsys-original-stage');dk=page.evaluate(BOX,'.mlsys-original-dock');nr=page.evaluate(BOX,'.mlsys-original-narration')
   # Adjacency in flow: a viewport tall enough that nothing pins.
   page.set_viewport_size({'width':width,'height':1600})
   page.evaluate('scrollTo(0, document.querySelector("#root").getBoundingClientRect().top+scrollY-20)');page.wait_for_timeout(200)
   st=page.evaluate(BOX,'.mlsys-original-stage');dk=page.evaluate(BOX,'.mlsys-original-dock');nr=page.evaluate(BOX,'.mlsys-original-narration')
   assert st['bottom']<=dk['top']+1 and abs(dk['bottom']-nr['top'])<=1,('diagram, bar, narration adjacent',st,dk,nr)
   page.set_viewport_size({'width':width,'height':800});page.wait_for_timeout(150)
   assert page.locator('.mlsys-original-status').inner_text()==('' if phone else 'Ready') or phone
   assert root.get_attribute('data-status')=='ready','Ready before first play, not Paused'
   live=page.locator('#root [role="status"]');assert live.count()==1 and live.text_content()=='Ready','one screen-reader status'
   assert page.locator('#root .mlsys-original-narration').get_attribute('aria-live') is None,'narration is not a second live region'
   # Mid-move: the menu keeps the committed step and shows the pending one with an arrow.
   page.evaluate('player.next()');page.wait_for_timeout(60)
   assert root.get_attribute('data-status')=='moving' and root.get_attribute('data-step')=='0'
   assert menu.evaluate('s=>s.value')=='0' and menu.evaluate('s=>s.selectedOptions[0].textContent')=='1 → 2 / 4 · Read the source (stage 2)'
   assert page.locator('.mlsys-original-face-count').text_content()=='1→2'
   page.evaluate('player.pause()');assert root.get_attribute('data-status')=='paused'
   narr=page.locator('.mlsys-original-narration strong').text_content();assert narr.endswith('Next · Read the source (stage 2)'),narr
   # Paused mid-move: only the target's title, never its body (the diagram has not reached it yet).
   assert page.locator('.mlsys-original-narration p').count()==0,'mid-move narration shows the target title only'
   announced=page.locator('#root [aria-live="polite"]').text_content();assert 'Read the source' not in announced,('no departure announcement',announced)
   page.evaluate('player.next()');assert root.get_attribute('data-step')=='1' and menu.evaluate('s=>s.selectedOptions[0].textContent')=='2 / 4 · Read the source (stage 2)'
   assert page.locator('.mlsys-original-narration strong').text_content().endswith('Read the source (stage 2)'),'narration keeps the full title'
   assert page.locator('#root [aria-live="polite"]').text_content()=='Step 2 of 4: Read the source (stage 2). Read the source value.','announced once on arrival'
   assert page.locator('.mlsys-original-narration p').text_content()=='Read the source value.','the explanation appears on arrival'
   # Idle mid-sequence: no status word (visible or spoken); Ready is for step 1 only.
   assert root.get_attribute('data-status')=='idle' and page.locator('.mlsys-original-status').text_content()=='' and live.text_content()=='','no status word mid-sequence'
   page.evaluate('player.go(0)');assert root.get_attribute('data-status')=='ready' and live.text_content()=='Ready','Ready at step 1'
   page.evaluate('player.go(1)')
   page.evaluate('player.play()');page.wait_for_timeout(30);page.evaluate('player.pause()');page.wait_for_timeout(30)
   assert root.get_attribute('data-status')=='paused' and live.text_content()=='Paused','interrupted play is Paused';page.evaluate('player.go(1)')
   assert root.get_attribute('data-status')=='idle','a manual step after a pause clears the status'
   page.evaluate('player.play()');page.wait_for_function("document.querySelector('#root').dataset.status==='reading'",timeout=3000)
   page.wait_for_function('player.state.index===3 && !player.state.playing',timeout=6000);assert root.get_attribute('data-status')=='complete'
   assert group.get_by_role('button',name='Replay',exact=True).count()==1
   # Speed: −/+ work in compact; the popover offers the four rates.
   root.focus();page.keyboard.press('+');assert page.evaluate('player.state.speed')==1.5
   page.keyboard.press('-');page.keyboard.press('-');assert page.evaluate('player.state.speed')==0.5
   keys=group.get_by_role('button',name=SPEED);keys.click()
   assert keys.get_attribute('aria-expanded')=='true'
   radios=root.get_by_role('radiogroup',name='Playback speed').get_by_role('radio')
   assert radios.evaluate_all('xs=>xs.map(x=>x.value)')==['0.5','1','1.5','2'] and page.evaluate('document.activeElement.value')=='0.5'
   root.get_by_role('radio',name='2×').click();assert page.evaluate('player.state.speed')==2
   assert page.evaluate('document.activeElement.value')=='2','the clicked speed radio keeps focus (Safari)'
   page.keyboard.press('ArrowLeft');assert page.evaluate('player.state.speed')==1.5,'arrow keys keep working'
   page.keyboard.press('ArrowRight');assert page.evaluate('player.state.speed')==2
   assert keys.inner_text().strip().endswith('2×')
   pop=page.evaluate(BOX,'.mlsys-original-popover');assert pop['top']>=-1 and pop['bottom']<=801,('popover inside the viewport',pop)
   assert page.evaluate("getComputedStyle(document.querySelector('.mlsys-original-popover')).maxHeight")!='none','popover height bounded'
   assert 'Close expanded view' in page.locator('.mlsys-original-keylist').inner_text() and page.locator('.mlsys-original-keylist kbd').count()==7
   if phone:root.get_by_role('button',name='Restart',exact=True).click();assert page.evaluate('player.state.index')==0;keys.click()
   page.keyboard.press('Escape');assert keys.get_attribute('aria-expanded')=='false' and page.evaluate('document.activeElement.classList.contains("mlsys-original-options-button")')
   for name in ['Back','Expand']:
    group.get_by_role('button',name=name,exact=True).focus();assert group.get_by_role('button',name=name,exact=True).count()==1,(name,'name stable on focus')
   # Tooltips (WCAG 1.4.13): hover shows the tip, the tip itself is hoverable (it persists while the
   # pointer moves onto it), and Escape hides it without moving focus or closing anything.
   page.evaluate('player.go(1)');root.focus();nxt=group.get_by_role('button',name='Next',exact=True);tip=nxt.locator('.mlsys-original-tip')
   nxt.hover();page.wait_for_timeout(50);assert tip.is_visible(),'hover shows the tooltip'
   tb=tip.bounding_box();page.mouse.move(tb['x']+tb['width']/2,tb['y']+tb['height']/2);page.wait_for_timeout(50);assert tip.is_visible(),'tooltip persists while hovered'
   page.keyboard.press('Escape');assert not tip.is_visible() and page.evaluate("document.activeElement===document.querySelector('#root')"),'Escape dismisses the tooltip in place'
   # One Escape does one thing: inside Expand, dismissing a tooltip does not also close the dialog.
   group.get_by_role('button',name='Expand',exact=True).click();dnext=page.locator('dialog .mlsys-original-btn-next');dnext.hover();page.wait_for_timeout(50)
   assert dnext.locator('.mlsys-original-tip').is_visible();page.keyboard.press('Escape');page.wait_for_timeout(50)
   assert page.locator('dialog[open]').count()==1 and not dnext.locator('.mlsys-original-tip').is_visible(),'Escape on a tooltip keeps the dialog open'
   page.keyboard.press('Escape');page.wait_for_timeout(50);assert page.locator('dialog').count()==0,'the next Escape closes the dialog'
   page.mouse.move(2,2);root.focus()
   group.get_by_role('button',name='Expand',exact=True).hover();nxt.hover();page.wait_for_timeout(50);assert tip.is_visible(),'the tooltip returns on the next hover'
   page.mouse.move(2,2)
   # Legend: the Esc row shows the close icon, not Expand.
   assert page.evaluate("[...document.querySelectorAll('.mlsys-original-keylist li')].at(-1).querySelector('svg').innerHTML.includes('M6.5 6.5')")
   page.screenshot(path=f'/tmp/original-player-compact-{engine.name}-{width}-{theme}.png')
   # Sticky dock: diagram on screen with the bar below the fold. Tall diagram (desktop): the bar
   # pins with its one-line cue, covering at most 15% of the visible diagram. Short diagram
   # (phone): pinning would exceed the cap, so the bar flows and the jump pill carries the cue.
   page.evaluate('player.go(1)')
   y=page.evaluate("document.querySelector('.mlsys-original-stage').getBoundingClientRect().bottom+scrollY-innerHeight+10")
   page.evaluate(f'scrollTo(0,{y})');page.wait_for_timeout(250)
   dock=page.locator('.mlsys-original-dock');cover=page.evaluate(COVER)
   assert cover['explained'] and cover['frac']<=0.15+1e-3,cover
   if not phone:
    assert dock.get_attribute('data-stuck')=='bottom' and dock.get_attribute('data-cue')=='above',(dock.get_attribute('data-stuck'),dock.get_attribute('data-cue'))
    dk=page.evaluate(BOX,'.mlsys-original-dock');ck=page.evaluate(BOX,'.mlsys-original-cue')
    assert abs(dk['bottom']-800)<=1 and abs(ck['bottom']-dk['top'])<=1 and ck['height']<=32 and page.locator('.mlsys-original-cue').inner_text()=='Read the source (stage 2)',ck
    # Pinned: no status word, and Speed shows only its rate, so the menu keeps room for the title.
    assert not page.locator('.mlsys-original-status').is_visible() and keys.locator('.mlsys-original-control-label').evaluate('n=>n.getBoundingClientRect().width')<=1 and keys.is_visible(),'pinned: no status, Speed shows its rate'
    # A keep-visible readout at the diagram's bottom edge: the bar no longer pins over it (no pinned
    # bar without its title either); the pill rides above the readout instead of covering it.
    page.evaluate('''()=>{const svg=document.querySelector('#root .mlsys-original-stage svg');const r=document.createElementNS('http://www.w3.org/2000/svg','rect');
      r.setAttribute('x','0');r.setAttribute('y','268');r.setAttribute('width','440');r.setAttribute('height','32');r.setAttribute('fill','#fc8');r.id='keep';r.setAttribute('data-mlsys-keep','');svg.append(r);scrollBy(0,1)}''')
    page.wait_for_timeout(250);cover=page.evaluate(COVER)
    assert dock.get_attribute('data-stuck')!='bottom' and dock.get_attribute('data-hold')=='natural',('no pin over a keep region',dock.get_attribute('data-stuck'))
    kp=page.evaluate(BOX,'#keep');jp=page.evaluate(BOX,'.mlsys-original-jump')
    assert page.locator('.mlsys-original-jump').is_visible() and jp['bottom']<=kp['top']+0.5,('pill rides above the keep region',jp,kp)
    page.evaluate("document.querySelector('#keep').remove()");page.evaluate('scrollBy(0,-1)');page.wait_for_timeout(250)
   else:
    assert dock.get_attribute('data-hold')=='natural' and page.locator('.mlsys-original-jump').is_visible(),cover
    assert page.locator('.mlsys-original-jump').inner_text().replace('\n',' ').startswith('2/4'),page.locator('.mlsys-original-jump').inner_text()
   # The pill is a named, keyboard-reachable button with a down arrow (not a menu chevron); it moves
   # focus to the narration. Pointer-sized hit area: at least 44 px tall.
   jump=page.locator('.mlsys-original-jump')
   if jump.is_visible():
    assert root.get_by_role('button',name='Go to step explanation').count()==1 and jump.get_attribute('tabindex') is None and jump.get_attribute('aria-hidden') is None
    assert jump.evaluate("b=>b.querySelector('svg').innerHTML.includes('M12 4.5v15')"),'down arrow icon'
    assert jump.evaluate("b=>parseFloat(getComputedStyle(b,'::after').height)")>=44,'44 px hit area'
    page.evaluate("document.querySelector('#root .mlsys-original-stage').tabIndex=-1");root.focus()
    # WebKit's Tab skips buttons unless the OS "Tab to all controls" setting is on; there, check the
    # button is in the tab order (tabIndex 0) and focus it directly.
    assert jump.evaluate('b=>b.tabIndex')==0
    reach=engine.name=='webkit' and (jump.focus() or True) and page.evaluate("document.activeElement===document.querySelector('.mlsys-original-jump')")
    for _ in range(0 if reach else 4):
     page.keyboard.press('Tab')
     if page.evaluate("document.activeElement===document.querySelector('.mlsys-original-jump')"):reach=True;break
    assert reach,'Tab reaches the pill'
    page.keyboard.press('Enter');page.wait_for_timeout(700)
    assert page.evaluate("document.activeElement===document.querySelector('#root .mlsys-original-narration')"),'the pill moves focus to the explanation'
    nb=page.evaluate(BOX,'#root .mlsys-original-narration');assert nb['top']>=0 and nb['bottom']<=801,('narration scrolled into view',nb)
    page.evaluate(f'scrollTo(0,{y})');page.wait_for_timeout(250)
   # Scrolled past the diagram -> the bar sits at the top; scrolling back restores the bottom state.
   page.evaluate("scrollTo(0, document.querySelector('.mlsys-original-narration').getBoundingClientRect().bottom+scrollY+5)");page.wait_for_timeout(250)
   assert dock.get_attribute('data-stuck') in ('top','none')
   page.evaluate(f'scrollTo(0,{y})');page.wait_for_timeout(250);assert dock.get_attribute('data-stuck')==('bottom' if not phone else 'none')
   group.get_by_role('button',name='Expand',exact=True).click();assert page.locator('dialog .mlsys-original-compact .mlsys-original-stage').is_visible()
   page.locator('dialog .mlsys-original-dialog-body').evaluate('d=>d.scrollTop=d.scrollHeight');page.wait_for_timeout(200)
   cl=page.evaluate(BOX,'.mlsys-original-dialog-close');dk=page.evaluate(BOX,'dialog .mlsys-original-dock');sb=page.evaluate(BOX,'dialog .mlsys-original-dialog-body')
   assert dk['top']>=cl['bottom']-1 and sb['top']>=cl['bottom']-1,('dialog dock and body below the Close bar',cl,dk,sb)
   page.keyboard.press('Escape');assert page.locator('dialog').count()==0 and page.locator('#outer #root').count()==1
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   page.emulate_media(media='print');assert not root.is_visible();page.emulate_media(media='screen')
   page.evaluate('player.destroy()');assert root.text_content()=='' and root.get_attribute('data-layout') is None and root.get_attribute('data-status') is None
   assert not errors,errors
   print('compact',engine.name,width,theme,'PASS')
  browser.close()

FOCUS_MARGIN = 8
FOCUS_IN_FRAME = '''(r,pad)=>{const st=r.querySelector('.mlsys-original-stage'),f=st.querySelector('svg').getBoundingClientRect();
 const foci=[...st.querySelectorAll('[data-mlsys-focus]')].filter(n=>getComputedStyle(n).visibility!=='hidden'&&n.getClientRects().length).map(n=>n.getBoundingClientRect()).filter(b=>b.width||b.height);
 const m=foci.map(b=>Math.min(b.left-f.left,f.right-b.right,b.top-f.top,f.bottom-b.bottom));return {n:foci.length,min:m.length?Math.min(...m):null}}'''
# Optional: lint the intros of emitted compact mounts in built chapters.
#   check_original_player.py --chapter-links links.json   ({"vol1:fig-x": "http://.../ch.html?view=explore#fig-x"})
if '--chapter-links' in sys.argv:
 links=json.loads(Path(sys.argv[sys.argv.index('--chapter-links')+1]).read_text())
 linted=0
 with sync_playwright() as p:
  browser=p.chromium.launch()
  for key,url in links.items():
   fid=url.split('#',1)[1]
   page=browser.new_page(viewport={'width':1440,'height':900},reduced_motion='reduce')
   page.goto(url,wait_until='domcontentloaded')
   page.wait_for_function('(id)=>document.getElementById(id)?.dataset.mlsysExploreState==="ready"',arg=fid)
   comp=page.locator('#'+fid+' .mlsys-original-player')
   if comp.count()!=1 or comp.get_attribute('data-layout')!='compact':page.close();continue
   intro=comp.locator('.mlsys-original-intro').inner_text()
   menu=comp.locator('.mlsys-original-sequence select');corpus=menu.locator('option').evaluate_all('xs=>xs.map(x=>x.textContent)')
   for i in range(len(corpus)):
    menu.select_option(str(i));corpus.append(comp.locator('.mlsys-original-narration').inner_text())
   problems=lint_intro(intro,corpus)
   assert not problems,(key,intro,problems)
   labels=[x for x in corpus if ' · ' in x[:12]]+comp.locator('.mlsys-original-narration strong').all_inner_texts()
   bad=[lint_label(x) for x in labels if lint_label(x)]
   assert not bad and comp.get_attribute('data-numbered-labels') is None,(key,'step labels must not start with a bare number',bad,comp.get_attribute('data-numbered-labels'))
   # Protection contract (2026-10-30, round four): every committed state of a compact module marks its
   # highlighted region with data-mlsys-focus inside the stage (the dock and pill avoid it), and in every
   # view the module offers, each focus region plus FOCUS_MARGIN px lies inside the drawing's frame
   # (no Follow crop or edge clips what the step is about).
   view=comp.locator('select[aria-label="Diagram view"]')
   for v in (view.locator('option').evaluate_all('xs=>xs.map(x=>x.value)') if view.count() else [None]):
    if v:view.select_option(v)
    for i in range(menu.locator('option').count()):
     menu.select_option(str(i))
     g=comp.evaluate(FOCUS_IN_FRAME,FOCUS_MARGIN)
     assert g['n']>0,(key,'view',v,'step',i+1,'has no visible data-mlsys-focus region')
     assert g['min']>=FOCUS_MARGIN,(key,'view',v,'step',i+1,'focus region within',g['min'],'px of the frame edge')
   print('intro',key,len(intro.split()),'words; focus regions in frame PASS')
   linted+=1;page.close()
  browser.close()
 assert linted,'no compact mounts found in --chapter-links'
