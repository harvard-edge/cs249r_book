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
   page.evaluate('''theme=>{document.body.classList.toggle('dark',theme==='dark');const root=document.querySelector('#root');const outer=document.querySelector('#outer');window.player=MLSysOriginalPlayer.create({root,outer,title:'Values moving through a diagram',scope:'Illustrative contributions; motion does not represent measured latency.',steps:[{label:'Ready',description:'The value remains at the source.'},{label:'Read',description:'Read the source value.'},{label:'Compute',description:'The compute stage receives the value.'},{label:'Write',description:'Commit the result to the destination.'}],render(index,{stage,details}){if(!stage.firstChild)stage.innerHTML='<svg viewBox="0 0 440 140" role="img" aria-label="Value flow"><path d="M50 70 H370" stroke="#667" stroke-width="2"/><circle cx="50" cy="70" r="18" fill="#3d7b9e"/></svg>';const c=stage.querySelector('circle');c.setAttribute('cx',50+100*index);details.textContent='Committed state: '+index;},transition({from,to,stage,animate,onFrame}){const c=stage.querySelector('circle');animate(c,[{opacity:1},{opacity:.3},{opacity:1}]);onFrame(t=>{c.setAttribute('cx',50+100*(from+(to-from)*t));root.dataset.progress=t;});}});}''',theme)
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
