"""Deterministic browser checks for per-figure playback speed; no book models are replaced.

Uses a synthetic four-endpoint scene and a manually advanced animation-frame
clock. This isolates movement and reading-time scaling from machine load.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
EXT = ROOT / 'books/_extensions/mlsysbook-ext/figure-explore'
RATES = ['0.5', '1', '1.5', '2']

FIXTURE = r'''() => {
  let now = 0, next = 0;
  const pending = new Map();
  performance.now = () => now;
  window.requestAnimationFrame = callback => {pending.set(++next, callback); return next;};
  window.cancelAnimationFrame = id => pending.delete(id);
  window.elapse = delta => { now += delta; };
  window.tick = delta => {
    now += delta;
    const callbacks = [...pending.values()]; pending.clear();
    callbacks.forEach(callback => callback(now));
  };
  window.players = {};
  window.createPlayer = id => {
    const root = document.getElementById(id);
    const outer = root.parentElement;
    return players[id] = MLSysOriginalPlayer.create({
      root, outer, title: 'Synthetic speed contract', scope: 'Clock test, not a measured ML workload.',
      steps: ['Ready', 'Read', 'Compute', 'Store'].map(label => ({label, description:label})),
      render(index, {stage, details}) {
        stage.innerHTML = '<svg viewBox="0 0 100 25"><circle r="5" cy="12"/></svg>';
        stage.querySelector('circle').setAttribute('cx', 10 + index * 25);
        details.textContent = 'Committed ' + index;
        root.dataset.committed = index;
        root.dataset.progress = '0';
      },
      transition({from,to,stage,animate,onFrame}) {
        root.dataset.transitionCalls = Number(root.dataset.transitionCalls || 0) + 1;
        const circle = stage.querySelector('circle');
        const animation = animate(circle, [{opacity:1},{opacity:0.5}]);
        onFrame(progress => {
          root.dataset.progress = progress;
          root.dataset.animationTime = animation.currentTime;
          circle.setAttribute('cx', 10 + 25 * (from + (to-from)*progress));
        });
      }
    });
  };
  createPlayer('first'); createPlayer('second');
}'''


def run(page, expanded_screenshot=None):
    page.set_content('''<style>body{font:16px system-ui;margin:8px}.mlsys-original-player{margin:0!important}
      svg{width:100%;height:45px}
      @media(prefers-color-scheme:dark){body{color:#eee;background:#202429;--bs-body-bg:#202429;
      --bs-body-color:#eee;--bs-border-color:#63717c;--bs-secondary-color:#b0bcc7;
      --bs-tertiary-bg:#2b3138}circle{fill:#84c6ed}}</style>
      <div data-mlsys-movement-ms="1000" data-mlsys-reading-pause-ms="500"><div id="first"></div></div>
      <div data-mlsys-movement-ms="1000" data-mlsys-reading-pause-ms="500"><div id="second"></div></div>''')
    page.add_style_tag(path=str(EXT / 'original-player.css'))
    page.add_script_tag(path=str(EXT / 'original-player.js'))
    page.evaluate(FIXTURE)
    # Resolve by visible accessible name; no rate mutation hook is needed.
    first = page.locator('#first')
    second = page.locator('#second')
    speed = first.get_by_role('combobox', name='Playback speed', exact=True)
    second_speed = second.get_by_role('combobox', name='Playback speed', exact=True)
    assert speed.locator('option').evaluate_all('(xs)=>xs.map(x=>x.value)') == RATES
    assert speed.input_value() == second_speed.input_value() == '1'
    assert page.evaluate('players.first.state.speed===1 && players.second.state.speed===1')
    assert page.evaluate('Object.keys(localStorage).length') == 0
    speed.select_option('1.5')
    assert second_speed.input_value() == '1'
    # Native editing controls retain their keys; region shortcuts bound at 0.5/2.
    speed.focus()
    page.keyboard.press('+')
    assert speed.input_value() == '1.5'
    for selector in ['#first input', '#first textarea', '#first [contenteditable]']:
        page.evaluate('''selector=>{let el=document.querySelector(selector);if(!el){
          el=document.createElement(selector.endsWith('input')?'input':selector.endsWith('textarea')?'textarea':'div');
          if(el.tagName==='DIV')el.contentEditable='true';document.querySelector('#first').append(el);}}''', selector)
        page.locator(selector).focus()
        page.keyboard.press('-')
        assert speed.input_value() == '1.5'
    first.focus()
    for _ in range(6): page.keyboard.press('+')
    assert speed.input_value() == '2'
    for _ in range(6): page.keyboard.press('-')
    assert speed.input_value() == '0.5'
    for rate in RATES:
        page.evaluate('players.first.go(0)')
        speed.select_option(rate)
        page.evaluate('players.first.next(); tick(0); tick(100)')
        assert abs(float(first.get_attribute('data-progress')) - float(rate) * .1) < 1e-8
        page.evaluate('players.first.next()')
        assert page.evaluate('players.first.state.index') == 1
    page.evaluate('players.first.go(0)')
    speed.select_option('1')
    page.evaluate("players.first.next(); tick(0); tick(100)")
    assert page.evaluate("players.first.state.index===0 && players.first.state.target===1")
    before = float(first.get_attribute('data-progress'))
    calls = first.get_attribute('data-transition-calls')
    page.evaluate('elapse(20)')  # Time since last frame belongs to the old rate.
    speed.select_option('2')
    assert float(first.get_attribute('data-progress')) == before
    assert first.get_attribute('data-transition-calls') == calls
    page.evaluate('tick(100)')
    after = float(first.get_attribute('data-progress'))
    assert abs(after - before - .22) < 1e-8, (before, after)
    assert page.evaluate('players.first.state.index') == 0
    # Pause freezes both semantic onFrame and WAAPI time across speed changes.
    page.evaluate('players.first.pause()')
    paused = first.locator('circle').get_attribute('cx')
    animation_time = first.get_attribute('data-animation-time')
    speed.select_option('0.5')
    page.evaluate('tick(2000)')
    assert first.locator('circle').get_attribute('cx') == paused
    assert first.get_attribute('data-animation-time') == animation_time
    assert page.evaluate('players.first.state.index') == 0
    # N completes just the pending endpoint; no auto-advance while paused.
    first.focus(); page.keyboard.press('n')
    assert page.evaluate('players.first.state.index') == 1
    page.evaluate('tick(3000)')
    assert page.evaluate('players.first.state.index') == 1
    first.focus(); page.keyboard.press('p')
    assert page.evaluate('players.first.state.index') == 0
    page.evaluate('players.first.restart()')
    assert speed.input_value() == '0.5'
    # 2x movement takes 500ms; 500ms reading hold takes 250ms.
    speed.select_option('2')
    page.evaluate('players.first.play(); tick(0); tick(499)')
    assert page.evaluate('players.first.state.index') == 0
    page.evaluate('tick(1)')
    assert page.evaluate("players.first.state.index===1 && players.first.state.phase==='reading'")
    page.evaluate('tick(0); tick(249)')
    assert page.evaluate("players.first.state.phase==='reading'")
    page.evaluate('tick(1)')
    assert page.evaluate("players.first.state.index===1 && players.first.state.target===2")
    page.evaluate('players.first.pause(); players.first.restart()')
    # Preferences are in-memory per instance, and restart preserves the selection.
    assert speed.input_value() == '2' and second_speed.input_value() == '1'
    # Full-window inspection moves this same component, pauses its clock, and
    # leaves both the pending action and chosen rate intact on open/close.
    page.evaluate('players.first.next(); tick(0); tick(100)')
    first.get_by_role('button', name='Expand', exact=True).focus()
    first.get_by_role('button', name='Expand', exact=True).evaluate('(button)=>button.click()')
    assert page.locator('dialog').is_visible()
    assert page.evaluate('players.first.state.index===0 && players.first.state.target===1 && !players.first.state.playing')
    expanded_x = first.locator('circle').get_attribute('cx')
    page.evaluate('tick(1000)')
    assert first.locator('circle').get_attribute('cx') == expanded_x
    assert speed.input_value() == '2'
    assert page.evaluate('''() => {
      const box=document.querySelector('dialog').getBoundingClientRect();
      return box.width>=innerWidth-2 && box.height>=innerHeight-2 && Math.abs(box.left)<=1 && Math.abs(box.top)<=1;
    }''')
    page.evaluate("document.querySelector('#first .mlsys-original-stage').style.minHeight='2200px'; document.querySelector('dialog').scrollTop=650; tick(0)")
    assert page.evaluate('''() => {
      const controls=document.querySelector('#first .mlsys-original-controls').getBoundingClientRect();
      const close=document.querySelector('.mlsys-original-dialog-close').getBoundingClientRect();
      return close.top>=-1 && close.bottom<=innerHeight && controls.top>=close.bottom-1 && controls.bottom<innerHeight;
    }'''), page.evaluate('''() => ({close:document.querySelector('.mlsys-original-dialog-close').getBoundingClientRect().toJSON(),controls:document.querySelector('#first .mlsys-original-controls').getBoundingClientRect().toJSON()})''')
    if expanded_screenshot:
        page.screenshot(path=str(expanded_screenshot))
    page.keyboard.press('Escape'); page.evaluate('tick(0)')
    assert page.locator('dialog').count() == 0
    assert page.evaluate('players.first.state.index===0 && players.first.state.target===1 && !players.first.state.playing')
    assert first.locator('circle').get_attribute('cx') == expanded_x
    assert speed.input_value() == '2'
    assert first.get_by_role('button', name='Expand', exact=True).evaluate('(button)=>button===document.activeElement')
    page.evaluate("document.querySelector('#first .mlsys-original-stage').style.minHeight=''; players.first.restart()")
    page.emulate_media(reduced_motion='reduce')
    page.wait_for_function("matchMedia('(prefers-reduced-motion: reduce)').matches", polling=20)
    page.wait_for_timeout(50)  # Settle the existing MediaQueryList change event.
    page.evaluate('players.first.next(); players.first.pause()')
    assert page.evaluate("players.first.state.index===1 && players.first.state.target===null"), page.evaluate("players.first.state")
    first.focus(); page.keyboard.press('p')
    assert page.evaluate('players.first.state.index') == 0
    assert speed.input_value() == '2'
    page.emulate_media(reduced_motion='no-preference')
    page.evaluate('players.first.play(); tick(0); tick(100); players.first.destroy(); tick(5000)')
    assert first.text_content() == ''
    assert page.evaluate('players.first.state.destroyed')
    page.evaluate("createPlayer('first')")
    speed = first.get_by_role('combobox', name='Playback speed', exact=True)
    assert speed.input_value() == '1'
    assert page.evaluate('Object.keys(localStorage).length') == 0
    assert second_speed.input_value() == '1'
    first.focus(); page.keyboard.press('+')
    assert speed.input_value() == '1.5'  # Old destroyed listener must not run twice.
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')



def actual_mounts(args):
    """Check the emitted HTML/player UI without injecting or replacing assets."""
    links = json.loads(args.chapter_links.read_text())
    if args.keys:
        selected = args.keys.split(',')
        missing = set(selected) - set(links)
        if missing:
            raise ValueError('Unknown chapter keys: ' + ', '.join(sorted(missing)))
        links = {key: links[key] for key in selected}
    records = []
    with sync_playwright() as p:
        for name in (['chromium', 'webkit'] if args.engine == 'all' else [args.engine]):
            browser = getattr(p, name).launch()
            for key, url in links.items():
                figure = url.split('#', 1)[1]
                for width, theme in [(1440, 'light'), (390, 'light'), (1440, 'dark'), (390, 'dark')]:
                    page = browser.new_page(viewport={'width':width, 'height':900}, color_scheme=theme)
                    errors = []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    page.goto(url, wait_until='domcontentloaded')
                    page.wait_for_function('(id)=>document.getElementById(id)?.dataset.mlsysExploreState==="ready"', arg=figure)
                    # Allow the now-visible original artwork to complete its initial responsive layout.
                    page.evaluate('()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
                    outer = page.locator('[id="' + figure + '"]')
                    comp = outer.locator('.mlsys-original-player,.life-panel')
                    component_id = comp.get_attribute('id')
                    assert component_id, (key, 'stable component identity')
                    comp = page.locator('[id="' + component_id + '"]')
                    speed = comp.get_by_role('combobox', name=re.compile(r'^(Playback|Animation) speed$'))
                    assert speed.is_visible(), key
                    assert speed.locator('option').evaluate_all('(xs)=>xs.map(x=>x.value)') == RATES, key
                    assert speed.input_value() == '1', key
                    initial = comp.get_attribute('data-step')
                    stage = comp.locator('.mlsys-original-stage,.life-workspace')
                    assert stage.count() == 1, (key, 'stage selector')
                    baseline = stage.inner_html()
                    for rate in RATES:
                        speed.select_option(rate)
                        assert comp.get_attribute('data-step') == initial, (key, rate, 'endpoint changed')
                        assert stage.inner_html() == baseline, (key, rate, 'paused diagram changed')
                        assert comp.get_attribute('data-playing') in (None, 'false'), key
                        assert comp.get_attribute('data-running') in (None, 'false'), key
                    comp.focus(); page.keyboard.press('+')
                    assert speed.input_value() == '2', key
                    page.keyboard.press('-')
                    assert speed.input_value() == '1.5', key
                    # Restart retains the reader preference; no target math is touched.
                    # CNN intentionally disables its redundant Restart button at0.
                    # The documented R shortcut exercises the same reset contract.
                    comp.focus(); page.keyboard.press('r')
                    assert speed.input_value() == '1.5', key
                    assert comp.get_attribute('data-step') == '0', key
                    expand = comp.get_by_role('button', name='Expand', exact=True)
                    expand.click()
                    dialog = page.locator('dialog[open]')
                    assert dialog.count() == 1, key
                    assert comp.get_attribute('data-step') == '0', (key, 'expand endpoint')
                    assert speed.input_value() == '1.5', (key, 'expand speed')
                    assert comp.get_attribute('data-playing') in (None, 'false'), key
                    assert comp.get_attribute('data-running') in (None, 'false'), key
                    bounds = dialog.evaluate('(node)=>{const r=node.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height,iw:innerWidth,ih:innerHeight}}')
                    assert bounds['w'] >= bounds['iw']-2 and bounds['h'] >= bounds['ih']-2, (key, bounds)
                    assert abs(bounds['x'])<=1 and abs(bounds['y'])<=1, (key, bounds)
                    close = dialog.get_by_role('button', name=re.compile(r'^Close expanded (view|animation)$'))
                    assert close.is_visible(), key
                    dialog.evaluate('(node)=>node.scrollTop=Math.min(500,node.scrollHeight-node.clientHeight)')
                    page.evaluate('()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
                    controls = comp.locator('.mlsys-original-controls,.life-controls')
                    cr = controls.bounding_box(); xr = close.bounding_box()
                    assert cr and xr and xr['y']>=-1 and xr['y']+xr['height']<=900, (key, 'close', xr)
                    assert cr['y']>=xr['y']+xr['height']-1 and cr['y']+cr['height']<=900, (key, 'toolbar', cr, xr)
                    screenshot = None
                    if name == 'chromium' and width == 1440 and theme == 'light':
                        screenshot = str(args.output / (key.replace(':','--')+'-expanded.png'))
                        page.screenshot(path=screenshot)
                    page.keyboard.press('Escape')
                    page.wait_for_function('()=>!document.querySelector("dialog[open]")')
                    page.evaluate('()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
                    assert comp.get_attribute('data-step') == '0' and speed.input_value() == '1.5', key
                    assert expand.evaluate('(button)=>button===document.activeElement'), (key, 'focus restore')
                    assert comp.get_attribute('data-playing') in (None, 'false'), key
                    assert comp.get_attribute('data-running') in (None, 'false'), key
                    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'), key
                    assert not errors, (key, errors)
                    if key == next(iter(links)):
                        comp.screenshot(path=str(args.output / f'actual-{name}-{width}-{theme}.png'))
                    records.append(dict(key=key, engine=name, width=width, height=900, theme=theme, passed=True, url=url, endpoint='0', speed='1.5', speed_choices=RATES, bounds=bounds, expanded_screenshot=screenshot, checks=['default_rate','rate_choices','paused_scene_identity','scoped_speed_shortcuts','restart_keeps_rate','expanded_viewport','expanded_paused_state','sticky_close_toolbar','Escape_focus_restore']))
                    (args.output/'actual-progress.json').write_text(json.dumps({'profiles':records}, indent=2)+'\n')
                    page.close()
            browser.close()
    data = {'schema':1, 'passed':True,
            'scope':'Unmodified emitted chapter mounts: speed choices/paused diagram identity/shortcuts/restart/full-window bounds/sticky controls/Escape.',
            'runtime_sha256':hashlib.sha256((EXT/'original-player.js').read_bytes()).hexdigest(),
            'stylesheet_sha256':hashlib.sha256((EXT/'original-player.css').read_bytes()).hexdigest(),
            'chapter_links_sha256':hashlib.sha256(args.chapter_links.read_bytes()).hexdigest(), 'profiles':records}
    (args.output/'actual-report.json').write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps({'figures':len(links), 'profiles':len(records), 'passed':True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/original-player-speed'))
    parser.add_argument('--chapter-links', type=Path, help='Optional key-to-actual-chapter-URL JSON; runs emitted speed UI checks instead.')
    parser.add_argument('--keys', help='Optional comma-separated keys within --chapter-links.')
    parser.add_argument('--engine', choices=['all', 'chromium', 'webkit'], default='all')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.chapter_links:
        actual_mounts(args)
        return
    records = []
    with sync_playwright() as p:
        for name in ['chromium', 'webkit']:
            browser = getattr(p, name).launch()
            for width, theme in [(1440, 'light'), (390, 'light'), (1440, 'dark'), (390, 'dark')]:
                page = browser.new_page(viewport={'width':width, 'height':900}, color_scheme=theme)
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                # An origin permits checking that no preference is persisted.
                page.route('http://speed.test/', lambda route: route.fulfill(body='<html></html>', content_type='text/html'))
                page.goto('http://speed.test/')
                run(page, args.output / f'expanded-{name}-{width}-{theme}.png')
                assert not errors, errors
                page.screenshot(path=str(args.output / f'{name}-{width}-{theme}.png'))
                records.append({'engine':name, 'width':width, 'theme':theme, 'passed':True})
                page.close()
            browser.close()
    data = {'runtime_sha256':hashlib.sha256((EXT/'original-player.js').read_bytes()).hexdigest(),
            'stylesheet_sha256':hashlib.sha256((EXT/'original-player.css').read_bytes()).hexdigest(),
            'scope':'Synthetic deterministic movement/reading clock plus full-window dialog contract; production chapter mount/visual review separate.',
            'viewport_height':900,
            'checks':['four_rates','per_figure_defaults','no_persistence','editing_exclusion','shortcut_bounds','midflight_old_rate_charge','paused_frame_and_WAAPI_freeze','atomic_N_P','restart_retains_rate','scaled_reading_pause','reduced_motion','destroy_cleanup','dialog_viewport_bounds','dialog_pause_and_state','sticky_close_and_toolbar','Escape_focus_restore'],
            'profiles':records}
    (args.output/'report.json').write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps(data))


if __name__ == '__main__':
    main()
