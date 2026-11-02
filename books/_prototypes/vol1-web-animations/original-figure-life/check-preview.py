from playwright.sync_api import sync_playwright
import argparse
from pathlib import Path
import json
parser=argparse.ArgumentParser(description='Inspect the original-artwork chapter preview in a headless browser.')
parser.add_argument('--url', default='http://127.0.0.1:8882/_build/original-figure-life/vol1/06_nn_architectures/06_nn_architectures.html#fig-cnn')
parser.add_argument('--screenshots', type=Path, default=Path('/tmp/original-convolution-review'))
args=parser.parse_args();args.screenshots.mkdir(parents=True,exist_ok=True)
url=args.url
with sync_playwright() as p:
    browser=p.chromium.launch()
    results=[]
    for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
        page=browser.new_page(viewport={'width':width,'height':1000},color_scheme=theme)
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(url)
        outer=page.locator('#fig-cnn');outer.scroll_into_view_if_needed()
        page.wait_for_function("document.querySelector('#fig-cnn').dataset.originalLifeReady === 'true'")
        outer.get_by_role('tab',name='Explore',exact=True).click()
        panel=outer.locator('.life-panel');panel.focus()
        assert panel.get_attribute('data-step')=='0'
        assert panel.locator('.life-keys').is_visible()
        page.evaluate("document.querySelector('#fig-cnn').scrollIntoView({block:'start'})")
        header_box=page.locator('#quarto-header').bounding_box()
        header_bottom=max(0,header_box['y']+header_box['height'])
        assert panel.locator('.life-controls').bounding_box()['y'] >= header_bottom
        for step in [1,2,3,4,5,6]:
            panel.press('n')
            if step==2 and width==1440 and theme=='light':
                page.wait_for_timeout(700)
                page.screenshot(path=str(args.screenshots/'weighted-contributions-in-motion.png'))
            page.wait_for_function("document.querySelector('#fig-cnn .life-panel').dataset.moving === 'false'")
            assert panel.get_attribute('data-step')==str(step)
            assert panel.get_attribute('data-running')=='false'
            if step in [2,3,6]:
                page.evaluate("document.querySelector('#fig-cnn').scrollIntoView({block:'start'})")
                page.screenshot(path=str(args.screenshots/f'{width}-{theme}-step{step}.png'))
        assert panel.get_attribute('data-outputs')=='2'
        # The blue connection must begin on the moving orange window.
        alignment=panel.locator('svg').evaluate(r"""svg => {
            const orange=[...svg.querySelectorAll('path')].find(p=>p.getAttribute('fill')==='rgb(100%, 50%, 0%)');
            const blue=[...svg.querySelectorAll('path')].find(p=>p.getAttribute('stroke')==='rgb(0%, 0%, 100%)');
            const first=p=>{const n=p.getAttribute('d').match(/-?\d*\.?\d+/g).map(Number);return new DOMPoint(n[0],n[1]).matrixTransform(p.getScreenCTM());};
            const a=first(orange),b=first(blue);return Math.hypot(a.x-b.x,a.y-b.y);
        }""")
        assert alignment<0.5,alignment

        panel.press('p');assert panel.get_attribute('data-step')=='5'
        panel.press('r');assert panel.get_attribute('data-step')=='0'
        menu=panel.get_by_role('combobox');menu.select_option('75')
        assert panel.get_attribute('data-outputs')=='25'
        page.screenshot(path=str(args.screenshots/f'{width}-{theme}-complete.png'))
        menu.select_option('0');panel.focus();panel.press('k')
        page.wait_for_timeout(300);panel.press('k')
        before=panel.get_attribute('data-step');page.wait_for_timeout(2000)
        assert panel.get_attribute('data-step')==before
        assert panel.get_attribute('data-moving')=='true'
        panel.press('k');page.wait_for_function("document.querySelector('#fig-cnn .life-panel').dataset.moving === 'false'")
        outer.get_by_role('tab',name='Diagram',exact=True).click()
        assert panel.get_attribute('data-running')=='false'
        assert not panel.is_visible()
        outer.get_by_role('tab',name='Explore',exact=True).click()
        page.emulate_media(reduced_motion='reduce');panel.focus();panel.press('r');panel.press('n')
        assert panel.get_attribute('data-moving')=='false'
        assert panel.get_attribute('data-step')=='1'
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        assert not errors,errors
        results.append({'width':width,'theme':theme,'passed':True})
        # Printing restores the complete static drawing even from Explore.
        page.emulate_media(media='print')
        assert not panel.is_visible()
        assert outer.locator('.mlsys-diagram-panel').is_visible()
        page.close()
    print(json.dumps(results,indent=2));browser.close()
