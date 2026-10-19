"""Check values embedded in the original drawing and the optional calculation."""
from playwright.sync_api import sync_playwright
import argparse
from pathlib import Path
import json

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url',default='http://127.0.0.1:8882/_build/original-figure-life/vol1/06_nn_architectures/06_nn_architectures.html?view=explore#fig-cnn')
parser.add_argument('--screenshots',type=Path,default=Path('/tmp/original-convolution-review'))
args=parser.parse_args();args.screenshots.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch();report=[]
    for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
        page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme,reduced_motion='reduce')
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(args.url)
        panel=page.locator('#fig-cnn .life-panel');panel.wait_for(state='visible')
        svg=panel.locator('.life-art svg');paths=svg.locator('path').count()
        assert svg.locator('[data-input-row]').count()==49
        assert svg.locator('[data-kernel-row]').all_text_contents()==['-1','0','1']*3
        assert svg.locator('[data-input-row="0"]').all_text_contents()==['0','0','1','0','0','0','0']
        assert svg.locator('[data-output-index]').count()==0
        inset=panel.locator('.life-values')
        assert inset.get_attribute('open') is None
        assert not inset.locator('.life-value-scope').is_visible()
        menu=panel.get_by_role('combobox')
        for n in range(25):
            row,column=divmod(n,5)
            expected=sum(int(column+c==2)*[-1,0,1][c] for r in range(3) for c in range(3))
            menu.select_option(str(n*3+1))
            assert svg.locator('[data-diagram-sum]').count()==0
            assert svg.locator('[data-output-index]').count()==n
            menu.select_option(str(n*3+2))
            assert svg.locator('[data-diagram-sum]').get_attribute('data-diagram-sum')==str(expected)
            assert svg.locator('[data-output-index]').count()==n
            menu.select_option(str(n*3+3))
            assert svg.locator(f'[data-output-index="{n}"]').text_content()==str(expected)
            assert panel.get_attribute('data-outputs')==str(n+1)
        assert svg.locator('path').count()==paths
        menu.select_option('9')  # A negative activation is visibly stored in the plane.
        page.evaluate("document.querySelector('#fig-cnn').scrollIntoView({block:'start'})")
        page.screenshot(path=str(args.screenshots/f'direct-values-{width}-{theme}.png'))
        svg.screenshot(path=str(args.screenshots/f'direct-drawing-{width}-{theme}.png'))
        before=panel.get_attribute('data-step')
        inset.locator('summary').click()
        assert panel.get_attribute('data-step')==before
        selected=sum(int(2+c==2)*[-1,0,1][c] for r in range(3) for c in range(3))
        assert inset.locator('.life-value-result').inner_text()==f'Output[0, 2] = {selected}'
        inset.locator('summary').click()
        if width==1440 and theme=='light':
            first=sum(int(c==2)*[-1,0,1][c] for r in range(3) for c in range(3))
            page.emulate_media(reduced_motion='no-preference');menu.select_option('0')
            panel.get_by_role('button',name='Next',exact=True).click()
            page.wait_for_function("document.querySelector('#fig-cnn .life-panel').dataset.moving === 'false'")
            panel.get_by_role('button',name='Next',exact=True).click()
            assert svg.locator('[data-diagram-sum]').count()==0
            assert svg.locator('[data-output-index]').count()==0
            page.wait_for_timeout(700)
            svg.screenshot(path=str(args.screenshots/'direct-values-in-motion.png'))
            page.wait_for_function("document.querySelector('#fig-cnn .life-panel').dataset.moving === 'false'")
            assert svg.locator('[data-diagram-sum]').get_attribute('data-diagram-sum')==str(first)
            panel.get_by_role('button',name='Next',exact=True).click()
            assert svg.locator('[data-output-index]').count()==0
            page.wait_for_function("document.querySelector('#fig-cnn .life-panel').dataset.moving === 'false'")
            assert svg.locator('[data-output-index="0"]').text_content()==str(first)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        assert not errors,errors
        report.append({'width':width,'theme':theme,'windows_checked':25,'passed':True})
        page.close()
    print(json.dumps(report,indent=2));browser.close()
