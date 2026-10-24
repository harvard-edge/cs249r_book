"""Inspect projection, live connector alignment, and diagram-linked arithmetic."""
from pathlib import Path
import argparse
import json
from playwright.sync_api import sync_playwright

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url',default='http://127.0.0.1:8882/_build/original-figure-life/vol1/06_nn_architectures/06_nn_architectures.html?view=explore#fig-cnn')
parser.add_argument('--screenshots',type=Path,default=Path('/tmp/convolution-geometry-review'))
parser.add_argument('--engine',choices=['chromium','webkit'],default='chromium')
args=parser.parse_args();args.screenshots.mkdir(parents=True,exist_ok=True)
geometry=r'''svg=>{
 const paths=[...svg.querySelectorAll('path')];
 const input=paths.find(p=>p.hasAttribute('stroke-dasharray'));
 const output=paths.find(p=>p.getAttribute('fill')==='rgb(89.99939%, 79.998779%, 89.99939%)');
 const orange=paths.find(p=>p.getAttribute('fill')==='rgb(100%, 50%, 0%)');
 const blue=paths.find(p=>p.getAttribute('stroke')==='rgb(0%, 0%, 100%)');
 const points=p=>{const n=p.getAttribute('d').match(/-?\d*\.?\d+/g).map(Number);return [0,2,4,6].map(i=>new DOMPoint(n[i],n[i+1]).matrixTransform(p.getScreenCTM()));};
 const subtract=(a,b)=>({x:a.x-b.x,y:a.y-b.y});
 const a=points(input),b=points(output),orangePoint=points(orange)[0],bluePoint=points(blue)[0];
 const axes=[1,3].map(i=>{const u=subtract(a[i],a[0]),v=subtract(b[i],b[0]);return Math.abs(u.x*v.y-u.y*v.x)/(Math.hypot(u.x,u.y)*Math.hypot(v.x,v.y));});
 return {axes,connectionError:Math.hypot(orangePoint.x-bluePoint.x,orangePoint.y-bluePoint.y),window:{x:orangePoint.x,y:orangePoint.y}};
}'''
with sync_playwright() as p:
    browser=getattr(p,args.engine).launch();report=[]
    for width,theme in [(1440,'light'),(390,'light'),(1440,'dark'),(390,'dark')]:
        page=browser.new_page(viewport={'width':width,'height':1100},color_scheme=theme)
        page.goto(args.url)
        panel=page.locator('#fig-cnn .life-panel');panel.wait_for(state='visible')
        svg=panel.locator('svg');menu=panel.get_by_role('combobox')
        sample=svg.evaluate(geometry)
        assert max(sample['axes'])<.002,sample
        panel.get_by_role('button',name='Next',exact=True).click()
        for frame in range(3):
            page.wait_for_timeout(180)
            sample=svg.evaluate(geometry)
            assert sample['connectionError']<.5,sample
            svg.screenshot(path=str(args.screenshots/f'move-{width}-{theme}-{frame}.png'))
        if panel.get_attribute('data-moving')=='true':
            panel.get_by_role('button',name='Play',exact=True).click()
            panel.get_by_role('button',name='Pause',exact=True).click()
            before=svg.evaluate(geometry)['window'];page.wait_for_timeout(250)
            assert svg.evaluate(geometry)['window']==before
            panel.get_by_role('button',name='Play',exact=True).click()
        page.wait_for_function("document.querySelector('#fig-cnn .life-panel').dataset.moving === 'false'")
        # Seek to a completed sum; disclosure controls may not advance the lesson.
        menu.select_option('8')
        before=panel.get_attribute('data-step')
        expected_row=sum(int(2+c==2)*[-1,0,1][c] for c in range(3))
        for row in range(3):
            panel.get_by_role('button',name=f'Row {row+1}',exact=True).click()
            assert panel.get_attribute('data-step')==before
            assert svg.locator('[data-focused-row]').get_attribute('data-focused-row')==str(row)
            assert panel.locator('.life-calculation').get_attribute('data-row')==str(row)
            assert panel.locator('.life-row-equation').inner_text().strip().endswith(f'= {expected_row}')
        calculation=panel.locator('.life-calculation')
        calculation.screenshot(path=str(args.screenshots/f'calculation-{width}-{theme}.png'))
        page.evaluate("document.querySelector('#fig-cnn').scrollIntoView({block:'start'})")
        page.screenshot(path=str(args.screenshots/f'chapter-{width}-{theme}.png'))
        svg.screenshot(path=str(args.screenshots/f'drawing-{width}-{theme}.png'))
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        report.append({'width':width,'theme':theme,'parallel_axes':sample['axes'],'live_alignment':'passed','calculation_rows':'passed'})
        page.close()
    print(json.dumps(report,indent=2));browser.close()
