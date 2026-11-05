import {test, expect, webkit} from '@playwright/test';
import {existsSync} from 'node:fs';
import {writeFile} from 'node:fs/promises';

// Constructed response and DOM selection; no native OS selection handles or physical-device gestures.
const answer = `${'Bandwidth moves data. '.repeat(180)}\n\n\`\`\`python\n${'transfer_time = bytes / bandwidth; '.repeat(18)}\n\`\`\`\n\n$$${Array.from({length:36}, (_, i) => `x_{${i}}`).join(' + ')} = B$$\n\nFinal paragraph.`;
const sse = text => `data: ${JSON.stringify({choices:[{delta:{content:text},finish_reason:null}]})}\n\ndata: ${JSON.stringify({choices:[{delta:{},finish_reason:'stop'}]})}\n\ndata: [DONE]\n\n`;
const viewports = [{name:'narrow-phone',width:320,height:568},{name:'phone',width:390,height:844},{name:'tablet-portrait',width:768,height:1024},{name:'tablet-landscape',width:1024,height:768},{name:'short-landscape',width:844,height:390},{name:'laptop',width:1366,height:768}];
async function select(page,id='passage') {
  await page.locator(`#${id}`).scrollIntoViewIfNeeded();
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  await page.locator(`#${id}`).evaluate(node => {const range=document.createRange();range.selectNodeContents(node);const s=getSelection();s.removeAllRanges();s.addRange(range);document.dispatchEvent(new Event('selectionchange'));});
}
async function open(page) {
  await page.route('https://proxy-worker-streaming.mlsysbook.workers.dev/**', route => route.fulfill({contentType:'text/event-stream',body:sse(answer)}));
  await page.goto('/vol1/chapter/'); await select(page);
  await page.getByRole('button',{name:'Explain this',exact:true}).tap();
  await expect(page.getByText('Explanation ready.',{exact:true})).toBeVisible();
  await expect(page.getByRole('textbox',{name:'Ask a follow-up question'})).not.toBeFocused();
  expect(await page.getByRole('textbox',{name:'Ask a follow-up question'}).evaluate(node=>parseFloat(getComputedStyle(node).fontSize))).toBeGreaterThanOrEqual(16);
}
async function geometry(page) {
  return page.locator('.reader-panel').evaluate(panel => {
    const rect=node=>{const r=node.getBoundingClientRect();return {left:r.left,right:r.right,top:r.top,bottom:r.bottom,width:r.width,height:r.height};};
    const body=panel.querySelector('.reader-body');
    return {panel:rect(panel),body:rect(body),composer:rect(panel.querySelector('.reader-composer')),scrollWidth:body.scrollWidth,clientWidth:body.clientWidth,pageWidth:document.documentElement.scrollWidth,viewportWidth:innerWidth};
  });
}
async function receipt(page,info,label) {
  await page.screenshot({path:info.outputPath(`${label}.png`),fullPage:true});
  await writeFile(info.outputPath(`${label}.json`),JSON.stringify({timestamp:new Date().toISOString(),title:info.title,viewport:page.viewportSize(),geometry:await geometry(page),limitations:'DOM selection and mocked visualViewport; no physical-device gesture or keyboard claim'},null,2));
}
function contained(box,v) {
  expect.soft(box.left).toBeGreaterThanOrEqual(v.left-1);expect.soft(box.right).toBeLessThanOrEqual(v.left+v.width+1);
  expect.soft(box.top).toBeGreaterThanOrEqual(v.top-1);expect.soft(box.bottom).toBeLessThanOrEqual(v.top+v.height+1);
}
test.use({hasTouch:true,reducedMotion:'reduce'});
test.describe('responsive Chromium',()=>{
  for(const v of viewports) test(`${v.name}: answer, controls, source, new selection`,async({page},info)=>{
    await page.setViewportSize({width:v.width,height:v.height});await open(page);await receipt(page,info,'answer');
    await page.locator('.reader-answer pre').scrollIntoViewIfNeeded();await receipt(page,info,'wide-code');
    const g=await geometry(page);contained(g.panel,{left:0,top:0,...v});contained(g.composer,{left:0,top:0,...v});
    expect.soft(g.body.height,'usable answer scroller').toBeGreaterThanOrEqual(80);
    expect.soft(g.scrollWidth,'code/math contained within answer').toBeLessThanOrEqual(g.clientWidth+1);
    expect.soft(g.pageWidth).toBeLessThanOrEqual(g.viewportWidth+1);
    for(const name of ['Close SocratiQ reader','Send question','Simplify','Show an example','Check my understanding','Start over']) {const target=page.getByRole('button',{name,exact:true});await target.scrollIntoViewIfNeeded();const box=await target.boundingBox();expect.soft(box.width,`${name} touch width`).toBeGreaterThanOrEqual(44);expect.soft(box.height,`${name} touch height`).toBeGreaterThanOrEqual(44);}
    await page.getByRole('textbox',{name:'Ask a follow-up question'}).fill('What does latency mean?');
    await page.getByRole('button',{name:'Send question',exact:true}).tap();
    await expect(page.locator('.reader-user-text').last()).toContainText('What does latency mean?');
    await expect(page.locator('.reader-source-link')).toHaveAttribute('href',/#bandwidth/);
    await page.locator('.reader-body').evaluate(node=>{node.scrollTop=0;});
    await page.locator('.reader-source-link').tap();await expect(page.getByRole('region',{name:'SocratiQ focused reader',exact:true})).toBeHidden();await expect(page.locator('#passage')).toBeFocused();await page.getByRole('button',{name:'Ask SocratiQ',exact:true}).tap();await expect(page.locator('.reader-user-text').last()).toContainText('What does latency mean?');
    await page.getByRole('button',{name:'Close SocratiQ reader'}).tap();await select(page,'math');
    await page.getByRole('button',{name:'Explain this',exact:true}).tap();
    await expect(page.locator('.reader-quote')).toContainText('Transfer time');await expect(page.locator('.reader-user-text')).toHaveCount(1);await expect(page.locator('.reader-user-text')).toContainText('Explain this passage.');
  });
  test('visualViewport keyboard and pinch offsets keep controls reachable',async({page},info)=>{
    await page.setViewportSize({width:390,height:844});
    await page.addInitScript(()=>{const viewport=new EventTarget();Object.assign(viewport,{width:390,height:844,offsetTop:0,offsetLeft:0,scale:1});Object.defineProperty(window,'visualViewport',{value:viewport,configurable:true});});
    await open(page);await page.getByRole('textbox',{name:'Ask a follow-up question'}).tap();
    for(const state of [{label:'keyboard',width:390,height:350,offsetTop:90,offsetLeft:0,scale:1},{label:'zoom-offset',width:260,height:430,offsetTop:70,offsetLeft:60,scale:1.5}]) {
      await page.evaluate(s=>{Object.assign(visualViewport,s);visualViewport.dispatchEvent(new Event('resize'));visualViewport.dispatchEvent(new Event('scroll'));},state);
      await receipt(page,info,state.label);const g=await geometry(page);const v={left:state.offsetLeft,top:state.offsetTop,width:state.width,height:state.height};contained(g.panel,v);contained(g.composer,v);expect.soft(g.body.height).toBeGreaterThanOrEqual(80);
    }
    await page.keyboard.press('Escape');await expect(page.getByRole('region',{name:'SocratiQ focused reader',exact:true})).toBeHidden();
  });
});

for(const v of viewports.filter(v=>['phone','tablet-portrait','tablet-landscape'].includes(v.name))) test(`WebKit ${v.name} touch smoke`, async ({}, info) => {
  test.skip(!existsSync(webkit.executablePath()), 'WebKit executable not installed; Safari remains untested.');
  const browser=await webkit.launch();
  try {const context=await browser.newContext({viewport:{width:v.width,height:v.height},hasTouch:true,reducedMotion:'reduce',baseURL:'http://127.0.0.1:4176'});const page=await context.newPage();await open(page);await receipt(page,info,'webkit-answer');const g=await geometry(page);contained(g.panel,{left:0,top:0,...v});expect(g.scrollWidth).toBeLessThanOrEqual(g.clientWidth+1);await page.locator('.reader-answer pre').scrollIntoViewIfNeeded();await receipt(page,info,'webkit-wide-code');await page.getByRole('button',{name:'Close SocratiQ reader'}).tap();await expect(page.getByRole('region',{name:'SocratiQ focused reader',exact:true})).toBeHidden();} finally {await browser.close();}
});
