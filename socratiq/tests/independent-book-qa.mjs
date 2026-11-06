// Read-only integration probe: pre-rendered book HTML/CSS plus the current widget.
// Constructed AI responses exercise rendering, not live inference quality.
import {createServer} from 'node:http';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {resolve, join, extname} from 'node:path';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {hostname, platform, arch} from 'node:os';
import {chromium, webkit, firefox, expect} from '@playwright/test';
const bookRoot = resolve(process.env.QA_BOOK_ROOT || new URL('../../books/_build', import.meta.url).pathname);
const output = resolve(process.env.SOCRATIQ_BOOK_QA_OUTPUT || '/tmp/socratiq-book-integration-qa');
const assets = new URL('../../books/shared/assets/scripts/socratiq/', import.meta.url);
const chapters = {vol1:'vol1/01_introduction/01_introduction.html',vol2:'vol2/01_introduction/01_introduction.html',vol3:'vol3/01_introduction/01_introduction.html',vol4:'vol4/01_boundary/01_boundary.html'};
const devices = {desktop:{width:1366,height:900},phone:{width:390,height:844},tablet:{width:768,height:1024}};
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const bundles = Object.fromEntries(await Promise.all(['reader.js','study.js'].map(async name=>[name,await readFile(new URL(name,assets))])));
const manifest = {command:'QA_BOOK_ROOT=<pre-rendered-books> node tests/independent-book-qa.mjs', bookRoot, codeSHA:execFileSync('git',['rev-parse','HEAD'],{cwd:new URL('../../',import.meta.url),encoding:'utf8'}).trim(), dirty:Boolean(execFileSync('git',['status','--porcelain'],{cwd:new URL('../../',import.meta.url),encoding:'utf8'}).trim()), environment:{hostname:hostname(),platform:platform(),arch:arch(),node:process.version},testHash:hash(await readFile(new URL(import.meta.url))), assets:{}, seeds:'Not applicable: authored fixture responses', classification:'Measured integration on existing pre-rendered book HTML/CSS with current widget substituted; constructed AI responses',started:new Date().toISOString(),bundleHashes:Object.fromEntries(Object.entries(bundles).map(([name,bytes])=>[name,hash(bytes)])),cases:[],limits:'No main-checkout writes, publication claim, live provider calls, physical-device or native keyboard certification.'};
await mkdir(output,{recursive:true});
const mime = {'.html':'text/html','.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg','.woff2':'font/woff2','.woff':'font/woff','.json':'application/json'};
const server = createServer(async(req,res)=>{
  try {
    const pathname = decodeURIComponent(new URL(req.url,'http://localhost').pathname);
    const volume = pathname.match(/^\/(vol[1-4])\//)?.[1];
    if (!volume) {res.writeHead(404).end();return;}
    if (/\/socratiq\/(reader|study)\.js$/.test(pathname)) {res.setHeader('Content-Type','text/javascript');res.end(bundles[pathname.endsWith('/reader.js')?'reader.js':'study.js']);return;}
    const root = resolve(bookRoot,`html-${volume}`), relative = pathname.slice(volume.length+2);
    const path = resolve(root,relative);
    if (!path.startsWith(`${root}/`)) {res.writeHead(403).end();return;}
    let bytes = await readFile(path);
    manifest.assets[`${volume}/${relative}`] = hash(bytes);
    if (extname(path)==='.html') {
      let html = bytes.toString();
      html = html.replace(/<script\b[^>]*src=["'][^"']*socrati[qQ][^"']*["'][^>]*>\s*<\/script>/gi,'');
      html = html.replace('</head>',`<script src="/${volume}/shared/assets/scripts/socratiq/reader.js" data-volume="${volume}" data-endpoint="https://mlsysbook.ai/api/socratiq/ai/stream" defer></script></head>`);
      bytes = Buffer.from(html);
    }
    res.setHeader('Content-Type',mime[extname(path)]||'application/octet-stream');res.end(bytes);
  } catch {res.writeHead(404).end();}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const origin=`http://127.0.0.1:${server.address().port}`;
let failures=0;
try {
  for (const [engine,type] of Object.entries({chromium,webkit,firefox}).filter(([name])=>!process.env.QA_BROWSERS||process.env.QA_BROWSERS.split(',').includes(name))) {
    const browser=await type.launch();
    try {
      for (const [volume,chapter] of Object.entries(chapters).filter(([name])=>!process.env.QA_VOLUMES||process.env.QA_VOLUMES.split(',').includes(name))) for (const [device,viewport] of Object.entries(devices).filter(([name])=>!process.env.QA_DEVICES||process.env.QA_DEVICES.split(',').includes(name))) {
        const directory=join(output,`${engine}-${volume}-${device}`);await mkdir(directory,{recursive:true});
        const context=await browser.newContext({viewport,hasTouch:device!=='desktop',colorScheme:volume==='vol2'||volume==='vol4'?'dark':'light',reducedMotion:'reduce'});
        const page=await context.newPage();page.setDefaultTimeout(20000);
        const receipt={engine,browserVersion:browser.version(),volume,device,viewport,chapter,htmlHash:hash(await readFile(resolve(bookRoot,`html-${volume}`,chapter))),errors:[],screenshots:[],status:'running'};
        page.on('pageerror',error=>receipt.errors.push(error.message));
        await page.route('**/*',route=>{
          const url=new URL(route.request().url());
          if(url.origin===origin)return route.continue();
          if(url.pathname==='/api/socratiq/ai/stream')return route.fulfill({contentType:'text/event-stream',body:`data: ${JSON.stringify({choices:[{delta:{content:'**Constructed QA explanation.** The selected source is used to exercise the rendered book interface.\n\n$$t = D/B$$\n\n```python\ntransfer_time = data_bytes / bandwidth\n```'},finish_reason:null}]})}\n\ndata: [DONE]\n\n`});
          return route.abort();
        });
        try {
          await page.goto(`${origin}/${volume}/${chapter}`,{waitUntil:'domcontentloaded'});
          await page.getByRole('button',{name:'Ask SocratiQ',exact:true}).click();
          await page.getByRole('dialog',{name:'SocratiQ guided tour'}).getByRole('button',{name:'Skip tour',exact:true}).click();
          await page.getByRole('button',{name:'Close SocratiQ reader',exact:true}).click();
          const source=page.locator('main#quarto-document-content section > p').filter({hasText:/\w{5}/}).first();
          await source.scrollIntoViewIfNeeded();
          receipt.selectedPassage=await source.textContent();
          await source.evaluate(node=>{const range=document.createRange();range.selectNodeContents(node);getSelection().removeAllRanges();getSelection().addRange(range);document.dispatchEvent(new Event('selectionchange'));});
          await page.getByRole('button',{name:'Explain this',exact:true}).click();
          await expect(page.getByText('Explanation ready.',{exact:true})).toBeVisible();
          await expect(page.locator('.reader-shell')).toHaveAttribute('data-theme',volume==='vol2'||volume==='vol4'?'dark':'light');
          await expect(page.locator('.reader-answer .katex')).toBeVisible();
          const panel=await page.locator('.reader-panel').boundingBox();
          assert.ok(panel.x>=0&&panel.y>=0&&panel.x+panel.width<=viewport.width+1&&panel.y+panel.height<=viewport.height+1);
          for(const [label,action] of [['book-reader',async()=>{}],['book-preferences',async()=>{await page.locator('.reader-preferences summary').click();}],['book-study',async()=>{await page.locator('.reader-preferences summary').click();await page.getByRole('button',{name:'Study tools',exact:true}).click();await expect(page.getByRole('button',{name:'Close study tools',exact:true})).toBeVisible();}],['book-settings',async()=>{await page.locator('#widget-chat-container #settings-btn').click();await expect(page.locator('#widget-chat-container #modal1')).toBeVisible();await page.locator('#widget-chat-container #study-learner-preferences summary').click();}]]) {
            await action();const file=join(directory,`${label}.png`);await page.screenshot({path:file,animations:'disabled'});receipt.screenshots.push({label,file,sha256:hash(await readFile(file))});
          }
          receipt.status='passed';
        }catch(error){failures++;receipt.status='failed';receipt.failure=error.stack;await page.screenshot({path:join(directory,'failure.png')}).catch(()=>{});}
        finally {await writeFile(join(directory,'receipt.json'),JSON.stringify(receipt,null,2));manifest.cases.push({engine,volume,device,status:receipt.status,receipt:join(directory,'receipt.json')});await writeFile(join(output,'manifest.json'),JSON.stringify(manifest,null,2));await context.close();console.log(`${engine} ${volume} ${device}: ${receipt.status}`);}
      }
    }finally{await browser.close();}
  }
}finally {await new Promise(resolve=>server.close(resolve));manifest.finished=new Date().toISOString();await writeFile(join(output,'manifest.json'),JSON.stringify(manifest,null,2));}
assert.equal(failures,0,'See raw receipts for failures');
