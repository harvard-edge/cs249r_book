// Independent source-only rerun, author-constructed page. No bundle builds.
import assert from 'node:assert/strict';
import {chromium} from '@playwright/test';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
const output=resolve(process.argv[2]);await mkdir(output,{recursive:true});
const source=await readFile('src_reader/context.js','utf8');const input=await readFile(process.argv[3] || resolve(output, '../constructed-input.html'),'utf8');
const browser=await chromium.launch();
try{
 const page=await browser.newPage();await page.route('**/*',route=>route.abort());await page.setContent(input);
 await page.addScriptTag({content:source.replace(/^export /gm,'')+'\nwindow.nightCaptureSelection=captureSelection;'});
 const raw=await page.evaluate(()=>{const p=document.querySelector('#passage');const range=document.createRange();range.selectNodeContents(p);const selection=getSelection();selection.removeAllRanges();selection.addRange(range);return window.nightCaptureSelection({window:{getSelection:()=>selection,location:new URL('http://constructed-user:constructed-pass@audit.invalid/vol1/chapter/?private=CONSTRUCTED_QUERY_SENTINEL')}});});
 await writeFile(resolve(output,'captured-source.json'),JSON.stringify(raw,null,2));await writeFile(resolve(output,'context.js.snapshot'),source);
 await writeFile(resolve(output,'manifest.json'),JSON.stringify({timestamp:new Date().toISOString(),command:process.argv,kind:'Independent source-only constructed context probe',sha:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),dirty:Boolean(execFileSync('git',['status','--porcelain'],{encoding:'utf8'}).trim()),sourceSha256:createHash('sha256').update(source).digest('hex'),inputSha256:createHash('sha256').update(input).digest('hex'),seed:'none',preregistration:'../acceptance.json'},null,2));
 assert.doesNotMatch(JSON.stringify(raw),/HIDDEN_SENTINEL|INERT_SENTINEL|FORM_SENTINEL|NAV_SENTINEL|INPUT_SENTINEL|CONSTRUCTED_QUERY_SENTINEL/);
 console.log(JSON.stringify({excludedHiddenFormNavigationAndQuery:true,urlUserinfoPresent:/constructed-user|constructed-pass/.test(raw.url)},null,2));
}finally{await browser.close();}
