// Constructed local lifecycle probe; worker requests are blocked.
import {chromium} from '@playwright/test';
import assert from 'node:assert/strict';
import {mkdir} from 'node:fs/promises';
import {resolve} from 'node:path';
import {tmpdir} from 'node:os';
const browser=await chromium.launch({headless:true});
const receipt=resolve(process.env.SOCRATIQ_PROBE_OUTPUT || `${tmpdir()}/socratiq-bridge-probe`);
await mkdir(receipt,{recursive:true});
try {
 const page=await browser.newPage({viewport:{width:1280,height:900}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/*.workers.dev/**',r=>r.abort());
 await page.route('**/runtime-fixture',r=>r.fulfill({contentType:'text/html',body:'<!DOCTYPE html><html><head><title>Constructed study fixture</title></head><body><main id="quarto-document-content"><h1>Constructed chapter</h1><section><h2>Memory bandwidth</h2><p>Memory bandwidth limits the accelerator during inference.</p></section></main><script type="module" src="/js/index.js"></script></body></html>'}));
 await page.goto('http://127.0.0.1:4187/runtime-fixture');
 await page.waitForTimeout(1500);
 const disabled=await page.evaluate(async()=>({host:!!document.querySelector('#widget-chat-container'),dbs:(await indexedDB.databases()).map(d=>d.name)}));
 assert.equal(disabled.host,false); assert.deepEqual(disabled.dbs,[]);
 await page.addInitScript(()=>{window.__SOCRATIQ_EMBEDDED__=true;window.studyEvents=[];for(const name of ['open','close'])window.addEventListener('socratiq:'+name,()=>window.studyEvents.push(name));});
 await page.reload();
 await page.waitForFunction(()=>!!window.SocratiqStudy);
 await page.evaluate(()=>window.SocratiqStudy.ready);
 assert.equal(await page.locator('#widget-chat-container').evaluate(e=>e.hidden),true);
 await page.evaluate(()=>window.SocratiqStudy.open());
 await page.getByRole('button',{name:'Close study tools'}).waitFor({state:'visible'});
 await page.screenshot({path:resolve(receipt,'embedded-desktop.png')});
 await page.evaluate(()=>{window.originalStudyHost=document.querySelector('#widget-chat-container');});
 await page.getByRole('button',{name:'Close study tools'}).click();
 assert.equal(await page.locator('#widget-chat-container').evaluate(e=>e.hidden),true);
 await page.evaluate(()=>window.SocratiqStudy.open());
 const reopen=await page.evaluate(()=>({sameHost:window.originalStudyHost===document.querySelector('#widget-chat-container'),hosts:document.querySelectorAll('#widget-chat-container').length,cookie:document.cookie,events:window.studyEvents,toggle:getComputedStyle(document.querySelector('#widget-chat-container').shadowRoot.getElementById('toggleButton')).display}));
 assert.equal(reopen.sameHost,true);assert.equal(reopen.hosts,1);assert.equal(reopen.cookie,'');assert.deepEqual(reopen.events,['open','close','open']);assert.equal(reopen.toggle,'none');
 await page.setViewportSize({width:390,height:844});
 await page.screenshot({path:resolve(receipt,'embedded-mobile.png')});
 assert.deepEqual(errors,[]);
 console.log(JSON.stringify({constructed:'Local browser lifecycle; external worker requests blocked',disabled,reopen,errors},null,2));
} finally {await browser.close();}
