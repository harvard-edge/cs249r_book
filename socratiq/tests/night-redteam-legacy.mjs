// Author-constructed corrupted persisted-chat probe in an isolated browser.
import {chromium} from '@playwright/test';
import {spawn,execFileSync} from 'node:child_process';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
const output=resolve(process.argv[2]);await mkdir(output,{recursive:true});
await writeFile(resolve(output,'acceptance.json'),JSON.stringify({criterion:'Restoring an author-constructed corrupted persisted chat must not execute its stored HTML event handlers or retain JavaScript links. Existing-chat content must not be logged in full.',failureSentence:'Report restored executable content as a confirmed local sink; remote exploitability remains unverified.',scope:'Isolated browser IndexedDB and local Vite; all non-local requests blocked'},null,2));
const server=spawn(process.execPath,['node_modules/vite/bin/vite.js','--config','vite.config.dev.mjs','--host','127.0.0.1','--port','4198','--strictPort'],{stdio:'ignore'});
let browser;
try{
 for(let attempt=0;attempt<100;attempt++){try{if((await fetch('http://127.0.0.1:4198')).ok)break;}catch{}await new Promise(resolve=>setTimeout(resolve,100));}
 browser=await chromium.launch();const page=await browser.newPage();const network=[];const logs=[];
 await page.route('**/*',route=>{network.push(route.request().url());if(route.request().url()==='http://127.0.0.1:4198/audit')return route.fulfill({contentType:'text/html',body:'<div id="host"></div>'});if(new URL(route.request().url()).hostname==='127.0.0.1')return route.continue();return route.abort();});
 page.on('console',message=>logs.push({type:message.type(),text:message.text()}));
 await page.goto('http://127.0.0.1:4198/audit');
 const raw=await page.evaluate(async()=>{
  const {getDBInstance}=await import('/js/libs/utils/indexDb.js');const {loadChat}=await import('/js/libs/utils/save_chats.js');
  const manager=await getDBInstance();
  const entry={id:1,title:'Constructed private chat title',html_content:'<p>CONSTRUCTED_PRIVATE_CHAT_CONTENT</p><svg onload="window.constructedStoredExecuted=true"></svg><a href="javascript:window.constructedLinkExecuted=true">Constructed stored link</a>',date:new Date().toISOString()};
  await manager.add('tinyMLDB_chats',entry);
  const root=document.querySelector('#host').attachShadow({mode:'open'});root.innerHTML='<div id="message-container"></div>';
  const loaded=await loadChat(root,1);await new Promise(resolve=>setTimeout(resolve,100));
  return {loaded,executed:!!window.constructedStoredExecuted,html:root.querySelector('#message-container').innerHTML,entry};
 });
 await writeFile(resolve(output,'raw.json'),JSON.stringify({raw,logs,network},null,2));
 const source=await readFile('src_shadow/js/libs/utils/save_chats.js');await writeFile(resolve(output,'save_chats.js.snapshot'),source);
 await writeFile(resolve(output,'manifest.json'),JSON.stringify({kind:'Author-constructed corrupted stored-chat local probe; no remote exploit claim',timestamp:new Date().toISOString(),command:process.argv,sha:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),dirty:Boolean(execFileSync('git',['status','--porcelain'],{encoding:'utf8'}).trim()),sourceSha256:createHash('sha256').update(source).digest('hex'),seeds:'none'},null,2));
 console.log(JSON.stringify({loaded:raw.loaded,executed:raw.executed,javascriptLink:raw.html.includes('href="javascript:'),fullContentInConsole:logs.some(log=>log.text.includes('CONSTRUCTED_PRIVATE_CHAT_CONTENT'))},null,2));
 process.exitCode=raw.executed||raw.html.includes('href="javascript:')?1:0;
}finally{await browser?.close();server.kill();}
