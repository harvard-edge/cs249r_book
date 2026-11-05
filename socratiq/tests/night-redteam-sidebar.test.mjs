// Constructed malicious import -> production IndexedDB merge -> sidebar renderers.
// Local source fixture; no provider calls or shared bundle build.
import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn,execFileSync} from 'node:child_process';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {testBrowser} from './legacy-test-browser.mjs';
const output=resolve(process.env.SOCRATIQ_TEST_RECEIPT_DIR||'/tmp/socratiq-sidebar-redteam');
test('imported title question and tag stay literal and inert in both production sidebar renderers',{timeout:90000},async()=>{
 await mkdir(output,{recursive:true});const server=spawn(process.execPath,['node_modules/vite/bin/vite.js','--config','vite.config.dev.mjs','--host','127.0.0.1','--port','4201','--strictPort'],{cwd:new URL('..',import.meta.url),stdio:'ignore'});let browser;
 try{
  for(let attempt=0;attempt<100;attempt++){try{if((await fetch('http://127.0.0.1:4201')).ok)break;}catch{}await new Promise(resolve=>setTimeout(resolve,100));}
  browser=await testBrowser.launch({headless:true});const page=await browser.newPage();const blocked=[];
  await page.route('**/*',route=>{const url=new URL(route.request().url());if(url.href==='http://127.0.0.1:4201/sidebar-audit')return route.fulfill({contentType:'text/html',body:'<!doctype html><html><head><title>Constructed sidebar import fixture</title></head><body><h1>Constructed sidebar import fixture</h1><div id="host"></div></body></html>'});if(url.hostname==='127.0.0.1')return route.continue();blocked.push(url.origin);return route.abort();});
  await page.goto('http://127.0.0.1:4201/sidebar-audit');
  const result=await page.evaluate(async()=>{
   const {SpacedRepetitionIndexDBHandler}=await import('/js/components/spaced_repetition/handlers/spaced-repetition-indexdb.js');
   const {SpacedRepetitionModal}=await import('/js/components/spaced_repetition/spaced-repetition-modal-handler.js');
   const {SpacedRepetitionUIHandler}=await import('/js/components/spaced_repetition/handlers/ui-handler.js');
   const {getDBInstance}=await import('/js/libs/utils/indexDb.js');
   const title='Constructed deck <img src="data:," onerror="window.constructedTitleInjected=true">';
   const question='Constructed question <img src="data:," onerror="window.constructedQuestionInjected=true">';
   const tag='Constructed tag "><img src="data:," onerror="window.constructedTagInjected=true">';
   const data={chapter_card_sets:[{chapter:876,title,cards:[{id:'constructed-sidebar-card',question,answer:'Constructed answer',tags:[tag]}]}]};
   const storage=Object.create(SpacedRepetitionIndexDBHandler.prototype);storage.dbManager=await getDBInstance();storage.inMemoryData={currentChapter:null};storage.useFallback=false;
   await storage.mergeImportedData(data);
   const cards=await new Promise((resolve,reject)=>{const request=storage.dbManager.db.transaction('cards','readonly').objectStore('cards').getAll();request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);});
   const chapters=await storage.getAllChapters();
   const root=document.querySelector('#host').attachShadow({mode:'open'});root.innerHTML='<style>svg{width:18px;height:18px;flex-shrink:0}button{display:flex;gap:8px;padding:8px;max-width:900px;white-space:normal;text-align:left}#chapterList,#tagList{max-height:300px;overflow:auto}h2{font-size:20px}</style><h2>Modal sidebar</h2><div id="chapterList"></div><div id="tagList"></div><span id="chapterCount"></span><span id="totalTagCount"></span>';
   const modal=Object.create(SpacedRepetitionModal.prototype);Object.assign(modal,{shadowRoot:root,storageHandler:storage,chapterSets:new Map([[876,cards.filter(card=>card.chapter===876)]]),currentChapter:null,themeManager:{getThemeClasses:()=>({})}});
   modal.renderChapters();modal.renderTags();
   for(let attempt=0;attempt<100&&!root.querySelector('#chapterList button[data-chapter]');attempt++)await new Promise(resolve=>setTimeout(resolve,10));
   await new Promise(resolve=>setTimeout(resolve,100));
   const snapshot=()=>({chapterHtml:root.querySelector('#chapterList').innerHTML,chapterText:root.querySelector('#chapterList').textContent,tagHtml:root.querySelector('#tagList').innerHTML,tagText:root.querySelector('#tagList').textContent,tags:[...root.querySelectorAll('#tagList [data-tag]')].map(node=>node.dataset.tag),active:root.querySelectorAll('#chapterList img,#tagList img,#chapterList [onerror],#tagList [onerror]').length,executed:{title:!!window.constructedTitleInjected,question:!!window.constructedQuestionInjected,tag:!!window.constructedTagInjected}});
   const modalResult=snapshot();
   window.constructedTitleInjected=false;window.constructedQuestionInjected=false;window.constructedTagInjected=false;
   const ui=Object.create(SpacedRepetitionUIHandler.prototype);Object.assign(ui,{shadowRoot:root,storageHandler:storage,themeManager:modal.themeManager,modal});modal.uiHandler=ui;
   await ui.renderChapters();await ui.renderTags();await new Promise(resolve=>setTimeout(resolve,100));
   const uiResult=snapshot();root.prepend(Object.assign(document.createElement('h2'),{textContent:'UI-handler sidebar after imported data'}));
   return{input:data,storedChapters:chapters,storedCards:cards,modal:modalResult,ui:uiResult};
  });
  await writeFile(resolve(output,'raw.json'),JSON.stringify({result,blocked},null,2));await page.screenshot({path:resolve(output,'sidebar.png')});
  const hashes={};for(const file of ['src_shadow/js/components/spaced_repetition/spaced-repetition-modal-handler.js','src_shadow/js/components/spaced_repetition/handlers/ui-handler.js','src_shadow/js/components/spaced_repetition/utils/sidebar-text.js','tests/night-redteam-sidebar.test.mjs']){const data=await readFile(new URL('../'+file,import.meta.url));hashes[file]=createHash('sha256').update(data).digest('hex');await writeFile(resolve(output,file.replaceAll('/','__')+'.snapshot'),data);}
  await writeFile(resolve(output,'manifest.json'),JSON.stringify({timestamp:new Date().toISOString(),command:process.argv,kind:'Constructed validated import and production sidebar browser regression',sha:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),dirty:Boolean(execFileSync('git',['status','--porcelain'],{encoding:'utf8'}).trim()),hashes,seeds:'none'},null,2));
  assert.equal(result.storedChapters.find(chapter=>chapter.chapter===876).title,result.input.chapter_card_sets[0].title);
  for(const name of ['modal','ui']){const raw=result[name];assert.equal(raw.active,0,`${name} creates active imported HTML`);assert.deepEqual(raw.executed,{title:false,question:false,tag:false});assert(raw.chapterText.includes(result.input.chapter_card_sets[0].title));assert(raw.tagText.includes(result.input.chapter_card_sets[0].cards[0].tags[0]));assert(raw.tags.includes(result.input.chapter_card_sets[0].cards[0].tags[0]));}
  assert(result.modal.chapterText.includes(result.input.chapter_card_sets[0].cards[0].question));
 }finally{await browser?.close();server.kill();}
});
