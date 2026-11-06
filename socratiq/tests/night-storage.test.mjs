import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn, execFileSync} from 'node:child_process';
import {testBrowser} from './legacy-test-browser.mjs';

const port = Number(process.env.NIGHT_STORAGE_PORT || 4191);
const origin = `http://127.0.0.1:${port}`;

test('legacy storage preserves constructed records through repairs and saves', {timeout: 90000}, async t => {
  const server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', '--config', 'vite.config.dev.mjs', '--host', '127.0.0.1', '--port', String(port), '--strictPort'], {cwd: new URL('..', import.meta.url), stdio: 'ignore'});
  let browser;
  try {
    let ready = false;
    for (let attempt = 0; attempt < 100; attempt++) {
      try { if ((await fetch(origin)).ok) { ready = true; break; } } catch {}
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    assert.ok(ready, 'isolated fixture server started');
    browser = await testBrowser.launch({headless: true});
    const withPage = async run => {
      const context = await browser.newContext();
      const page = await context.newPage();
      await page.route('**/night-storage-fixture', route => route.fulfill({contentType:'text/html',body:'<!doctype html><title>Constructed storage regression fixture</title><main id="message-container"><p>Constructed saved message</p></main>'}));
      if (process.env.NIGHT_STORAGE_BASELINE) {
        for (const file of ['js/libs/utils/indexDb.js', 'js/libs/utils/save_chats.js', 'js/components/quiz/chapterMapService.js', 'js/components/spaced_repetition/handlers/spaced-repetition-indexdb.js']) {
          const body = execFileSync('git', ['show', `HEAD:socratiq/src_shadow/${file}`], {encoding: 'utf8'});
          await page.route(`**/${file}`, async route => {
            if (file.endsWith('spaced-repetition-indexdb.js')) {
              // Keep Vite's bare-import transformation while restoring the
              // baseline save method under test.
              const response = await route.fetch();
              const current = await response.text();
              const start = source => source.indexOf('  async saveToLocalStorage(');
              const end = source => source.indexOf('  purgeCardFromMemory(', start(source));
              const originalMethod = body.slice(start(body), end(body));
              return route.fulfill({response, body:current.slice(0,start(current))+originalMethod+current.slice(end(current))});
            }
            return route.fulfill({contentType:'text/javascript', body});
          });
        }
      }
      await page.goto(`${origin}/night-storage-fixture`);
      try { await run(page); } finally { await context.close(); }
    };

    await t.test('schema repair preserves unknown stores and adds missing indexes above installed versions', () => withPage(async page => {
      const result = await page.evaluate(async () => {
        const {IndexDBManager} = await import('/js/libs/utils/indexDb.js');
        await new Promise((resolve, reject) => {
          const request = indexedDB.open('constructed-schema-db', 7);
          request.onupgradeneeded = () => {
            request.result.createObjectStore('items', {keyPath:'id'}).put({id:1,date:'2026-01-01',text:'keep item'});
            request.result.createObjectStore('future-store', {keyPath:'id'}).put({id:2,text:'keep unknown store'});
          };
          request.onsuccess = () => {request.result.close();resolve();};
          request.onerror = () => reject(request.error);
        });
        const manager = new IndexDBManager('constructed-schema-db', 3);
        await manager.initDB([{name:'items',keyPath:'id',indexes:[{name:'date',keyPath:'date'}]},{name:'new-store',keyPath:'id'}]);
        const version = manager.db.version;
        const item = await manager.get('items', 1);
        const unknown = await manager.get('future-store', 2);
        const latest = await manager.getLatest('items');
        await manager.initDB(manager._lastKnownStores);
        return {version, stableVersion:manager.db.version, item, unknown, latest, stores:Array.from(manager.db.objectStoreNames)};
      });
      assert.ok(result.version > 7);
      assert.equal(result.stableVersion, result.version);
      assert.equal(result.item.text, 'keep item');
      assert.equal(result.unknown.text, 'keep unknown store');
      assert.equal(result.latest.id, 1);
      assert.ok(result.stores.includes('new-store'));
    }));

    await t.test('startup repair and concurrent initialization keep unknown user data', () => withPage(async page => {
      const result = await page.evaluate(async () => {
        await new Promise((resolve,reject)=>{
          const request=indexedDB.open('socratiqDB',7);
          request.onupgradeneeded=()=>request.result.createObjectStore('future-user-data',{keyPath:'id'}).put({id:1,text:'retain'});
          request.onsuccess=()=>{request.result.close();resolve();};
          request.onerror=()=>reject(request.error);
        });
        const storage=await import('/js/libs/utils/indexDb.js');
        const managers=await Promise.all([storage.getDBInstance(),storage.getDBInstance(),storage.getDBInstance()]);
        const repaired=await storage.forceInitializeSocratiqDB();
        return {same:managers.every(manager=>manager===managers[0]), record:await repaired.get('future-user-data',1)};
      });
      assert.ok(result.same);
      assert.equal(result.record.text,'retain');
    }));

    await t.test('aborted writes reject and closed connections reopen', () => withPage(async page => {
      const result = await page.evaluate(async () => {
        const {IndexDBManager}=await import('/js/libs/utils/indexDb.js');
        const manager=new IndexDBManager('constructed-transaction-db');
        await manager.initDB([{name:'items',keyPath:'id'}]);
        let rejected=false;
        try {
          await manager.performTransaction('items','readwrite',store=>{
            const request=store.put({id:1});
            request.addEventListener('success',()=>store.transaction.abort());
            return request;
          });
        } catch {rejected=true;}
        const absent=await manager.get('items',1);
        manager.db.close();
        await manager.put('items',{id:2,text:'reconnected'});
        return {rejected,absent,record:await manager.get('items',2)};
      });
      assert.ok(result.rejected);
      assert.equal(result.absent,undefined);
      assert.equal(result.record.text,'reconnected');
    }));

    await t.test('chat deletion and concurrent saves never overwrite surviving chats', () => withPage(async page => {
      const result=await page.evaluate(async()=>{
        const {getDBInstance}=await import('/js/libs/utils/indexDb.js');
        const manager=await getDBInstance();
        for(const id of [1,2,3]) await manager.put('tinyMLDB_chats',{id,title:`Constructed chat ${id}`,html_content:`<p>Constructed original ${id}</p>`});
        await manager.delete('tinyMLDB_chats',2);
        const storage=await import('/js/libs/utils/save_chats.js');
        const ids=await Promise.all([storage.determineAndSaveChat(document,null,'Constructed new one'),storage.determineAndSaveChat(document,null,'Constructed new two')]);
        await storage.determineAndSaveChat(document,String(ids[0]),'Constructed update');
        const rows=await storage.listAllRows('socratiqDB','tinyMLDB_chats');
        const malformed=['3junk','NaN','-1','0','Infinity','{}','1.5'].map(value=>{localStorage.setItem('mostRecentID',value);return storage.retrieveRecentIDFromLocal();});
        return {ids,rows,malformed};
      });
      assert.equal(new Set(result.ids).size,2);
      assert.ok(result.ids.every(id=>id>3));
      assert.equal(result.rows.find(row=>row.id===3).html_content,'<p>Constructed original 3</p>');
      assert.equal(result.rows.length,4);
      assert.ok(result.malformed.every(value=>value===null));
    }));

    await t.test('corrupt recent preferences restore latest chat through sanitized HTML', () => withPage(async page => {
      const result=await page.evaluate(async()=>{
        const {getDBInstance}=await import('/js/libs/utils/indexDb.js');
        const manager=await getDBInstance();
        const html='<div class="ai-message-chat"><p>Constructed restored message</p><button data-quiz-id="constructed" class="quiz-btn">Study quiz</button><input type="radio" name="constructed-answer"><svg><circle cx="5" cy="5" r="4"></circle></svg><img src="bad-fixture-image" onerror="window.constructedStoredScript=1"><a href="javascript:window.constructedStoredScript=2">Unsafe link</a><script>window.constructedStoredScript=3</script></div>';
        await manager.put('tinyMLDB_chats',{id:7,title:'Constructed corrupted chat fixture',html_content:html});
        const storage=await import('/js/libs/utils/save_chats.js');
        localStorage.setItem('mostRecentID','missing-not-an-id');
        const first=await storage.loadLastSavedRowIntoElement(document);
        localStorage.setItem('mostRecentID','999');
        const stale=await storage.loadLastSavedRowIntoElement(document);
        await new Promise(resolve=>setTimeout(resolve,50));
        const container=document.getElementById('message-container');
        return {first,stale,text:container.textContent,script:window.constructedStoredScript,
          unsafe:container.querySelectorAll('script, [onerror], [href^="javascript:"]').length,
          controls:!!container.querySelector('button[data-quiz-id="constructed"]') && !!container.querySelector('input[type="radio"]'),
          svg:!!container.querySelector('svg circle'), saved:(await manager.get('tinyMLDB_chats',7)).html_content};
      });
      if (process.env.NIGHT_STORAGE_SCREENSHOT) await page.screenshot({path:process.env.NIGHT_STORAGE_SCREENSHOT,fullPage:true});
      assert.equal(result.first.id,7);
      assert.equal(result.stale.id,7);
      assert.ok(result.text.includes('Constructed restored message'));
      assert.equal(result.script,undefined);
      assert.equal(result.unsafe,0);
      assert.ok(result.controls);
      assert.ok(result.svg);
      assert.ok(result.saved.includes('onerror='), 'restore leaves the original stored record intact');
    }));

    await t.test('flashcard snapshot saves roll back together and preserve unrelated decks and metadata', () => withPage(async page => {
      const result=await page.evaluate(async()=>{
        const {getDBInstance}=await import('/js/libs/utils/indexDb.js');
        const {SpacedRepetitionIndexDBHandler}=await import('/js/components/spaced_repetition/handlers/spaced-repetition-indexdb.js');
        const manager=await getDBInstance();
        const names=['chapters','cards','card_tags','current_chapter'];
        await manager.put('chapters',{chapter:1,title:'Constructed original deck',is_current:1,customMetadata:'retain chapter metadata'});
        await manager.put('chapters',{chapter:2,title:'Constructed untouched deck',is_current:0,customMetadata:'retain untouched chapter'});
        await manager.put('cards',{id:'constructed-old',chapter:1,question:'Constructed original question',answer:'Constructed answer',tags:['old'],customMetadata:'retain card metadata'});
        await manager.put('cards',{id:'constructed-other',chapter:2,question:'Constructed untouched question',tags:['other']});
        await manager.put('card_tags',{id:'constructed-old_old',card_id:'constructed-old',tag:'old'});
        await manager.put('card_tags',{id:'constructed-other_other',card_id:'constructed-other',tag:'other'});
        await manager.put('current_chapter',{chapter:1,title:'Constructed original deck',customMetadata:'retain current metadata'});
        const dump=async()=>Object.fromEntries(await Promise.all(names.map(async name=>[name,await manager.getAll(name)])));
        const before=await dump();
        const modalSets=new Map([[1,[{id:'constructed-memory'}]]]);
        const handler={dbManager:manager,inMemoryData:{currentChapter:{chapter:1,title:'Constructed renamed deck'},currentDeck:[],allTags:new Set()},modal:{chapterSets:modalSets},ensureInitialized:async()=>true,getChapterTitle:chapter=>`Constructed deck ${chapter}`};
        const save=sets=>SpacedRepetitionIndexDBHandler.prototype.saveToLocalStorage.call(handler,sets);
        const originalPut=IDBObjectStore.prototype.put;
        IDBObjectStore.prototype.put=function(value,...rest){
          const request=originalPut.call(this,value,...rest);
          if(this.name==='cards' && value.question==='Constructed forced abort') request.addEventListener('success',()=>this.transaction.abort());
          return request;
        };
        let rejected=false;
        try {await Promise.race([save(new Map([[1,[{id:'constructed-new',question:'Constructed forced abort',answer:'Constructed answer',tags:['new']}]]])),new Promise((_,reject)=>setTimeout(()=>reject(new Error('Constructed save timeout')),2000))]);}
        catch {rejected=true;}
        finally {IDBObjectStore.prototype.put=originalPut;}
        const afterAbort=await dump();
        const memoryUnchanged=modalSets.get(1)[0].id==='constructed-memory';
        if (JSON.stringify(afterAbort)!==JSON.stringify(before)) return {before,afterAbort,rejected,memoryUnchanged};
        await save(new Map([[1,[{id:'constructed-old',question:'Constructed updated question',answer:'Constructed updated answer',tags:['new']}]]]));
        const afterSave=await dump();
        // The inline editor's boolean argument is a current-deck save.
        handler.inMemoryData.currentDeck=[{id:'constructed-old',question:'Constructed editor update',answer:'Constructed answer',tags:[]}];
        await save(true);
        const afterBoolean=await dump();
        let invalidRejected=false;
        try {await save({chapter:1,cards:[]});} catch {invalidRejected=true;}
        const afterInvalid=await dump();
        let collisionRejected=false;
        try {await save(new Map([[1,[{id:'constructed-other',question:'Constructed collision',tags:[]}]]]));}
        catch {collisionRejected=true;}
        const afterCollision=await dump();
        return {before,afterAbort,rejected,memoryUnchanged,afterSave,afterBoolean,invalidRejected,afterInvalid,collisionRejected,afterCollision};
      });
      assert.ok(result.rejected);
      assert.deepEqual(result.afterAbort,result.before);
      assert.ok(result.memoryUnchanged);
      assert.equal(result.afterSave.chapters.find(chapter=>chapter.chapter===1).customMetadata,'retain chapter metadata');
      assert.equal(result.afterSave.cards.find(card=>card.id==='constructed-old').customMetadata,'retain card metadata');
      assert.equal(result.afterSave.current_chapter[0].customMetadata,'retain current metadata');
      assert.deepEqual(result.afterSave.cards.find(card=>card.id==='constructed-other'),result.before.cards.find(card=>card.id==='constructed-other'));
      assert.deepEqual(result.afterSave.card_tags.map(tag=>tag.tag).sort(),['new','other']);
      assert.equal(result.afterBoolean.cards.find(card=>card.id==='constructed-old').question,'Constructed editor update');
      assert.deepEqual(result.afterBoolean.card_tags.map(tag=>tag.tag),['other']);
      assert.ok(result.invalidRejected);
      assert.deepEqual(result.afterInvalid,result.afterBoolean);
      assert.ok(result.collisionRejected);
      assert.deepEqual(result.afterCollision,result.afterBoolean);
    }));

    await t.test('typed tag keys avoid numeric/string and delimiter collisions and migrate affected legacy keys', () => withPage(async page => {
      const result=await page.evaluate(async()=>{
        const {getDBInstance}=await import('/js/libs/utils/indexDb.js');
        const {SpacedRepetitionIndexDBHandler:Handler}=await import('/js/components/spaced_repetition/handlers/spaced-repetition-indexdb.js');
        const manager=await getDBInstance();
        const cards=[
          {id:1,chapter:1,question:'Constructed numeric ID',answer:'Constructed answer',tags:['common']},
          {id:'1',chapter:1,question:'Constructed string ID',answer:'Constructed answer',tags:['common']},
          {id:'a_b',chapter:1,question:'Constructed delimiter ID',answer:'Constructed answer',tags:['c']},
          {id:'a',chapter:1,question:'Constructed delimiter tag',answer:'Constructed answer',tags:['b_c']}
        ];
        for(const card of cards) await manager.put('cards',card);
        await manager.put('chapters',{chapter:1,title:'Constructed collision deck'});
        await manager.put('card_tags',{id:'1_common',card_id:1,tag:'common',customMetadata:'retain numeric tag metadata'});
        await manager.put('card_tags',{id:['number',1,'common'],card_id:1,tag:'common',canonicalMetadata:'retain existing canonical metadata'});
        await manager.put('card_tags',{id:'a_b_c',card_id:'a_b',tag:'c',customMetadata:'retain delimiter tag metadata'});
        await manager.put('card_tags',{id:'untouched_keep',card_id:'untouched',tag:'keep',customMetadata:'retain untouched legacy key'});
        const handler={dbManager:manager,inMemoryData:{currentChapter:{chapter:1,title:'Constructed collision deck'},currentDeck:cards,allTags:new Set()},modal:{chapterSets:new Map()},ensureInitialized:async()=>true,getChapterTitle:()=> 'Constructed collision deck'};
        handler.performTransaction=Handler.prototype.performTransaction.bind(handler);
        await Handler.prototype.saveToLocalStorage.call(handler,new Map([[1,cards]]));
        const afterSave=await manager.getAll('card_tags');
        // Exercise the import writer too, with both colliding ID types in one file.
        handler.inMemoryData.currentChapter=null;
        await Handler.prototype.mergeImportedData.call(handler,{chapter_card_sets:[{chapter:1,title:'Constructed imported collision deck',cards:cards.map(card=>({...card,answer:'Constructed imported answer'}))}]});
        const afterImport=await manager.getAll('card_tags');
        // insertCard historically resolved after card request success. A forced
        // later tag failure must now reject and roll the card back as well.
        const originalPut=IDBObjectStore.prototype.put;
        IDBObjectStore.prototype.put=function(value,...rest){
          const request=originalPut.call(this,value,...rest);
          if(this.name==='card_tags' && value.card_id==='constructed-aborted-insert') request.addEventListener('success',()=>this.transaction.abort());
          return request;
        };
        let insertRejected=false;
        try {await Handler.prototype.insertCard.call(handler,{id:'constructed-aborted-insert',chapter:1,question:'Constructed abort',answer:'Constructed answer',tags:['abort']},1);}
        catch {insertRejected=true;}
        finally {IDBObjectStore.prototype.put=originalPut;}
        return {afterSave,afterImport,insertRejected,expectedTags:cards.flatMap(card=>card.tags).length+1,abortedCard:await manager.get('cards','constructed-aborted-insert')};
      });
      for(const rows of [result.afterSave,result.afterImport]) {
        assert.equal(rows.length,result.expectedTags);
        assert.equal(rows.find(row=>row.card_id===1).canonicalMetadata,'retain existing canonical metadata');
        assert.equal(rows.find(row=>row.card_id===1).customMetadata,'retain numeric tag metadata');
        assert.equal(rows.find(row=>row.card_id==='a_b').customMetadata,'retain delimiter tag metadata');
        assert.ok(rows.filter(row=>row.card_id!=='untouched').every(row=>Array.isArray(row.id)));
        assert.equal(rows.find(row=>row.card_id==='untouched').id,'untouched_keep');
        assert.equal(new Set(rows.map(row=>JSON.stringify(row.id))).size,rows.length);
        assert.ok(rows.some(row=>row.card_id===1 && row.tag==='common'));
        assert.ok(rows.some(row=>row.card_id==='1' && row.tag==='common'));
        assert.ok(rows.some(row=>row.card_id==='a_b' && row.tag==='c'));
        assert.ok(rows.some(row=>row.card_id==='a' && row.tag==='b_c'));
      }
      assert.ok(result.insertRejected);
      assert.equal(result.abortedCard,undefined);
    }));

    await t.test('message flashcards use the named canonical deck without a legacy localStorage selection', () => withPage(async page => {
      const result=await page.evaluate(async()=>{
        const {getDBInstance}=await import('/js/libs/utils/indexDb.js');
        const {SpacedRepetitionIndexDBHandler:Handler}=await import('/js/components/spaced_repetition/handlers/spaced-repetition-indexdb.js');
        const {DuckAI}=await import('/js/libs/agents/duck-ai-cloudflare.js');
        const manager=await getDBInstance();
        const chapter={chapter:42,title:'Constructed named message deck',is_current:1};
        const handler={dbManager:manager,inMemoryData:{currentChapter:chapter,currentDeck:[],allTags:new Set()},modal:{chapterSets:new Map(),flashcards:[]},notificationHandler:{showLoadingState:()=> 'constructed-loading',updateNotification:()=>{}},ensureInitialized:async()=>true,getChapterTitle:()=>chapter.title};
        handler.performTransaction=Handler.prototype.performTransaction.bind(handler);
        handler.processFlashcards=Handler.prototype.processFlashcards.bind(handler);
        await Handler.prototype.addChapter.call(handler,chapter);
        localStorage.removeItem('current_chapter');
        const calls=[];
        const getInstance= DuckAI.getInstance;
        let generated=[{question:'Constructed generated question',answer:'Constructed generated answer',chapter:999}];
        DuckAI.getInstance=async()=>({async *generateFlashcards(text){calls.push(text);yield generated;}});
        try {
          const first=await Handler.prototype.addFlashcardsFromText.call(handler,'Constructed source message');
          handler.inMemoryData.currentChapter=null;
          const recovered=await Handler.prototype.addFlashcardsFromText.call(handler,'Constructed second source message');
          const multiline={question:'Constructed "quoted" question\n```python\nprint("bandwidth")\n```',answer:'Constructed first line\nSecond line with "quotes" and \\backslash'};
          generated=JSON.stringify([multiline]);
          const exactText=await Handler.prototype.addFlashcardsFromText.call(handler,'Constructed multiline source message');
          const {SpacedRepetitionModal}=await import('/js/components/spaced_repetition/spaced-repetition-modal-handler.js');
          const {copy_download}=await import('/js/components/settings/copy_download.js');
          const previousModal=SpacedRepetitionModal.instance;
          SpacedRepetitionModal.instance={storageHandler:{addFlashcardsFromText:Handler.prototype.addFlashcardsFromText.bind(handler)}};
          const host=document.createElement('div');document.body.appendChild(host);
          const root=host.attachShadow({mode:'open'});
          root.innerHTML='<div data-markdown="Constructed UI source message"><button class="st-send-btn">Send to flashcards</button></div>';
          generated=[{question:'Constructed UI-generated question',answer:'Constructed UI-generated answer'}];
          copy_download(root,root.firstElementChild);
          root.querySelector('.st-send-btn').click();
          for(let attempt=0;attempt<100&&!root.querySelector('.popover-socratiq');attempt++) await new Promise(resolve=>setTimeout(resolve,10));
          const notification=root.querySelector('.popover-socratiq')?.textContent;
          SpacedRepetitionModal.instance=previousModal;
          const beforeInvalid=await manager.getAll('cards');
          generated=[{question:'Constructed valid prefix',answer:'Constructed answer'},{question:'Constructed malformed generated card'}];
          let invalidRejected=false;
          try {await Handler.prototype.addFlashcardsFromText.call(handler,'Constructed invalid response request');} catch {invalidRejected=true;}
          const afterInvalid=await manager.getAll('cards');
          handler.inMemoryData.currentChapter=null;
          await manager.performTransaction('current_chapter','readwrite',store=>store.clear());
          await manager.put('chapters',{...chapter,is_current:0});
          localStorage.setItem('current_chapter',JSON.stringify({chapter:999,title:'Constructed stale legacy deck'}));
          let missingError='';
          try {await Handler.prototype.addFlashcardsFromText.call(handler,'Constructed missing selection request');} catch(error) {missingError=error.message;}
          return {first,recovered,multiline,exactText,notification,beforeInvalid,afterInvalid,invalidRejected,missingError,calls,
            chapterCards:await new Promise((resolve,reject)=>{const request=manager.db.transaction('cards','readonly').objectStore('cards').index('chapter').getAll(chapter.chapter);request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);}),
            modalCards:handler.modal.chapterSets.get(chapter.chapter)};
        } finally {DuckAI.getInstance=getInstance;}
      });
      assert.ok(result.first.success);
      assert.ok(result.recovered.success);
      assert.ok(Array.isArray(result.first.flashcards));
      assert.ok(Array.isArray(result.recovered.flashcards));
      assert.equal(result.exactText.flashcards[0].question,result.multiline.question);
      assert.equal(result.exactText.flashcards[0].answer,result.multiline.answer);
      assert.equal(result.notification,'Created 1 flashcard!');
      assert.ok(result.beforeInvalid.every(card=>card.chapter===42));
      assert.equal(result.chapterCards.length,result.beforeInvalid.length);
      assert.equal(result.modalCards.length,result.beforeInvalid.length);
      assert.ok(result.invalidRejected);
      assert.deepEqual(result.afterInvalid,result.beforeInvalid);
      assert.match(result.missingError,/Choose or create a flashcard deck/);
      assert.ok(!result.calls.includes('Constructed missing selection request'));
    }));

    await t.test('dashboard reports recorded results and cumulative outcomes without an invented chapter denominator', () => withPage(async page => {
      await page.route('**/js/index.js',route=>route.fulfill({contentType:'text/javascript',body:'export async function saveChatHistory() {}'}));
      await page.route('**/js/components/quiz/showQuizStats.js',async route=>{
        const response=await route.fetch();
        return route.fulfill({response,body:(await response.text())+'\nwindow.constructedChart = Chart;'});
      });
      const result=await page.evaluate(async()=>{
        const {getDBInstance}=await import('/js/libs/utils/indexDb.js');
        const manager=await getDBInstance();
        const pages=[{url:'https://constructed.test/section-results.html',title:'Constructed section results'},{url:'https://constructed.test/cumulative.html',title:'Constructed cumulative results'},{url:'https://constructed.test/no-results.html',title:'Constructed no recorded results'}];
        for(const page of pages) await manager.put('chapterMap',page);
        const history=[{quizType:'section',score:{passed:true}},{quizType:'section',score:{passed:false}}];
        await manager.put('cumulativeQuizScores',{normalizedUrl:pages[0].url,quizHistory:history});
        await manager.put('cumulativeQuizScores',{normalizedUrl:pages[1].url,completedByCumulativeQuiz:true,cumulativeQuizTitle:'Constructed cumulative pass',quizHistory:[{quizType:'cumulative',score:{passed:true}}]});
        const {getChapterProgressFromChapterMap,getDashboardChapterData}=await import('/js/components/quiz/chapterMapService.js');
        const progress=Object.fromEntries(await getChapterProgressFromChapterMap());
        const dashboard=await getDashboardChapterData();
        const host=document.createElement('div');document.body.appendChild(host);
        const root=host.attachShadow({mode:'open'});
        const {styles}=await import('/cssStyles.js');
        root.innerHTML='<style>'+styles+'</style><button id="chat-quiz-btn">Open results</button>';
        const {initStatsDisplay}=await import('/js/components/quiz/showQuizStats.js');
        initStatsDisplay(root);
        root.querySelector('#chat-quiz-btn').click();
        let chart;
        for(let attempt=0;attempt<100;attempt++) {
          chart=window.constructedChart.getChart(root.querySelector('#chapterProgressChart'));
          if(chart) break;
          await new Promise(resolve=>setTimeout(resolve,20));
        }
        if(!chart) throw new Error('Constructed results chart did not render');
        await new Promise(resolve=>setTimeout(resolve,1200));
        return {progress,dashboard,labels:chart.data.labels,datasets:chart.data.datasets.map(dataset=>({label:dataset.label,data:dataset.data})),axis:chart.options.scales.x.title.text,tooltip:chart.options.plugins.tooltip.callbacks.label({dataIndex:0}),legend:root.querySelector('.custom-legend').textContent};
      });
      const section=result.progress['https://constructed.test/section-results.html'];
      const cumulative=result.progress['https://constructed.test/cumulative.html'];
      const empty=result.progress['https://constructed.test/no-results.html'];
      assert.equal(section.totalQuizzes,null);
      assert.equal(section.completionPercentage,null);
      assert.equal(section.totalAttempted,2);
      assert.equal(section.passedQuizzes,1);
      assert.equal(section.passRate,50);
      assert.equal(cumulative.totalQuizzes,null);
      assert.equal(cumulative.passedQuizzes,0);
      assert.ok(cumulative.completedByCumulativeQuiz);
      assert.equal(cumulative.completionBasis,'cumulative-quiz');
      assert.equal(empty.completionPercentage,null);
      assert.equal(empty.passRate,null);
      assert.ok(result.dashboard.every(chapter=>chapter.totalQuizzes===null));
      assert.equal(result.axis,'Recorded section quiz results');
      assert.equal(result.datasets[0].data.reduce((sum,value)=>sum+value,0),1);
      assert.equal(result.datasets[1].data.reduce((sum,value)=>sum+value,0),1);
      assert.ok(result.labels.flat().includes('Cumulative quiz passed'));
      assert.ok(result.datasets.every(dataset=>!/(Completed|Remaining)/.test(dataset.label)));
      assert.match(result.legend,/Recorded results do not measure chapter completion/);
      if(process.env.NIGHT_STORAGE_PROGRESS_SCREENSHOT) {
        await page.screenshot({path:process.env.NIGHT_STORAGE_PROGRESS_SCREENSHOT,fullPage:true});
        await page.locator('.stats-body').evaluate(element=>{element.scrollTop=300;});
        await page.screenshot({path:process.env.NIGHT_STORAGE_PROGRESS_SCREENSHOT.replace('.png','-chart.png'),fullPage:true});
        await page.setViewportSize({width:375,height:812});
        await page.locator('.stats-body').evaluate(element=>{element.scrollTop=0;});
        await page.waitForTimeout(1200);
        await page.screenshot({path:process.env.NIGHT_STORAGE_PROGRESS_SCREENSHOT.replace('.png','-phone.png'),fullPage:true});
        await page.locator('.stats-body').evaluate(element=>{element.scrollTop=350;});
        await page.screenshot({path:process.env.NIGHT_STORAGE_PROGRESS_SCREENSHOT.replace('.png','-phone-chart.png'),fullPage:true});
      }
    }));

    await t.test('cumulative completion keeps its passing cumulative result after later section failures', () => withPage(async page => {
      const result=await page.evaluate(async()=>{
        const {getDBInstance}=await import('/js/libs/utils/indexDb.js');
        const manager=await getDBInstance();
        const pages=['history','legacy','unknown'].map(name=>({url:`https://constructed.test/cumulative-${name}.html`,title:`Constructed cumulative ${name}`}));
        for(const page of pages) await manager.put('chapterMap',page);
        await manager.put('cumulativeQuizScores',{normalizedUrl:pages[0].url,completedByCumulativeQuiz:true,lastQuizType:'section',lastQuizScore:{percentage:10},cumulativeQuizTitle:'Constructed later failed cumulative',quizHistory:[
          {quizType:'cumulative',quizTitle:'Constructed earlier passing cumulative',score:{passed:true,percentage:70}},
          {quizType:'cumulative',quizTitle:'Constructed retained passing cumulative',score:{passed:true,percentage:80}},
          {quizType:'cumulative',quizTitle:'Constructed later failed cumulative',score:{passed:false,percentage:30}},
          {quizType:'section',quizTitle:'Constructed latest section fail',score:{passed:false,percentage:10}}
        ]});
        const legacyScore={correct:3,total:4};
        await manager.put('pageCompletions',{id:'constructed-legacy-completion',pageUrl:pages[1].url,completedByCumulativeQuiz:true,cumulativeQuizTitle:'Constructed legacy cumulative pass',score:legacyScore});
        await manager.put('cumulativeQuizScores',{normalizedUrl:pages[1].url,completedByCumulativeQuiz:true,lastQuizType:'section',lastQuizScore:{percentage:20},quizHistory:[]});
        await manager.put('cumulativeQuizScores',{normalizedUrl:pages[2].url,completedByCumulativeQuiz:true,lastQuizType:'section',lastQuizScore:{percentage:15},quizHistory:[{quizType:'section',score:{passed:false,percentage:15}}]});
        const {getChapterProgressFromChapterMap}=await import('/js/components/quiz/chapterMapService.js');
        return {progress:Object.fromEntries(await getChapterProgressFromChapterMap()),legacyPercentage:(legacyScore.correct/legacyScore.total)*100};
      });
      const history=result.progress['https://constructed.test/cumulative-history.html'];
      const legacy=result.progress['https://constructed.test/cumulative-legacy.html'];
      const unknown=result.progress['https://constructed.test/cumulative-unknown.html'];
      assert.ok(history.completedByCumulativeQuiz);
      assert.equal(history.cumulativeScore,80);
      assert.equal(history.cumulativeQuizTitle,'Constructed retained passing cumulative');
      assert.equal(legacy.cumulativeScore,result.legacyPercentage);
      assert.equal(legacy.cumulativeQuizTitle,'Constructed legacy cumulative pass');
      assert.equal(unknown.cumulativeScore,null);
      assert.equal(unknown.cumulativeQuizTitle,null);
    }));

    await t.test('chapter progress uses canonical URL and avoids shared-number or shared-title credit', () => withPage(async page => {
      const result=await page.evaluate(async()=>{
        const {getDBInstance}=await import('/js/libs/utils/indexDb.js');
        const manager=await getDBInstance();
        const pages=[
          {url:'https://constructed.test/vol1/training.html',title:'1 Training Systems',tocData:[{text:'1.2 Memory Planning'}]},
          {url:'https://constructed.test/vol2/training.html',title:'1 Training Systems'},
          {url:'https://constructed.test/vol2/data.html',title:'2 Data Systems'}
        ];
        for(const page of pages) await manager.put('chapterMap',page);
        for(const score of [
          {quizTitle:'1 Training Systems',percentageScore:90},
          {quizTitle:'2.1 Unrelated Topic',percentageScore:90},
          {quizTitle:'1.2 Memory Planning',percentageScore:90},
          {quizTitle:'URL identified quiz',percentageScore:90,pageUrl:'https://constructed.test/vol2/training.html?scroll=5#part'},
          {quizId:'constructed-unassigned',quizTitle:'1.2 Memory Planning unassigned',title:'1.2 Memory Planning',pageUrl:null,percentageScore:90},
          {quizId:'constructed-multipage',quizTitle:'Constructed multi-page quiz',pageUrl:'https://constructed.test/vol2/training.html',sourcePageUrls:['https://constructed.test/vol1/training.html','https://constructed.test/vol2/training.html'],percentageScore:90}
        ]) await manager.put('quizHighScores',score);
        const {getChapterProgressFromChapterMap}=await import('/js/components/quiz/chapterMapService.js');
        return Object.fromEntries(await getChapterProgressFromChapterMap());
      });
      assert.equal(result['https://constructed.test/vol1/training.html'].passedQuizzes,1);
      assert.equal(result['https://constructed.test/vol2/training.html'].passedQuizzes,1);
      assert.equal(result['https://constructed.test/vol2/data.html'].passedQuizzes,0);
    }));
  } finally {
    if (browser) await browser.close();
    server.kill();
  }
});
