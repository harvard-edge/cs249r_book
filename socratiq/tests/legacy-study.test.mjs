import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {readFile} from 'node:fs/promises';
import {chromium} from '@playwright/test';

test('legacy study reviews, deck ownership, backups, and saved-message export', {timeout:90000}, async () => {
  const server=spawn(process.execPath,['node_modules/vite/bin/vite.js','--config','vite.config.dev.mjs','--host','127.0.0.1','--port','4187','--strictPort'],{cwd:new URL('..',import.meta.url),stdio:'ignore'});
  let browser;
  try {
    for(let attempt=0;attempt<100;attempt++) {
      try {if((await fetch('http://127.0.0.1:4187')).ok) break;} catch {}
      await new Promise(resolve=>setTimeout(resolve,100));
    }
    browser=await chromium.launch({headless:true});
    const page=await browser.newPage();
    await page.route('**/*',route=>(new URL(route.request().url()).origin === 'https://mlsysbook.ai' && new URL(route.request().url()).pathname.startsWith('/api/socratiq/'))?route.abort():route.continue());
    const failures=[];
    page.on('pageerror',error=>failures.push(error.message));
    page.on('console',message=>{if(/Error (parsing imported|importing data)/.test(message.text())) failures.push(message.text());});
    await page.goto('http://127.0.0.1:4187/test_website/encryption_textbook/index.html?socratiq=true');
    const click=async selector=>{await page.locator(selector).evaluate(element=>element.click());await page.waitForTimeout(700);};
    await page.locator('#flashcards-button').waitFor({state:'attached'});
    await page.waitForTimeout(3000);
    await click('#flashcards-button');
    await click('#newDeckBtn');
    await page.locator('#deckNameInput').fill('Constructed regression deck');
    await click('#createDeck');
    await click('#showAddCard');
    await page.locator('[contenteditable=true]').nth(0).fill('Constructed regression question');
    await page.locator('[contenteditable=true]').nth(1).fill('Constructed regression answer');
    await click('#saveCard');
    await click('#startReview');
    await click('#flipCardBtn');
    await page.getByRole('button',{name:'Good',exact:true}).last().evaluate(element=>element.click());
    await page.waitForTimeout(500);
    // Reuse the existing review modal and grade again: one action is one review.
    await page.locator('#cardReviewModal button').filter({hasText:'ESC'}).evaluate(element=>element.click());
    await click('#startReview');
    await page.getByRole('button',{name:'Good',exact:true}).last().evaluate(element=>element.click());
    await page.waitForTimeout(500);
    await page.reload();
    await page.waitForTimeout(2000);
    await click('#flashcards-button');
    await click('#sr-download-btn');
    const downloading=page.waitForEvent('download');
    await click('.json-btn');
    const backup=JSON.parse(await readFile(await (await downloading).path(),'utf8'));
    const card=backup.chapter_card_sets.flatMap(set=>set.cards).find(card=>card.question==='Constructed regression question');
    assert.equal(card.repetitions,2);
    assert.equal(card.reviewHistory.length,2);
    assert.notEqual(card.chapter,0);
    assert.equal(backup.current_chapter.title,'Constructed regression deck');
    assert.equal(card.chapter,backup.current_chapter.chapter);
    const currentChapters=await page.evaluate(async()=>{
      const {getDBInstance}=await import('/js/libs/utils/indexDb.js');
      const manager=await getDBInstance();
      return new Promise((resolve,reject)=>{
        const request=manager.db.transaction('chapters','readonly').objectStore('chapters').getAll();
        request.onsuccess=()=>resolve(request.result.filter(chapter=>chapter.is_current===1));
        request.onerror=()=>reject(request.error);
      });
    });
    assert.equal(currentChapters.length,1);
    assert.equal(currentChapters[0].title,'Constructed regression deck');

    card.question='Constructed imported regression question';
    const choosing=page.waitForEvent('filechooser');
    await click('#sr-import-btn');
    await (await choosing).setFiles({name:'constructed-backup.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(backup))});
    await page.waitForTimeout(1000);
    await page.reload();
    await page.waitForTimeout(2000);
    await click('#flashcards-button');
    assert.ok(await page.getByText('Constructed imported regression question',{exact:true}).count());
    await click('#sr-download-btn');
    const downloadingAnki=page.waitForEvent('download');
    await click('.anki-btn');
    const anki=await readFile(await (await downloadingAnki).path(),'utf8');
    assert.ok(anki.includes('Constructed imported regression question'));
    const importingAnki=page.waitForEvent('filechooser');
    await click('#sr-import-btn');
    await (await importingAnki).setFiles({name:'constructed-anki.txt',mimeType:'text/plain',buffer:Buffer.from(anki.replace('Constructed imported regression question','Constructed Anki imported question'))});
    await page.waitForTimeout(1000);
    await page.reload();
    await page.waitForTimeout(2000);
    await click('#flashcards-button');
    await click('#sr-download-btn');
    const exportedAnki=page.waitForEvent('download');
    await click('.json-btn');
    const importedAnki=JSON.parse(await readFile(await (await exportedAnki).path(),'utf8'));
    const importedAnkiCard=importedAnki.chapter_card_sets.flatMap(set=>set.cards).find(card=>card.question==='Constructed Anki imported question');
    assert.ok(importedAnkiCard);
    assert.equal(importedAnkiCard.chapter,0);
    await click('[data-chapter="0"]:not([data-card-index])');
    assert.ok(await page.getByText('Constructed Anki imported question',{exact:true}).last().isVisible());
    const history=await page.locator('#message-container').evaluate(async element=>{
      const root=element.getRootNode();
      const storage=await import('/js/libs/utils/save_chats.js');
      const helpers=await import('/js/libs/utils/utils.js');
      element.innerHTML='<div class="ai-message-chat"><div class="text-sm text-zinc-800">Constructed history regression fixture</div></div>';
      helpers.add_copy_paste_share_buttons(root,element.firstElementChild,null,'Constructed history regression fixture');
      const id=await storage.determineAndSaveChat(root,null,'Constructed history');
      element.innerHTML='';
      return {loaded:await storage.loadChat(root,id),text:element.textContent};
    });
    assert.equal(history.loaded,true);
    assert.ok(history.text.includes('Constructed history regression fixture'));
    await page.reload();
    await page.waitForTimeout(2000);
    assert.ok((await page.locator('#message-container').textContent()).includes('Constructed history regression fixture'));
    const exportingMessage=page.waitForEvent('download');
    await page.locator('#message-container [title="Download"]').evaluate(element=>element.click());
    assert.equal(await readFile(await (await exportingMessage).path(),'utf8'),'Constructed history regression fixture');
    assert.deepEqual(failures,[]);
  } finally {await browser?.close();server.kill();}
});
