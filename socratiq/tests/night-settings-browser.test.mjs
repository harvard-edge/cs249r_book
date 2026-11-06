import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { testBrowser } from './legacy-test-browser.mjs';
import { mkdir, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { tmpdir } from 'node:os';
const receipt = process.env.SOCRATIQ_TEST_RECEIPT_DIR
  ? pathToFileURL(`${resolve(process.env.SOCRATIQ_TEST_RECEIPT_DIR)}/`)
  : pathToFileURL(`${resolve(tmpdir(), 'socratiq-settings-tests')}/`);

test('constructed legacy settings and import corruption preserve stored data', { timeout: 90000 }, async () => {
  await mkdir(receipt, { recursive: true });
  const server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', '--config', 'vite.config.dev.mjs', '--host', '127.0.0.1', '--port', '4193', '--strictPort'], { cwd: new URL('..', import.meta.url), stdio: 'ignore' });
  let browser;
  const errors = [];
  try {
    for (let attempt = 0; attempt < 100; attempt++) {
      try { if ((await fetch('http://127.0.0.1:4193')).ok) break; } catch {}
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    browser = await testBrowser.launch({ headless: true });
    const page = await browser.newPage();
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', route => (new URL(route.request().url()).origin === 'https://mlsysbook.ai' && new URL(route.request().url()).pathname.startsWith('/api/socratiq/')) ? route.abort() : route.continue());
    await page.addInitScript(() => localStorage.setItem('userSettings', '{constructed broken JSON'));
    await page.goto('http://127.0.0.1:4193/test_website/encryption_textbook/index.html?socratiq=true');
    await page.locator('#flashcards-button').waitFor({ state: 'attached' });
    assert.equal(await page.evaluate(() => localStorage.getItem('userSettings')), '{constructed broken JSON');
    const settings = await page.evaluate(async () => {
      const { SettingsManager } = await import('/js/components/settings/settings.js');
      const cases = ['null', '[]', '42', '{}', '{"sliderValue":99,"checkboxes":null,"customAPI":[]}', '{"sliderValue":"3","checkboxes":{"show-answers":true},"customAPI":{"provider":"ollama"},"legacyField":"keep"}'];
      return cases.map(raw => {
        localStorage.setItem('userSettings', raw);
        const host = document.createElement('div');
        const root = host.attachShadow({ mode: 'open' });
        root.innerHTML = '<div id="modal1"><input id="understanding-slider" value="1"><input type="checkbox" id="show-answers"></div>';
        document.body.append(host);
        const manager = new SettingsManager(root);
        manager.init();
        const result = { raw, stored: localStorage.getItem('userSettings'), slider: manager.settings.sliderValue, checkboxes: manager.settings.checkboxes, customAPI: manager.settings.customAPI, legacyField: manager.settings.legacyField };
        host.remove();
        return result;
      });
    });
    for (const result of settings) { assert.equal(result.stored, result.raw); assert.ok(result.slider >= 0 && result.slider <= 3); assert.equal(typeof result.checkboxes, 'object'); assert.equal(typeof result.customAPI, 'object'); }
    assert.equal(settings.at(-1).slider, 3);
    assert.equal(settings.at(-1).checkboxes['show-answers'], true);
    assert.equal(settings.at(-1).legacyField, 'keep');
    const imports = await page.evaluate(async () => {
      const { SpacedRepetitionIndexDBHandler } = await import('/js/components/spaced_repetition/handlers/spaced-repetition-indexdb.js');
      const { getDBInstance } = await import('/js/libs/utils/indexDb.js');
      const handler = Object.create(SpacedRepetitionIndexDBHandler.prototype);
      handler.dbManager = await getDBInstance();
      handler.inMemoryData = { currentChapter: null };
      handler.useFallback = false;
      const snapshot = async () => {
        const result = {};
        for (const name of ['chapters', 'cards', 'card_tags', 'current_chapter']) result[name] = await new Promise((resolve, reject) => { const request = handler.dbManager.db.transaction(name, 'readonly').objectStore(name).getAll(); request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error); });
        return JSON.stringify(result);
      };
      const data = { chapter_card_sets: [{ chapter: 876, title: 'Constructed import deck', cards: [{ id: 'constructed-import', question: 'Constructed import question', answer: 'Constructed import answer', tags: ['constructed'], compatibility: 'keep' }] }] };
      await handler.mergeImportedData(data);
      data.chapter_card_sets[0].cards[0].tags = ['constructed-replacement'];
      await handler.mergeImportedData(data);
      const before = await snapshot();
      let rejected = false;
      try { await handler.mergeImportedData({ chapter_card_sets: [data.chapter_card_sets[0], { chapter: 877, cards: [null] }] }); } catch { rejected = true; }
      const after = await snapshot();
      const originalTransaction = handler.performTransaction;
      handler.performTransaction = function(names, mode, operation) {
        return originalTransaction.call(this, names, mode, (tx, resolve, reject) => {
          const originalStore = tx.objectStore.bind(tx);
          let puts = 0;
          tx.objectStore = name => {
            const store = originalStore(name);
            if (name === 'cards') { const originalPut = store.put.bind(store); store.put = card => { if (++puts === 2) throw new Error('Constructed write failure'); return originalPut(card); }; }
            return store;
          };
          operation(tx, resolve, reject);
        });
      };
      let writeRejected = false;
      try { await handler.mergeImportedData({ chapter_card_sets: [{ chapter: 879, cards: [{ id: 'constructed-rollback-1', question: 'Q', answer: 'A' }, { id: 'constructed-rollback-2', question: 'Q2', answer: 'A2' }] }] }); } catch { writeRejected = true; }
      await new Promise(resolve => setTimeout(resolve, 100));
      return { rejected, unchanged: before === after, writeRejected, rolledBack: before === await snapshot(), snapshot: JSON.parse(after) };
    });
    assert.equal(imports.rejected, true);
    assert.equal(imports.unchanged, true);
    assert.equal(imports.writeRejected, true);
    assert.equal(imports.rolledBack, true);
    const card = imports.snapshot.cards.find(card => card.id === 'constructed-import');
    assert.equal(card.chapter, 876);
    assert.equal(card.compatibility, 'keep');
    assert.deepEqual(imports.snapshot.card_tags.filter(tag => tag.card_id === card.id).map(tag => tag.tag), ['constructed-replacement']);
    assert.equal(imports.snapshot.chapters.find(chapter => chapter.chapter === 876).title, 'Constructed import deck');
    await page.screenshot({ path: new URL('legacy-settings-desktop.png', receipt).pathname, fullPage: false });
    await page.evaluate(async () => {
      const { getDBInstance } = await import('/js/libs/utils/indexDb.js');
      const manager = await getDBInstance();
      await new Promise((resolve, reject) => {
        const tx = manager.db.transaction('tinyMLDB_chats', 'readwrite');
        tx.objectStore('tinyMLDB_chats').put({ id: 'constructed-title', title: '<img src=x onerror="window.constructedTitleInjected=true">Constructed title' });
        tx.oncomplete = () => resolve(); tx.onerror = () => reject(tx.error);
      });
    });
    await page.evaluate(async () => {
      const { injectLoad_chats, setupModal_loadchats } = await import('/js/components/settings/previous_conversations.js');
      const host = document.createElement('div'); host.id = 'constructed-history-fixture';
      const root = host.attachShadow({ mode: 'open' });
      root.innerHTML = '<button id="new-chat-btn">Constructed history</button><div id="load_chats"></div>';
      document.body.append(host);
      const widgetRoot = document.querySelector('#socratiq-shadow-host')?.shadowRoot;
      const sourceRoot = widgetRoot || [...document.querySelectorAll('*')].map(element => element.shadowRoot).find(candidate => candidate?.querySelector('#message-container'));
      if (sourceRoot) { root.adoptedStyleSheets = sourceRoot.adoptedStyleSheets; sourceRoot.querySelectorAll('style').forEach(style => root.append(style.cloneNode(true))); }
      injectLoad_chats(root); setupModal_loadchats(root);
      root.querySelector('#new-chat-btn').click();
    });
    await page.locator('[id="chat-constructed-title"]').waitFor({ state: 'attached' });
    assert.equal(await page.locator('[id="chat-constructed-title"] img').count(), 0);
    assert.ok((await page.locator('[id="chat-constructed-title"]').textContent()).includes('<img src=x'));
    assert.equal(await page.evaluate(() => Boolean(window.constructedTitleInjected)), false);
    assert.deepEqual(errors, []);
    await page.screenshot({ path: new URL('history-title-desktop.png', receipt).pathname, fullPage: false });
    await writeFile(new URL('browser-results.json', receipt), JSON.stringify({ settings, imports, historyTitleRenderedAsText: true, pageErrors: errors }, null, 2));
  } finally { await browser?.close(); server.kill(); }
});
