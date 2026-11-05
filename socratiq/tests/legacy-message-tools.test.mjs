// Authored chapter and constructed SSE responses verify UI wiring, not model quality.
import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile, mkdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {chromium, expect} from '@playwright/test';
const assets = new URL('../../books/shared/assets/scripts/socratiq/', import.meta.url);
const passage = 'Memory bandwidth measures how much data can move per unit time.';
const cardQuestion = 'What does memory bandwidth measure?';
async function fixture(run) {
  const server = createServer(async (request, response) => {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (pathname.startsWith('/assets/')) {
      response.setHeader('Content-Type', 'text/javascript; charset=utf-8');
      response.end(await readFile(new URL(pathname.endsWith('/study.js') ? 'study.js' : 'reader.js', assets)));
    } else {
      response.setHeader('Content-Type', 'text/html; charset=utf-8');
      response.end(`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Constructed message chapter</title><script src="/assets/reader.js" data-volume="vol2" defer></script></head><body><main id="quarto-document-content"><h1>Memory and latency</h1><section id="memory"><h2>Memory bandwidth</h2><p>${passage}</p><p>Latency measures how long an individual request takes.</p></section></main></body></html>`);
    }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const errors = [], requests = [];
  page.setDefaultTimeout(10000);
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error' && message.text().startsWith('Failed to create flashcards:')) errors.push(message.text());
  });
  await page.route('**/*', async route => {
    const url = new URL(route.request().url());
    if (url.origin === origin) return route.continue();
    if (!url.hostname.endsWith('workers.dev')) return route.abort();
    const body = route.request().postData() || '';
    requests.push(body);
    const paragraphId = body.match(/\(id=([^:]+):url=/)?.[1];
    const content = body.includes('flashcards') ? JSON.stringify([{question: cardQuestion, answer: passage}]) : `${passage}\n\n{{ref:${paragraphId}:Source passage:${origin}/vol2/chapter/}}`;
    return route.fulfill({status: 200, contentType: 'text/event-stream', body: `data: ${JSON.stringify({choices: [{delta: {content}, finish_reason: null}]})}\n\ndata: ${JSON.stringify({choices: [{delta: {}, finish_reason: 'stop'}]})}\n\ndata: [DONE]\n\n`});
  });
  const study = page.locator('#widget-chat-container');
  async function open() {
    await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).click();
    await page.getByRole('button', {name: 'Study tools', exact: true}).click();
    await expect(study.getByRole('button', {name: 'Close study tools', exact: true})).toBeVisible({timeout: 60000});
  }
  try {
    await page.goto(`${origin}/vol2/chapter/`); await open();
    await run({page, study, origin, requests, open});
    assert.deepEqual(errors, []);
  } catch (error) {
    const output = process.env.SOCRATIQ_MESSAGE_TEST_OUTPUT || join(tmpdir(), 'socratiq-message-tools');
    await mkdir(output, {recursive: true});
    await page.screenshot({path: join(output, `failure-${Date.now()}.png`)}).catch(() => {});
    throw error;
  } finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
}
async function ask(study) {
  await study.locator('#user-input').fill('Explain memory bandwidth.');
  await study.locator('#enter-btn').click();
  const message = study.locator('#message-container .ai-message-chat').last();
  await expect(message).toContainText(passage);
  await expect(message.locator('.st-send-btn')).toBeVisible();
  return message;
}
test('built message saves a generated card in the visibly selected named deck without legacy chapter storage', {timeout: 90000}, async () => {
  await fixture(async ({page, study, requests, open}) => {
    await study.locator('#flashcards-button').click();
    await study.locator('#newDeckBtn').click();
    await study.locator('#deckNameInput').fill('Constructed message deck');
    await study.locator('#createDeck').click();
    await study.getByText('Constructed message deck', {exact: true}).click();
    await study.locator('#closeModal_sr').click();
    assert.equal(await page.evaluate(() => localStorage.getItem('current_chapter')), null);
    const message = await ask(study);
    await message.locator('.st-send-btn').click();
    await expect.poll(() => requests.some(body => body.includes('flashcards')), {timeout: 15000}).toBe(true);
    await study.locator('#flashcards-button').click();
    await expect(study.getByText(cardQuestion, {exact: true}).filter({visible: true}).first()).toBeVisible({timeout: 15000});
    await study.locator('#closeModal_sr').click();
    await page.reload(); await open();
    await study.locator('#flashcards-button').click();
    await study.getByText('Constructed message deck', {exact: true}).click();
    await expect(study.getByText(cardQuestion, {exact: true}).filter({visible: true}).first()).toBeVisible({timeout: 15000});
  });
});
test('built source pill highlights its passage and preserves the source URL after reload', {timeout: 90000}, async () => {
  await fixture(async ({page, study, origin, open}) => {
    const message = await ask(study);
    const reference = message.getByText('Source passage', {exact: true});
    await expect(reference).toBeVisible();
    assert.equal(await reference.getAttribute('data-source-url'), `${origin}/vol2/chapter/`);
    await reference.click();
    await expect(page.locator('main .custom-reference-highlight')).toHaveText(passage);
    await expect.poll(() => page.evaluate(() => new Promise(resolve => {
      const request = indexedDB.open('socratiqDB');
      request.onerror = () => resolve(false);
      request.onsuccess = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains('tinyMLDB_chats')) { db.close(); resolve(false); return; }
        const rows = db.transaction('tinyMLDB_chats', 'readonly').objectStore('tinyMLDB_chats').getAll();
        rows.onsuccess = () => { db.close(); resolve(JSON.stringify(rows.result).includes('Source passage')); };
        rows.onerror = () => { db.close(); resolve(false); };
      };
    })), {timeout: 15000}).toBe(true);
    await page.reload(); await open();
    const restored = study.getByText('Source passage', {exact: true});
    await expect(restored).toBeVisible({timeout: 60000});
    assert.equal(await restored.getAttribute('data-source-url'), `${origin}/vol2/chapter/`);
    await restored.click();
    await expect(page.locator('main .custom-reference-highlight')).toHaveText(passage);
  });
});
