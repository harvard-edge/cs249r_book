import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile, mkdtemp} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import {chromium, expect} from '@playwright/test';
import tailwindcss from 'tailwindcss';

test('embedded context button and keyboard mentions select context before sending', {timeout: 60000}, async () => {
  const packageInfo = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));
  const server = await createServer({
    root: fileURLToPath(new URL('../src_shadow', import.meta.url)), configFile: false,
    cacheDir: await mkdtemp(join(tmpdir(), 'socratiq-context-test-')),
    server: {host: '127.0.0.1', port: 0, strictPort: true},
    optimizeDeps: {noDiscovery: true, include: Object.keys(packageInfo.dependencies)},
    css: {postcss: {plugins: [tailwindcss(fileURLToPath(new URL('../tailwind.config.cjs', import.meta.url)))]}},
    plugins: [{name: 'constructed-context-chapter', configureServer(server) {
      server.middlewares.use((request, response, next) => {
        if (request.url !== '/qa/chapter/') return next();
        response.setHeader('Content-Type', 'text/html; charset=utf-8');
        response.end(`<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><title>Constructed context chapter</title><main id="quarto-document-content"><h1><span class="chapter-number">1</span> Memory systems</h1><section class="level2" id="bandwidth"><h2 class="anchored" data-number="1.1">Memory bandwidth</h2><h3>Memory details</h3><p>Constructed bandwidth context marker: data volume transferred per unit time.</p></section><section class="level2" id="latency"><h2 class="anchored" data-number="1.2">Latency</h2><p>Constructed latency context marker: duration of an individual request.</p></section></main><script type="module">window.__SOCRATIQ_EMBEDDED__=true;import('/js/index.js').then(()=>window.SocratiqStudy.open());</script>`);
      });
    }}],
  });
  let browser;
  try {
    await server.listen();
    browser = await chromium.launch({headless: true});
    const page = await browser.newPage();
    await page.addInitScript(() => {
      window.qaSent = [];
      window.addEventListener('aiActionCompleted', event => {
        if (event.detail?.type === 'query') window.qaSent.push(event.detail);
      });
    });
    const origin = `http://127.0.0.1:${server.httpServer.address().port}`;
    const requests = [];
    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      if (url.hostname.endsWith('workers.dev')) {
        requests.push(JSON.parse(route.request().postData()));
        return route.fulfill({contentType: 'text/event-stream', body: `data: ${JSON.stringify({choices: [{delta: {content: 'Constructed context response.'}}]})}\n\ndata: ${JSON.stringify({choices: [{delta: {}, finish_reason: 'stop'}]})}\n\ndata: [DONE]\n\n`});
      }
      return url.origin === origin ? route.continue() : route.abort();
    });
    await page.goto(`${origin}/qa/chapter/`);
    const study = page.locator('#widget-chat-container');
    await expect(study.getByRole('button', {name: 'Close study tools'})).toBeVisible();
    let sentRequests = 0;
    if (!process.env.SOCRATIQ_CONTEXT_KEYBOARD_ONLY) {
    await study.locator('#context-button').click();
    const contextMenu = study.locator('#context-menu');
    await expect(contextMenu).toBeVisible();
    await contextMenu.locator('[data-filter="section"]').click();
    await contextMenu.locator('.at-mention-item').filter({hasText: 'Memory bandwidth'}).click();
    await expect(study.locator('#copy-container')).toContainText('Constructed bandwidth context marker');
    await study.locator('#user-input').fill('Explain this selected section.');
    await study.locator('#enter-btn').click();
    await expect.poll(() => requests.length).toBe(1);
    assert.ok(requests[0].messages.some(message => message.content.includes('Constructed bandwidth context marker')));
    const selectedQuery = await page.evaluate(() => window.qaSent.at(-1).text);
    assert.ok(selectedQuery.includes('Constructed bandwidth context marker'));
    assert.ok(selectedQuery.includes('Explain this selected section.'));
    await expect(study.locator('#message-container')).toContainText('Constructed context response.');
    sentRequests++;
    }

    const sentBefore = await page.evaluate(() => window.qaSent.length);
    const input = study.locator('#user-input');
    await input.fill('@Memory');
    const mentionMenu = study.locator('.at-mentions-dropup:not(#context-menu)');
    await expect(mentionMenu).toBeVisible();
    await input.press('ArrowDown');
    await input.press('Enter');
    assert.equal(await page.evaluate(() => window.qaSent.length), sentBefore, 'Enter selects a mention instead of sending a question');
    assert.match(await input.inputValue(), /@\[/);
    await expect(mentionMenu).toBeHidden();
    await expect(study.locator('#copy-container')).toContainText('Constructed bandwidth context marker');
    await input.fill('Explain the keyboard-selected section.');
    await study.locator('#enter-btn').click();
    await expect.poll(() => requests.length).toBe(sentRequests + 1);
    assert.ok(requests.at(-1).messages.some(message => message.content.includes('Constructed bandwidth context marker')));
    assert.ok((await page.evaluate(() => window.qaSent.at(-1).text)).includes('Constructed bandwidth context marker'));
  } finally {
    await browser?.close();
    await server.close();
  }
});
