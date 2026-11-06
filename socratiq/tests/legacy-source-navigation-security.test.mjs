// Constructed saved-chat metadata, actual restore/tooltip code, and local browser
// fixtures. No model calls, external navigation, or operator credentials.
import test from 'node:test';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {readFile} from 'node:fs/promises';
import {createServer as createHttpServer} from 'node:http';
import {createHash} from 'node:crypto';
import {createServer} from 'vite';
import {testBrowser} from './legacy-test-browser.mjs';
import {safeSourceUrl, safeSourcePosition} from '../src_shadow/js/components/tooltip/source-url.js';

const invalidUrls = [
  'javascript:void(window.sourceNavigationProbe=true)',
  'java\nscript:void(window.sourceNavigationProbe=true)',
  'data:text/html,<script>window.sourceNavigationProbe=true</script>',
  'file:///tmp/constructed-source.html',
  'http://[',
  'https://user:password@example.test/source',
];
const htmlAttribute = value => value.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
let server, browser, origin, bundleServer, bundleOrigin;

test.before(async () => {
  server = await createServer({
    configFile: fileURLToPath(new URL('../vite.config.dev.mjs', import.meta.url)),
    root: fileURLToPath(new URL('../src_shadow', import.meta.url)),
    server: {host: '127.0.0.1', port: 0, open: false},
    logLevel: 'error',
  });
  await server.listen();
  origin = `http://127.0.0.1:${server.httpServer.address().port}`;
  browser = await testBrowser.launch({headless: true});
  const reader = await readFile(new URL('../../books/shared/assets/scripts/socratiq/reader.js', import.meta.url));
  const study = await readFile(new URL('../../books/shared/assets/scripts/socratiq/study.js', import.meta.url));
  console.info(`Source-navigation study bundle: ${createHash('sha256').update(study).digest('hex')}`);
  bundleServer = createHttpServer((request, response) => {
    const url = new URL(request.url, 'http://fixture');
    if (url.pathname === '/assets/reader.js' || url.pathname === '/assets/study.js') {
      response.setHeader('Content-Type', 'text/javascript');
      return response.end(url.pathname.endsWith('reader.js') ? reader : study);
    }
    if (url.pathname === '/favicon.ico') {response.writeHead(204); return response.end();}
    response.setHeader('Content-Type', 'text/html');
    response.end('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Constructed built source navigation fixture</title><script src="/assets/reader.js" data-volume="vol1" defer></script></head><body style="min-height:2000px"><main id="quarto-document-content"><h1>Constructed chapter</h1><section><h2>Source</h2><p>Training adjusts parameters; inference applies them.</p></section></main></body></html>');
  });
  await new Promise(resolve => bundleServer.listen(0, '127.0.0.1', resolve));
  bundleOrigin = `http://127.0.0.1:${bundleServer.address().port}`;
});
test.after(async () => {await browser?.close(); await server?.close(); if (bundleServer) await new Promise(resolve => bundleServer.close(resolve));});

async function fixture(run) {
  const context = await browser.newContext({viewport: {width: 1100, height: 800}, reducedMotion: 'reduce'});
  const page = await context.newPage();
  const errors = [], external = [], navigationFixtures = [];
  page.setDefaultTimeout(10000);
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/*', route => {
    const url = new URL(route.request().url());
    if (url.href === 'https://example.test/constructed-source') {
      navigationFixtures.push(url.href);
      return route.fulfill({contentType: 'text/html', body: '<!doctype html><title>Constructed HTTPS source</title><p>Local intercepted destination.</p>'});
    }
    if (url.origin !== origin) {external.push(url.href); return route.abort();}
    if (url.pathname.startsWith('/navigation-security-fixture')) {
      return route.fulfill({contentType: 'text/html', body: '<!doctype html><html lang="en"><title>Constructed source navigation fixture</title><body style="min-height:2000px;padding-top:180px"><div id="host" style="margin-left:320px"></div></body></html>'});
    }
    return route.continue();
  });
  await page.goto(`${origin}/navigation-security-fixture`);
  try {
    await run({page, external, errors, navigationFixtures});
    assert.deepEqual(errors, [], 'Source metadata must not produce uncaught browser errors.');
    assert.deepEqual(external, [], 'All external requests are blocked and unexpected attempts fail the test.');
  } finally {await context.close();}
}

async function restore(page, sourceUrl, position = '0', sourceLabel = 'Constructed source') {
  const html = `<div class="ai-message-chat"><span class="reference-number" data-source-reference="Constructed saved passage" data-source-label="${htmlAttribute(sourceLabel)}" data-source-position="${htmlAttribute(String(position))}" data-source-url="${htmlAttribute(sourceUrl)}">Saved source</span><a id="javascript-href-control" href="javascript:void(window.hrefControlProbe=true)">href control</a></div>`;
  const restored = await page.evaluate(async html => {
    const {getDBInstance} = await import('/js/libs/utils/indexDb.js');
    const {loadChat} = await import('/js/libs/utils/save_chats.js');
    const manager = await getDBInstance();
    await manager.add('tinyMLDB_chats', {id: 1, title: 'Constructed saved source', html_content: html, date: new Date().toISOString()});
    const root = document.querySelector('#host').attachShadow({mode: 'open'});
    root.innerHTML = '<div id="message-container"></div>';
    const loaded = await loadChat(root, 1);
    return {loaded, href: root.querySelector('#javascript-href-control')?.getAttribute('href')};
  }, html);
  assert.equal(restored.loaded, true, 'Actual saved-chat restore must finish.');
  assert.equal(restored.href, null, 'DOMPurify removes the standard javascript href control.');
  const pill = page.locator('#host .reference-number');
  await pill.hover();
  const action = page.locator('#reference-tooltip .go-to-source');
  await action.waitFor({state: 'visible'});
  return {pill, action};
}

test('source URL parser rejects executable, local, malformed, credentialed, and nontext metadata', () => {
  const base = 'https://mlsysbook.ai/vol1/chapter/';
  for (const value of [...invalidUrls, '', ' ', undefined, null, {}, []]) assert.equal(safeSourceUrl(value, base), '', String(value));
  assert.equal(safeSourceUrl('../source/#passage', base), 'https://mlsysbook.ai/vol1/source/#passage');
  assert.equal(safeSourceUrl('#passage', base), `${base}#passage`);
  assert.equal(safeSourceUrl('https://example.test/source', base), 'https://example.test/source');
  for (const value of [Infinity, -1, NaN, 'Infinity', 'not-a-number', {}, null]) assert.equal(safeSourcePosition(value), 0);
  assert.equal(safeSourcePosition('80'), 80);
});

for (const sourceUrl of invalidUrls) {
  test(`restored source tooltip disables unsafe navigation: ${JSON.stringify(sourceUrl)}`, {timeout: 60000}, async () => {
    await fixture(async ({page}) => {
      const {action} = await restore(page, sourceUrl, 'Infinity', '');
      assert.equal(await action.isDisabled(), true);
      assert.equal(await action.getAttribute('data-url'), null);
      const before = page.url();
      const box = await action.boundingBox();
      // A physical click on a disabled native button must have no effect.
      await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
      assert.equal(page.url(), before);
      assert.equal(await page.evaluate(() => Boolean(window.sourceNavigationProbe || window.hrefControlProbe)), false);
    });
  });
}

test('raw navigation events reject unsafe URLs and missing detail without browser errors', {timeout: 60000}, async () => {
  await fixture(async ({page}) => {
    await restore(page, 'javascript:void(window.sourceNavigationProbe=true)');
    const before = page.url();
    await page.evaluate(values => {
      document.dispatchEvent(new CustomEvent('navigateToSource', {bubbles: true, composed: true}));
      for (const detail of [null, {}, [], 'invalid', ...values.map(url => ({url, position: 0}))]) {
        document.dispatchEvent(new CustomEvent('navigateToSource', {detail, bubbles: true, composed: true}));
      }
    }, invalidUrls);
    assert.equal(page.url(), before);
    assert.equal(await page.evaluate(() => Boolean(window.sourceNavigationProbe)), false);
  });
});

test('restored relative source navigates through a native tooltip click', {timeout: 60000}, async () => {
  await fixture(async ({page}) => {
    const {action} = await restore(page, '/navigation-security-fixture-relative#passage');
    assert.equal(await action.isEnabled(), true);
    await action.click();
    await page.waitForURL(`${origin}/navigation-security-fixture-relative#passage`);
  });
});

test('restored same-page source scrolls without navigation', {timeout: 60000}, async () => {
  await fixture(async ({page}) => {
    const before = page.url();
    const {action} = await restore(page, before, '80');
    await action.click();
    await page.waitForFunction(() => window.scrollY > 0);
    assert.equal(page.url(), before);
  });
});

test('restored HTTPS source uses its validated destination without external network', {timeout: 60000}, async () => {
  await fixture(async ({page, navigationFixtures}) => {
    const {action} = await restore(page, 'https://example.test/constructed-source');
    await action.click();
    await page.waitForURL('https://example.test/constructed-source');
    assert.deepEqual(navigationFixtures, ['https://example.test/constructed-source']);
  });
});

async function builtFixture(sourceUrl, position, run) {
  const context = await browser.newContext({viewport: {width: 1200, height: 900}, reducedMotion: 'reduce'});
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  const errors = [], external = [], httpsNavigations = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/*', route => {
    const url = new URL(route.request().url());
    if (url.href === 'https://example.test/constructed-source') {
      httpsNavigations.push(url.href);
      return route.fulfill({contentType: 'text/html', body: '<!doctype html><title>Constructed HTTPS source</title>'});
    }
    if (url.origin === bundleOrigin) return route.continue();
    external.push(url.href);
    return route.abort();
  });
  try {
    await page.goto(`${bundleOrigin}/vol1/chapter/`);
    const resolvedSource = sourceUrl === 'same-page' ? page.url() : sourceUrl;
    const html = `<div class="ai-message-chat"><span class="reference-number" data-source-label="Constructed source" data-source-reference="Constructed saved passage" data-source-position="${position}" data-source-url="${htmlAttribute(resolvedSource)}">Saved source</span></div>`;
    await page.evaluate(async html => {
      localStorage.setItem('socratiq_reader_tour_v1', 'true');
      localStorage.setItem('mostRecentID', '1');
      await new Promise((resolve, reject) => {
        const request = indexedDB.open('socratiqDB', 1);
        request.onupgradeneeded = () => request.result.createObjectStore('tinyMLDB_chats', {keyPath: 'id', autoIncrement: true});
        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
          const db = request.result;
          const transaction = db.transaction('tinyMLDB_chats', 'readwrite');
          transaction.objectStore('tinyMLDB_chats').put({id: 1, title: 'Constructed stored source', html_content: html, date: new Date().toISOString()});
          transaction.oncomplete = () => {db.close(); resolve();};
          transaction.onerror = () => reject(transaction.error);
        };
      });
    }, html);
    await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).click();
    const tour = page.getByRole('dialog', {name: 'SocratiQ guided tour'});
    if (await tour.isVisible()) await tour.getByRole('button', {name: 'Skip tour', exact: true}).click();
    await page.getByRole('button', {name: 'Study tools', exact: true}).click();
    const pill = page.locator('#widget-chat-container .reference-number');
    await pill.waitFor({state: 'visible', timeout: 60000});
    await pill.hover();
    const action = page.locator('#reference-tooltip .go-to-source');
    await action.waitFor({state: 'visible'});
    await run({page, pill, action, httpsNavigations});
    assert.deepEqual(errors, [], 'The complete built widget must reject malformed navigation events without errors.');
    assert.deepEqual(external, [], 'The complete built widget must not attempt an unexpected external request.');
  } finally {await context.close();}
}

test('complete built widget blocks restored executable source metadata and every raw navigation listener', {timeout: 90000}, async () => {
  await builtFixture(invalidUrls[0], 0, async ({page, action}) => {
    assert.equal(await action.isDisabled(), true);
    assert.equal(await action.getAttribute('data-url'), null);
    assert.notEqual(await action.evaluate(element => getComputedStyle(element).cursor), 'pointer', 'A disabled source action must not advertise a clickable pointer.');
    const before = page.url();
    const box = await action.boundingBox();
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
    await page.evaluate(values => {
      document.dispatchEvent(new CustomEvent('navigateToSource', {bubbles: true, composed: true}));
      for (const detail of [null, {}, [], 'invalid', ...values.map(url => ({url, position: 0}))]) {
        document.dispatchEvent(new CustomEvent('navigateToSource', {detail, bubbles: true, composed: true}));
      }
    }, invalidUrls);
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    assert.equal(page.url(), before);
    assert.equal(await page.evaluate(() => Boolean(window.sourceNavigationProbe)), false);
  });
});

test('complete built widget preserves native relative source navigation', {timeout: 90000}, async () => {
  await builtFixture('/constructed-relative-source#passage', 0, async ({page, action}) => {
    await action.click();
    await page.waitForURL(`${bundleOrigin}/constructed-relative-source#passage`);
  });
});

test('complete built widget preserves native same-page source scrolling', {timeout: 90000}, async () => {
  await builtFixture('same-page', 80, async ({page, action}) => {
    const before = page.url();
    await action.click();
    await page.waitForFunction(() => window.scrollY > 0);
    assert.equal(page.url(), before);
  });
});

test('complete built widget preserves validated HTTPS navigation through an intercepted fixture', {timeout: 90000}, async () => {
  await builtFixture('https://example.test/constructed-source', 0, async ({page, action, httpsNavigations}) => {
    await action.click();
    await page.waitForURL('https://example.test/constructed-source');
    assert.deepEqual(httpsNavigations, ['https://example.test/constructed-source']);
  });
});
