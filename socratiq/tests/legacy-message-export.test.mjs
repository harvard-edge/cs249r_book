// Constructed message exercises the production Copy/Download/Share handlers.
// No provider request, inference, posting, or deployment is performed.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile, mkdtemp} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import {chromium, expect} from '@playwright/test';

const markdown = 'Constructed shared roundtrip **passage**.\n\n```python\nx = 1\nprint(x)\n```';

test('message copy, download, and share preserve multiline Markdown verbatim', {timeout: 60000}, async () => {
  const packageInfo = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));
  const root = fileURLToPath(new URL('../src_shadow', import.meta.url));
  const cacheDir = await mkdtemp(join(tmpdir(), 'socratiq-export-test-'));
  const server = await createServer({
    root, configFile: false, cacheDir,
    server: {host: '127.0.0.1', port: 0, strictPort: true, fs: {allow: [fileURLToPath(new URL('..', import.meta.url)), cacheDir]}},
    optimizeDeps: {noDiscovery: true, include: Object.keys(packageInfo.dependencies)},
    plugins: [{name: 'constructed-export-message', configureServer(server) {
      server.middlewares.use((request, response, next) => {
        if (request.url !== '/export-fixture') return next();
        response.setHeader('Content-Type', 'text/html; charset=utf-8');
        response.end('<!doctype html><title>Constructed export fixture</title><h1>Constructed message exports</h1><div id="message-host"></div>');
      });
    }}],
  });
  let browser;
  try {
    await server.listen();
    const origin = `http://127.0.0.1:${server.httpServer.address().port}`;
    browser = await chromium.launch({headless: true});
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', route => new URL(route.request().url()).origin === origin ? route.continue() : route.abort());
    await page.goto(`${origin}/export-fixture`);
    await page.evaluate(async markdown => {
      Object.defineProperty(navigator, 'clipboard', {configurable: true, value: {writeText: async text => {window.copiedExport = text;}}});
      const shadow = document.querySelector('#message-host').attachShadow({mode: 'open'});
      shadow.innerHTML = '<style>button{display:block;width:120px;height:44px;margin:8px}button svg{width:20px;height:20px}</style><div class="p-3 relative max-w-xs bg-white"><p class="text-sm text-zinc-800">Rendered message text</p><button title="Copy">Copy</button><button title="Download">Download</button><button title="Share">Share</button></div>';
      const message = shadow.querySelector('div');
      message.setAttribute('data-markdown', markdown);
      const {copy_download} = await import('/js/components/settings/copy_download.js');
      copy_download(shadow, message);
    }, markdown);
    const host = page.locator('#message-host');
    await host.locator('[title="Copy"]').click();
    await expect.poll(() => page.evaluate(() => window.copiedExport)).toBe(markdown);
    const downloading = page.waitForEvent('download');
    await host.locator('[title="Download"]').click();
    assert.equal(await readFile(await (await downloading).path(), 'utf8'), markdown);
    await host.locator('[title="Share"]').click();
    const url = new URL(await page.evaluate(() => window.copiedExport));
    const shared = JSON.parse(decodeURIComponent(url.searchParams.get('shared_messages')));
    assert.equal(shared.markdown, markdown);
    assert.equal(shared.type, 'ai-message');
    assert.deepEqual(errors, []);
  } finally {
    await browser?.close();
    await server.close();
  }
});
