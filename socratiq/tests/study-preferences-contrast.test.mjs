import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile, mkdir, writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {testBrowser} from './legacy-test-browser.mjs';

test('production study learner preference text meets normal-text contrast in both themes', {timeout: 60000}, async () => {
  const bundle = await readFile(new URL('../../books/shared/assets/scripts/socratiq/study.js', import.meta.url));
  const output = process.env.SOCRATIQ_TEST_RECEIPT_DIR || join(tmpdir(), 'socratiq-study-preferences-contrast');
  await mkdir(output, {recursive: true});
  const server = createServer((request, response) => {
    if (request.url === '/study.js') {response.setHeader('Content-Type', 'text/javascript'); response.end(bundle); return;}
    response.setHeader('Content-Type', 'text/html');
    response.end('<!doctype html><html><meta name="viewport" content="width=device-width,initial-scale=1"><body><main><h1>Constructed contrast fixture</h1><p>A textbook passage.</p></main><script>window.__SOCRATIQ_EMBEDDED__=true</script><script type="module" src="/study.js"></script></body></html>');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const browser = await testBrowser.launch({headless: true});
  const receipts = [];
  try {
    for (const theme of ['light', 'dark']) {
      const page = await browser.newPage({viewport: {width: 390, height: 844}, colorScheme: theme});
      await page.addInitScript(theme => {window.SocratiqWidgetTheme = theme;}, theme);
      await page.route('https://**/*', route => route.abort());
      await page.goto(`http://127.0.0.1:${server.address().port}/`);
      await page.waitForFunction(() => window.SocratiqStudy?.open);
      await page.evaluate(() => window.SocratiqStudy.open());
      const study = page.locator('#widget-chat-container');
      await study.locator('#settings-btn').click();
      await study.locator('#study-learner-preferences summary').click();
      await study.locator('#study-learner-save').click();
      const colors = await study.locator('#study-learner-preferences').evaluate(root => {
        const rgb = value => (value.match(/[\d.]+/g) || []).map(Number).slice(0, 3);
        const luminance = color => rgb(color).map(channel => {
          const value = channel / 255; return value <= .04045 ? value / 12.92 : ((value + .055) / 1.055) ** 2.4;
        }).reduce((sum, value, index) => sum + value * [.2126, .7152, .0722][index], 0);
        return [...root.querySelectorAll('p, label, summary')].filter(element => element.textContent.trim()).map(element => {
          let node = element; let background;
          while (node) {
            const candidate = getComputedStyle(node).backgroundColor;
            if (candidate !== 'rgba(0, 0, 0, 0)' && candidate !== 'transparent') {background = candidate; break;}
            node = node.parentElement || node.getRootNode().host;
          }
          if (!background) throw new Error('No opaque background for contrast measurement');
          const color = getComputedStyle(element).color;
          const foregroundLuminance = luminance(color), backgroundLuminance = luminance(background);
          return {text: element.textContent.trim().slice(0, 100), color, background,
            ratio: (Math.max(foregroundLuminance, backgroundLuminance) + .05) / (Math.min(foregroundLuminance, backgroundLuminance) + .05)};
        });
      });
      receipts.push({theme, colors});
      await writeFile(join(output, 'contrast-results.json'), JSON.stringify(receipts, null, 2));
      for (const result of colors) assert.ok(result.ratio >= 4.5, `${theme}: ${result.text} contrast ${result.ratio} must be >= 4.5`);
      await page.screenshot({path: join(output, `settings-${theme}.png`)});
      await page.close();
    }
  } finally {await browser.close(); await new Promise(resolve => server.close(resolve));}
});
