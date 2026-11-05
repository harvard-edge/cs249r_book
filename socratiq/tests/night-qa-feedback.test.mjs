import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {chromium} from '@playwright/test';

test('feedback assigns the existing form URL on open without submitting', async () => {
  const source = await readFile(new URL('../src_shadow/js/components/feedback/openFeedback.js', import.meta.url));
  const server = createServer((request, response) => {
    if (request.url === '/feedback.js') {
      response.setHeader('Content-Type', 'text/javascript'); response.end(source); return;
    }
    response.setHeader('Content-Type', 'text/html');
    response.end(`<!doctype html><style>.hidden{display:none}</style><button id="openModal-feedback">Feedback</button><div id="modal_feedback"></div><script type="module">import {injectFeedback,openFeedback} from '/feedback.js';injectFeedback(document);openFeedback(document);</script>`);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({headless: true});
    const page = await browser.newPage();
    const requests = [];
    await page.route('https://docs.google.com/**', route => {
      requests.push({url: route.request().url(), method: route.request().method()});
      return route.fulfill({contentType: 'text/html', body: '<title>Constructed blocked external form</title>'});
    });
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.locator('#feedback-iframe').waitFor({state: 'attached'});
    assert.equal(await page.locator('#feedback-iframe').getAttribute('src'), '');
    assert.deepEqual(requests, []);
    const expected = 'https://docs.google.com/forms/d/e/1FAIpQLSeK8RXgc6kbT1IbWVLjyUhwowp3x1ySbAjUQQqztdDs5ccmmQ/viewform?embedded=true';
    const requested = page.waitForRequest(request => request.url() === expected);
    await page.getByRole('button', {name: 'Feedback', exact: true}).click();
    assert.equal(await page.locator('#feedback-iframe').getAttribute('src'), expected);
    await requested;
    assert.deepEqual(requests, [{url: expected, method: 'GET'}]);
    await page.getByRole('button', {name: 'Close', exact: true}).click();
    assert.equal(await page.locator('#myModal').isVisible(), false);
  } finally {
    await browser?.close();
    await new Promise(resolve => server.close(resolve));
  }
});
