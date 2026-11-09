import {test, expect} from '@playwright/test';

// Constructed responses exercise UI behavior; live inference is a separate opt-in smoke check.
const answer = 'Bandwidth is the amount transferred per unit time. For a transfer:\n\n$$t = D / B$$\n\n**Example:** a wider road carries more cars.';
function sse(text) {
  return `data: ${JSON.stringify({choices: [{delta: {content: text}, finish_reason: null}]})}\n\ndata: ${JSON.stringify({choices: [{delta: {}, finish_reason: 'stop'}]})}\n\ndata: [DONE]\n\n`;
}
async function select(page, selector = '#passage') {
  await page.locator(selector).evaluate(element => {
    const range = document.createRange(); range.selectNodeContents(element);
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    document.dispatchEvent(new Event('selectionchange'));
  });
}
async function explain(page) {
  await select(page);
  await page.getByRole('button', {name: 'Explain this', exact: true}).click();
}

for (const volume of ['vol1', 'vol2', 'vol3', 'vol4']) {
  test(`${volume}: selection, contextual explanation, math, follow-up`, async ({page}, testInfo) => {
    const requests = [];
    await page.route('https://mlsysbook.ai/api/socratiq/ai/stream**', async route => {
      const request = route.request();
      const url = new URL(request.url());
      expect(url.origin + url.pathname).toBe('https://mlsysbook.ai/api/socratiq/ai/stream');
      expect(url.searchParams.get('provider')).toBe('groq');
      expect(url.searchParams.get('url')).toBe('https://api.groq.com/openai/v1/chat/completions');
      expect(request.headers()).not.toHaveProperty('authorization');
      expect(request.headers()).not.toHaveProperty('x-api-key');
      requests.push(route.request().postDataJSON());
      await route.fulfill({contentType: 'text/event-stream', body: sse(answer)});
    });
    await page.goto(`/${volume}/chapter/?private=do-not-send`);
    await expect(page.getByRole('button', {name: 'Ask SocratiQ'})).toBeVisible();
    expect(requests).toHaveLength(0);
    await explain(page);
    await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
    await expect(page.locator('#socratiq-reader .reader-footnote')).toHaveText(
      'SocratiQ uses AI and can make mistakes. Check claims in the book.',
    );
    await expect(page.getByRole('button', {name: 'Stop explanation'})).toBeHidden();
    await expect(page.getByRole('button', {name: 'Retry', exact: true})).toBeHidden();
    await expect(page.locator('#socratiq-reader .reader-thread .katex')).toBeVisible();
    const data = JSON.stringify(requests[0]);
    expect(requests[0].messages[0].role).toBe('system');
    expect(requests[0].messages[0].content).toContain('Learner calibration');
    expect(requests[0].messages[1].role).toBe('user');
    expect(data).not.toMatch(/apiKey|GROQ_API_KEY/);
    expect(data).toContain(volume);
    expect(data).toContain('Memory bandwidth measures');
    expect(data).not.toContain('Navigation must not');
    expect(data).not.toContain('Unrelated content');
    expect(data).not.toContain('private=');
    await page.getByRole('textbox', {name: 'Ask a follow-up question'}).fill('How is that different from latency?');
    await page.getByRole('button', {name: 'Send question'}).click();
    await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
    expect(requests).toHaveLength(2);
    expect(JSON.stringify(requests[1])).toContain('wider road');
    if (volume === 'vol1') await page.screenshot({path: testInfo.outputPath('desktop.png')});
    await page.keyboard.press('Escape');
    await expect(page.getByRole('region', {name: 'SocratiQ focused reader'})).toBeHidden();
  });
}

test('mobile selection and dark panel fit viewport', async ({page}, testInfo) => {
  await page.setViewportSize({width: 390, height: 844});
  await page.emulateMedia({colorScheme: 'dark'});
  await page.route('https://mlsysbook.ai/api/socratiq/ai/stream**', route => route.fulfill({contentType: 'text/event-stream', body: sse(answer)}));
  await page.goto('/vol4/chapter/');
  await explain(page);
  await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
  const panel = page.getByRole('region', {name: 'SocratiQ focused reader'});
  const rect = await panel.boundingBox();
  expect(rect.x).toBeGreaterThanOrEqual(0); expect(rect.x + rect.width).toBeLessThanOrEqual(390);
  expect(rect.y).toBeGreaterThanOrEqual(0); expect(rect.y + rect.height).toBeLessThanOrEqual(844);
  await expect(page.getByRole('button', {name: 'Send question'})).toBeInViewport();
  await expect(page.locator('#socratiq-reader .reader-footnote')).toBeInViewport();
  await page.screenshot({path: testInfo.outputPath('mobile-dark.png')});
});

test('unavailable service offers retry without fabricating an explanation', async ({page}) => {
  let attempts = 0;
  await page.route('https://mlsysbook.ai/api/socratiq/ai/stream**', route => {
    attempts++;
    return route.fulfill(attempts === 1 ? {status: 503, body: 'private provider diagnostic'} : {contentType: 'text/event-stream', body: sse(answer)});
  });
  await page.goto('/vol1/chapter/'); await explain(page);
  await expect(page.getByRole('button', {name: 'Retry', exact: true})).toBeVisible();
  await expect(page.locator('#socratiq-reader')).not.toContainText('private provider diagnostic');
  await page.getByRole('button', {name: 'Retry', exact: true}).click();
  await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
});

test('model HTML is inert; math and code remain distinct', async ({page}) => {
  const malicious = '<img src=x onerror="window.readerInjected=true"><script>window.readerInjected=true</script>\n\n![tracking image](https://example.com/tracker.png)\n\n[unsafe](javascript:alert(1))\n\n[Source](https://example.com/source)\n\n`$code$` and \\(t = D/B\\)';
  await page.route('https://mlsysbook.ai/api/socratiq/ai/stream**', route => route.fulfill({contentType: 'text/event-stream', body: sse(malicious)}));
  await page.goto('/vol2/chapter/'); await explain(page);
  await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
  expect(await page.evaluate(() => window.readerInjected)).toBeUndefined();
  await expect(page.locator('#socratiq-reader .reader-thread img')).toHaveCount(0);
  expect(await page.locator('#socratiq-reader .reader-thread [onerror], #socratiq-reader .reader-thread script, #socratiq-reader .reader-thread a[href^="javascript:"]').count()).toBe(0);
  await expect(page.locator('#socratiq-reader .reader-thread code')).toHaveText('$code$');
  await expect(page.locator('#socratiq-reader .reader-thread .katex')).toBeVisible();
  await expect(page.getByRole('link', {name: 'Source', exact: true})).toHaveAttribute('target', '_blank');
  await expect(page.getByRole('link', {name: 'Source', exact: true})).toHaveAttribute('rel', 'noopener noreferrer');
});

test('context excludes navigation, preserves TeX, and rejects oversized selections', async ({page}) => {
  await page.goto('/vol3/chapter/');
  await select(page, '#math');
  const source = await page.evaluate(async () => (await import('/context.js')).captureSelection({volume: 'vol3'}));
  expect(source.quote).toContain('$t = D / B$');
  expect(source.quote).not.toContain('duplicated visual math');
  await select(page, 'nav');
  expect(await page.evaluate(async () => (await import('/context.js')).captureSelection())).toBeNull();
  await page.locator('#passage').evaluate(element => {element.textContent = 'long '.repeat(1000);});
  await select(page);
  expect(await page.evaluate(async () => {try {(await import('/context.js')).captureSelection(); return '';} catch (error) {return error.message;}})).toContain('shorter passage');
});

test('stopping a request prevents late output from replacing a new passage', async ({page}) => {
  let releaseFirst;
  let startedFirst;
  const firstStarted = new Promise(resolve => {startedFirst = resolve;});
  const firstResponse = new Promise(resolve => {releaseFirst = resolve;});
  let calls = 0;
  await page.route('https://mlsysbook.ai/api/socratiq/ai/stream**', async route => {
    calls++;
    if (calls === 1) {
      startedFirst();
      await firstResponse;
      await route.fulfill({contentType: 'text/event-stream', body: sse('Obsolete answer from the cancelled request.')}).catch(() => {});
    } else await route.fulfill({contentType: 'text/event-stream', body: sse('Explanation of the new passage.')});
  });
  await page.goto('/vol1/chapter/'); await explain(page); await firstStarted;
  await page.getByRole('button', {name: 'Stop explanation'}).click();
  await page.getByRole('button', {name: 'Close SocratiQ reader'}).click();
  await select(page, '#other p');
  await page.getByRole('button', {name: 'Explain this', exact: true}).click();
  await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
  releaseFirst();
  await expect(page.locator('#socratiq-reader .reader-thread')).toContainText('Explanation of the new passage.');
  await expect(page.locator('#socratiq-reader .reader-thread')).not.toContainText('Obsolete answer');
});
