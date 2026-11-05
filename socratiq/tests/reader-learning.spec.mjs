import {test, expect} from '@playwright/test';

const endpoint = 'https://proxy-worker-streaming.mlsysbook.workers.dev/**';
const answer = '**Bandwidth** moves data.';
const sse = text => `data: ${JSON.stringify({choices: [{delta: {content: text}, finish_reason: null}]})}\n\ndata: ${JSON.stringify({choices: [{delta: {}, finish_reason: 'stop'}]})}\n\ndata: [DONE]\n\n`;
const payload = request => {
  const content = request.postDataJSON().messages.at(-1).content;
  return JSON.parse(content.slice(content.indexOf('{'), content.lastIndexOf('}') + 1));
};
async function openSelected(page) {
  await page.goto('/vol1/chapter/');
  await page.locator('#passage').evaluate(node => {
    const range = document.createRange(); range.selectNodeContents(node);
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    document.dispatchEvent(new Event('selectionchange'));
  });
  await expect(page.getByRole('button', {name: 'Explain this', exact: true})).toBeVisible();
  await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).click();
}

test('learning actions retain source and completed history, and launcher sends nothing', async ({page}) => {
  const requests = [];
  await page.route(endpoint, route => {
    requests.push(payload(route.request()));
    return route.fulfill({contentType: 'text/event-stream', body: sse(answer)});
  });
  await openSelected(page);
  expect(requests).toHaveLength(0);
  const input = page.getByRole('textbox', {name: 'Ask a follow-up question'});
  await input.fill('Keep my draft');
  for (const label of ['Simplify', 'Show an example', 'Check my understanding']) {
    await page.getByRole('button', {name: label, exact: true}).click();
    await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
    await expect(input).toHaveValue('Keep my draft');
  }
  expect(requests[0].history).toEqual([]);
  expect(requests[1].history).toEqual([{role: 'user', content: requests[0].question}, {role: 'assistant', content: answer}]);
  expect(requests[2].source).toEqual(requests[0].source);
  expect(requests[2].question).toContain('one conceptual question at a time');
  expect(requests[2].question).toContain('Do not reveal the answer until I respond');
  await input.fill('I think it measures data movement.');
  await page.getByRole('button', {name: 'Send question'}).click();
  await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
  expect(requests[3].history.at(-2).content).toEqual(requests[2].question);
});

test('start over aborts pending work, clears draft and history, and retains passage', async ({page}) => {
  const requests = [];
  let release;
  const pending = new Promise(resolve => {release = resolve;});
  await page.route(endpoint, async route => {
    requests.push(payload(route.request()));
    if (requests.length === 1) await pending;
    await route.fulfill({contentType: 'text/event-stream', body: sse(answer)}).catch(() => {});
  });
  await openSelected(page);
  const input = page.getByRole('textbox', {name: 'Ask a follow-up question'});
  await input.fill('Draft to clear');
  await page.getByRole('button', {name: 'Simplify', exact: true}).click();
  await expect.poll(() => requests.length).toBe(1);
  await expect(page.getByRole('button', {name: 'Show an example', exact: true})).toBeDisabled();
  await page.getByRole('button', {name: 'Start over', exact: true}).click();
  release();
  await expect(input).toHaveValue('');
  await expect(page.locator('#socratiq-reader .reader-turn')).toHaveCount(0);
  await expect(page.locator('#socratiq-reader .reader-source-link')).toBeVisible();
  await page.getByRole('button', {name: 'Show an example', exact: true}).click();
  await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
  expect(requests[1].history).toEqual([]);
  expect(requests[1].source).toEqual(requests[0].source);
  await expect(page.locator('#socratiq-reader .reader-turn')).toHaveCount(2);
});

test('copy keeps Markdown and source provenance and reports clipboard rejection', async ({page}) => {
  await page.addInitScript(() => {
    window.copiedNotes = [];
    Object.defineProperty(navigator, 'clipboard', {configurable: true, value: {writeText: async text => {
      if (window.rejectCopy) throw new Error('Denied');
      window.copiedNotes.push(text);
    }}});
  });
  await page.route(endpoint, route => route.fulfill({contentType: 'text/event-stream', body: sse(answer)}));
  await openSelected(page);
  await page.getByRole('button', {name: 'Simplify', exact: true}).click();
  const copy = page.getByRole('button', {name: 'Copy answer as Markdown'});
  await copy.click();
  await expect(page.getByText('Answer copied as Markdown.', {exact: true})).toBeVisible();
  const copied = await page.evaluate(() => window.copiedNotes[0]);
  expect(copied).toContain(answer);
  expect(copied).toContain('SocratiQ AI explanation');
  expect(copied).toContain('Source:');
  expect(copied).toContain('/vol1/chapter/');
  expect(copied).toContain('> Memory bandwidth');
  await page.evaluate(() => {window.rejectCopy = true;});
  await copy.click();
  await expect(page.getByText('Could not copy the answer. Select the answer text and copy it manually.', {exact: true})).toBeVisible();
  await expect(copy).toBeEnabled();
});

test('failed quick action keeps draft and start over clears retry state', async ({page}) => {
  let calls = 0;
  await page.route(endpoint, route => {
    calls += 1;
    return route.fulfill({status: 400, contentType: 'application/json', body: JSON.stringify({error: 'Invalid request'})});
  });
  await openSelected(page);
  const input = page.getByRole('textbox', {name: 'Ask a follow-up question'});
  await input.fill('A question I am still writing');
  await page.getByRole('button', {name: 'Simplify', exact: true}).click();
  await expect(page.getByRole('button', {name: 'Retry', exact: true})).toBeVisible();
  await expect(input).toHaveValue('A question I am still writing');
  await page.getByRole('button', {name: 'Start over', exact: true}).click();
  await expect(input).toHaveValue('');
  await expect(page.getByRole('button', {name: 'Retry', exact: true})).toBeHidden();
  await expect(page.locator('#socratiq-reader .reader-turn')).toHaveCount(0);
  expect(calls).toBe(1);
});

test('keyboard users can focus and scroll the answer region without trapping book navigation', async ({page}) => {
  const longAnswer = 'A paragraph about bandwidth and latency.\n\n'.repeat(60);
  await page.route(endpoint, route => route.fulfill({contentType: 'text/event-stream', body: sse(longAnswer)}));
  await openSelected(page);
  await page.getByRole('button', {name: 'Simplify', exact: true}).click();
  await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
  const body = page.getByRole('region', {name: 'Selected passage and answers'});
  await page.getByRole('button', {name: 'Close SocratiQ reader'}).focus();
  await page.keyboard.press('Tab');
  await expect(body).toBeFocused();
  await body.evaluate(node => {node.scrollTop = 0;});
  await page.keyboard.press('PageDown');
  await expect.poll(() => body.evaluate(node => node.scrollTop)).toBeGreaterThan(0);
  await page.keyboard.press('Escape');
  await expect(page.getByRole('button', {name: 'Ask SocratiQ', exact: true})).toBeFocused();
});

test('short mobile viewport leaves room to read an answer with the composer visible', async ({page}, testInfo) => {
  await page.setViewportSize({width: 390, height: 360});
  await page.route(endpoint, route => route.fulfill({contentType: 'text/event-stream', body: sse(answer)}));
  await openSelected(page);
  await page.getByRole('button', {name: 'Simplify', exact: true}).click();
  await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
  const body = page.getByRole('region', {name: 'Selected passage and answers'});
  await body.evaluate(node => {node.scrollTop = node.scrollHeight;});
  const readingSpace = await body.evaluate(node => {
    const source = node.querySelector('.reader-source');
    const answerNode = node.querySelector('.reader-answer');
    const pinnedHeight = getComputedStyle(source).position === 'sticky' ? source.offsetHeight : 0;
    return {height: node.clientHeight - pinnedHeight, answerBottom: answerNode.getBoundingClientRect().bottom, sourceBottom: source.getBoundingClientRect().bottom};
  });
  expect(readingSpace.height).toBeGreaterThan(40);
  expect(readingSpace.answerBottom).toBeGreaterThan(readingSpace.sourceBottom);
  await expect(page.getByRole('button', {name: 'Send question'})).toBeInViewport();
  await page.screenshot({path: testInfo.outputPath('short-mobile.png')});
});
