import {test, expect} from '@playwright/test';

const endpoint = 'https://mlsysbook.ai/api/socratiq/ai/stream**';
const sse = text => `data: ${JSON.stringify({choices: [{delta: {content: text}, finish_reason: null}]})}\n\ndata: ${JSON.stringify({choices: [{delta: {}, finish_reason: 'stop'}]})}\n\ndata: [DONE]\n\n`;
function payload(request) {
  const content = request.postDataJSON().messages.at(-1).content;
  return JSON.parse(content.slice(content.indexOf('{'), content.lastIndexOf('}') + 1));
}
async function explain(page, selector) {
  await page.locator(selector).evaluate(node => {
    const range = document.createRange(); range.selectNodeContents(node);
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
  });
  await page.keyboard.press('Control+Shift+E');
}
const picker = page => page.getByRole('combobox', {name: 'Recent passages'});
const input = page => page.getByRole('textbox', {name: 'Ask a follow-up question'});
const ready = page => expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
async function restore(page, quote) {
  const value = await picker(page).locator('option').evaluateAll((options, text) => options.find(option => option.textContent.includes(text)).value, quote);
  await picker(page).selectOption(value);
}

test('recent passages retain independent history, draft, copy listeners and Start over state', async ({page}) => {
  const requests = [];
  await page.addInitScript(() => Object.defineProperty(navigator, 'clipboard', {configurable: true, value: {writeText: async text => {window.copied = text;}}}));
  await page.route(endpoint, route => {
    requests.push(payload(route.request()));
    return route.fulfill({contentType: 'text/event-stream', body: sse(`Answer ${requests.length}`)});
  });
  await page.goto('/vol1/chapter/');
  await explain(page, '#passage'); await ready(page);
  await expect(picker(page)).toBeHidden();
  await input(page).fill('A draft about bandwidth');
  await explain(page, '#other p'); await ready(page);
  expect(requests[1].history).toEqual([]);
  expect(requests[1].source.quote).toContain('Unrelated content');
  await restore(page, 'Memory bandwidth');
  await expect(input(page)).toHaveValue('A draft about bandwidth');
  await expect(page.getByText('Answer 1', {exact: true})).toBeVisible();
  expect(requests).toHaveLength(2);
  await page.getByRole('button', {name: 'Copy answer as Markdown'}).click();
  expect(await page.evaluate(() => window.copied)).toContain('Answer 1');
  await input(page).fill('What is latency?');
  await page.getByRole('button', {name: 'Send question'}).click(); await ready(page);
  expect(requests[2].source).toEqual(requests[0].source);
  expect(requests[2].history).toEqual([{role: 'user', content: 'Explain this passage.'}, {role: 'assistant', content: 'Answer 1'}]);
  await explain(page, '#passage');
  expect(requests).toHaveLength(3);
  await page.getByRole('button', {name: 'Start over', exact: true}).click();
  await expect(page.locator('#socratiq-reader .reader-turn')).toHaveCount(0);
  await restore(page, 'Unrelated content');
  await expect(page.getByText('Answer 2', {exact: true})).toBeVisible();
  await page.getByRole('button', {name: 'Close SocratiQ reader'}).click();
  await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).click();
  await expect(page.locator('#socratiq-reader .reader-quote')).toContainText('Unrelated content');
  expect(requests).toHaveLength(3);
});

test('cancelled late response cannot modify restored passage and retry remains per passage', async ({page}) => {
  let release;
  const gate = new Promise(resolve => {release = resolve;});
  let calls = 0;
  await page.route(endpoint, async route => {
    calls += 1;
    if (calls === 1) {
      await gate;
      await route.fulfill({contentType: 'text/event-stream', body: sse('Late answer')}).catch(() => {});
    } else {
      await route.fulfill({status: 400, contentType: 'application/json', body: JSON.stringify({error: 'Invalid request'})});
    }
  });
  await page.goto('/vol1/chapter/');
  await explain(page, '#passage');
  await expect.poll(() => calls).toBe(1);
  await explain(page, '#other p');
  await expect(page.getByRole('button', {name: 'Retry', exact: true})).toBeVisible();
  await restore(page, 'Memory bandwidth');
  await expect(page.locator('#socratiq-reader .reader-answer')).toHaveText('Response stopped.');
  release();
  await expect(page.getByRole('button', {name: 'Retry', exact: true})).toBeHidden();
  await expect(page.getByText('Late answer', {exact: true})).toHaveCount(0);
  await restore(page, 'Unrelated content');
  await expect(page.getByRole('button', {name: 'Retry', exact: true})).toBeVisible();
  expect(calls).toBe(2);
});

test('recent passage memory evicts the oldest inactive passage', async ({page}) => {
  await page.route(endpoint, route => route.fulfill({contentType: 'text/event-stream', body: sse('Answer')}));
  await page.goto('/vol1/chapter/');
  for (let index = 0; index < 6; index += 1) {
    await page.locator('#passage').evaluate((node, value) => {node.textContent = `Passage number ${value}`;}, index);
    await explain(page, '#passage'); await ready(page);
  }
  await expect(picker(page).locator('option')).toHaveCount(5);
  await expect(picker(page).locator('option').filter({hasText: 'Passage number 0'})).toHaveCount(0);
  await expect(picker(page).locator('option').filter({hasText: 'Passage number 5'})).toHaveCount(1);
});
