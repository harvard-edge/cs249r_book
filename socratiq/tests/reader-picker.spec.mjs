import {test, expect} from '@playwright/test';

test.use({hasTouch: true, viewport: {width: 390, height: 844}});
const endpoint = 'https://proxy-worker-streaming.mlsysbook.workers.dev/**';
const sse = text => `data: ${JSON.stringify({choices: [{delta: {content: text}, finish_reason: null}]})}\n\ndata: [DONE]\n\n`;

test('tap a paragraph previews it without changing selection or sending until asked', async ({page}, info) => {
  const requests = [];
  await page.route(endpoint, route => {
    requests.push(route.request().postDataJSON());
    return route.fulfill({contentType: 'text/event-stream', body: sse('Constructed explanation for the selected paragraph.')});
  });
  await page.goto('/vol1/chapter/');
  await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).tap();
  await page.getByRole('button', {name: 'Choose a paragraph', exact: true}).tap();
  await expect(page.getByRole('region', {name: 'Choose a book paragraph'})).toBeVisible();
  await page.screenshot({path: info.outputPath('choosing-phone.png')});
  await page.locator('#passage').tap();
  await expect(page.locator('.reader-quote')).toContainText('Memory bandwidth');
  await expect(page.getByRole('button', {name: 'Explain passage', exact: true})).toBeVisible();
  expect(requests).toHaveLength(0);
  expect(await page.evaluate(() => getSelection().toString())).toBe('');
  await page.screenshot({path: info.outputPath('preview-phone.png')});
  await page.getByRole('button', {name: 'Explain passage', exact: true}).tap();
  await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
  expect(requests).toHaveLength(1);
  expect(JSON.stringify(requests[0])).toContain('Memory bandwidth');
  await page.getByRole('button', {name: 'Choose another paragraph', exact: true}).tap();
  await page.locator('#other p').tap();
  expect(requests).toHaveLength(1);
  await page.screenshot({path: info.outputPath('recent-phone.png')});
  const recent = page.getByRole('combobox', {name: 'Recent passages'});
  await recent.selectOption({index: 0});
  await expect(page.locator('.reader-answer')).toContainText('Constructed explanation');
  expect(requests).toHaveLength(1);
});

test('picking is opt-in, cancellable, and leaves navigation links alone', async ({page}) => {
  let calls = 0;
  await page.route(endpoint, route => {calls += 1; return route.abort();});
  await page.goto('/vol1/chapter/');
  await page.locator('#passage').tap();
  await expect(page.getByRole('region', {name: 'SocratiQ focused reader'})).toBeHidden();
  await page.locator('#other').evaluate(node => node.insertAdjacentHTML('beforeend', '<p><a id="book-link" href="#bandwidth">Book navigation</a></p>'));
  await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).tap();
  await page.getByRole('button', {name: 'Choose a paragraph', exact: true}).tap();
  await page.locator('#book-link').tap();
  await expect(page).toHaveURL(/#bandwidth$/);
  await expect(page.getByRole('region', {name: 'Choose a book paragraph'})).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('region', {name: 'Choose a book paragraph'})).toBeHidden();
  await expect(page.getByRole('region', {name: 'SocratiQ focused reader'})).toBeVisible();
  expect(calls).toBe(0);
});

test('an oversized paragraph stays unsent and offers recovery', async ({page}) => {
  await page.goto('/vol1/chapter/');
  await page.locator('#passage').evaluate(node => {node.textContent = 'long '.repeat(1000);});
  await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).tap();
  await page.getByRole('button', {name: 'Choose a paragraph', exact: true}).tap();
  await page.locator('#passage').tap({position: {x: 20, y: 20}});
  await expect(page.getByRole('region', {name: 'Choose a book paragraph'})).toContainText('too long');
  await page.getByRole('button', {name: 'Cancel picking', exact: true}).tap();
  await expect(page.getByRole('region', {name: 'SocratiQ focused reader'})).toBeVisible();
});
