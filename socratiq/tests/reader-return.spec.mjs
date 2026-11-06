import {test, expect} from '@playwright/test';

test('returning to the selected passage keeps the page and conversation intact', async ({page, context}) => {
  let calls = 0;
  await page.emulateMedia({reducedMotion: 'reduce'});
  await page.route('https://mlsysbook.ai/api/socratiq/ai/stream**', route => {
    calls += 1;
    return route.fulfill({contentType: 'text/event-stream', body: `data: ${JSON.stringify({choices: [{delta: {content: 'The saved explanation connects bandwidth and latency.'}, finish_reason: null}]})}\n\ndata: [DONE]\n\n`});
  });
  await page.goto('/vol1/chapter/?reading=preserved');
  await page.locator('#passage').evaluate(element => {
    element.style.marginTop = '1200px';
    window.readerReturnMarker = 'same document';
    const range = document.createRange();
    range.selectNodeContents(element);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    document.dispatchEvent(new Event('selectionchange'));
  });
  await page.keyboard.press('Control+Shift+E');
  await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
  await page.evaluate(() => window.scrollTo(0, 0));
  const url = page.url();
  await page.locator('.reader-source-link').click();
  await expect(page.getByRole('region', {name: 'SocratiQ focused reader'})).toBeHidden();
  await expect(page.locator('#passage')).toBeFocused();
  expect(page.url()).toBe(url);
  expect(context.pages()).toHaveLength(1);
  expect(await page.evaluate(() => window.readerReturnMarker)).toBe('same document');
  expect(await page.evaluate(() => window.scrollY)).toBeGreaterThan(0);
  await page.getByRole('button', {name: 'Ask SocratiQ'}).click();
  expect(await page.locator('#passage').getAttribute('tabindex')).toBeNull();
  await expect(page.locator('.reader-answer')).toContainText('The saved explanation');
  expect(calls).toBe(1);
});

test('source return safely falls back after the selected node is detached', async ({page}) => {
  await page.goto('/vol1/chapter/');
  const result = await page.evaluate(async () => {
    const {captureSelection, revealSource} = await import('/context.js');
    const paragraph = document.querySelector('#passage');
    const range = document.createRange();
    range.selectNodeContents(paragraph);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    const source = captureSelection();
    paragraph.remove();
    const returned = revealSource(source);
    const focused = document.activeElement.id;
    const serialized = JSON.parse(JSON.stringify(source));
    const fallback = revealSource(serialized);
    document.querySelector('#bandwidth').remove();
    return {returned, focused, fallback, missing: revealSource(source), remote: revealSource({url: 'https://example.com/#bandwidth'})};
  });
  expect(result).toEqual({returned: true, focused: 'bandwidth', fallback: true, missing: false, remote: false});
});
