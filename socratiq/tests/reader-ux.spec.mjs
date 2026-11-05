import {test, expect} from '@playwright/test';

function sse(text) {
  return `data: ${JSON.stringify({choices: [{delta: {content: text}, finish_reason: null}]})}\n\ndata: ${JSON.stringify({choices: [{delta: {}, finish_reason: 'stop'}]})}\n\ndata: [DONE]\n\n`;
}

async function selectPassage(page) {
  await page.locator('#passage').evaluate(element => {
    const range = document.createRange();
    range.selectNodeContents(element);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    document.dispatchEvent(new Event('selectionchange'));
  });
}

const longAnswer = Array.from({length: 36}, (_, index) => `Paragraph ${index + 1}: ${'Bandwidth moves data through the system. '.repeat(5)}`).join('\n\n');

test('keyboard shortcut explains a selected passage and keeps the answer start and source available', async ({page}, testInfo) => {
  const requests = [];
  await page.route('https://proxy-worker-streaming.mlsysbook.workers.dev/**', async route => {
    requests.push(route.request().postDataJSON());
    await route.fulfill({contentType: 'text/event-stream', body: sse(longAnswer)});
  });
  await page.goto('/vol1/chapter/');

  const unselectedKey = await page.evaluate(() => {
    const event = new KeyboardEvent('keydown', {key: 'e', ctrlKey: true, shiftKey: true, bubbles: true, cancelable: true});
    document.dispatchEvent(event);
    return event.defaultPrevented;
  });
  expect(unselectedKey).toBe(false);
  await expect(page.getByRole('region', {name: 'SocratiQ focused reader'})).toBeHidden();

  await selectPassage(page);
  await page.keyboard.press('Control+Shift+E');
  await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
  expect(requests).toHaveLength(1);

  const body = page.locator('#socratiq-reader .reader-body');
  const answer = page.locator('#socratiq-reader .reader-answer');
  await expect(page.locator('#socratiq-reader .reader-source-link')).toBeVisible();
  const layout = await body.evaluate(node => {
    const answerNode = node.querySelector('.reader-answer');
    const box = node.getBoundingClientRect();
    return {
      answerTop: answerNode.getBoundingClientRect().top,
      bodyTop: box.top,
      bodyBottom: box.bottom,
      scrollTop: node.scrollTop,
      maxScroll: node.scrollHeight - node.clientHeight,
    };
  });
  expect(layout.answerTop).toBeGreaterThanOrEqual(layout.bodyTop);
  expect(layout.answerTop).toBeLessThan(layout.bodyBottom);
  expect(layout.scrollTop).toBeLessThan(layout.maxScroll);
  await expect(answer).toContainText('Paragraph 1:');
  await page.screenshot({path: testInfo.outputPath('initial-answer.png')});

  const contextMenuWasPrevented = await page.locator('#passage').evaluate(node => {
    const event = new MouseEvent('contextmenu', {bubbles: true, cancelable: true, button: 2});
    node.dispatchEvent(event);
    return event.defaultPrevented;
  });
  expect(contextMenuWasPrevented).toBe(false);
});

test('follow-up streams follow only when the reader is already at the bottom', async ({page}) => {
  let calls = 0;
  await page.route('https://proxy-worker-streaming.mlsysbook.workers.dev/**', route => {
    calls += 1;
    return route.fulfill({contentType: 'text/event-stream', body: sse(longAnswer)});
  });
  await page.goto('/vol1/chapter/');
  await selectPassage(page);
  await page.getByRole('button', {name: 'Explain this', exact: true}).click();
  await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
  const body = page.locator('#socratiq-reader .reader-body');

  await body.evaluate(node => {node.scrollTop = 0;});
  await page.getByRole('textbox', {name: 'Ask a follow-up question'}).fill('Explain that another way.');
  await page.getByRole('button', {name: 'Send question'}).click();
  await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
  expect(await body.evaluate(node => node.scrollTop)).toBe(0);

  await body.evaluate(node => {node.scrollTop = node.scrollHeight;});
  await page.getByRole('textbox', {name: 'Ask a follow-up question'}).fill('One more detail.');
  await page.getByRole('button', {name: 'Send question'}).click();
  await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
  const scroll = await body.evaluate(node => ({top: node.scrollTop, max: node.scrollHeight - node.clientHeight}));
  expect(scroll.top).toBe(scroll.max);
  expect(calls).toBe(3);
});

test('touch cancellation drops the pending source and transparent backgrounds use the color-scheme fallback', async ({page}) => {
  await page.emulateMedia({colorScheme: 'dark'});
  await page.goto('/vol1/chapter/');
  await selectPassage(page);
  const explainButton = page.getByRole('button', {name: 'Explain this', exact: true});
  await expect(explainButton).toBeVisible();
  await explainButton.evaluate(button => button.dispatchEvent(new PointerEvent('pointercancel', {
    bubbles: true,
    pointerId: 1,
    pointerType: 'touch',
  })));
  await expect(explainButton).toBeHidden();
  await page.evaluate(() => {
    document.body.style.backgroundColor = 'transparent';
    document.documentElement.style.backgroundColor = 'transparent';
  });
  await expect(page.getByRole('button', {name: 'Explain this', exact: true})).toBeHidden();
  await expect.poll(() => page.locator('#socratiq-reader .reader-shell').getAttribute('data-theme')).toBe('dark');
  await page.getByRole('button', {name: 'Ask SocratiQ'}).click();
  await expect(page.getByRole('textbox', {name: 'Ask a follow-up question'})).toBeDisabled();
});
