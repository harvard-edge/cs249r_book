// Author-constructed integration fixture, served with both production bundles.
import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile, mkdir, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {resolve, sep} from 'node:path';
import {pathToFileURL} from 'node:url';
import {chromium, expect} from '@playwright/test';

const assets = new URL('../../books/shared/assets/scripts/socratiq/', import.meta.url);
const receipts = pathToFileURL(resolve(process.env.SOCRATIQ_STUDY_TEST_OUTPUT || `${tmpdir()}/socratiq-study-integration`) + sep);

async function withinBrowserBudget(promise, milliseconds = 5000) {
  let timer;
  try {
    return await Promise.race([promise, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error('Browser did not finish the theme transition within the test budget')), milliseconds);
    })]);
  } finally { clearTimeout(timer); }
}

async function fixture(run) {
  // Missing builds must fail, rather than testing a stand-in legacy API.
  await Promise.all(['reader.js', 'study.js'].map(name => readFile(new URL(name, assets))));
  const requests = [];
  const server = createServer(async (request, response) => {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    requests.push(pathname);
    if (pathname.endsWith('/reader.js') || pathname.endsWith('/study.js')) {
      response.setHeader('Content-Type', 'text/javascript; charset=utf-8');
      response.end(await readFile(new URL(pathname.endsWith('/study.js') ? 'study.js' : 'reader.js', assets)));
      return;
    }
    response.setHeader('Content-Type', 'text/html; charset=utf-8');
    response.end(`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Study integration chapter</title><style>body{font:18px/1.6 Georgia;margin:32px;background:#fff;color:#222}main{max-width:720px;margin:auto}</style><script src="/preview/vol2/assets/scripts/socratiq/reader.js" data-volume="vol2" defer></script></head><body><main id="quarto-document-content"><h1>Understanding memory bandwidth</h1><section id="bandwidth"><h2>Bandwidth</h2><p id="passage">Memory bandwidth measures how much data can move per unit time. Latency measures how long an individual request takes.</p></section></main></body></html>`);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch({headless: true});
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/*', route => new URL(route.request().url()).origin === origin ? route.continue() : route.abort());
  try { await run({page, requests, origin, errors}); }
  catch (error) {
    await mkdir(receipts, {recursive: true});
    await page.screenshot({path: new URL(`failure-${Date.now()}.png`, receipts).pathname, fullPage: true, timeout: 5000}).catch(() => {});
    console.error('Browser errors:', errors);
    throw error;
  }
  finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
}

test('built study workspace loads lazily once beside the reader, preserves a deck, and returns focus', {timeout: 90000}, async () => {
  await fixture(async ({page, requests, origin, errors}) => {
    await page.goto(`${origin}/preview/vol2/chapter/`);
    await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).click();
    assert.equal(requests.some(path => path.endsWith('/study.js')), false);
    await page.getByRole('button', {name: 'Study tools', exact: true}).click();
    const legacy = page.locator('#widget-chat-container');
    await expect(legacy.getByRole('button', {name: 'Close study tools', exact: true})).toBeVisible({timeout: 60000});
    await mkdir(receipts, {recursive: true});
    for (const [name, width, height] of [['desktop', 1280, 900], ['tablet', 768, 1024], ['phone', 390, 844]]) {
      await page.setViewportSize({width, height});
      const close = legacy.getByRole('button', {name: 'Close study tools', exact: true});
      await expect(close).toBeInViewport();
      await page.screenshot({path: new URL(`${name}-study-workspace.png`, receipts).pathname, fullPage: true});
    }
    await page.setViewportSize({width: 1280, height: 900});
    await expect(page.getByRole('button', {name: 'Ask SocratiQ', exact: true})).toBeHidden();
    await expect(page.locator('#socratiq-reader .reader-panel')).toBeHidden();
    await page.locator('#passage').evaluate(node => {
      const range = document.createRange(); range.selectNodeContents(node);
      window.getSelection().removeAllRanges(); window.getSelection().addRange(range);
    });
    await page.keyboard.press('Control+Shift+E');
    await expect(page.locator('#socratiq-reader .selection-action')).toBeHidden();
    await expect(page.locator('#socratiq-reader .reader-panel')).toBeHidden();
    await legacy.locator('#flashcards-button').click();
    await legacy.locator('#newDeckBtn').click();
    await legacy.locator('#deckNameInput').fill('Constructed embedded integration deck');
    await legacy.locator('#createDeck').click();
    await expect(legacy.getByText('Constructed embedded integration deck', {exact: true}).first()).toBeVisible();
    await mkdir(receipts, {recursive: true});
    await page.screenshot({path: new URL('desktop-study.png', receipts).pathname, fullPage: true});
    for (const [name, width, height] of [['tablet', 768, 1024], ['phone', 390, 844]]) {
      await page.setViewportSize({width, height});
      await expect(legacy.locator('#newDeckBtn')).toBeInViewport();
      await page.screenshot({path: new URL(`${name}-study-deck.png`, receipts).pathname, fullPage: true});
      if (name === 'phone') {
        const targets = await legacy.locator('#sr-import-btn, #sr-download-btn, #copyButton, #sr-share-btn, #closeModal_sr').evaluateAll(buttons => buttons.map(button => {
          const {width, height} = button.getBoundingClientRect();
          return {id: button.id, width, height};
        }));
        await writeFile(new URL('phone-control-targets.json', receipts), JSON.stringify({classification: 'measured browser UI dimensions in constructed fixture', targets}, null, 2));
        for (const target of targets) {
          assert.ok(target.width >= 44 && target.height >= 44, `${target.id} must keep a 44px phone touch target`);
        }
        assert.equal(await legacy.locator('#sr-modal-content_inner').evaluate(element => element.scrollWidth <= element.clientWidth), true);
      }
    }
    await legacy.locator('#sr-download-btn').click();
    const downloading = page.waitForEvent('download');
    await legacy.locator('.json-btn').click();
    const backup = JSON.parse(await readFile(await (await downloading).path(), 'utf8'));
    assert.ok(backup.chapter_card_sets.some(deck => deck.title === 'Constructed embedded integration deck') || backup.current_chapter.title === 'Constructed embedded integration deck');
    await legacy.locator('#newDeckBtn').click();
    await legacy.locator('#deckNameInput').fill('Constructed phone deck');
    await legacy.locator('#createDeck').click();
    await expect(legacy.getByText('Constructed phone deck', {exact: true}).first()).toBeVisible();
    await page.setViewportSize({width: 1280, height: 900});
    await legacy.locator('#closeModal_sr').click();
    await legacy.getByRole('button', {name: 'Close study tools', exact: true}).click();
    await expect(legacy).toHaveJSProperty('hidden', true);
    await expect(page.getByRole('button', {name: 'Study tools', exact: true})).toBeFocused();
    await page.getByRole('button', {name: 'Study tools', exact: true}).click();
    await expect(legacy.getByRole('button', {name: 'Close study tools', exact: true})).toBeVisible();
    await legacy.locator('#flashcards-button').click();
    await expect(legacy.getByText('Constructed embedded integration deck', {exact: true}).first()).toBeVisible();
    assert.equal(requests.filter(path => path.endsWith('/study.js')).length, 1);
    assert.ok(requests.includes('/preview/vol2/assets/scripts/socratiq/study.js'));
    assert.equal(await page.locator('script[data-socratiq-study]').count(), 1);
    assert.equal(await page.locator('#widget-chat-container').count(), 1);
    assert.equal((await page.context().cookies()).some(cookie => /socratiq/i.test(cookie.name)), false);
    await legacy.locator('#closeModal_sr').click();
    const downloads = [];
    page.on('download', download => downloads.push(download));
    const welcomeDownload = page.waitForEvent('download');
    await legacy.locator('#ai-message_2 [title="Download"]').click();
    assert.ok((await readFile(await (await welcomeDownload).path(), 'utf8')).includes('SocratiQ is an AI Generative Learning Assistant'));
    await page.waitForTimeout(200);
    assert.equal(downloads.length, 1, 'Repeated startup initialization must keep one download per click');
    assert.deepEqual(errors, []);
  });
});

test('built ink editor adds and edits a phone flashcard across study close and reopen', {timeout: 90000}, async () => {
  await fixture(async ({page, origin, errors}) => {
    await page.setViewportSize({width: 390, height: 844});
    await page.goto(`${origin}/preview/vol2/chapter/`);
    await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).click();
    await page.getByRole('button', {name: 'Study tools', exact: true}).click();
    const legacy = page.locator('#widget-chat-container');
    await legacy.locator('#flashcards-button').click();
    await legacy.locator('#newDeckBtn').click();
    await legacy.locator('#deckNameInput').fill('Constructed editor deck');
    await legacy.locator('#createDeck').click();
    await legacy.locator('#showAddCard').click();
    await legacy.locator('#questionEditor [contenteditable=true]').fill('Constructed bandwidth question');
    await legacy.locator('#answerEditor [contenteditable=true]').fill('Constructed bandwidth answer');
    await legacy.locator('#saveCard').click();
    const question = legacy.locator('#question-0 [contenteditable=true]');
    await expect(question).toHaveText('Constructed bandwidth question');
    await question.fill('Constructed edited bandwidth question');
    await legacy.locator('#answer-0 [contenteditable=true]').fill('Constructed edited bandwidth answer');
    await legacy.locator('#closeModal_sr').click();
    await legacy.getByRole('button', {name: 'Close study tools', exact: true}).click();
    await page.getByRole('button', {name: 'Study tools', exact: true}).click();
    await legacy.locator('#flashcards-button').click();
    await expect(question).toHaveText('Constructed edited bandwidth question');
    await expect(legacy.locator('#answer-0 [contenteditable=true]')).toHaveText('Constructed edited bandwidth answer');
    await mkdir(receipts, {recursive: true});
    await page.screenshot({path: new URL('phone-study-editor.png', receipts).pathname, fullPage: true});
    assert.deepEqual(errors, []);
  });
});

test('built legacy chat renders KaTeX and Mermaid from a constructed provider response', {timeout: 90000}, async () => {
  await fixture(async ({page, origin, errors}) => {
    const markdown = 'Constructed renderer regression fixture.\n\nTransfer time is $t = D / B$.\n\n```mermaid\ngraph TD\n  Memory --> Accelerator\n```\n';
    const sse = `data: ${JSON.stringify({choices: [{delta: {content: markdown}, finish_reason: null}]})}\n\ndata: ${JSON.stringify({choices: [{delta: {}, finish_reason: 'stop'}]})}\n\ndata: [DONE]\n\n`;
    let calls = 0;
    await page.route('https://proxy-worker-streaming.mlsysbook.workers.dev/**', route => {
      calls += 1;
      return route.fulfill({contentType: 'text/event-stream', body: sse});
    });
    await page.goto(`${origin}/preview/vol2/chapter/`);
    await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).click();
    await page.getByRole('button', {name: 'Study tools', exact: true}).click();
    const legacy = page.locator('#widget-chat-container');
    await legacy.locator('#user-input').fill('Render the constructed regression fixture.');
    await legacy.locator('#enter-btn').click();
    await expect(legacy.locator('#message-container .katex')).toBeVisible({timeout: 30000});
    await expect(legacy.locator('#message-container .mermaid-diagram svg')).toBeVisible({timeout: 30000});
    await expect(legacy.locator('#message-container .mermaid-diagram svg')).toContainText('Memory');
    await expect(legacy.locator('#message-container .mermaid-diagram svg')).toContainText('Accelerator');
    assert.ok(calls > 0);
    await mkdir(receipts, {recursive: true});
    await page.screenshot({path: new URL('desktop-study-renderers.png', receipts).pathname, fullPage: true});
    assert.deepEqual(errors, []);
  });
});

test('built settings, progress, graph, and help overlays receive actual desktop and phone clicks', {timeout: 90000}, async () => {
  await fixture(async ({page, origin, errors}) => {
    await page.goto(`${origin}/preview/vol2/chapter/`);
    await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).click();
    await page.getByRole('button', {name: 'Study tools', exact: true}).click();
    const legacy = page.locator('#widget-chat-container');
    for (const [name, width, height] of [['desktop', 1280, 900], ['phone', 390, 844]]) {
      await page.setViewportSize({width, height});
      await legacy.locator('#settings-btn').click();
      await expect(legacy.locator('#modal1')).toBeVisible();
      await legacy.locator('#close-btn').click();
      await expect(legacy.locator('#modal1')).toBeHidden();
      await legacy.locator('#chat-quiz-btn').click();
      await expect(legacy.locator('.stats-modal')).toBeVisible();
      if (name === 'phone') {
        const title = await legacy.locator('.stats-header h2').boundingBox();
        const prompt = await legacy.locator('#badges-prompt').boundingBox();
        const close = await legacy.locator('.stats-modal .close-modal').boundingBox();
        const overlaps = (a, b) => a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;
        assert.equal(overlaps(title, prompt), false);
        assert.equal(overlaps(title, close), false);
        assert.ok(title.y < 100, 'Phone dashboard should start in the visible viewport');
        await page.screenshot({path: new URL('phone-study-progress.png', receipts).pathname, animations: 'disabled'});
      }
      await legacy.locator('.stats-modal .close-modal').click();
      await expect(legacy.locator('.stats-modal')).toBeHidden();
      await legacy.locator('#knowledge-graph-btn').click();
      const graph = legacy.locator('#socratiq-knowledge-graph-modal');
      if (name === 'phone') {
        await expect(graph.locator('#list-view .list-item').first()).toBeVisible();
        await expect(graph.getByRole('switch', {name: 'List view', exact: true})).toBeChecked();
        await graph.locator('#list-view .list-item').first().click();
        await expect.poll(() => graph.locator('#list-view .list-item:visible').count()).toBeGreaterThan(1);
        await expect(graph.locator('#list-view .list-item').first()).toContainText('Knowledge Graph');
        await graph.locator('#list-view .list-item:visible').last().click();
        await expect(graph.locator('#generate-summative-btn')).toBeEnabled();
        const selected = await graph.locator('.quiz-items-container').boundingBox();
        const actions = await graph.locator('.knowledge-graph-actions').boundingBox();
        assert.ok(selected.y + selected.height <= actions.y, 'Selected components must remain above the graph action buttons');
        const plot = await graph.locator('.knowledge-graph-viewport').boundingBox();
        assert.ok(plot.width >= 300, 'Phone graph must keep usable plot width beside its stacked component panel');
        await graph.getByRole('switch', {name: 'List view', exact: true}).click();
        await expect(graph.locator('.knowledge-graph-viewport > svg')).toBeVisible();
        await graph.getByRole('switch', {name: 'List view', exact: true}).click();
        await expect(graph.locator('#list-view .list-item').first()).toBeVisible();
        await graph.locator('#list-view .list-item').first().click();
        await expect.poll(() => graph.locator('#list-view .list-item:visible').count()).toBeGreaterThan(1);
        await expect(graph.locator('#list-view .list-item').first()).toContainText('Knowledge Graph');
      } else {
        await expect(graph.locator('.knowledge-graph-viewport > svg')).toBeVisible();
      }
      await page.screenshot({path: new URL(`${name}-study-graph.png`, receipts).pathname, fullPage: true, animations: 'disabled'});
      await graph.getByText('Graph legend and guide', {exact: true}).click();
      await expect(graph.getByText('Pages & Topics (clickable)', {exact: true})).toBeVisible();
      await graph.locator(':scope > div > button').click();
      await expect(graph).toBeHidden();
      await legacy.locator('#help-btn').click();
      await expect(legacy.locator('#helpModalContainer')).toBeVisible();
      await legacy.locator('#closeButtonHelpModal').click();
      await expect(legacy.locator('#helpModalContainer')).toBeHidden();
    }
    assert.deepEqual(errors, []);
  });
});

test('failed study download leaves reader usable and retries the production bundle', {timeout: 90000}, async () => {
  await fixture(async ({page, origin, errors}) => {
    let attempts = 0;
    await page.route('**/study.js', route => ++attempts === 1 ? route.abort() : route.continue());
    await page.goto(`${origin}/preview/vol2/chapter/`);
    await page.setViewportSize({width: 390, height: 844});
    await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).click();
    await page.getByRole('button', {name: 'Study tools', exact: true}).click();
    await expect(page.getByText('Study tools could not open. Check your connection and try again.')).toBeVisible();
    await expect(page.getByRole('button', {name: 'Choose a paragraph', exact: true})).toBeEnabled();
    await mkdir(receipts, {recursive: true});
    await page.screenshot({path: new URL('mobile-study-error.png', receipts).pathname, fullPage: true});
    await page.getByRole('button', {name: 'Retry study tools', exact: true}).click();
    await expect(page.locator('#widget-chat-container').getByRole('button', {name: 'Close study tools', exact: true})).toBeVisible({timeout: 60000});
    assert.equal(attempts, 2);
    assert.equal(await page.locator('script[data-socratiq-study]').count(), 1);
    assert.deepEqual(errors, []);
  });
});

test('built theme transitions remain responsive with a saved deck and stable subscriptions', {timeout: 90000}, async () => {
  await fixture(async ({page, origin, errors}) => {
    const themeErrors = [];
    page.on('console', message => {
      if (message.type() === 'error' && /ThemeManager|DOMToken|InvalidCharacter/.test(message.text())) themeErrors.push(message.text());
    });
    await page.setViewportSize({width: 390, height: 844});
    await page.goto(`${origin}/preview/vol2/chapter/`);
    await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).click();
    await page.getByRole('button', {name: 'Study tools', exact: true}).click();
    const legacy = page.locator('#widget-chat-container');
    await legacy.locator('#flashcards-button').click();
    await legacy.locator('#newDeckBtn').click();
    await legacy.locator('#deckNameInput').fill('Constructed theme transition deck');
    await legacy.locator('#createDeck').click();
    await legacy.locator('#closeModal_sr').click();
    const before = await legacy.evaluate(host => host.shadowRoot.modalInstance.themeManager.themeChangeCallbacks.size);
    for (const theme of ['dark', 'light', 'dark']) {
      await page.emulateMedia({colorScheme: theme});
      await withinBrowserBudget(page.evaluate(value => {
        const dark = value === 'dark';
        document.documentElement.classList.toggle('quarto-dark', dark);
        document.body.classList.toggle('dark-mode', dark);
        document.body.style.backgroundColor = dark ? '#111827' : '#ffffff';
        document.body.style.color = dark ? '#e5e7eb' : '#252525';
      }, theme));
      await expect(legacy).toHaveAttribute('data-socratiq-theme', theme);
      await withinBrowserBudget(page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => resolve(true)))));
      assert.equal(await legacy.evaluate(host => host.shadowRoot.modalInstance.themeManager.themeChangeCallbacks.size), before);
    }
    await legacy.locator('#flashcards-button').click();
    await expect(legacy.getByText('Constructed theme transition deck', {exact: true}).first()).toBeVisible();
    await expect(legacy.locator('#sr-floating-controls > div')).toBeHidden();
    await legacy.locator('#dashboardSearch').click();
    await legacy.locator('#dashboardSearch').fill('Constructed filter interaction');
    await expect(legacy.locator('#dashboardSearch')).toHaveValue('Constructed filter interaction');
    await legacy.locator('#dashboardSearch').fill('');
    await expect(legacy.locator('#showAddCard')).toBeVisible();
    await expect(legacy.locator('#startReview')).toBeVisible();
    await page.screenshot({path: new URL('phone-study-dark-deck.png', receipts).pathname, animations: 'disabled'});
    await legacy.locator('#closeModal_sr').click();
    await page.screenshot({path: new URL('phone-study-dark-workspace.png', receipts).pathname, animations: 'disabled'});
    assert.deepEqual(themeErrors, []);
    assert.deepEqual(errors, []);
  });
});

test('initial dark selected deck titles retain readable phone and tablet contrast', {timeout: 90000}, async () => {
  await fixture(async ({page, origin, errors}) => {
    await page.emulateMedia({colorScheme: 'dark'});
    await page.addInitScript(() => { window.SocratiqWidgetTheme = 'dark'; });
    await page.goto(`${origin}/preview/vol2/chapter/`);
    await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).click();
    await page.getByRole('button', {name: 'Study tools', exact: true}).click();
    const legacy = page.locator('#widget-chat-container');
    await legacy.locator('#flashcards-button').click();
    await legacy.locator('#newDeckBtn').click();
    await legacy.locator('#deckNameInput').fill('Constructed initial dark selected deck');
    await legacy.locator('#createDeck').click();
    const measurements = [];
    await mkdir(receipts, {recursive: true});
    for (const [name, width, height] of [['phone', 390, 844], ['tablet', 768, 1024]]) {
      await page.setViewportSize({width, height});
      const row = legacy.locator('#chapterList button[data-chapter]').filter({hasText: 'Constructed initial dark selected deck'}).first();
      await expect(row).toBeVisible();
      const measurement = await row.evaluate(button => {
        const parse = color => color.match(/[\d.]+/g)?.map(Number) || [0, 0, 0, 0];
        const ancestors = [];
        for (let element = button; element; element = element.parentElement || element.getRootNode().host) ancestors.unshift(element);
        let background = [255, 255, 255];
        for (const element of ancestors) {
          const rgba = parse(getComputedStyle(element).backgroundColor);
          const alpha = rgba[3] ?? 1;
          background = background.map((channel, index) => rgba[index] * alpha + channel * (1 - alpha));
        }
        const foreground = parse(getComputedStyle(button).color).slice(0, 3);
        const luminance = rgb => rgb.map(channel => {
          const s = channel / 255;
          return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
        }).reduce((sum, channel, index) => sum + channel * [0.2126, 0.7152, 0.0722][index], 0);
        const values = [luminance(foreground), luminance(background)].sort((a, b) => a - b);
        return {foreground, background, contrast: (values[1] + 0.05) / (values[0] + 0.05)};
      });
      measurements.push({viewport: name, ...measurement});
      await writeFile(new URL('initial-dark-deck-contrast.json', receipts), JSON.stringify({classification: 'measured browser CSS contrast in constructed fixture', measurements}, null, 2));
      assert.ok(measurement.contrast >= 4.5, `Selected deck title must remain readable in initial dark ${name} view`);
      await page.screenshot({path: new URL(`initial-dark-${name}-deck.png`, receipts).pathname, animations: 'disabled'});
    }
    assert.deepEqual(errors, []);
  });
});
