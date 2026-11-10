import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile, mkdtemp} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import {chromium, expect} from '@playwright/test';
import tailwindcss from 'tailwindcss';

test('embedded context button and keyboard mentions select context before sending', {timeout: 60000}, async () => {
  const packageInfo = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));
  const server = await createServer({
    root: fileURLToPath(new URL('../src_shadow', import.meta.url)), configFile: false,
    cacheDir: await mkdtemp(join(tmpdir(), 'socratiq-context-test-')),
    server: {host: '127.0.0.1', port: 0, strictPort: true},
    optimizeDeps: {noDiscovery: true, include: Object.keys(packageInfo.dependencies)},
    css: {postcss: {plugins: [tailwindcss(fileURLToPath(new URL('../tailwind.config.cjs', import.meta.url)))]}},
    plugins: [{name: 'constructed-context-chapter', configureServer(server) {
      server.middlewares.use(async (request, response, next) => {
        if (request.url === '/qa/reader-context.js') {
          response.setHeader('Content-Type', 'text/javascript; charset=utf-8');
          response.end(await readFile(new URL('../src_reader/context.js', import.meta.url), 'utf8'));
          return;
        }
        if (request.url !== '/qa/chapter/') return next();
        response.setHeader('Content-Type', 'text/html; charset=utf-8');
        response.end(`<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><title>Constructed context chapter</title><nav><ul id="qa-nav"><li id="qa-nav-li">Navigation item</li></ul></nav><aside><blockquote id="qa-aside-quote">Sidebar quote</blockquote></aside><main id="quarto-document-content"><div class="chapter-concept-map">CONCEPT_MAP_NOISE</div><h1><span class="chapter-number">1</span> Memory systems</h1><section class="level2" id="bandwidth"><h2 class="anchored" data-number="1.1">Memory bandwidth</h2><h3>Memory details</h3><p>Constructed bandwidth context marker: data volume transferred per unit time.</p></section><section class="level2" id="latency"><h2 class="anchored" data-number="1.2">Latency</h2><p>Constructed latency context marker: duration of an individual request.</p><ul><li id="qa-li">List item latency bound</li><li id="qa-control-li">Interactive item <button type="button">Action</button></li></ul><figure><figcaption id="qa-figcaption">Figure 1.2: Latency distribution</figcaption></figure><blockquote id="qa-blockquote">Tail latency dominates distributed inference.</blockquote><p id="qa-math">Inline KaTeX <span class="katex"><span class="katex-mathml"><math><semantics><annotation encoding="application/x-tex">L = N / B</annotation></semantics></math></span><span class="katex-html" aria-hidden="true">L=N/B</span></span> and MathJax <mjx-container id="qa-mjx"><mjx-math><mjx-mi>T</mjx-mi></mjx-math><mjx-assistive-mml>ASSISTIVE_MML_DUPLICATE</mjx-assistive-mml></mjx-container>.</p><details class="callout-definition" id="qa-callout-def"><summary><strong>Definition 1.1</strong>: Operational Intensity</summary><p>Operations performed per byte of DRAM traffic.</p></details></section></main><script type="module">window.__SOCRATIQ_EMBEDDED__=true;import('/js/index.js').then(()=>window.SocratiqStudy.open());</script>`);
      });
    }}],
  });
  let browser;
  try {
    await server.listen();
    browser = await chromium.launch({headless: true});
    const page = await browser.newPage();
    await page.addInitScript(() => {
      window.qaSent = [];
      window.addEventListener('aiActionCompleted', event => {
        if (event.detail?.type === 'query') window.qaSent.push(event.detail);
      });
    });
    const origin = `http://127.0.0.1:${server.httpServer.address().port}`;
    const requests = [];
    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      if (url.origin + url.pathname === 'https://mlsysbook.ai/api/socratiq/ai/stream') {
        requests.push(JSON.parse(route.request().postData()));
        return route.fulfill({contentType: 'text/event-stream', body: `data: ${JSON.stringify({choices: [{delta: {content: 'Constructed context response.'}}]})}\n\ndata: ${JSON.stringify({choices: [{delta: {}, finish_reason: 'stop'}]})}\n\ndata: [DONE]\n\n`});
      }
      return url.origin === origin ? route.continue() : route.abort();
    });
    await page.goto(`${origin}/qa/chapter/`);
    const study = page.locator('#widget-chat-container');
    await expect(study.getByRole('button', {name: 'Close study tools'})).toBeVisible();
    let sentRequests = 0;
    if (!process.env.SOCRATIQ_CONTEXT_KEYBOARD_ONLY) {
    await study.locator('#context-button').click();
    const contextMenu = study.locator('#context-menu');
    await expect(contextMenu).toBeVisible();
    await contextMenu.locator('[data-filter="section"]').click();
    await contextMenu.locator('.at-mention-item').filter({hasText: 'Memory bandwidth'}).click();
    await expect(study.locator('#copy-container')).toContainText('Constructed bandwidth context marker');
    await study.locator('#user-input').fill('Explain this selected section.');
    await study.locator('#enter-btn').click();
    await expect.poll(() => requests.length).toBe(1);
    assert.ok(requests[0].messages.some(message => message.content.includes('Constructed bandwidth context marker')));
    const selectedQuery = await page.evaluate(() => window.qaSent.at(-1).text);
    assert.ok(selectedQuery.includes('Constructed bandwidth context marker'));
    assert.ok(selectedQuery.includes('Explain this selected section.'));
    await expect(study.locator('#message-container')).toContainText('Constructed context response.');
    sentRequests++;
    }

    const sentBefore = await page.evaluate(() => window.qaSent.length);
    const input = study.locator('#user-input');
    await input.fill('@Memory');
    const mentionMenu = study.locator('.at-mentions-dropup:not(#context-menu)');
    await expect(mentionMenu).toBeVisible();
    await input.press('ArrowDown');
    await input.press('Enter');
    assert.equal(await page.evaluate(() => window.qaSent.length), sentBefore, 'Enter selects a mention instead of sending a question');
    assert.match(await input.inputValue(), /@\[/);
    await expect(mentionMenu).toBeHidden();
    await expect(study.locator('#copy-container')).toContainText('Constructed bandwidth context marker');
    await input.fill('Explain the keyboard-selected section.');
    await study.locator('#enter-btn').click();
    await expect.poll(() => requests.length).toBe(sentRequests + 1);
    assert.ok(requests.at(-1).messages.some(message => message.content.includes('Constructed bandwidth context marker')));
    assert.ok((await page.evaluate(() => window.qaSent.at(-1).text)).includes('Constructed bandwidth context marker'));

    // Goal 2 & Goal 3 regression checks:
    // 1. captureParagraph captures <li>, <figcaption>, and <blockquote> inside <main> while rejecting nav/aside/controls
    const paragraphCaptures = await page.evaluate(async () => {
      const {captureParagraph} = await import('/qa/reader-context.js');
      return {
        li: captureParagraph(document.getElementById('qa-li'))?.quote || null,
        figcaption: captureParagraph(document.getElementById('qa-figcaption'))?.quote || null,
        blockquote: captureParagraph(document.getElementById('qa-blockquote'))?.quote || null,
        navLi: captureParagraph(document.getElementById('qa-nav-li')),
        asideQuote: captureParagraph(document.getElementById('qa-aside-quote')),
        controlLi: captureParagraph(document.getElementById('qa-control-li')),
      };
    });
    assert.equal(paragraphCaptures.li, 'List item latency bound');
    assert.equal(paragraphCaptures.figcaption, 'Figure 1.2: Latency distribution');
    assert.equal(paragraphCaptures.blockquote, 'Tail latency dominates distributed inference.');
    assert.equal(paragraphCaptures.navLi, null);
    assert.equal(paragraphCaptures.asideQuote, null);
    assert.equal(paragraphCaptures.controlLi, null);

    // 2. extractReadableBookText reconstructs inline $...$ from MathJax/KaTeX and strips mjx-assistive-mml and .chapter-concept-map
    const extractionCheck = await page.evaluate(async () => {
      const {extractReadableBookText} = await import('/js/utils/textExtractor.js');
      const mjxNode = document.getElementById('qa-mjx');
      window.MathJax = {
        startup: {
          document: {
            getMathItemsWithin: root => (root.contains(mjxNode) ? [{typesetRoot: mjxNode, math: 'T_{total}'}] : []),
          },
        },
      };
      return extractReadableBookText(document.getElementById('quarto-document-content'));
    });
    assert.ok(extractionCheck.includes('$L = N / B$'), 'reconstructs inline KaTeX $...$');
    assert.ok(extractionCheck.includes('$T_{total}$'), 'reconstructs inline MathJax $...$');
    assert.ok(!extractionCheck.includes('ASSISTIVE_MML_DUPLICATE'), 'strips mjx-assistive-mml');
    assert.ok(!extractionCheck.includes('CONCEPT_MAP_NOISE'), 'strips .chapter-concept-map');

    // 3. create_text_ref.js indexes <details class="callout-definition"> into window.subsections and populates non-empty .text for <summary><strong>Definition 1.1</strong></summary> in window.keywords
    const indexedRefs = await page.evaluate(async () => {
      const {get_text_ref} = await import('/js/components/quizBtns/create_text_ref.js');
      get_text_ref();
      return {
        subsections: window.subsections,
        keywords: window.keywords,
      };
    });
    assert.ok(indexedRefs.subsections['Definition 1.1: Operational Intensity'], 'indexes callout-definition into window.subsections');
    assert.ok(
      indexedRefs.subsections['Definition 1.1: Operational Intensity'].text.includes('Operations performed per byte of DRAM traffic.'),
      'callout-definition subsection includes full body text'
    );
    assert.ok(indexedRefs.keywords['Definition 1.1'], 'indexes <summary><strong>Definition 1.1</strong></summary> into window.keywords');
    assert.ok(
      indexedRefs.keywords['Definition 1.1'].text.includes('Operations performed per byte of DRAM traffic.'),
      'keyword inside closed <details> summary populates non-empty full definition .text'
    );

    // 4. Focused Reader active passage (window.__SOCRATIQ_ACTIVE_SOURCE__) appears as "Current reader passage" in Study Tools + Context and @ mentions
    await page.evaluate(() => {
      window.__SOCRATIQ_ACTIVE_SOURCE__ = {
        quote: 'Constructed active reader passage quote for bridge verification.',
        section: 'Latency',
      };
    });
    await study.locator('#context-button').click();
    const contextMenu = study.locator('#context-menu');
    await expect(contextMenu).toBeVisible();
    const activeContextItem = contextMenu.locator('.at-mention-item').filter({hasText: 'Current reader passage'});
    await expect(activeContextItem).toBeVisible();
    await activeContextItem.click();
    await expect(study.locator('#copy-container')).toContainText('Constructed active reader passage quote for bridge verification.');

    await input.fill('@Current');
    await expect(mentionMenu).toBeVisible();
    await expect(mentionMenu.locator('.at-mention-item').filter({hasText: 'Current reader passage'})).toBeVisible();
    await input.press('ArrowDown');
    await input.press('Enter');
    assert.match(await input.inputValue(), /@\[Current reader passage\]/);
  } finally {
    await browser?.close();
    await server.close();
  }
});

test('research agent preserves API helpers and handles array topics without ReferenceError', async () => {
  const {fetchPapers} = await import('../src_shadow/js/libs/agents/research.js');
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new Error('constructed offline test');
  };
  try {
    const result = await fetchPapers(['roofline model', 'memory bandwidth'], 2);
    assert.deepEqual(result, []);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
