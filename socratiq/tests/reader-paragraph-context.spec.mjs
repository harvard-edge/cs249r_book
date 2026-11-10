import {test, expect} from '@playwright/test';

test('paragraph capture needs no selection and preserves bounded source context', async ({page}) => {
  await page.goto('/vol3/chapter/?private=value');
  const source = await page.evaluate(async () => {
    const before = document.createElement('p'); before.textContent = 'Earlier material. '.repeat(1200);
    document.querySelector('#bandwidth').prepend(before);
    const p = document.querySelector('#passage'); p.innerHTML = '<em>Memory bandwidth</em> measures data movement.';
    return (await import('/context.js')).captureParagraph(p.querySelector('em').firstChild, {version: 'test-release'});
  });
  expect(source.quote).toBe('Memory bandwidth measures data movement.');
  expect(source.context).toContain(source.quote);
  expect(source.context).toContain('A system can have high bandwidth');
  expect(source.context.length).toBeLessThanOrEqual(12000);
  expect(source.context).not.toContain('Unrelated content');
  expect(source.volume).toBe('vol3');
  expect(source.version).toBe('test-release');
  expect(source.url).toBe('http://127.0.0.1:4176/vol3/chapter/#bandwidth');
});

test('capture preserves live selection and markup, restores TeX, and supports reveal', async ({page}) => {
  await page.goto('/vol1/chapter/');
  const result = await page.evaluate(async () => {
    const {captureParagraph, revealSource} = await import('/context.js');
    const p = document.querySelector('#math');
    p.innerHTML = '<span>Transfer time is </span><mjx-container><svg><text>fraction glyph</text></svg></mjx-container>.';
    const root = p.querySelector('mjx-container');
    const item = {math: String.raw`\frac{D}{B}`, typesetRoot: root};
    window.MathJax = {startup: {document: {getMathItemsWithin: container => container.contains(root) ? [item] : []}}};
    const text = document.querySelector('#passage').firstChild;
    const range = document.createRange(); range.setStart(text, 7); range.setEnd(text, 16);
    const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
    const snapshot = () => ({html: p.innerHTML, text: selection.toString(), start: selection.anchorOffset, end: selection.focusOffset});
    const before = snapshot(); const source = captureParagraph(root.querySelector('text').firstChild); const after = snapshot();
    return {before, after, source, revealed: revealSource(source), focused: document.activeElement.id};
  });
  expect(result.after).toEqual(result.before);
  expect(result.revealed).toBe(true);
  expect(result.focused).toBe('math');
  expect(result.source.quote).toBe(String.raw`Transfer time is $\frac{D}{B}$ .`);
  expect(result.source.context).toContain(String.raw`$\frac{D}{B}$`);
});

test('capture rejects navigation, controls, links, editable and detached content', async ({page}) => {
  await page.goto('/vol1/chapter/');
  const results = await page.evaluate(async () => {
    const {captureParagraph} = await import('/context.js');
    document.querySelector('main').insertAdjacentHTML('beforeend', '<nav><p id="nav">Navigation</p></nav><aside><p id="aside">Sidebar</p></aside><form><p id="form">Private form</p></form><p id="input">Private <input value="secret"></p><p id="link">Read <a href="/">another page</a></p><p id="editable"><span contenteditable>Private draft</span></p><p id="button"><span role="button">control</span></p>');
    const detached = document.createElement('p'); detached.textContent = 'Detached';
    return [...['nav', 'aside', 'form', 'input', 'editable', 'button'].map(id => captureParagraph(document.getElementById(id))), captureParagraph(document.querySelector('#link a')), captureParagraph(document.querySelector('input')), captureParagraph(detached), captureParagraph(null)];
  });
  expect(results.every(source => source === null)).toBe(true);
});

test('paragraphs with citations remain selectable by tapping their prose', async ({page}) => {
  await page.goto('/vol1/chapter/');
  const source = await page.evaluate(async () => {
    const paragraph = document.querySelector('#passage');
    paragraph.innerHTML = 'Memory bandwidth is a rate. See <a href="#other">the worked example</a>.';
    return (await import('/context.js')).captureParagraph(paragraph);
  });
  expect(source.quote).toContain('See the worked example.');
});

test('oversized paragraphs report an actionable error', async ({page}) => {
  await page.goto('/vol1/chapter/');
  const error = await page.evaluate(async () => {
    const p = document.querySelector('#passage'); p.textContent = 'Long passage. '.repeat(400);
    try { (await import('/context.js')).captureParagraph(p); } catch (error) { return error.message; }
  });
  expect(error).toContain('Select a shorter passage');
});

test('captureParagraph captures li, figcaption, and blockquote inside main while rejecting nav/aside/controls', async ({page}) => {
  await page.goto('/vol1/chapter/');
  const result = await page.evaluate(async () => {
    const {captureParagraph} = await import('/context.js');
    document.querySelector('#bandwidth').insertAdjacentHTML(
      'beforeend',
      '<ul><li id="main-li">List item bandwidth bound</li><li id="ctrl-li">Button <button>Click</button></li></ul><figure><figcaption id="main-fig">Figure 1.1: Roofline</figcaption></figure><blockquote id="main-quote">Bandwidth limits operational intensity.</blockquote><nav><ul><li id="nav-li">Nav item</li></ul></nav><aside><blockquote id="aside-quote">Sidebar quote</blockquote></aside>'
    );
    return {
      li: captureParagraph(document.getElementById('main-li'))?.quote || null,
      figcaption: captureParagraph(document.getElementById('main-fig'))?.quote || null,
      blockquote: captureParagraph(document.getElementById('main-quote'))?.quote || null,
      ctrlLi: captureParagraph(document.getElementById('ctrl-li')),
      navLi: captureParagraph(document.getElementById('nav-li')),
      asideQuote: captureParagraph(document.getElementById('aside-quote')),
    };
  });
  expect(result.li).toBe('List item bandwidth bound');
  expect(result.figcaption).toBe('Figure 1.1: Roofline');
  expect(result.blockquote).toBe('Bandwidth limits operational intensity.');
  expect(result.ctrlLi).toBeNull();
  expect(result.navLi).toBeNull();
  expect(result.asideQuote).toBeNull();
});
