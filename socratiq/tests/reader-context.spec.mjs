import {test, expect} from '@playwright/test';

test('long-section context stays centered on selection after excluding unrelated nodes', async ({page}) => {
  await page.goto('/vol1/chapter/');
  const source = await page.evaluate(async () => {
    const section = document.querySelector('#bandwidth');
    section.replaceChildren();
    const aside = document.createElement('aside');
    aside.textContent = 'Excluded navigation text. '.repeat(1500);
    const before = document.createElement('p');
    before.textContent = 'Earlier section material. '.repeat(700);
    const selected = document.createElement('p');
    selected.textContent = 'The selected mechanism must remain in the surrounding context.';
    const after = document.createElement('p');
    after.textContent = 'The next paragraph clarifies the selected mechanism.';
    section.append(aside, before, selected, after);
    const range = document.createRange(); range.selectNodeContents(selected);
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    return (await import('/context.js')).captureSelection();
  });
  expect(source.context).toContain(source.quote);
  expect(source.context).toContain('The next paragraph clarifies');
  expect(source.context).not.toContain('Excluded navigation');
  expect(source.context.length).toBeLessThanOrEqual(12000);
});

test('MathJax registry preserves original TeX for partial math selections without mutating the page', async ({page}) => {
  await page.goto('/vol1/chapter/');
  const before = await page.evaluate(() => {
    const paragraph = document.querySelector('#math');
    paragraph.parentElement.insertBefore(document.createElement('nav'), paragraph);
    paragraph.innerHTML = `Transfer time is <mjx-container class="MathJax mjx-container" jax="SVG"><svg><text>visible fraction glyph</text></svg><mjx-assistive-mml><math><mtext>accessible text that loses the fraction</mtext></math></mjx-assistive-mml></mjx-container>, where D is data volume.`;
    const root = paragraph.querySelector('mjx-container');
    const tex = String.raw`\frac{D}{B}`;
    const text = root.querySelector('text').firstChild;
    const range = document.createRange();
    range.setStart(text, 8);
    range.setEnd(text, 15);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    const item = {math: tex, typesetRoot: root};
    window.MathJax = {startup: {document: {getMathItemsWithin: container => container.contains(root) ? [item] : []}}};
    return {html: paragraph.innerHTML, start: selection.anchorOffset, text: selection.toString()};
  });

  const source = await page.evaluate(async () => (await import('/context.js')).captureSelection());
  expect(source.quote).toContain(String.raw`$\frac{D}{B}$`);
  expect(source.quote).not.toContain('visible fraction glyph');
  expect(source.quote).not.toContain('accessible text that loses the fraction');
  expect(source.context).toContain(String.raw`$\frac{D}{B}$`);

  const after = await page.evaluate(() => {
    const paragraph = document.querySelector('#math');
    const selection = window.getSelection();
    return {html: paragraph.innerHTML, start: selection.anchorOffset, text: selection.toString()};
  });
  expect(after).toEqual(before);
});

test('MathJax context removes navigation and controls while retaining neighboring prose', async ({page}) => {
  await page.goto('/vol2/chapter/');
  const result = await page.evaluate(async () => {
    const section = document.querySelector('#bandwidth');
    section.insertAdjacentHTML('beforeend', '<form><label>Private control</label><input value="secret"></form><p>Local context stays nearby.</p>');
    const paragraph = document.querySelector('#passage');
    const range = document.createRange();
    range.selectNodeContents(paragraph);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    return (await import('/context.js')).captureSelection();
  });
  expect(result.context).toContain('Memory bandwidth measures');
  expect(result.context).toContain('Local context stays nearby.');
  expect(result.context).not.toContain('Navigation must not');
  expect(result.context).not.toContain('Private control');
  expect(result.context).not.toContain('secret');
});
