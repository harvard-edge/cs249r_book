import {test, expect} from '@playwright/test';

// Constructed hidden sentinels exercise source capture; no model requests.
test.beforeEach(async ({page}) => {
  await page.route('https://mlsysbook.ai/api/socratiq/ai/stream**', route => route.abort());
  await page.goto('/vol1/chapter/');
});

test('quote, context, and headings omit hidden live content without changing DOM or selection', async ({page}) => {
  const result = await page.evaluate(async () => {
    const {captureSelection, captureParagraph} = await import('/context.js');
    const style = document.createElement('style');
    style.textContent = '.private-layer {display:none} .private-visibility {visibility:hidden}';
    document.head.append(style);
    const section = document.querySelector('#bandwidth');
    section.innerHTML = `<h2 hidden>PRIVATE_HEADING</h2><h2>Visible heading <span hidden>PRIVATE_HEADING_CHILD</span></h2>
      <p id="selected">Visible beginning <span hidden>PRIVATE_ATTRIBUTE</span><span class="private-layer">PRIVATE_CLASS</span><span inert>PRIVATE_INLINE_INERT</span>visible ending.</p>
      <p hidden>PRIVATE_SIBLING</p><p style="display:none">PRIVATE_INLINE_STYLE</p>
      <div inert><p>PRIVATE_INERT</p></div><div class="private-layer"><p>PRIVATE_ANCESTOR</p></div>
      <p class="private-visibility">PRIVATE_VISIBILITY</p><p style="opacity:0">PRIVATE_OPACITY</p>
      <p style="content-visibility:hidden">PRIVATE_CONTENT_VISIBILITY</p><p aria-hidden="true">PRIVATE_ARIA</p>
      <details><summary>Visible summary</summary><p>PRIVATE_COLLAPSED</p>PRIVATE_COLLAPSED_TEXT</details>
      <p>Visible neighboring paragraph.</p>`;
    const title = document.querySelector('h1');
    title.insertAdjacentHTML('beforeend', '<span hidden>PRIVATE_TITLE</span>');
    const paragraph = document.querySelector('#selected');
    const range = document.createRange(); range.selectNodeContents(paragraph);
    const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
    const snapshot = () => ({html: document.querySelector('main').innerHTML, text: selection.toString(), start: selection.anchorOffset, end: selection.focusOffset});
    const before = snapshot();
    const source = captureSelection();
    const tapped = captureParagraph(paragraph);
    return {source, tapped, before, after: snapshot()};
  });
  expect(result.source.quote).toBe('Visible beginning visible ending.');
  expect(result.tapped.quote).toBe(result.source.quote);
  expect(result.source.context).toContain('Visible neighboring paragraph.');
  expect(result.source.context).toContain('Visible summary');
  expect(result.source.section).toBe('Visible heading');
  expect(JSON.stringify([result.source, result.tapped])).not.toContain('PRIVATE_');
  expect(result.after).toEqual(result.before);
});

test('capture rejects programmatic selections and paragraph taps inside hidden ancestors', async ({page}) => {
  const results = await page.evaluate(async () => {
    const {captureSelection, captureParagraph} = await import('/context.js');
    const section = document.querySelector('#bandwidth');
    section.innerHTML = '<div style="display:none"><p id="hidden">PRIVATE_PARENT</p></div><div inert><p id="inert">PRIVATE_INERT</p></div><details><summary>Visible</summary><p id="collapsed">PRIVATE_COLLAPSED</p></details>';
    return ['hidden', 'inert', 'collapsed'].map(id => {
      const paragraph = document.getElementById(id);
      const range = document.createRange(); range.selectNodeContents(paragraph);
      const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
      return [captureSelection(), captureParagraph(paragraph)];
    });
  });
  expect(results.flat().every(source => source === null)).toBe(true);
});

test('visible math keeps hidden TeX accessibility sources while hidden expressions stay excluded', async ({page}) => {
  const result = await page.evaluate(async () => {
    const {captureSelection} = await import('/context.js');
    const section = document.querySelector('#bandwidth');
    section.innerHTML = `<h2>Math context</h2><p hidden><mjx-container id="private-math"><svg><text>PRIVATE_GLYPH</text></svg></mjx-container></p>
      <p id="selected">Visible math <mjx-container id="visible-math"><svg aria-hidden="true"><text>fraction glyph</text></svg><mjx-assistive-mml style="display:none">hidden accessibility layer</mjx-assistive-mml></mjx-container> and <span class="katex"><span aria-hidden="true">duplicated glyph</span><math style="display:none"><semantics><annotation encoding="application/x-tex">E = mc^2</annotation></semantics></math></span>.</p>
      <p style="display:none"><span class="katex"><annotation encoding="application/x-tex">PRIVATE_KATEX</annotation></span></p>
      <p>Fallback <mjx-container><span aria-hidden="true">visible fallback expression</span><span hidden>PRIVATE_FALLBACK</span></mjx-container>.</p>`;
    const hidden = document.querySelector('#private-math');
    const visible = document.querySelector('#visible-math');
    const items = [{math: 'PRIVATE_REGISTRY', typesetRoot: hidden}, {math: String.raw`\frac{D}{B}`, typesetRoot: visible}];
    window.MathJax = {startup: {document: {getMathItemsWithin: () => items}}};
    const selection = getSelection();
    const range = document.createRange(); range.selectNodeContents(document.querySelector('#selected'));
    selection.removeAllRanges(); selection.addRange(range);
    const source = captureSelection();
    const glyph = visible.querySelector('text').firstChild;
    range.setStart(glyph, 2); range.setEnd(glyph, 6);
    selection.removeAllRanges(); selection.addRange(range);
    const before = {html: section.innerHTML, text: selection.toString(), start: selection.anchorOffset, end: selection.focusOffset};
    const partial = captureSelection();
    return {source, partial, before, after: {html: section.innerHTML, text: selection.toString(), start: selection.anchorOffset, end: selection.focusOffset}};
  });
  for (const text of [result.source.quote, result.source.context]) {
    expect(text).toContain(String.raw`$\frac{D}{B}$`);
    expect(text).toContain('$E = mc^2$');
    expect(text).not.toContain('PRIVATE_');
    expect(text).not.toContain('glyph');
    expect(text).not.toContain('accessibility layer');
  }
  expect(result.partial.quote).toBe(String.raw`$\frac{D}{B}$`);
  expect(result.source.context).toContain('visible fallback expression');
  expect(result.after).toEqual(result.before);
});

test('source URLs omit query strings and embedded credentials', async ({page}) => {
  const source = await page.evaluate(async () => {
    const paragraph = document.querySelector('#passage');
    const range = document.createRange(); range.selectNodeContents(paragraph);
    const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
    return (await import('/context.js')).captureSelection({window: {
      getSelection: () => selection,
      location: new URL('https://PRIVATE_USER:PRIVATE_PASSWORD@example.invalid/vol1/chapter/?PRIVATE_QUERY=value'),
    }});
  });
  expect(source.url).toBe('https://example.invalid/vol1/chapter/#bandwidth');
  expect(JSON.stringify(source)).not.toContain('PRIVATE_');
});

test('hidden text does not displace a deep selection from bounded context', async ({page}) => {
  const source = await page.evaluate(async () => {
    const section = document.querySelector('#bandwidth');
    section.replaceChildren();
    const before = document.createElement('p'); before.textContent = 'Earlier visible material. '.repeat(800);
    const hidden = document.createElement('div'); hidden.style.display = 'none'; hidden.textContent = 'PRIVATE_PREFIX '.repeat(3000);
    const selected = document.createElement('p'); selected.textContent = 'Selected mechanism remains in the bounded context.';
    const after = document.createElement('p'); after.textContent = 'Visible explanation immediately after the selection.';
    section.append(before, hidden, selected, after);
    const range = document.createRange(); range.selectNodeContents(selected);
    const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
    return (await import('/context.js')).captureSelection();
  });
  expect(source.context).toContain(source.quote);
  expect(source.context).toContain('Visible explanation immediately after');
  expect(source.context).not.toContain('PRIVATE_');
  expect(source.context.length).toBeLessThanOrEqual(12000);
});
