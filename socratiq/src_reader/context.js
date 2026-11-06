// Capture only the reader's selection and its local section, never navigation or forms.
const EXCLUDED = 'nav, aside, form, fieldset, script, style, button, input, textarea, select, [contenteditable]:not([contenteditable="false"]), #socratiq-reader';
const PARAGRAPH_INTERACTIVE = 'a, button, input, textarea, select, option, label, summary, iframe, object, embed, audio[controls], video[controls], [role="button"], [role="link"], [role="textbox"], [contenteditable]:not([contenteditable="false"])';
const PARAGRAPH_CONTROLS = PARAGRAPH_INTERACTIVE.replace(/^a, /, '').replace(', [role="link"]', '');
const elementOf = node => node?.nodeType === 1 ? node : node?.parentElement;
// DOM references stay local: source snapshots remain plain JSON for transport.
const sourceRanges = new WeakMap();

export function revealSource(source, {document: doc = document, window: win = doc.defaultView} = {}) {
  if (!source || typeof source !== 'object') return false;
  const saved = sourceRanges.get(source);
  const node = saved?.range.startContainer;
  let target = saved?.element.isConnected && node?.isConnected && node.ownerDocument === doc
    ? (node.nodeType === 1 ? node : node.parentElement) : null;
  if (target) target = target.closest('p, li, blockquote, pre, figure, h1, h2, h3, h4, h5, h6, section, main, article') || target;
  if (!target) {
    try {
      const url = new URL(source.url, win.location.href);
      const current = new URL(win.location.href);
      if (url.origin !== current.origin || url.pathname !== current.pathname) return false;
      const id = decodeURIComponent(url.hash.slice(1));
      target = id ? doc.getElementById(id) : null;
    } catch { return false; }
  }
  if (!target?.isConnected || !target.closest('main, article') || target.closest(EXCLUDED)) return false;
  target.scrollIntoView({block: 'center', behavior: win.matchMedia?.('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
  const tabindex = target.getAttribute('tabindex');
  if (tabindex === null) {
    target.setAttribute('tabindex', '-1');
    target.addEventListener('blur', () => target.removeAttribute('tabindex'), {once: true});
  }
  target.focus({preventScroll: true});
  return true;
}

function mathItemsWithin(win, element) {
  try {
    return win.MathJax?.startup?.document?.getMathItemsWithin?.(element) || [];
  } catch {
    // A page can expose MathJax before its startup document is ready.
    return [];
  }
}

function itemRoot(item) {
  return item?.typesetRoot?.nodeType === 1 ? item.typesetRoot : null;
}

function itemTex(item) {
  return typeof item?.math === 'string' && item.math ? item.math : null;
}

function excludedElement(element, {mathLayer = false} = {}) {
  if (element.matches(`${EXCLUDED}, [hidden], [inert]`) || (!mathLayer && element.getAttribute('aria-hidden') === 'true')) return true;
  // Read computed styles on the live page: detached clones lose stylesheet and
  // ancestor visibility, including classes used for private/collapsed content.
  const style = element.ownerDocument.defaultView?.getComputedStyle(element);
  return style?.display === 'none' || style?.visibility === 'hidden' || style?.visibility === 'collapse'
    || style?.contentVisibility === 'hidden' || style?.opacity === '0';
}

function excludedAncestor(element) {
  if (element?.closest(EXCLUDED)) return true;
  // Visible math is one expression. Its accessibility/visual layers may be
  // hidden independently, but the expression's root and ancestors must qualify.
  const math = element?.closest('mjx-container, .katex');
  for (let node = math || element; node; node = node.parentElement) {
    if (excludedElement(node)) return true;
    const parent = node.parentElement;
    if (parent?.matches('details:not([open])') && node !== parent.querySelector('summary')) return true;
  }
  return false;
}

function visibleMathText(node) {
  if (node.nodeType === 3) return node.data;
  if (node.nodeType !== 1 || excludedElement(node, {mathLayer: true})) return '';
  return [...node.childNodes].map(visibleMathText).join('');
}

export function readableText(element, {mathItems = [], range = null} = {}) {
  if (excludedAncestor(element)) return '';
  const registered = new Map(mathItems.map(item => [itemRoot(item), itemTex(item)]));
  const parts = [];
  const visit = node => {
    if (range && !range.intersectsNode(node)) return;
    if (node.nodeType === 3) {
      const begin = range?.startContainer === node ? range.startOffset : 0;
      const end = range?.endContainer === node ? range.endOffset : node.length;
      parts.push(node.data.slice(begin, end));
      return;
    }
    if (node.nodeType !== 1 || excludedElement(node)) return;
    if (registered.has(node) || node.matches('mjx-container, .katex')) {
      // Resolve TeX before inspecting hidden implementation layers. Never read
      // a hidden expression, and never pair registry entries by clone indexes.
      const tex = registered.get(node) || node.querySelector('annotation[encoding="application/x-tex"]')?.textContent;
      const accessible = node.querySelector('mjx-assistive-mml')?.textContent;
      parts.push(tex ? ` $${tex}$ ` : accessible || visibleMathText(node));
      return;
    }
    if (node.matches('details:not([open])')) {
      const summary = node.querySelector('summary');
      if (summary?.parentElement === node) visit(summary);
    } else {
      for (const child of node.childNodes) visit(child);
    }
  };
  visit(element);
  return parts.join('').replace(/\s+/g, ' ').trim();
}

export function captureSelection({document: doc = document, window: win = window, volume, version = ''} = {}) {
  const selection = win.getSelection();
  if (!selection || selection.isCollapsed || !selection.rangeCount) return null;
  const range = selection.getRangeAt(0);
  return captureRange(range, {document: doc, window: win, volume, version});
}

// Capture a paragraph with a detached Range; the reader's native selection is untouched.
export function captureParagraph(node, {document: doc = document, window: win = doc.defaultView, volume, version = ''} = {}) {
  const target = elementOf(node);
  if (!target?.isConnected || target.ownerDocument !== doc || excludedAncestor(target) || target.closest(PARAGRAPH_INTERACTIVE)) return null;
  const paragraph = target.closest('p');
  if (!paragraph?.closest('main, article') || paragraph.closest(EXCLUDED) || paragraph.querySelector(`${EXCLUDED}, ${PARAGRAPH_CONTROLS}`)) return null;
  const range = doc.createRange();
  range.selectNodeContents(paragraph);
  return captureRange(range, {document: doc, window: win, volume, version});
}

function captureRange(range, {document: doc, window: win, volume, version}) {
  const start = elementOf(range.startContainer);
  const end = elementOf(range.endContainer);
  const main = start?.closest('main, article');
  if (!main || !main.contains(end) || excludedAncestor(start) || excludedAncestor(end)) return null;
  const section = elementOf(range.commonAncestorContainer).closest('section') || main;
  const mathItems = mathItemsWithin(win, section);
  const quote = readableText(section, {mathItems, range});
  if (!quote) return null;
  if (quote.length > 4000) throw new Error('Select a shorter passage (a few paragraphs) so SocratiQ can focus its explanation.');
  const heading = [...section.querySelectorAll('h1,h2,h3,h4,h5,h6')].find(node => !excludedAncestor(node));
  const anchor = section.id || heading?.id || start.closest('[id]')?.id;
  const url = new URL(win.location.href);
  // Do not send query strings, which can contain private or tracking parameters.
  url.search = '';
  url.username = '';
  url.password = '';
  url.hash = anchor || '';
  let context = readableText(section, {mathItems});
  if (context.length > 12000) {
    // Center the bounded context on the selection, even deep inside a long section.
    const prefix = doc.createRange();
    prefix.selectNodeContents(section);
    prefix.setEnd(range.startContainer, range.startOffset);
    for (const item of mathItems) {
      const root = itemRoot(item);
      if (root?.contains(range.startContainer)) {
        // Count the expression as a unit when the selection begins inside it.
        prefix.setEndAfter(root);
        break;
      }
    }
    const offset = readableText(section, {mathItems, range: prefix}).length;
    const begin = Math.max(0, offset - 4000);
    context = context.slice(begin, begin + 12000);
  }
  const rect = range.getBoundingClientRect();
  const title = [...doc.querySelectorAll('h1')].find(node => !excludedAncestor(node));
  const source = {
    quote, context, title: title ? readableText(title) || doc.title : doc.title,
    section: heading ? readableText(heading) : '', url: url.href,
    volume: volume || win.location.pathname.match(/(?:^|\/)(vol[1-4])(?:\/|$)/)?.[1] || '',
    version, rect: {top: rect.top, bottom: rect.bottom, left: rect.left, right: rect.right, width: rect.width, height: rect.height},
  };
  sourceRanges.set(source, {range: range.cloneRange(), element: start});
  return source;
}
