// Capture only the reader's selection and its local section, never navigation or forms.
const EXCLUDED = 'nav, aside, form, fieldset, script, style, button, input, textarea, select, [contenteditable="true"], #socratiq-reader';

function mathItemsWithin(win, element) {
  try {
    return win.MathJax?.startup?.document?.getMathItemsWithin?.(element) || [];
  } catch {
    // A page can expose MathJax before its startup document is ready.
    return [];
  }
}

function nodePath(root, target) {
  const path = [];
  for (let node = target; node !== root; node = node.parentNode) {
    if (!node?.parentNode) return null;
    path.unshift(Array.prototype.indexOf.call(node.parentNode.childNodes, node));
  }
  return path;
}

function atPath(root, path) {
  return path?.reduce((node, index) => node?.childNodes[index], root) || null;
}

function itemRoot(item) {
  return item?.typesetRoot?.nodeType === 1 ? item.typesetRoot : null;
}

function itemTex(item) {
  return typeof item?.math === 'string' && item.math ? item.math : null;
}

function replaceMathInClone(copy, source, mathItems, range = null) {
  const intersecting = mathItems.filter(item => {
    const root = itemRoot(item);
    if (!root || !(source === root || source.contains(root))) return false;
    if (!range) return true;
    try { return range.intersectsNode(root); } catch { return false; }
  });

  if (range) {
    // cloneContents() preserves partial MathJax containers. Replace each one
    // intersected by the source range with the complete original expression.
    const clonedRoots = [...copy.querySelectorAll('mjx-container')];
    intersecting.forEach((item, index) => {
      const clone = clonedRoots[index];
      if (clone) clone.replaceWith(copy.ownerDocument.createTextNode(` $${itemTex(item) || clone.textContent}$ `));
    });
    return;
  }

  for (const item of intersecting) {
    const root = itemRoot(item);
    const clone = atPath(copy, nodePath(source, root));
    if (clone?.parentNode) clone.replaceWith(copy.ownerDocument.createTextNode(` $${itemTex(item) || clone.textContent}$ `));
  }
}

export function readableText(element, {mathItems = []} = {}) {
  const copy = element.cloneNode(true);
  // Resolve source-to-clone paths before removing siblings that shift indexes.
  replaceMathInClone(copy, element, mathItems);
  copy.querySelectorAll(EXCLUDED).forEach(node => node.remove());
  // Preserve TeX annotations for KaTeX and pages without the MathJax registry.
  copy.querySelectorAll('mjx-container, .katex').forEach(math => {
    const tex = math.querySelector('annotation[encoding="application/x-tex"]')?.textContent;
    const accessible = math.querySelector('mjx-assistive-mml')?.textContent;
    math.replaceWith(element.ownerDocument.createTextNode(tex ? ` $${tex}$ ` : accessible || math.textContent));
  });
  return (copy.textContent || '').replace(/\s+/g, ' ').trim();
}

export function captureSelection({document: doc = document, window: win = window, volume, version = ''} = {}) {
  const selection = win.getSelection();
  if (!selection || selection.isCollapsed || !selection.rangeCount) return null;
  const range = selection.getRangeAt(0);
  const elementOf = node => node.nodeType === 1 ? node : node.parentElement;
  const start = elementOf(range.startContainer);
  const end = elementOf(range.endContainer);
  const main = start?.closest('main, article');
  if (!main || !main.contains(end) || start.closest(EXCLUDED) || end.closest(EXCLUDED)) return null;
  const section = elementOf(range.commonAncestorContainer).closest('section') || main;
  const mathItems = mathItemsWithin(win, section);
  const quoteRange = range.cloneRange();
  for (const item of mathItems) {
    const root = itemRoot(item);
    if (!root || !(section === root || section.contains(root))) continue;
    try {
      if (!range.intersectsNode(root)) continue;
      // A range wholly inside MathJax's SVG/HTML output has no container in
      // cloneContents(). Expand only the local copy so the quote gets the full
      // expression while the user's live selection remains exactly as made.
      if (root.contains(range.startContainer)) quoteRange.setStartBefore(root);
      if (root.contains(range.endContainer)) quoteRange.setEndAfter(root);
    } catch { /* Ignore MathItems whose output root is no longer in the DOM. */ }
  }
  const fragment = doc.createElement('div');
  fragment.append(quoteRange.cloneContents());
  replaceMathInClone(fragment, section, mathItems, quoteRange);
  const quote = readableText(fragment);
  if (!quote) return null;
  if (quote.length > 4000) throw new Error('Select a shorter passage (a few paragraphs) so SocratiQ can focus its explanation.');
  const heading = section.querySelector('h1,h2,h3,h4,h5,h6');
  const anchor = section.id || heading?.id || start.closest('[id]')?.id;
  const url = new URL(win.location.href);
  // Do not send query strings, which can contain private or tracking parameters.
  url.search = '';
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
    const wrapper = doc.createElement('div');
    wrapper.append(prefix.cloneContents());
    replaceMathInClone(wrapper, section, mathItems, prefix);
    const offset = readableText(wrapper).length;
    const begin = Math.max(0, offset - 4000);
    context = context.slice(begin, begin + 12000);
  }
  const rect = range.getBoundingClientRect();
  return {
    quote, context, title: doc.querySelector('h1')?.textContent.trim() || doc.title,
    section: heading?.textContent.trim() || '', url: url.href,
    volume: volume || win.location.pathname.match(/(?:^|\/)(vol[1-4])(?:\/|$)/)?.[1] || '',
    version, rect: {top: rect.top, bottom: rect.bottom, left: rect.left, right: rect.right, width: rect.width, height: rect.height},
  };
}
