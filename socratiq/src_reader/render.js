import markdownit from 'markdown-it';
import DOMPurify from 'dompurify';
import katex from 'katex';

const markdown = markdownit({
  html: false,
  linkify: true,
  breaks: true,
  typographer: false,
});

markdown.block.ruler.before('fence', 'reader_math_block', (state, startLine, endLine, silent) => {
  const start = state.bMarks[startLine] + state.tShift[startLine];
  const firstLine = state.src.slice(start, state.eMarks[startLine]);
  const delimiter = firstLine.startsWith('$$') ? '$$' : firstLine.startsWith('\\[') ? '\\[' : '';
  if (!delimiter) return false;

  const closing = delimiter === '$$' ? '$$' : '\\]';
  const openingLine = firstLine.slice(delimiter.length);
  const sameLineEnd = openingLine.indexOf(closing);
  let math = '';
  let nextLine = startLine;
  if (sameLineEnd >= 0 && openingLine.slice(sameLineEnd + closing.length).trim() === '') {
    math = openingLine.slice(0, sameLineEnd);
  } else {
    const lines = [openingLine];
    let found = false;
    for (let line = startLine + 1; line < endLine; line += 1) {
      const lineStart = state.bMarks[line] + state.tShift[line];
      const lineText = state.src.slice(lineStart, state.eMarks[line]);
      const closeIndex = lineText.indexOf(closing);
      if (closeIndex >= 0 && lineText.slice(closeIndex + closing.length).trim() === '') {
        lines.push(lineText.slice(0, closeIndex));
        nextLine = line;
        found = true;
        break;
      }
      lines.push(lineText);
    }
    if (!found) return false;
    math = lines.join('\n');
  }

  if (silent) return true;
  const token = state.push('reader_math_block', 'div', 0);
  token.block = true;
  token.content = math;
  token.map = [startLine, nextLine + 1];
  token.meta = { displayMode: true };
  state.line = nextLine + 1;
  return true;
});

markdown.inline.ruler.before('escape', 'reader_math_inline', (state, silent) => {
  const start = state.pos;
  const source = state.src;
  let delimiter = '';
  let closing = '';
  let displayMode = false;
  if (source.startsWith('\\(', start)) {
    delimiter = '\\(';
    closing = '\\)';
  } else if (source.startsWith('\\[', start)) {
    delimiter = '\\[';
    closing = '\\]';
    displayMode = true;
  } else if (source.startsWith('$$', start)) {
    delimiter = '$$';
    closing = '$$';
    displayMode = true;
  } else if (source.startsWith('$', start) && !source.startsWith('$$', start)) {
    let slashCount = 0;
    for (let index = start - 1; index >= 0 && source[index] === '\\'; index -= 1) slashCount += 1;
    if (slashCount % 2 === 1) return false;
    delimiter = '$';
    closing = '$';
  } else {
    return false;
  }

  const contentStart = start + delimiter.length;
  const closeIndex = source.indexOf(closing, contentStart);
  if (closeIndex < 0 || source.slice(contentStart, closeIndex).includes('\n')) return false;
  const tex = source.slice(contentStart, closeIndex);
  if (!tex.trim() || (delimiter === '$' && /^\s|\s$/.test(tex))) return false;
  if (delimiter === '$' && source.startsWith('$$', closeIndex)) return false;
  if (silent) return true;

  const token = state.push('reader_math_inline', '', 0);
  token.content = tex;
  token.meta = { displayMode };
  state.pos = closeIndex + closing.length;
  return true;
});

const mathRenderer = (tokens, index, options, env) => {
  const item = tokens[index];
  const id = env.readerMath.push({ tex: item.content, displayMode: Boolean(item.meta?.displayMode) }) - 1;
  return `SQRMTOKEN${id}END`;
};
markdown.renderer.rules.reader_math_inline = mathRenderer;
markdown.renderer.rules.reader_math_block = (tokens, index, options, env) => {
  const marker = mathRenderer(tokens, index, options, env);
  return `<div class="reader-math-block">${marker}</div>\n`;
};

/** Render model text as sanitized Markdown, then add KaTeX to safe text nodes. */
export function renderSafeMarkdown(source = '') {
  const env = { readerMath: [] };
  const safeHtml = DOMPurify.sanitize(markdown.render(String(source), env), {
    FORBID_TAGS: ['script', 'style', 'iframe', 'object', 'embed', 'form', 'img'],
    FORBID_ATTR: ['onerror', 'onload'],
  });
  const container = document.createElement('div');
  container.innerHTML = safeHtml;
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  while (walker.nextNode()) textNodes.push(walker.currentNode);
  const markerPattern = /SQRMTOKEN(\d+)END/g;
  for (const textNode of textNodes) {
    const text = textNode.textContent || '';
    markerPattern.lastIndex = 0;
    let match;
    let cursor = 0;
    const fragment = document.createDocumentFragment();
    let replaced = false;
    while ((match = markerPattern.exec(text))) {
      const item = env.readerMath[Number(match[1])];
      if (!item) continue;
      replaced = true;
      if (match.index > cursor) fragment.appendChild(document.createTextNode(text.slice(cursor, match.index)));
      const mathNode = document.createElement('span');
      try {
        katex.render(item.tex, mathNode, {
          displayMode: item.displayMode,
          throwOnError: false,
          trust: false,
        });
      } catch {
        const fallback = document.createElement('code');
        fallback.className = 'reader-math-error';
        fallback.textContent = item.tex;
        mathNode.appendChild(fallback);
      }
      fragment.appendChild(mathNode);
      cursor = match.index + match[0].length;
    }
    if (replaced) {
      if (cursor < text.length) fragment.appendChild(document.createTextNode(text.slice(cursor)));
      textNode.parentNode.replaceChild(fragment, textNode);
    }
  }

  return container.innerHTML;
}
