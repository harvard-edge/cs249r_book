import { completeReaderStyles } from './styles.js';
import { renderSafeMarkdown } from './render.js';

const MAX_HISTORY_MESSAGES = 12;
const MAX_VISIBLE_TURNS = 12;
const INITIAL_QUESTION = 'Explain this passage.';

/**
 * Mount the focused SocratiQ reader.
 *
 * `captureSelection()` should return a plain snapshot with `quote`, `context`,
 * `title`, `section`, `url`, `volume`, `version`, and a viewport `rect`, or null.
 * `requestExplanation()` receives the active source, current question, prior
 * completed turns, an AbortSignal, and an onChunk callback. `history` contains
 * prior turns only; the current `question` is passed separately. Each history
 * item has the shape `{ role: 'user' | 'assistant', content: string }`.
 */
export function mountReader({ captureSelection, requestExplanation }) {
  if (typeof captureSelection !== 'function' || typeof requestExplanation !== 'function') {
    throw new TypeError('mountReader requires captureSelection and requestExplanation functions.');
  }

  const host = document.createElement('div');
  host.id = 'socratiq-reader';
  document.body.appendChild(host);
  const root = host.attachShadow({ mode: 'open' });

  const style = document.createElement('style');
  style.textContent = completeReaderStyles;
  root.appendChild(style);

  const shell = document.createElement('div');
  shell.className = 'reader-shell';
  shell.dataset.theme = detectTheme();
  shell.innerHTML = `
    <button class="reader-launcher" type="button" aria-expanded="false" aria-controls="reader-panel">
      <span class="reader-mark" aria-hidden="true">S</span>
      <span>Ask SocratiQ</span>
    </button>
    <button class="selection-action" type="button" hidden>
      <span class="reader-mark" aria-hidden="true">S</span>
      <span>Explain this</span>
    </button>
    <section class="reader-panel" id="reader-panel" role="region" aria-label="SocratiQ focused reader" hidden>
      <header class="reader-header">
        <div class="reader-heading">
          <p class="reader-eyebrow">SocratiQ reader</p>
          <h2 class="reader-title">Ask about the text</h2>
          <p class="reader-header-note">A focused explanation, grounded in this passage.</p>
        </div>
        <button class="reader-icon-button reader-close" type="button" aria-label="Close SocratiQ reader" title="Close">
          <span aria-hidden="true">×</span>
        </button>
      </header>
      <p class="reader-privacy">The selected passage and surrounding section are sent for AI explanation when you ask.</p>
      <div class="reader-body">
        <section class="reader-source" aria-label="Selected source" hidden>
          <p class="reader-source-label">Selected passage</p>
          <a class="reader-source-link" target="_blank" rel="noopener noreferrer"></a>
          <p class="reader-quote"></p>
        </section>
        <div class="reader-instruction">
          <strong>Select a passage to begin.</strong>
          Highlight text in the book, then choose “Explain this” or press <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>E</kbd>.
        </div>
        <div class="reader-thread" role="log" aria-label="Explanation conversation" aria-live="off"></div>
      </div>
      <p class="reader-status" role="status" aria-live="polite"></p>
      <div class="reader-error" role="alert" hidden>
        <span class="reader-error-text"></span>
        <button class="reader-retry" type="button">Retry</button>
      </div>
      <form class="reader-composer" aria-disabled="true">
        <textarea rows="1" aria-label="Ask a follow-up question" placeholder="Select a passage to begin" disabled></textarea>
        <button class="reader-send" type="submit" aria-label="Send question" title="Send" disabled>
          <span aria-hidden="true">↑</span>
        </button>
        <button class="reader-stop" type="button" aria-label="Stop explanation" hidden>Stop</button>
      </form>
    </section>
  `;
  root.appendChild(shell);

  const launcher = shell.querySelector('.reader-launcher');
  const panel = shell.querySelector('.reader-panel');
  const closeButton = shell.querySelector('.reader-close');
  const selectionButton = shell.querySelector('.selection-action');
  const sourceCard = shell.querySelector('.reader-source');
  const sourceLink = shell.querySelector('.reader-source-link');
  const sourceQuote = shell.querySelector('.reader-quote');
  const instruction = shell.querySelector('.reader-instruction');
  const thread = shell.querySelector('.reader-thread');
  const status = shell.querySelector('.reader-status');
  const errorBox = shell.querySelector('.reader-error');
  const errorText = shell.querySelector('.reader-error-text');
  const retryButton = shell.querySelector('.reader-retry');
  const composer = shell.querySelector('.reader-composer');
  const input = shell.querySelector('textarea');
  input.maxLength = 2000;
  const sendButton = shell.querySelector('.reader-send');
  const stopButton = shell.querySelector('.reader-stop');

  let pendingSource = null;
  let activeSource = null;
  let history = [];
  let activeController = null;
  let lastFailedQuestion = null;
  let failedUserTurn = null;
  let visibleTurns = 0;
  let selectionTimer = 0;
  let selectionActionPressed = false;

  updateViewportVars();
  launcher.addEventListener('click', () => {
    if (pendingSource) setActiveSource(pendingSource);
    openPanel(launcher, 'launcher');
  });

  selectionButton.addEventListener('pointerdown', (event) => {
    selectionActionPressed = true;
    if (event.pointerType !== 'touch') event.preventDefault();
  });
  selectionButton.addEventListener('pointercancel', () => {
    selectionActionPressed = false;
    pendingSource = null;
    hideSelectionAction();
  });
  selectionButton.addEventListener('click', () => {
    selectionActionPressed = false;
    if (!pendingSource) return;
    const source = pendingSource;
    setActiveSource(source);
    selectionButton.hidden = true;
    openPanel(selectionButton, 'selection');
    void submitQuestion(INITIAL_QUESTION, { initial: true });
  });

  closeButton.addEventListener('click', closePanel);
  stopButton.addEventListener('click', () => stopRequest('Response stopped.'));
  retryButton.addEventListener('click', () => {
    if (!lastFailedQuestion || !activeSource) return;
    errorBox.hidden = true;
    void submitQuestion(lastFailedQuestion, { reuseUserTurn: failedUserTurn });
  });

  composer.addEventListener('submit', (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question || !activeSource || activeController) return;
    input.value = '';
    input.style.height = '';
    void submitQuestion(question);
  });

  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      composer.requestSubmit();
    }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 110)}px`;
  });

  document.addEventListener('selectionchange', scheduleSelectionCapture);
  document.addEventListener('pointerup', scheduleSelectionCapture, true);
  // Keep compatibility with browsers that do not dispatch Pointer Events.
  document.addEventListener('mouseup', scheduleSelectionCapture, true);
  document.addEventListener('scroll', hideSelectionAction, { capture: true, passive: true });
  document.addEventListener('keydown', handleEscape, true);
  document.addEventListener('keydown', handleSelectionShortcut, true);
  window.addEventListener('resize', updateViewportVars, { passive: true });
  window.visualViewport?.addEventListener('resize', updateViewportVars, { passive: true });
  window.visualViewport?.addEventListener('scroll', updateViewportVars, { passive: true });

  function openPanel(opener, origin) {
    panel.dataset.launchOrigin = origin;
    panel.hidden = false;
    launcher.hidden = true;
    launcher.setAttribute('aria-expanded', 'true');
    selectionButton.hidden = true;
    updateComposerState();
    requestAnimationFrame(() => {
      if (!panel.hidden && activeSource) input.focus({ preventScroll: true });
      else if (!panel.hidden) closeButton.focus({ preventScroll: true });
    });
  }

  function closePanel() {
    if (panel.hidden) return;
    stopRequest('Response stopped because the reader was closed.');
    panel.hidden = true;
    launcher.hidden = false;
    launcher.setAttribute('aria-expanded', 'false');
    selectionButton.hidden = true;
    requestAnimationFrame(() => launcher.focus({ preventScroll: true }));
  }

  function handleEscape(event) {
    if (event.key !== 'Escape') return;
    if (!panel.hidden) {
      event.preventDefault();
      event.stopPropagation();
      closePanel();
    } else if (!selectionButton.hidden) {
      event.preventDefault();
      selectionButton.hidden = true;
    }
  }

  function handleSelectionShortcut(event) {
    if (event.key.toLowerCase() !== 'e' || !event.ctrlKey || !event.shiftKey || event.altKey || event.metaKey) return;
    let source;
    try {
      source = captureSelection();
    } catch {
      return;
    }
    if (!source || !String(source.quote || '').trim()) return;
    event.preventDefault();
    event.stopPropagation();
    pendingSource = source;
    setActiveSource(source);
    selectionButton.hidden = true;
    openPanel(launcher, 'selection');
    void submitQuestion(INITIAL_QUESTION, { initial: true });
  }

  function stopRequest(message) {
    if (!activeController) return;
    const controller = activeController;
    activeController = null;
    controller.abort();
    stopButton.hidden = true;
    sendButton.hidden = false;
    input.disabled = !activeSource;
    sendButton.disabled = !activeSource;
    status.textContent = message;
  }

  function scheduleSelectionCapture(event) {
    if (event?.composedPath?.().includes(host)) return;
    if (selectionActionPressed) return;
    if (selectionIsInsideReader()) return;
    clearTimeout(selectionTimer);
    selectionTimer = window.setTimeout(readSelection, event?.type === 'selectionchange' ? 90 : 40);
  }

  function readSelection() {
    try {
      const source = captureSelection();
      if (!source || !String(source.quote || '').trim()) {
        pendingSource = null;
        if (!panel.hidden) return;
        hideSelectionAction();
        return;
      }
      pendingSource = source;
      positionSelectionAction(source.rect);
      selectionButton.hidden = false;
    } catch (error) {
      pendingSource = null;
      hideSelectionAction();
      openPanel(launcher, 'launcher');
      showCaptureError(error);
    }
  }

  function selectionIsInsideReader() {
    const selection = window.getSelection();
    return Boolean(selection?.anchorNode && selection.anchorNode.getRootNode() === root);
  }

  function hideSelectionAction() {
    selectionButton.hidden = true;
  }

  function positionSelectionAction(rect = {}) {
    const viewport = window.visualViewport;
    const leftEdge = viewport?.offsetLeft ?? 0;
    const topEdge = viewport?.offsetTop ?? 0;
    const width = viewport?.width ?? window.innerWidth;
    const height = viewport?.height ?? window.innerHeight;
    const buttonWidth = 144;
    const buttonHeight = 42;
    const selectionLeft = Number(rect.left ?? rect.x ?? 0);
    const selectionTop = Number(rect.top ?? rect.y ?? 0);
    const selectionWidth = Number(rect.width ?? Math.max(0, Number(rect.right) - selectionLeft));
    const selectionBottom = Number(rect.bottom ?? (selectionTop + Number(rect.height ?? 0)));
    let left = selectionLeft + Math.max(0, (selectionWidth - buttonWidth) / 2);
    let top = selectionTop - buttonHeight - 9;
    if (top < topEdge + 8) top = selectionBottom + 9;
    left = Math.min(Math.max(left, leftEdge + 8), leftEdge + width - buttonWidth - 8);
    top = Math.min(Math.max(top, topEdge + 8), topEdge + height - buttonHeight - 8);
    selectionButton.style.left = `${left}px`;
    selectionButton.style.top = `${top}px`;
  }

  function setActiveSource(source) {
    if (!source) return;
    const isNewSource = sourceKey(activeSource) !== sourceKey(source);
    activeSource = source;
    if (isNewSource) clearConversation();
    renderSource(source);
    updateComposerState();
  }

  function sourceKey(source) {
    if (!source) return '';
    return [source.url, source.section, source.quote].map((value) => String(value || '')).join('\u0000');
  }

  function renderSource(source) {
    const url = safeSourceUrl(source.url);
    const heading = String(source.section || source.title || 'Textbook passage');
    sourceLink.textContent = heading;
    if (url) {
      sourceLink.href = url;
      sourceLink.hidden = false;
    } else {
      sourceLink.removeAttribute('href');
      sourceLink.hidden = false;
    }
    const volumeNumber = {vol1: 'I', vol2: 'II', vol3: 'III', vol4: 'IV'}[source.volume] || String(source.volume || '');
    const volumeLabel = source.volume ? ` · Volume ${volumeNumber || source.volume}` : '';
    sourceCard.title = source.version ? `Published version: ${source.version}` : '';
    const versionLabel = '';
    sourceCard.querySelector('.reader-source-label').textContent = `Selected passage${volumeLabel}${versionLabel}`;
    sourceQuote.textContent = String(source.quote || '');
    sourceCard.hidden = false;
    instruction.hidden = true;
  }

  function safeSourceUrl(value) {
    try {
      const url = new URL(String(value || ''), window.location.href);
      return ['http:', 'https:'].includes(url.protocol) ? url.href : '';
    } catch {
      return '';
    }
  }

  function clearConversation() {
    stopRequest('');
    history = [];
    visibleTurns = 0;
    thread.replaceChildren();
    errorBox.hidden = true;
    retryButton.hidden = false;
    status.textContent = '';
    lastFailedQuestion = null;
    failedUserTurn = null;
  }

  function updateComposerState() {
    composer.setAttribute('aria-disabled', String(!activeSource));
    input.disabled = !activeSource || Boolean(activeController);
    sendButton.disabled = !activeSource || Boolean(activeController);
    input.placeholder = activeSource ? 'Ask a follow-up…' : 'Select a passage to begin';
  }

  function appendUserTurn(question) {
    const turn = document.createElement('article');
    turn.className = 'reader-turn reader-turn-user';
    const label = document.createElement('p');
    label.className = 'reader-turn-label';
    label.textContent = 'You';
    const text = document.createElement('p');
    text.className = 'reader-user-text';
    text.textContent = question;
    turn.append(label, text);
    thread.appendChild(turn);
    visibleTurns += 1;
    trimVisibleTurns();
    return turn;
  }

  function appendAssistantTurn() {
    const turn = document.createElement('article');
    turn.className = 'reader-turn reader-turn-assistant';
    const label = document.createElement('p');
    label.className = 'reader-turn-label';
    label.textContent = 'SocratiQ';
    const answer = document.createElement('div');
    answer.className = 'reader-answer';
    turn.append(label, answer);
    thread.appendChild(turn);
    return answer;
  }

  async function submitQuestion(question, { initial = false, reuseUserTurn = null } = {}) {
    if (!activeSource || activeController) return;
    const body = shell.querySelector('.reader-body');
    const followStream = !initial && isAtBottom(body);
    const userTurn = reuseUserTurn || appendUserTurn(question);
    const answerNode = appendAssistantTurn();
    answerNode.textContent = 'Thinking…';
    if (initial) scrollAnswerToStart(answerNode);
    errorBox.hidden = true;
    status.textContent = 'Generating an explanation…';
    input.disabled = true;
    sendButton.hidden = true;
    stopButton.hidden = false;
    sendButton.disabled = true;
    const controller = new AbortController();
    activeController = controller;
    updateViewportVars();

    try {
      const answer = await requestExplanation({
        source: activeSource,
        question,
        history: boundHistory(history),
        signal: controller.signal,
        onChunk: (accumulatedAnswer) => {
          if (controller.signal.aborted || activeController !== controller) return;
          answerNode.innerHTML = renderSafeMarkdown(accumulatedAnswer || '');
          if (followStream) scrollThreadToEnd();
        },
      });
      if (controller.signal.aborted || activeController !== controller) return;
      const completeAnswer = String(answer || '').trim();
      answerNode.innerHTML = renderSafeMarkdown(completeAnswer || 'SocratiQ did not return an explanation.');
      history.push({ role: 'user', content: question }, { role: 'assistant', content: completeAnswer });
      history = boundHistory(history);
      trimVisibleTurns();
      lastFailedQuestion = null;
      failedUserTurn = null;
      status.textContent = 'Explanation ready.';
    } catch (error) {
      if (controller.signal.aborted || activeController !== controller) {
        answerNode.textContent = 'Response stopped.';
        return;
      }
      answerNode.closest('.reader-turn')?.remove();
      lastFailedQuestion = question;
      failedUserTurn = userTurn;
      errorText.textContent = getReadableError(error);
      retryButton.hidden = false;
      errorBox.hidden = false;
      status.textContent = '';
    } finally {
      if (activeController === controller) {
        activeController = null;
        stopButton.hidden = true;
        sendButton.hidden = false;
        updateComposerState();
      if (followStream) scrollThreadToEnd();
      }
    }
  }

  function trimVisibleTurns() {
    while (visibleTurns > MAX_VISIBLE_TURNS) {
      const firstTurn = thread.querySelector('.reader-turn-user');
      const nextTurn = firstTurn?.nextElementSibling;
      const followingAnswer = nextTurn?.classList.contains('reader-turn-assistant') ? nextTurn : null;
      firstTurn?.remove();
      followingAnswer?.remove();
      visibleTurns -= 1;
    }
  }

  function boundHistory(turns) {
    const bounded = [];
    let totalCharacters = 0;
    for (const turn of turns.slice(-MAX_HISTORY_MESSAGES).reverse()) {
      const content = String(turn.content || '').slice(0, 6000);
      const remaining = 12000 - totalCharacters;
      if (remaining <= 0) break;
      const keptContent = content.length > remaining ? content.slice(0, remaining) : content;
      bounded.push({ role: turn.role, content: keptContent });
      totalCharacters += keptContent.length;
    }
    return bounded.reverse();
  }

  function scrollThreadToEnd() {
    const body = shell.querySelector('.reader-body');
    body.scrollTop = body.scrollHeight;
  }

  function isAtBottom(body) {
    return body.scrollHeight - body.scrollTop - body.clientHeight <= 24;
  }

  function scrollAnswerToStart(answerNode) {
    const body = shell.querySelector('.reader-body');
    const bodyTop = body.getBoundingClientRect().top;
    const answerTop = answerNode.getBoundingClientRect().top;
    const sourceHeight = sourceCard.hidden ? 0 : sourceCard.getBoundingClientRect().height;
    body.scrollTop = Math.max(0, body.scrollTop + answerTop - bodyTop - sourceHeight);
  }

  function showCaptureError(error) {
    const message = getReadableError(error, true);
    errorText.textContent = message;
    retryButton.hidden = true;
    errorBox.hidden = false;
    status.textContent = '';
  }

  function getReadableError(error, captureFailure = false) {
    const rawMessage = String(error?.message || '');
    if (captureFailure && /large|length|too many|limit|maximum|exceed|shorter passage/i.test(rawMessage)) {
      return 'That passage is too long to explain at once. Select a shorter passage and try again.';
    }
    if (captureFailure) return 'That passage could not be captured. Select a shorter passage and try again.';
    return rawMessage && rawMessage.length < 180
      ? rawMessage
      : 'The explanation could not be completed. Check your connection and try again.';
  }

  function updateViewportVars() {
    const viewport = window.visualViewport;
    const visibleHeight = viewport?.height ?? window.innerHeight;
    const visibleBottom = (viewport?.offsetTop ?? 0) + visibleHeight;
    const bottomClearance = Math.max(0, window.innerHeight - visibleBottom);
    host.style.setProperty('--reader-visible-height', `${visibleHeight}px`);
    host.style.setProperty('--reader-bottom-clearance', `${bottomClearance}px`);
  }

  function detectTheme() {
    const override = window.SocratiqWidgetTheme;
    if (override === 'dark' || override === 'light') return override;
    if (document.documentElement.matches('.quarto-dark, [data-bs-theme="dark"]') || document.body?.classList.contains('dark-mode')) return 'dark';
    let background = [255, 255, 255];
    let hasBackground = false;
    for (const element of [document.documentElement, document.body].filter(Boolean)) {
      const color = window.getComputedStyle(element).backgroundColor;
      const rgb = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:\s*,\s*([\d.]+))?\s*\)/i);
      if (!rgb) continue;
      const [red, green, blue] = rgb.slice(1, 4).map(Number);
      const alpha = rgb[4] === undefined ? 1 : Number(rgb[4]);
      if (alpha <= 0) continue;
      hasBackground = true;
      background = [red, green, blue].map((channel, index) => channel * alpha + background[index] * (1 - alpha));
    }
    if (hasBackground) {
      const [red, green, blue] = background;
      const luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255;
      if (luminance < 0.43) return 'dark';
      if (luminance > 0.57) return 'light';
    }
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function updateTheme() {
    shell.dataset.theme = detectTheme();
  }

  const themeObserver = installThemeTracking();

  function installThemeTracking() {
    const observer = new MutationObserver(updateTheme);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'style', 'data-bs-theme'] });
    if (document.body) observer.observe(document.body, { attributes: true, attributeFilter: ['class', 'style'] });
    document.addEventListener('socratiqThemeChanged', updateTheme);
    return observer;
  }

  return {
    host,
    destroy() {
      stopRequest('Response stopped.');
      clearTimeout(selectionTimer);
      document.removeEventListener('selectionchange', scheduleSelectionCapture);
      document.removeEventListener('pointerup', scheduleSelectionCapture, true);
      document.removeEventListener('mouseup', scheduleSelectionCapture, true);
      document.removeEventListener('scroll', hideSelectionAction, true);
      document.removeEventListener('keydown', handleEscape, true);
      document.removeEventListener('keydown', handleSelectionShortcut, true);
      document.removeEventListener('socratiqThemeChanged', updateTheme);
      window.removeEventListener('resize', updateViewportVars);
      window.visualViewport?.removeEventListener('resize', updateViewportVars);
      window.visualViewport?.removeEventListener('scroll', updateViewportVars);
      themeObserver.disconnect();
      host.remove();
    },
  };
}
