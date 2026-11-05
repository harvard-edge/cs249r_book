import { completeReaderStyles } from './styles.js';
import { renderSafeMarkdown } from './render.js';
import { createParagraphPicker } from './picker.js';
import { createPassageSessions, passageKey } from './passage-sessions.js';
import { createReaderEnvironment } from './environment.js';

const MAX_HISTORY_MESSAGES = 12;
const MAX_VISIBLE_TURNS = 12;
const INITIAL_QUESTION = 'Explain this passage.';
const LEARNING_QUESTIONS = {
  simplify: 'Explain this passage in simpler language, defining any essential technical terms.',
  example: 'Show a concrete example that helps me understand this passage.',
  check: 'Check my understanding of this passage. Ask me one conceptual question at a time. Do not reveal the answer until I respond; then give feedback before asking another question.',
};

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
export function mountReader({ captureSelection, captureParagraph, requestExplanation, onRevealSource, onOpenStudy }) {
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
      <label class="reader-sessions" hidden>Recent passages
        <select class="reader-session-select" aria-label="Recent passages"></select>
      </label>
      <p class="reader-privacy">The selected passage and surrounding section are sent for AI explanation when you ask.</p>
      <div class="reader-body" role="region" aria-label="Selected passage and answers" tabindex="0">
      <div class="reader-tools" hidden>
        <button class="reader-study" type="button">Study tools</button>
        <span class="reader-study-status" role="status" aria-live="polite"></span>
      </div>
        <section class="reader-source" aria-label="Selected source" hidden>
          <p class="reader-source-label">Selected passage</p>
          <a class="reader-source-link" target="_blank" rel="noopener noreferrer"></a>
          <p class="reader-quote"></p>
          <p class="reader-caution">AI explanations can be wrong. Check them against the source.</p>
        </section>
        <div class="reader-instruction">
          <strong>Select a passage to begin.</strong>
          Highlight text in the book, then choose “Explain this” or press <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>E</kbd>.
          <button class="reader-choose" type="button">Choose a paragraph</button>
        </div>
        <div class="reader-thread" role="log" aria-label="Explanation conversation" aria-live="off"></div>
        <div class="reader-learning" role="group" aria-label="Learning actions" hidden>
          <button class="reader-explain" type="button">Explain passage</button>
          <button type="button" data-learning="simplify">Simplify</button>
          <button type="button" data-learning="example">Show an example</button>
          <button type="button" data-learning="check">Check my understanding</button>
          <button type="button" class="reader-reset">Start over</button>
          <button class="reader-choose" type="button">Choose another paragraph</button>
        </div>
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

  const sessionPicker = shell.querySelector('.reader-session-select');
  const studyButton = shell.querySelector('.reader-study');
  const studyStatus = shell.querySelector('.reader-study-status');
  shell.querySelector('.reader-tools').hidden = typeof onOpenStudy !== 'function';
  const sessionControl = shell.querySelector('.reader-sessions');
  const launcher = shell.querySelector('.reader-launcher');
  const panel = shell.querySelector('.reader-panel');
  const closeButton = shell.querySelector('.reader-close');
  const selectionButton = shell.querySelector('.selection-action');
  const sourceCard = shell.querySelector('.reader-source');
  const sourceLink = shell.querySelector('.reader-source-link');
  const sourceQuote = shell.querySelector('.reader-quote');
  const instruction = shell.querySelector('.reader-instruction');
  const learningActions = shell.querySelector('.reader-learning');
  const learningButtons = [...shell.querySelectorAll('[data-learning]')];
  const resetButton = shell.querySelector('.reader-reset');
  const explainButton = shell.querySelector('.reader-explain');
  const chooseButtons = [...shell.querySelectorAll('.reader-choose')];
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
  const readerBody = shell.querySelector('.reader-body');
  const sessions = createPassageSessions({
    thread, body: readerBody, input, status, errorText, errorBox, retryButton,
    select: sessionPicker, control: sessionControl,
  });
  const environment = createReaderEnvironment({ host, shell });

  let pendingSource = null;
  let activeSource = null;
  let history = [];
  let activeController = null;
  let activeAnswerNode = null;
  let lastFailedQuestion = null;
  let failedUserTurn = null;
  let visibleTurns = 0;
  let selectionTimer = 0;
  let selectionActionPressed = false;
  let studyBusy = false;
  let studyActive = false;
  let destroyed = false;

  studyButton.addEventListener('click', async () => {
    if (studyBusy || studyActive || typeof onOpenStudy !== 'function') return;
    studyBusy = true;
    studyButton.disabled = true;
    studyStatus.textContent = 'Loading study tools…';
    stopRequest('Response stopped because study tools were opened.');
    clearTimeout(selectionTimer);
    picker?.stop();
    pendingSource = null;
    hideSelectionAction();
    try {
      await onOpenStudy();
      if (!destroyed) suspendForStudy();
    } catch {
      if (destroyed) return;
      studyStatus.textContent = 'Study tools could not open. Check your connection and try again.';
      studyButton.textContent = 'Retry study tools';
    } finally {
      studyBusy = false;
      studyButton.disabled = false;
    }
  });

  function suspendForStudy() {
    if (destroyed) return;
    studyActive = true;
    clearTimeout(selectionTimer);
    picker?.stop();
    pendingSource = null;
    hideSelectionAction();
    closePanel({restoreFocus: false});
    shell.hidden = true;
  }

  function resumeFromStudy() {
    if (!studyActive || destroyed) return;
    studyActive = false;
    shell.hidden = false;
    studyStatus.textContent = '';
    studyButton.textContent = 'Study tools';
    openPanel('study');
    requestAnimationFrame(() => studyButton.focus({preventScroll: true}));
  }
  window.addEventListener('socratiq:open', suspendForStudy);
  window.addEventListener('socratiq:close', resumeFromStudy);

  const picker = typeof captureParagraph === 'function' ? createParagraphPicker({
    shell, host, captureParagraph,
    onPick(source) {
      setActiveSource(source);
      openPanel('picker');
      if (!visibleTurns) status.textContent = 'Passage ready. Choose Explain passage or ask a question.';
    },
    onCancel() { openPanel('launcher'); },
  }) : null;
  chooseButtons.forEach(button => {
    button.hidden = !picker;
    button.addEventListener('click', () => {
      clearTimeout(selectionTimer);
      pendingSource = null;
      hideSelectionAction();
      closePanel({restoreFocus: false});
      launcher.hidden = true;
      picker.start();
    });
  });
  explainButton.addEventListener('click', () => void submitQuestion(INITIAL_QUESTION, {initial: true}));

  sessionPicker.addEventListener('change', () => {
    const source = sessions.sourceForKey(sessionPicker.value);
    if (source) setActiveSource(source);
    pendingSource = null;
    hideSelectionAction();
  });

  launcher.addEventListener('click', () => {
    if (pendingSource) setActiveSource(pendingSource);
    openPanel('launcher');
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
    openPanel('selection');
    if (!history.length && !thread.childElementCount) void submitQuestion(INITIAL_QUESTION, { initial: true });
  });

  learningButtons.forEach(button => button.addEventListener('click', () => {
    void submitQuestion(LEARNING_QUESTIONS[button.dataset.learning]);
  }));
  resetButton.addEventListener('click', () => {
    clearConversation();
    updateComposerState();
    status.textContent = 'Conversation cleared. Ask about the same passage.';
    input.focus({ preventScroll: true });
  });
  closeButton.addEventListener('click', closePanel);
  sourceLink.addEventListener('click', event => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    if (typeof onRevealSource !== 'function' || !activeSource) return;
    if (onRevealSource(activeSource)) {
      event.preventDefault();
      closePanel({restoreFocus: false});
    }
  });
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

  function openPanel(origin) {
    if (studyActive) return;
    panel.dataset.launchOrigin = origin;
    panel.hidden = false;
    launcher.hidden = true;
    launcher.setAttribute('aria-expanded', 'true');
    selectionButton.hidden = true;
    updateComposerState();
    requestAnimationFrame(() => {
      if (!panel.hidden && activeSource) {
        // Reading an explanation on touch should not open the software keyboard.
        const target = !input.disabled && !window.matchMedia('(pointer: coarse)').matches
          ? input : readerBody;
        target.focus({ preventScroll: true });
      }
      else if (!panel.hidden) closeButton.focus({ preventScroll: true });
    });
  }

  function closePanel({restoreFocus = true} = {}) {
    if (panel.hidden) return;
    stopRequest('Response stopped because the reader was closed.');
    panel.hidden = true;
    launcher.hidden = false;
    launcher.setAttribute('aria-expanded', 'false');
    selectionButton.hidden = true;
    if (restoreFocus) requestAnimationFrame(() => launcher.focus({ preventScroll: true }));
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
    if (picker?.active || studyBusy || studyActive) return;
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
    openPanel('selection');
    if (!history.length && !thread.childElementCount) void submitQuestion(INITIAL_QUESTION, { initial: true });
  }

  function stopRequest(message) {
    if (!activeController) return;
    const controller = activeController;
    activeController = null;
    controller.abort();
    if (activeAnswerNode?.textContent === 'Thinking…') activeAnswerNode.textContent = 'Response stopped.';
    activeAnswerNode = null;
    stopButton.hidden = true;
    sendButton.hidden = false;
    input.disabled = !activeSource;
    updateComposerState();
    status.textContent = message;
  }

  function scheduleSelectionCapture(event) {
    if (picker?.active || studyBusy || studyActive) return;
    if (event?.composedPath?.().includes(host)) return;
    if (selectionActionPressed) return;
    if (selectionIsInsideReader()) return;
    clearTimeout(selectionTimer);
    selectionTimer = window.setTimeout(readSelection, event?.type === 'selectionchange' ? 90 : 40);
  }

  function readSelection() {
    if (studyBusy || studyActive) return;
    try {
      const source = captureSelection();
      if (!source || !String(source.quote || '').trim()) {
        pendingSource = null;
        if (!panel.hidden) return;
        hideSelectionAction();
        return;
      }
      pendingSource = source;
      selectionButton.hidden = false;
      environment.positionSelectionAction(selectionButton, source.rect);
    } catch (error) {
      pendingSource = null;
      hideSelectionAction();
      openPanel('launcher');
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

  function setActiveSource(source) {
    if (!source) return;
    pendingSource = null;
    clearTimeout(selectionTimer);
    if (passageKey(source) !== passageKey(activeSource)) {
      stopRequest('Response stopped.');
      sessions.save(activeSource, { history, visibleTurns, lastFailedQuestion, failedUserTurn });
      clearConversation();
      const restored = sessions.activate(source);
      activeSource = restored.source;
      if (restored.conversation) {
        ({ history, visibleTurns, lastFailedQuestion, failedUserTurn } = restored.conversation);
      }
    } else {
      activeSource = source;
    }
    renderSource(activeSource);
    sessions.render(activeSource);
    updateComposerState();
  }

  function renderSource(source) {
    const url = safeSourceUrl(source.url);
    const heading = String(source.section || source.title || 'Textbook passage');
    sourceLink.textContent = heading;
    sourceLink.setAttribute('aria-label', `Return to passage: ${heading}`);
    sourceLink.title = 'Return to this passage in the book';
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
    sourceCard.querySelector('.reader-source-label').textContent = `Selected passage${volumeLabel}`;
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
    input.value = '';
    input.style.height = '';
    errorText.textContent = '';
  }

  function updateComposerState() {
    learningActions.hidden = !activeSource;
    explainButton.hidden = !activeSource || visibleTurns > 0;
    explainButton.disabled = !activeSource || Boolean(activeController);
    learningButtons.forEach(button => { button.disabled = !activeSource || Boolean(activeController); });
    // Stop remains available while the input and send controls are disabled.
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
    const followStream = !initial && isAtBottom(readerBody);
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
    const requestSource = activeSource;
    const controller = new AbortController();
    activeController = controller;
    activeAnswerNode = answerNode;
    updateComposerState();
    environment.updateViewport();

    try {
      const answer = await requestExplanation({
        source: requestSource,
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
      if (completeAnswer) addCopyButton(answerNode, completeAnswer, requestSource);
      history.push({ role: 'user', content: question }, { role: 'assistant', content: completeAnswer });
      history = boundHistory(history);
      trimVisibleTurns();
      lastFailedQuestion = null;
      failedUserTurn = null;
      status.textContent = 'Explanation ready.';
    } catch (error) {
      if (controller.signal.aborted || activeController !== controller) return;
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
        activeAnswerNode = null;
        stopButton.hidden = true;
        sendButton.hidden = false;
        updateComposerState();
        if (followStream) scrollThreadToEnd();
      }
    }
  }

  function addCopyButton(answerNode, markdown, source) {
    const title = String(source.section || source.title || 'Textbook passage');
    const url = safeSourceUrl(source.url);
    const quote = String(source.quote || '').split('\n').map(line => `> ${line}`).join('\n');
    const notes = `SocratiQ AI explanation — check against the source.\n\n${markdown}\n\nSource: ${title}${url ? ` — ${url}` : ''}\n\n${quote}`;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'reader-copy';
    button.textContent = 'Copy answer';
    button.setAttribute('aria-label', 'Copy answer as Markdown');
    button.addEventListener('click', async () => {
      button.disabled = true;
      try {
        if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
        await navigator.clipboard.writeText(notes);
        if (!button.isConnected) return;
        status.textContent = 'Answer copied as Markdown.';
      } catch {
        if (!button.isConnected) return;
        status.textContent = 'Could not copy the answer. Select the answer text and copy it manually.';
      } finally {
        button.disabled = false;
      }
    });
    answerNode.after(button);
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
    readerBody.scrollTop = readerBody.scrollHeight;
  }

  function isAtBottom(body) {
    return body.scrollHeight - body.scrollTop - body.clientHeight <= 24;
  }

  function scrollAnswerToStart(answerNode) {
    const bodyTop = readerBody.getBoundingClientRect().top;
    const answerTop = answerNode.getBoundingClientRect().top;
    const sourceHeight = !sourceCard.hidden && getComputedStyle(sourceCard).position === 'sticky'
      ? sourceCard.getBoundingClientRect().height : 0;
    readerBody.scrollTop = Math.max(0, readerBody.scrollTop + answerTop - bodyTop - sourceHeight);
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

  return {
    host,
    destroy() {
      destroyed = true;
      window.removeEventListener('socratiq:open', suspendForStudy);
      window.removeEventListener('socratiq:close', resumeFromStudy);
      picker?.destroy();
      stopRequest('Response stopped.');
      clearTimeout(selectionTimer);
      document.removeEventListener('selectionchange', scheduleSelectionCapture);
      document.removeEventListener('pointerup', scheduleSelectionCapture, true);
      document.removeEventListener('mouseup', scheduleSelectionCapture, true);
      document.removeEventListener('scroll', hideSelectionAction, true);
      document.removeEventListener('keydown', handleEscape, true);
      document.removeEventListener('keydown', handleSelectionShortcut, true);
      environment.destroy();
      host.remove();
    },
  };
}
