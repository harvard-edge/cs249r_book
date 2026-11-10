import { completeReaderStyles } from './styles.js';
import { renderSafeMarkdown } from './render.js';
import { createParagraphPicker } from './picker.js';
import { createPassageSessions, passageKey } from './passage-sessions.js';
import { createReaderEnvironment } from './environment.js';
import {
  readLearnerPreferences,
  saveLearnerPreferences,
  clearLearnerPreferences,
  SELECTION_CLIP_OPTIONS,
  DEFAULT_CLIP_ACTIONS,
  readClipActions,
  saveClipActions,
} from './preferences.js';
import {createReaderTour} from './onboarding.js';
import {
  BYOK_PROVIDERS,
  readActiveByokConfig,
  saveByokConfig,
  clearByokConfig,
  startOpenRouterPKCE,
  completeOpenRouterPKCE,
} from '../src_shared/byok-provider.js';

const MAX_HISTORY_MESSAGES = 12;
const MAX_VISIBLE_TURNS = 12;
const INITIAL_QUESTION = 'Explain this passage.';
const LEARNING_QUESTIONS = {
  simplify: 'Explain this passage in simpler language, defining any essential technical terms.',
  example: 'Show a concrete example that helps me understand this passage.',
  check: 'Check my understanding of this passage. Ask me one conceptual question at a time. Do not reveal the answer until I respond; then give feedback before asking another question.',
  taxonomy: "Locate this passage in the ML Systems Taxonomy (Foundations F, Stack S, Lifecycle P, Qualities Q, System Types T): name the relevant taxonomy subject area(s), connect it to this volume's analytical lens, and explain what upstream or downstream system layers it constrains.",
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
  if (!document.getElementById('socratiq-host-layout')) {
    const hostLayoutStyle = document.createElement('style');
    hostLayoutStyle.id = 'socratiq-host-layout';
    hostLayoutStyle.textContent = `
      #tbl-socratiq-controls table.table, #tbl-socratiq-controls table { display: table !important; width: 100% !important; table-layout: fixed !important; }
      #tbl-socratiq-controls th, #tbl-socratiq-controls td { white-space: normal !important; overflow-wrap: break-word !important; word-break: normal !important; }
      @media (max-width: 800px) {
        #quarto-back-to-top { bottom: 4.5rem !important; }
        body { padding-bottom: 4.25rem; }
      }
    `;
    document.head.appendChild(hostLayoutStyle);
  }
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
    <div class="selection-clip" role="toolbar" aria-label="Selection actions" hidden>
      <div class="selection-clip-bar">
        <button class="selection-action" type="button" data-clip-action="explain">
          <span class="reader-mark" aria-hidden="true">S</span>
          <span>Explain this</span>
        </button>
        <button class="selection-clip-btn" type="button" data-clip-action="ask">Ask…</button>
        <button class="selection-clip-btn" type="button" data-clip-action="check">Check me</button>
        <button class="selection-clip-btn" type="button" data-clip-action="taxonomy">Taxonomy</button>
        <button class="selection-clip-btn" type="button" data-clip-action="simplify" hidden>Simplify</button>
        <button class="selection-clip-btn" type="button" data-clip-action="example" hidden>Example</button>
        <button class="selection-clip-btn" type="button" data-clip-action="study" hidden>Study tools</button>
        <button class="selection-clip-config-btn" type="button" aria-label="Configure selection actions" aria-expanded="false" title="Configure selection actions">
          <span aria-hidden="true">⚙</span>
        </button>
      </div>
      <div class="selection-clip-menu" role="group" aria-label="Selection clip actions" hidden>
        <p class="selection-clip-menu-title">Selection bar actions</p>
        <div class="selection-clip-options"></div>
      </div>
    </div>
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
      <p class="reader-privacy">The selected passage and surrounding section are sent with your learning preferences for AI explanation when you ask.</p>
      <div class="reader-body" role="region" aria-label="Selected passage and answers" tabindex="0">
      <details class="reader-preferences">
        <summary>Learning preferences</summary>
        <div class="reader-preferences-fields">
          <label>Explanation level<select class="reader-level"><option value="beginner">Beginner — define the essentials</option><option value="intermediate">Intermediate — connect concepts</option><option value="advanced">Advanced — examine trade-offs</option></select></label>
          <label>What do you already know? <span>(optional)</span><textarea class="reader-background" rows="2" maxlength="1200" placeholder="For example: I know Python; hardware is new to me."></textarea></label>
          <label>How should explanations help you? <span>(optional)</span><textarea class="reader-teaching" rows="2" maxlength="800" placeholder="For example: Define symbols and use everyday analogies."></textarea></label>
          <fieldset class="reader-clip-prefs">
            <legend>Selection bar actions</legend>
            <div class="reader-clip-prefs-options"></div>
          </fieldset>
          <fieldset class="reader-byok-box">
            <legend>Connect your own AI (optional)</legend>
            <button type="button" class="reader-byok-sso">Sign in with OpenRouter (Gemini / Claude / GPT)</button>
            <label>Provider<select class="reader-byok-provider"><option value="google-gemini">Google Gemini (gemini-2.5-flash)</option><option value="anthropic">Anthropic Claude (claude-sonnet-4-20250514)</option><option value="openrouter">OpenRouter (Gemini / Claude / GPT)</option><option value="openai">OpenAI (gpt-4.1-mini)</option><option value="groq">Groq (openai/gpt-oss-120b)</option></select></label>
            <label>Model<input type="text" class="reader-byok-model" value="gemini-2.5-flash" maxlength="160"></label>
            <label>API key <a class="reader-byok-portal" href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener noreferrer">Get free Gemini key ↗</a><input type="password" class="reader-byok-key" autocomplete="off" spellcheck="false" maxlength="512" placeholder="Paste API key (session-only by default)"></label>
            <label class="selection-clip-option"><input type="checkbox" class="reader-byok-remember"><span>Remember key in this browser</span></label>
            <div class="reader-preference-actions"><button type="button" class="reader-byok-connect">Connect model</button><button type="button" class="reader-byok-disconnect">Use free shared model</button></div>
            <p class="reader-byok-status" role="status" aria-live="polite"></p>
          </fieldset>
          <p>Saved in this browser. Included when you ask AI; avoid sensitive details. Changes apply to your next question.</p>
          <div class="reader-preference-actions"><button type="button" class="reader-preferences-save">Save preferences</button><button type="button" class="reader-preferences-clear">Clear preferences</button></div>
          <p class="reader-preferences-status" role="status" aria-live="polite"></p>
        </div>
      </details>
      <div class="reader-tools" hidden>
        <button class="reader-study" type="button">Study tools</button>
        <button class="reader-tour-replay" type="button">Replay tour</button>
        <span class="reader-study-status" role="status" aria-live="polite"></span>
      </div>
        <section class="reader-source" aria-label="Selected source" hidden>
          <p class="reader-source-label">Selected passage</p>
          <a class="reader-source-link" target="_blank" rel="noopener noreferrer"></a>
          <p class="reader-quote"></p>
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
          <button type="button" data-learning="taxonomy">Map to taxonomy</button>
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
        <textarea rows="1" aria-label="Ask a follow-up question" aria-describedby="socratiq-reader-footnote" placeholder="Select a passage to begin" disabled></textarea>
        <button class="reader-send" type="submit" aria-label="Send question" title="Send" disabled>
          <span aria-hidden="true">↑</span>
        </button>
        <button class="reader-stop" type="button" aria-label="Stop explanation" hidden>Stop</button>
      </form>
      <p id="socratiq-reader-footnote" class="reader-footnote">SocratiQ uses AI and can make mistakes. Check claims in the book.</p>
      <p class="reader-model-bar"><span class="reader-model-status">Free shared model</span> · <button type="button" class="reader-footnote-link">Connect your own Gemini or Claude</button></p>
    </section>
  `;
  root.appendChild(shell);

  const sessionPicker = shell.querySelector('.reader-session-select');
  const studyButton = shell.querySelector('.reader-study');
  const studyStatus = shell.querySelector('.reader-study-status');
  shell.querySelector('.reader-tools').hidden = false;
  studyButton.hidden = typeof onOpenStudy !== 'function';
  const sessionControl = shell.querySelector('.reader-sessions');
  const launcher = shell.querySelector('.reader-launcher');
  const panel = shell.querySelector('.reader-panel');
  const closeButton = shell.querySelector('.reader-close');
  const selectionClip = shell.querySelector('.selection-clip');
  const selectionButton = shell.querySelector('.selection-action');
  const clipButtons = [...shell.querySelectorAll('[data-clip-action]')];
  const clipConfigButton = shell.querySelector('.selection-clip-config-btn');
  const clipMenu = shell.querySelector('.selection-clip-menu');
  const clipOptionsBox = shell.querySelector('.selection-clip-options');
  const prefsClipOptionsBox = shell.querySelector('.reader-clip-prefs-options');
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
  const input = shell.querySelector('.reader-composer textarea');
  input.maxLength = 2000;
  const sendButton = shell.querySelector('.reader-send');
  const stopButton = shell.querySelector('.reader-stop');
  const readerBody = shell.querySelector('.reader-body');
  const sessions = createPassageSessions({
    thread, body: readerBody, input, status, errorText, errorBox, retryButton,
    select: sessionPicker, control: sessionControl,
  });
  const environment = createReaderEnvironment({ host, shell });
  const preferencesControl = shell.querySelector('.reader-preferences');
  const levelInput = shell.querySelector('.reader-level');
  const backgroundInput = shell.querySelector('.reader-background');
  const teachingInput = shell.querySelector('.reader-teaching');
  const preferenceStatus = shell.querySelector('.reader-preferences-status');
  const modelStatusNode = shell.querySelector('.reader-model-status');
  const footnoteLink = shell.querySelector('.reader-footnote-link');
  const byokBox = shell.querySelector('.reader-byok-box');
  const byokSsoBtn = shell.querySelector('.reader-byok-sso');
  const byokProviderSelect = shell.querySelector('.reader-byok-provider');
  const byokModelInput = shell.querySelector('.reader-byok-model');
  const byokPortalLink = shell.querySelector('.reader-byok-portal');
  const byokKeyInput = shell.querySelector('.reader-byok-key');
  const byokRememberInput = shell.querySelector('.reader-byok-remember');
  const byokConnectBtn = shell.querySelector('.reader-byok-connect');
  const byokDisconnectBtn = shell.querySelector('.reader-byok-disconnect');
  const byokStatusNode = shell.querySelector('.reader-byok-status');

  function updateByokPortal(providerId) {
    const spec = BYOK_PROVIDERS[providerId] || BYOK_PROVIDERS['google-gemini'];
    if (byokPortalLink) {
      if (spec.portalUrl) {
        byokPortalLink.href = spec.portalUrl;
        byokPortalLink.textContent = spec.portalLabel || 'Get API key ↗';
        byokPortalLink.hidden = false;
      } else {
        byokPortalLink.hidden = true;
      }
    }
  }

  function refreshByokUI() {
    const cfg = readActiveByokConfig();
    const spec = BYOK_PROVIDERS[cfg.provider] || BYOK_PROVIDERS['google-gemini'];
    if (byokProviderSelect) byokProviderSelect.value = cfg.provider;
    if (byokModelInput) byokModelInput.value = cfg.model || spec.defaultModel;
    if (byokKeyInput) byokKeyInput.value = cfg.apiKey || '';
    if (byokRememberInput) byokRememberInput.checked = Boolean(cfg.saveLocally);
    updateByokPortal(cfg.provider);
    if (cfg.enabled) {
      if (modelStatusNode) modelStatusNode.textContent = `Connected: ${spec.shortLabel} (${cfg.model})`;
      if (footnoteLink) footnoteLink.textContent = 'Manage model';
    } else {
      if (modelStatusNode) modelStatusNode.textContent = 'Free shared model';
      if (footnoteLink) footnoteLink.textContent = 'Connect your own Gemini or Claude';
    }
  }
  refreshByokUI();

  footnoteLink?.addEventListener('click', () => {
    preferencesControl.open = true;
    refreshPreferences();
    refreshByokUI();
    requestAnimationFrame(() => {
      byokBox?.scrollIntoView({ block: 'nearest' });
      byokKeyInput?.focus({ preventScroll: true });
    });
  });

  byokProviderSelect?.addEventListener('change', () => {
    const spec = BYOK_PROVIDERS[byokProviderSelect.value] || BYOK_PROVIDERS['google-gemini'];
    byokModelInput.value = spec.defaultModel;
    updateByokPortal(spec.id);
  });

  byokSsoBtn?.addEventListener('click', () => {
    void startOpenRouterPKCE().catch((err) => {
      if (byokStatusNode) byokStatusNode.textContent = err?.message || 'Could not start OpenRouter sign-in.';
    });
  });

  byokConnectBtn?.addEventListener('click', () => {
    const provider = byokProviderSelect.value;
    const spec = BYOK_PROVIDERS[provider] || BYOK_PROVIDERS['google-gemini'];
    const apiKey = byokKeyInput.value.trim();
    if (!apiKey) {
      byokStatusNode.textContent = 'Paste an API key or use Sign in with OpenRouter.';
      return;
    }
    const saved = saveByokConfig({
      enabled: true,
      provider,
      model: byokModelInput.value.trim() || spec.defaultModel,
      endpoint: spec.endpoint,
      apiKey,
      saveLocally: byokRememberInput.checked,
      authMethod: 'manual',
    });
    refreshByokUI();
    byokStatusNode.textContent = saved.saveLocally
      ? `Connected to ${spec.shortLabel} (${saved.model}). Key remembered in this browser.`
      : `Connected to ${spec.shortLabel} (${saved.model}) for this browser tab only.`;
  });

  byokDisconnectBtn?.addEventListener('click', () => {
    clearByokConfig();
    refreshByokUI();
    byokStatusNode.textContent = 'Switched back to the free shared model. Personal key cleared.';
  });

  const handleByokChanged = () => refreshByokUI();
  window.addEventListener('socratiq:byok-changed', handleByokChanged);

  if (typeof window !== 'undefined' && window.location?.search?.includes('code=')) {
    void completeOpenRouterPKCE().then((res) => {
      if (res?.enabled) {
        refreshByokUI();
        if (byokStatusNode) byokStatusNode.textContent = 'Connected via OpenRouter SSO for this session.';
      }
    }).catch(() => {});
  }
  function renderClipOptionCheckboxes(container) {
    if (!container) return;
    container.replaceChildren();
    for (const option of SELECTION_CLIP_OPTIONS) {
      if (option.id === 'study' && typeof onOpenStudy !== 'function') continue;
      const label = document.createElement('label');
      label.className = 'selection-clip-option';
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.value = option.id;
      checkbox.dataset.clipOption = option.id;
      checkbox.addEventListener('change', () => {
        const checked = [...container.querySelectorAll('input[data-clip-option]:checked')].map(el => el.value);
        saveClipActions(checked);
        syncClipButtons();
        if (!selectionClip.hidden && pendingSource?.rect) {
          environment.positionSelectionAction(selectionClip, pendingSource.rect);
        }
      });
      const text = document.createElement('span');
      text.textContent = option.shortLabel;
      label.append(checkbox, text);
      container.appendChild(label);
    }
  }
  renderClipOptionCheckboxes(clipOptionsBox);
  renderClipOptionCheckboxes(prefsClipOptionsBox);

  function syncClipButtons() {
    const enabled = new Set(readClipActions());
    for (const button of clipButtons) {
      const action = button.dataset.clipAction;
      if (action === 'study' && typeof onOpenStudy !== 'function') {
        button.hidden = true;
        continue;
      }
      button.hidden = !enabled.has(action);
    }
    for (const checkbox of shell.querySelectorAll('input[data-clip-option]')) {
      checkbox.checked = enabled.has(checkbox.dataset.clipOption);
    }
  }
  syncClipButtons();

  function closeClipMenu() {
    if (!clipMenu) return;
    clipMenu.hidden = true;
    clipConfigButton?.setAttribute('aria-expanded', 'false');
  }

  function refreshPreferences() {
    const preferences = readLearnerPreferences();
    levelInput.value = preferences.level;
    backgroundInput.value = preferences.background;
    teachingInput.value = preferences.instructions;
    syncClipButtons();
  }
  refreshPreferences();
  preferencesControl.querySelector('summary').addEventListener('click', () => {
    if (!preferencesControl.open) {refreshPreferences(); preferenceStatus.textContent = '';}
  });
  shell.querySelector('.reader-preferences-save').addEventListener('click', () => {
    const {persisted} = saveLearnerPreferences({level: levelInput.value, background: backgroundInput.value, instructions: teachingInput.value});
    refreshPreferences();
    preferenceStatus.textContent = persisted ? 'Preferences saved in this browser.' : 'Preferences apply on this page. Browser storage is unavailable.';
  });
  shell.querySelector('.reader-preferences-clear').addEventListener('click', () => {
    const {persisted} = clearLearnerPreferences();
    saveClipActions([...DEFAULT_CLIP_ACTIONS]);
    refreshPreferences();
    preferenceStatus.textContent = persisted ? 'Preferences cleared. Explanation level reset to Intermediate.' : 'Preferences cleared for this page. Browser storage is unavailable.';
  });
  const tour = createReaderTour({shell, targets: {
    preferences: () => preferencesControl.querySelector('summary'),
    choose: () => [...shell.querySelectorAll('.reader-choose')].find(button => button.getClientRects().length) || sourceCard,
    study: () => studyButton.hidden ? shell.querySelector('.reader-tour-replay') : studyButton,
  }});
  shell.querySelector('.reader-tour-replay').addEventListener('click', () => tour.start({replay: true}));

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

  function publishActiveSource(source = activeSource) {
    try {
      window.__SOCRATIQ_ACTIVE_SOURCE__ = source ? {
        quote: source.quote,
        section: source.section,
        title: source.title,
        volume: source.volume,
        url: source.url,
      } : null;
    } catch {
      // Ignore environments where window properties are read-only.
    }
  }
  publishActiveSource(null);

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
    publishActiveSource(activeSource);
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
    publishActiveSource(activeSource);
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
    refreshPreferences();
    shell.hidden = false;
    studyStatus.textContent = '';
    studyButton.textContent = 'Study tools';
    openPanel('study');
    requestAnimationFrame(() => studyButton.focus({preventScroll: true}));
  }
  window.addEventListener('socratiq:open', suspendForStudy);
  window.addEventListener('socratiq:study-open', suspendForStudy);
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

  selectionClip.addEventListener('pointerdown', (event) => {
    selectionActionPressed = true;
    if (event.pointerType !== 'touch' && !event.target.closest('input, label')) {
      event.preventDefault();
    }
  });
  selectionClip.addEventListener('pointercancel', () => {
    selectionActionPressed = false;
    pendingSource = null;
    hideSelectionAction();
  });
  clipConfigButton.addEventListener('click', () => {
    selectionActionPressed = false;
    const nextOpen = clipMenu.hidden;
    clipMenu.hidden = !nextOpen;
    clipConfigButton.setAttribute('aria-expanded', String(nextOpen));
    if (pendingSource?.rect) {
      environment.positionSelectionAction(selectionClip, pendingSource.rect);
    }
  });
  clipButtons.forEach((button) => {
    button.addEventListener('click', () => {
      selectionActionPressed = false;
      const source = pendingSource || activeSource;
      if (!source) return;
      const action = button.dataset.clipAction;
      hideSelectionAction();
      setActiveSource(source);
      if (action === 'study') {
        publishActiveSource(source);
        studyButton.click();
        return;
      }
      if (action === 'ask') {
        openPanel('selection', { focusInput: true });
        if (!visibleTurns) status.textContent = 'Passage ready. Type your question below.';
        return;
      }
      openPanel('selection');
      if (action === 'explain') {
        if (!history.length && !thread.childElementCount) {
          void submitQuestion(INITIAL_QUESTION, { initial: true });
        }
      } else if (LEARNING_QUESTIONS[action]) {
        void submitQuestion(LEARNING_QUESTIONS[action], { initial: !history.length && !thread.childElementCount });
      }
    });
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

  function openPanel(origin, { focusInput = false } = {}) {
    if (studyActive) return;
    panel.dataset.launchOrigin = origin;
    panel.hidden = false;
    launcher.hidden = true;
    launcher.setAttribute('aria-expanded', 'true');
    hideSelectionAction();
    updateComposerState();
    requestAnimationFrame(() => {
      if (!panel.hidden && activeSource) {
        // Reading an explanation on touch should not open the software keyboard unless Ask… was explicitly chosen.
        const target = !input.disabled && (focusInput || !window.matchMedia('(pointer: coarse)').matches)
          ? input : readerBody;
        target.focus({ preventScroll: true });
      }
      else if (!panel.hidden) closeButton.focus({ preventScroll: true });
      if (!panel.hidden) tour.start();
    });
  }

  function closePanel({restoreFocus = true} = {}) {
    if (panel.hidden) return;
    tour.stop({restoreFocus: false});
    stopRequest('Response stopped because the reader was closed.');
    panel.hidden = true;
    launcher.hidden = false;
    launcher.setAttribute('aria-expanded', 'false');
    hideSelectionAction();
    if (restoreFocus) requestAnimationFrame(() => launcher.focus({ preventScroll: true }));
  }

  function handleEscape(event) {
    if (event.key !== 'Escape') return;
    if (!panel.hidden) {
      event.preventDefault();
      event.stopPropagation();
      closePanel();
    } else if (!selectionClip.hidden) {
      event.preventDefault();
      if (!clipMenu.hidden) closeClipMenu();
      else hideSelectionAction();
    }
  }

  function handleSelectionShortcut(event) {
    if (picker?.active || studyBusy || studyActive) return;
    if (event.key.toLowerCase() !== 'e' || !event.shiftKey || event.altKey || (event.ctrlKey === event.metaKey)) return;
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
    hideSelectionAction();
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
      syncClipButtons();
      closeClipMenu();
      selectionClip.hidden = false;
      environment.positionSelectionAction(selectionClip, source.rect);
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
    closeClipMenu();
    selectionClip.hidden = true;
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
    publishActiveSource(activeSource);
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
    let contextReduced = false;
    activeController = controller;
    activeAnswerNode = answerNode;
    updateComposerState();
    environment.updateViewport();

    try {
      const answer = await requestExplanation({
        source: requestSource,
        question,
        history: boundHistory(history),
        learnerPreferences: readLearnerPreferences(),
        onContextBudget: () => {
          contextReduced = true;
          if (activeController === controller) status.textContent = 'Generating an explanation… Some surrounding text or older turns were omitted to fit this request.';
        },
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
      status.textContent = contextReduced ? 'Explanation ready. Some surrounding text or older turns were omitted to fit this request.' : 'Explanation ready.';
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
      publishActiveSource(null);
      window.removeEventListener('socratiq:open', suspendForStudy);
      window.removeEventListener('socratiq:study-open', suspendForStudy);
      window.removeEventListener('socratiq:close', resumeFromStudy);
      window.removeEventListener('socratiq:byok-changed', handleByokChanged);
      document.getElementById('socratiq-host-layout')?.remove();
      picker?.destroy();
      stopRequest('Response stopped.');
      clearTimeout(selectionTimer);
      document.removeEventListener('selectionchange', scheduleSelectionCapture);
      document.removeEventListener('pointerup', scheduleSelectionCapture, true);
      document.removeEventListener('mouseup', scheduleSelectionCapture, true);
      document.removeEventListener('scroll', hideSelectionAction, true);
      document.removeEventListener('keydown', handleEscape, true);
      document.removeEventListener('keydown', handleSelectionShortcut, true);
      tour.destroy();
      environment.destroy();
      host.remove();
    },
  };
}
