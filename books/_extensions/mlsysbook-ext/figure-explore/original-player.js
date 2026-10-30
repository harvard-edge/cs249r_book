/* Shared, endpoint-driven playback for enhancements of original textbook artwork.
 * render(index, {stage, details, index, step}) commits a deterministic endpoint.
 * transition({from, to, duration, signal, stage, details, animate, onFrame}) sets
 * up movement on the source DOM. onFrame registers a progress callback; animate
 * registers WAAPI interpolation driven by this player's pausable clock. Do not
 * await animation.finished: the player owns its currentTime and endpoint commit.
 * Returned transition promises may finish setup, but cannot advance lesson state.
 */
(function () {
  'use strict';
  let serial = 0;
  const abortError = () => new DOMException('Playback interrupted', 'AbortError');
  // Compact-layout icons: 24-unit stroke glyphs in currentColor. Back/Next are single
  // chevrons (not skip-to-start/end bars); Expand uses outward corner brackets, not an
  // arrow that reads as an external link; Options is a pair of sliders.
  const ICONS = {
    play: '<path d="M8 5.5v13l10.5-6.5z" fill="currentColor" stroke="none"/>',
    pause: '<path d="M7.5 5.5h3v13h-3zM13.5 5.5h3v13h-3z" fill="currentColor" stroke="none"/>',
    replay: '<path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3"/><path d="M4.5 4.5v4h4"/><path d="M10.5 9.2v5.6l4.4-2.8z" fill="currentColor" stroke="none"/>',
    back: '<path d="M14.5 5.5L8 12l6.5 6.5"/>',
    next: '<path d="M9.5 5.5L16 12l-6.5 6.5"/>',
    restart: '<path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3"/><path d="M4.5 4.5v4h4"/>',
    expand: '<path d="M4 9.5V4h5.5M14.5 4H20v5.5M20 14.5V20h-5.5M9.5 20H4v-5.5"/>',
    close: '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
    options: '<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>',
    down: '<path d="M12 4.5v15M6.5 14L12 19.5 17.5 14"/>'
  };
  function icon(name) {
    const glyph = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    glyph.setAttribute('viewBox', '0 0 24 24');
    glyph.setAttribute('class', 'mlsys-original-icon');
    glyph.setAttribute('aria-hidden', 'true');
    glyph.setAttribute('focusable', 'false');
    glyph.setAttribute('fill', 'none');
    glyph.setAttribute('stroke', 'currentColor');
    glyph.setAttribute('stroke-width', '2');
    glyph.setAttribute('stroke-linecap', 'round');
    glyph.setAttribute('stroke-linejoin', 'round');
    glyph.innerHTML = ICONS[name];
    return glyph;
  }
  // Compact layout: the menu counter ("9 / 16") is the only step number. A label that
  // begins with a diagram stage number ("4. Backprop") reads "Backprop (stage 4)".
  // Modules should author labels that way; check_original_player.py flags any that do not.
  const NUMBERED = /^\s*(\d+)[.)]\s+(.+)$/;
  const stageLabel = label => { const m = NUMBERED.exec(String(label)); return m ? m[2] + ' (stage ' + m[1] + ')' : String(label); };
  // Pinned dock, cue, and jump pill never cover more than this share of the visible diagram.
  const COVER_CAP = 0.15;
  // Protected regions inside the stage: the step's highlighted region (data-mlsys-focus) and
  // step-critical readouts, legends, and labels (data-mlsys-keep). The dock and pill avoid both.
  const PROTECTED = '[data-mlsys-focus], [data-mlsys-keep]';
  // Reading dwell scales with narration length around this reference, within [0.6, 2] x base.
  const REF_WORDS = 30;

  // One header watcher serves every mounted player. Read the transformed header
  // bounds during its scroll transition; never reposition or scroll the page.
  const stickyPlayers = new Set();
  let headerFrame = 0, headerUntil = 0, watchedHeader = null;
  let headerResize = null, headerMutation = null, lastHeaderOffset = null;
  function updateHeaderOffset(now) {
    headerFrame = 0;
    const header = document.querySelector('#quarto-header');
    if (header !== watchedHeader) {
      headerResize?.disconnect(); headerMutation?.disconnect();
      watchedHeader = header;
      if (header) {
        headerResize?.observe(header);
        headerMutation?.observe(header, {attributes:true, subtree:true, childList:true, attributeFilter:['class','style','hidden']});
      }
    }
    const offset = Math.max(0, Math.ceil(header?.getBoundingClientRect().bottom || 0));
    if (offset !== lastHeaderOffset) {
      document.documentElement.style.setProperty('--mlsys-original-header-offset', offset + 'px');
      lastHeaderOffset = offset;
    }
    if (stickyPlayers.size && now < headerUntil) headerFrame = requestAnimationFrame(updateHeaderOffset);
  }
  function scheduleHeaderOffset() {
    if (!stickyPlayers.size) return;
    headerUntil = performance.now() + 650;
    if (!headerFrame) headerFrame = requestAnimationFrame(updateHeaderOffset);
  }
  function watchHeader(root) {
    stickyPlayers.add(root);
    if (stickyPlayers.size === 1) {
      headerResize = typeof ResizeObserver === 'function' ? new ResizeObserver(scheduleHeaderOffset) : null;
      headerMutation = new MutationObserver(scheduleHeaderOffset);
      window.addEventListener('scroll', scheduleHeaderOffset, {passive:true});
      window.addEventListener('resize', scheduleHeaderOffset, {passive:true});
      document.addEventListener('visibilitychange', scheduleHeaderOffset);
    }
    scheduleHeaderOffset();
    return () => {
      stickyPlayers.delete(root);
      if (stickyPlayers.size) return;
      cancelAnimationFrame(headerFrame); headerFrame = 0;
      headerResize?.disconnect(); headerMutation?.disconnect();
      headerResize = null; headerMutation = null; watchedHeader = null; lastHeaderOffset = null;
      window.removeEventListener('scroll', scheduleHeaderOffset);
      window.removeEventListener('resize', scheduleHeaderOffset);
      document.removeEventListener('visibilitychange', scheduleHeaderOffset);
      document.documentElement.style.removeProperty('--mlsys-original-header-offset');
    };
  }

  function create(options) {
    const { root, outer, title, scope, steps, render, transition } = options;
    if (!root || !outer || !Array.isArray(steps) || !steps.length || typeof render !== 'function') {
      throw new Error('Original figure requires root, outer, nonempty steps, and render.');
    }
    // Opt-in compact layout: title, intro, Assumptions, optional view controls, diagram,
    // a sticky dock (control bar + one-line step cue), narration, details. Every figure
    // that omits `layout` keeps the default DOM below unchanged.
    if (options.layout != null && options.layout !== 'compact') throw new Error('Unknown original figure layout: ' + options.layout);
    const compact = options.layout === 'compact';
    const intro = compact && typeof options.intro === 'string' ? options.intro.trim() : '';
    if (compact && !intro) throw new Error('Compact original figure requires a nonempty intro.');
    function timing(name) {
      const raw = outer.dataset[name];
      const value = Number(raw);
      if (raw == null || raw.trim() === '' || !Number.isFinite(value) || value < 0) {
        throw new Error('Original figure requires shared build-time timing: ' + name);
      }
      return value;
    }
    const movementMs = timing('mlsysMovementMs');
    const readingMs = timing('mlsysReadingPauseMs');
    const speeds = [0.5, 1, 1.5, 2];
    let speed = 1;
    const id = 'mlsys-original-' + (++serial);
    root.classList.add('mlsys-original-player');
    root.tabIndex = 0;
    root.setAttribute('role', 'region');
    root.setAttribute('aria-labelledby', id + '-title');
    const make = (tag, className, text) => {
      const el = document.createElement(tag);
      if (className) el.className = className;
      if (text != null) el.textContent = text;
      return el;
    };
    const heading = make('h4', 'mlsys-original-title', title);
    heading.id = id + '-title';
    const assumptions = make('details', 'mlsys-original-assumptions');
    assumptions.append(make('summary', 'mlsys-original-assumptions-summary', compact ? 'Assumptions' : 'Example and assumptions'),
      make('p', 'mlsys-original-scope', scope));
    const controls = make('div', 'mlsys-original-controls');
    controls.setAttribute('role', 'group');
    controls.setAttribute('aria-label', 'Animation controls');
    const button = (label, hint) => {
      const el = make('button', '', label);
      el.type = 'button';
      el.title = hint;
      controls.append(el);
      return el;
    };
    const playButton = button('Play', 'Play or pause (K)');
    const backButton = button('Back', 'Previous step (P)');
    const nextButton = button('Next', 'Next step (N)');
    const restartButton = button('Restart', 'Restart (R)');
    const expandButton = button('Expand', 'Inspect the diagram in an expanded view');
    let speedSelect = null, cueStep = null, cueState = null;
    // Compact-only parts; all remain null in the default layout.
    let introText = null, keysDesc = null, dock = null, cue = null, cueText = null, viewSlot = null;
    let jumpAnchor = null, jump = null, jumpCount = null, jumpText = null, statusLive = null, announcer = null, rateText = null;
    let statusChip = null, faceCount = null, faceStatus = null, keysButton = null, popover = null;
    let speedInputs = [], playLabel = null, playIcon = null, playTip = null, popoverRestart = null, narrationFocusable = false;
    if (!compact) {
      for (const [control, label, symbol] of [[restartButton,'Restart','↺'],[expandButton,'Expand','↗']]) {
        control.setAttribute('aria-label', label);
        const full = make('span', 'mlsys-original-control-full', label);
        const short = make('span', 'mlsys-original-control-short', symbol);
        short.setAttribute('aria-hidden', 'true');
        control.replaceChildren(full, short);
      }
      const speedLabel = make('label', 'mlsys-original-speed', 'Speed ');
      speedSelect = make('select');
      speedSelect.id = id + '-speed';
      speedSelect.setAttribute('aria-label', 'Playback speed');
      speedSelect.title = 'Playback speed (− / +); affects movement and reading pauses';
      speedLabel.htmlFor = speedSelect.id;
      speeds.forEach(rate => {
        const option = make('option', '', rate + '×');
        option.value = String(rate); speedSelect.append(option);
      });
      speedSelect.value = String(speed);
      speedLabel.append(speedSelect); controls.append(speedLabel);
      const currentCue = make('div', 'mlsys-original-current-cue');
      currentCue.setAttribute('aria-hidden', 'true');
      cueStep = make('span', 'mlsys-original-cue-step');
      cueState = make('span', 'mlsys-original-cue-state');
      currentCue.append(cueStep, cueState); controls.append(currentCue);
    } else {
      root.classList.add('mlsys-original-compact');
      root.dataset.layout = 'compact';
      introText = make('p', 'mlsys-original-intro', intro);
      // Shortcut description for the group: referenced, never rendered or traversed twice.
      keysDesc = make('p', 'mlsys-original-keys-desc', 'Shortcuts: N next, P back, K play or pause, R restart, minus slower, plus faster.');
      keysDesc.id = id + '-keys';
      keysDesc.hidden = true;
      controls.setAttribute('aria-describedby', keysDesc.id);
      // Name from visible text (hidden on phones); icons are decorative. The tooltip is an
      // aria-hidden child, so it never joins the accessible name (a ::after would).
      const decorate = (control, label, glyph, tip) => {
        control.removeAttribute('title');
        control.classList.add('mlsys-original-btn', 'mlsys-original-btn-' + glyph);
        const text = make('span', 'mlsys-original-control-label', label);
        const mark = icon(glyph);
        const bubble = make('span', 'mlsys-original-tip', tip);
        bubble.setAttribute('aria-hidden', 'true');
        control.replaceChildren(mark, text, bubble);
        return [text, mark, bubble];
      };
      [playLabel, playIcon, playTip] = decorate(playButton, 'Play', 'play', 'Play (K)');
      decorate(backButton, 'Back', 'back', 'Back one step (P)');
      decorate(nextButton, 'Next', 'next', 'Next step (N)');
      decorate(restartButton, 'Restart', 'restart', 'Restart (R)');
      decorate(expandButton, 'Expand', 'expand', 'Expand to the full window');
      statusChip = make('span', 'mlsys-original-status');
      statusChip.setAttribute('aria-hidden', 'true');
      playButton.after(statusChip);
      // Speed lives behind a button whose name says so and whose face shows the current rate.
      keysButton = make('button', 'mlsys-original-options-button');
      keysButton.type = 'button';
      decorate(keysButton, 'Speed', 'options', 'Speed, restart, and shortcuts');
      rateText = make('span', 'mlsys-original-rate', '1×');
      keysButton.querySelector('.mlsys-original-tip').before(rateText);
      keysButton.setAttribute('aria-expanded', 'false');
      popover = make('div', 'mlsys-original-popover');
      popover.id = id + '-options';
      popover.hidden = true;
      popover.setAttribute('role', 'group');
      popover.setAttribute('aria-label', 'Speed and options');
      keysButton.setAttribute('aria-controls', popover.id);
      const speedSet = make('fieldset', 'mlsys-original-speed-set');
      speedSet.setAttribute('role', 'radiogroup');
      speedSet.setAttribute('aria-label', 'Playback speed');
      speedSet.append(make('legend', '', 'Speed'));
      speeds.forEach(rate => {
        const choice = make('label', 'mlsys-original-speed-choice');
        const input = make('input');
        input.type = 'radio';
        input.name = id + '-speed';
        input.value = String(rate);
        input.checked = rate === speed;
        choice.append(input, document.createTextNode(rate + '×'));
        speedSet.append(choice);
        speedInputs.push(input);
      });
      popoverRestart = make('button', 'mlsys-original-popover-restart');
      popoverRestart.type = 'button';
      popoverRestart.append(icon('restart'), make('span', '', 'Restart'));
      // Key legend: keyboard users only. Touch-only devices hide it (CSS, pointer: coarse).
      const keySection = make('div', 'mlsys-original-popover-keys');
      const keyList = make('ul', 'mlsys-original-keylist');
      for (const [glyph, keys, text] of [['next', ['N'], 'Next step'], ['back', ['P'], 'Back one step'], ['play', ['K'], 'Play or pause'],
        ['restart', ['R'], 'Restart'], [null, ['−', '+'], 'Slower or faster'], ['close', ['Esc'], 'Close expanded view']]) {
        const row = make('li');
        row.append(glyph ? icon(glyph) : make('span', 'mlsys-original-icon'));
        keys.forEach(key => row.append(make('kbd', '', key)));
        row.append(make('span', '', text));
        keyList.append(row);
      }
      keySection.append(make('p', 'mlsys-original-popover-title', 'Keys, while the figure has focus'), keyList);
      popover.append(speedSet, make('p', 'mlsys-original-popover-note', 'Speed changes motion and reading pauses, never the values.'),
        popoverRestart, keySection);
      // One line attached to the bar: the current step title, shown only while the dock is
      // pinned away from the narration. Decorative copy of the narration heading.
      cue = make('div', 'mlsys-original-cue');
      cue.setAttribute('aria-hidden', 'true');
      cueText = make('span', 'mlsys-original-cue-text');
      cue.append(cueText);
      // Pill for scroll positions where the diagram is on screen but the bar cannot pin without
      // covering it: the step cue plus a way down to the explanation. A named button, reachable
      // by keyboard while shown (visibility: hidden removes it from the tab order otherwise).
      jumpAnchor = make('div', 'mlsys-original-jump-anchor');
      jump = make('button', 'mlsys-original-jump');
      jump.type = 'button';
      jump.setAttribute('aria-label', 'Go to step explanation');
      jumpCount = make('span', 'mlsys-original-jump-count');
      jumpCount.setAttribute('aria-hidden', 'true');
      jumpText = make('span', 'mlsys-original-jump-text');
      jumpText.setAttribute('aria-hidden', 'true');
      jump.append(jumpCount, jumpText, icon('down'));
      jumpAnchor.append(jump);
      // Screen readers: one concise playback status, and each committed step announced once.
      statusLive = make('span', 'mlsys-original-sr', 'Ready');
      // The pill moves focus here, so the explanation is a programmatic focus target.
      narrationFocusable = true;
      statusLive.setAttribute('role', 'status');
      announcer = make('div', 'mlsys-original-sr');
      announcer.setAttribute('aria-live', 'polite');
      announcer.setAttribute('aria-atomic', 'true');
      dock = make('div', 'mlsys-original-dock');
      dock.dataset.cue = 'none';
      dock.dataset.stuck = 'none';
      dock.dataset.hold = 'sticky';
      const views = options.viewControls == null ? [] : [].concat(options.viewControls).filter(Boolean);
      if (views.length) {
        viewSlot = make('div', 'mlsys-original-view-controls');
        viewSlot.append(...views);
      }
    }
    const owningFigure = outer.matches('figure') ? outer : outer.querySelector(':scope > figure');
    let stopHeaderWatch = () => {};
    const sequenceLabel = compact ? make('label', 'mlsys-original-sequence') : make('label', 'mlsys-original-sequence', 'Sequence ');
    if (compact) {
      // Phones show a short face ("9/16" over the status word); the native menu sits on top of it.
      const face = make('span', 'mlsys-original-sequence-face');
      face.setAttribute('aria-hidden', 'true');
      faceCount = make('span', 'mlsys-original-face-count');
      faceStatus = make('span', 'mlsys-original-face-status');
      face.append(faceCount, faceStatus);
      sequenceLabel.append(make('span', 'mlsys-original-sequence-name', 'Sequence'), face);
    }
    const sequence = make('select');
    sequence.id = id + '-sequence';
    sequenceLabel.htmlFor = sequence.id;
    const titleOf = index => compact ? stageLabel(steps[index].label) : steps[index].label;
    const optionText = index => compact ? (index + 1) + ' / ' + steps.length + ' · ' + titleOf(index) : (index + 1) + '. ' + steps[index].label;
    steps.forEach((step, index) => {
      const option = make('option', '', optionText(index));
      option.value = String(index);
      sequence.append(option);
    });
    sequenceLabel.append(sequence);
    const narration = make('div', 'mlsys-original-narration');
    if (narrationFocusable) narration.tabIndex = -1;
    // Compact layout announces through its own announcer (once per committed step).
    if (!compact) {
      narration.setAttribute('aria-live', 'polite');
      narration.setAttribute('aria-atomic', 'true');
    }
    const stage = make('div', 'mlsys-original-stage');
    const details = make('div', 'mlsys-original-details');
    if (compact) {
      // DOM order is the visual order: Play, status, menu, Back, Next, Restart, Expand, Speed.
      statusChip.after(sequenceLabel);
      controls.append(keysButton);
      dock.append(controls, popover, cue, keysDesc, statusLive, announcer);
      root.append(heading, introText, assumptions, ...(viewSlot ? [viewSlot] : []), stage, jumpAnchor, dock, narration, details);
    } else {
      const hints = make('p', 'mlsys-original-shortcuts', 'Keyboard: N next · P back · K play/pause · R restart · −/+ speed.');
      root.append(heading, assumptions, sequenceLabel, narration, controls, hints, stage, details);
    }

    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
    let index = 0, target = null, playing = false, destroyed = false;
    let phase = 'idle', frame = 0, clock = null, controller = null, epoch = 0;
    let visible = true;
    let dialog = null, placeholder = null, restoreFocus = null;
    const listeners = [];
    function listen(node, type, fn, opts) {
      node.addEventListener(type, fn, opts);
      listeners.push(() => node.removeEventListener(type, fn, opts));
    }
    let alteredOption = null, narrationKey = '', dockFrame = 0, resize = null;
    // Compact status: Ready only at step 1 (before play, after Restart); Complete only at the end;
    // Moving / Reading / Paused while a play is under way or interrupted; otherwise no word, since
    // the counter already says where the reader is. Never names a destination as reached.
    let interrupted = false, spokenStatus = 'Ready', announceSteps = false, tipsOff = null;
    function statusWord() {
      if (target != null) return clock && !clock.running ? 'Paused' : 'Moving';
      if (phase === 'reading') return clock && !clock.running ? 'Paused' : 'Reading';
      if (index === steps.length - 1) return 'Complete';
      if (interrupted) return 'Paused';
      return index === 0 ? 'Ready' : '';
    }
    function syncCompact() {
      const moving = target != null;
      const word = statusWord();
      statusChip.textContent = word;
      faceStatus.textContent = word;
      root.dataset.status = word ? word.toLowerCase() : 'idle';
      // Screen readers hear a coarse status once per change (same rule; an idle mid-sequence step
      // clears it silently); Moving/Reading flips are not spoken.
      const spoken = playing ? 'Playing' : word === 'Moving' || word === 'Reading' ? null : word;
      if (spoken != null && spoken !== spokenStatus) { spokenStatus = spoken; statusLive.textContent = spoken; }
      if (alteredOption != null) { sequence.options[alteredOption].textContent = optionText(alteredOption); alteredOption = null; }
      // Mid-move the menu keeps the committed step and shows the pending one with an arrow.
      if (moving) {
        sequence.options[index].textContent = (index + 1) + ' → ' + (target + 1) + ' / ' + steps.length + ' · ' + titleOf(target);
        alteredOption = index;
      }
      sequence.value = String(index);
      faceCount.textContent = moving ? (index + 1) + '→' + (target + 1) : (index + 1) + '/' + steps.length;
      const label = playing || clock?.running || (phase === 'moving' && !clock) ? 'Pause' : index === steps.length - 1 && target == null ? 'Replay' : 'Play';
      if (playLabel.textContent !== label) {
        playLabel.textContent = label;
        playIcon.replaceWith(playIcon = icon(label.toLowerCase()));
        playTip.textContent = label + ' (K)';
      }
      backButton.disabled = index === 0 && target == null;
      nextButton.disabled = index === steps.length - 1 && target == null;
      root.dataset.playback = phase;
      root.dataset.playing = String(playing);
      root.dataset.step = String(index);
      stage.setAttribute('aria-busy', phase === 'moving' ? 'true' : 'false');
      if (moving) showNarration(target);
      jumpCount.textContent = faceCount.textContent;
      scheduleDock();
    }
    function syncControls() {
      if (compact) { syncCompact(); return; }
      const selected = target == null ? index : target;
      cueStep.textContent = (target != null ? 'Next ' : '') + (selected + 1) + ' / ' + steps.length + ' · ' + steps[selected].label;
      cueStep.title = cueStep.textContent;
      cueState.textContent = target != null ? (clock?.running || !clock ? 'Moving' : 'Paused') : clock && !clock.running ? 'Paused' : playing ? 'Reading' : index === steps.length - 1 ? 'Complete' : 'Step';
      sequence.value = String(selected);
      playButton.textContent = playing || clock?.running || (phase === 'moving' && !clock) ? 'Pause' : index === steps.length - 1 && target == null ? 'Replay' : 'Play';
      backButton.disabled = index === 0 && target == null;
      nextButton.disabled = index === steps.length - 1 && target == null;
      root.dataset.playback = phase;
      root.dataset.playing = String(playing);
      root.dataset.step = String(index);
      stage.setAttribute('aria-busy', phase === 'moving' ? 'true' : 'false');
      if (target != null) showNarration(target, clock && !clock.running ? 'Paused · ' : 'In motion · ');
    }
    function showNarration(next, prefix = '') {
      if (compact) {
        // Visible heading: the full title (one counter lives in the menu). While a move is under
        // way or paused, only "Next · <target title>" shows: the target's explanation would assert
        // a state the diagram has not reached, so it appears on arrival. The box keeps its height
        // through the move so the page below does not jump. Nothing is spoken here.
        const pending = next !== index || target != null;
        const description = steps[next].description || '';
        const key = next + '|' + pending + '|' + description;
        if (key === narrationKey) return;
        const wasPending = narrationKey.split('|')[1] === 'true';
        narrationKey = key;
        if (pending && !wasPending && narration.offsetHeight) narration.style.minHeight = narration.offsetHeight + 'px';
        if (!pending) narration.style.removeProperty('min-height');
        const shown = (pending ? 'Next · ' : '') + titleOf(next);
        const head = make('strong');
        head.append(make('span', 'mlsys-original-sr', (pending ? 'Next, step ' : 'Step ') + (next + 1) + ' of ' + steps.length + ': '), shown);
        if (pending) narration.replaceChildren(head);
        else narration.replaceChildren(head, make('p', '', description));
        cueText.textContent = shown;
        jumpText.textContent = shown;
        // Each committed step is announced once, on arrival (never at departure).
        if (!pending && announceSteps) announcer.textContent = 'Step ' + (next + 1) + ' of ' + steps.length + ': ' + titleOf(next) + '. ' + description;
        scheduleDock();
        return;
      }
      narration.replaceChildren(
        make('strong', '', prefix + (next + 1) + ' / ' + steps.length + ' · ' + steps[next].label),
        make('p', '', steps[next].description || '')
      );
    }
    function endpoint(next) {
      index = next;
      target = null;
      phase = 'idle';
      render(index, compact ? { stage, details, assumptions, index, step: steps[index] } : { stage, details, index, step: steps[index] });
      showNarration(index);
      syncControls();
    }
    function stopClock() {
      cancelAnimationFrame(frame);
      frame = 0;
      if (clock) clock.last = null;
    }
    function clearAction() {
      ++epoch;
      stopClock();
      if (controller) controller.abort();
      controller = null;
      if (clock) {
        clock.animations.forEach(animation => animation.cancel());
        clock.reject(abortError());
      }
      clock = null;
      target = null;
      phase = 'idle';
    }
    function advanceClock(now) {
      frame = 0;
      if (!clock || !clock.running || destroyed) return;
      const current = clock;
      if (current.last != null) current.elapsed = Math.min(current.duration, current.elapsed + (now - current.last) * speed);
      current.last = now;
      const progress = current.duration ? current.elapsed / current.duration : 1;
      try {
        current.animations.forEach(animation => { animation.currentTime = current.elapsed; });
        current.callbacks.forEach(callback => callback(progress));
        // Protected regions move with the animation; re-run dock and pill avoidance each frame.
        if (compact) scheduleDock();
      } catch (error) {
        current.running = false;
        current.reject(error);
        return;
      }
      if (current.elapsed >= current.duration) {
        current.running = false;
        current.resolve();
      } else frame = requestAnimationFrame(advanceClock);
    }
    function startClock(duration) {
      let resolve, reject;
      const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
      clock = { duration, elapsed: 0, last: null, running: true, animations: [], callbacks: [], resolve, reject };
      frame = requestAnimationFrame(advanceClock);
      return promise;
    }
    function resumeClock() {
      if (!clock || clock.elapsed >= clock.duration) return;
      clock.running = true;
      clock.last = null;
      if (!frame) frame = requestAnimationFrame(advanceClock);
    }
    function setSpeed(value) {
      const rate = Number(value);
      if (!speeds.includes(rate)) throw new RangeError('Unsupported playback speed.');
      if (destroyed) return;
      // Charge time before this change at the old rate; retain the same
      // timeline position, pending endpoint, and paused state.
      if (clock?.running && clock.last != null) {
        const now = performance.now();
        clock.elapsed = Math.min(clock.duration, clock.elapsed + (now - clock.last) * speed);
        clock.last = now;
      }
      speed = rate;
      if (speedSelect) speedSelect.value = String(speed);
      speedInputs.forEach(input => { input.checked = Number(input.value) === speed; });
      if (rateText) rateText.textContent = speed + '×';
      root.dataset.speed = String(speed);
    }
    function changeSpeed(direction) {
      setSpeed(speeds[Math.max(0, Math.min(speeds.length - 1, speeds.indexOf(speed) + direction))]);
    }
    function pause() {
      if (playing || clock?.running) interrupted = true;
      playing = false;
      if (clock) clock.running = false;
      stopClock();
      syncControls();
    }
    function blocked() {
      return document.hidden || (!visible && !dialog?.open) || !!outer.closest('[hidden], [aria-hidden="true"]') || !!root.closest('[hidden], [aria-hidden="true"]') ||
        root.getClientRects().length === 0;
    }
    // Compact: the automatic reading pause scales with the narration's word count, bounded
    // to [0.6, 2] times the build-time pause. It paces reading; it never encodes measured time.
    function readingFor(i) {
      const words = String(steps[i].description || '').trim().split(/\s+/).filter(Boolean).length;
      return Math.min(readingMs * 2, Math.max(readingMs * 0.6, readingMs * words / REF_WORDS));
    }
    async function dwell() {
      if (!playing || destroyed) return;
      if (index === steps.length - 1) { playing = false; syncControls(); return; }
      phase = 'reading';
      const token = epoch;
      if (compact) syncControls();
      try {
        await startClock(compact ? readingFor(index) : readingMs);
        if (destroyed || token !== epoch) return;
        clock = null;
        phase = 'idle';
        if (playing) move(index + 1);
      } catch (error) { if (error.name !== 'AbortError') fail(error); }
    }
    function fail(error) {
      pause();
      clearAction();
      endpoint(index);
      root.dispatchEvent(new CustomEvent('mlsys-original-error', { detail: error, bubbles: true }));
      console.error('Original figure playback failed:', error);
    }
    async function move(next) {
      if (destroyed) return;
      clearAction();
      const token = epoch;
      const from = index;
      // Reduced motion has no intermediate state to pause. Commit directly;
      // an observer pause must not strand a zero-duration animation clock.
      if (reduce.matches) {
        endpoint(next);
        if (playing) dwell();
        return;
      }
      target = next;
      phase = 'moving';
      syncControls();
      controller = new AbortController();
      const duration = movementMs;
      try {
        const done = startClock(transition ? duration : 0);
        done.catch(() => {}); // A synchronous setup failure must not leak the clock promise.
        const current = clock;
        let setup;
        if (transition) {
          setup = transition({
            from, to: next, duration, signal: controller.signal, stage, details,
            animate(element, keyframes, animationOptions = {}) {
              const animation = element.animate(keyframes, { ...animationOptions, duration, fill: 'both' });
              animation.pause();
              animation.currentTime = 0;
              current.animations.push(animation);
              return animation;
            },
            onFrame(callback) { current.callbacks.push(callback); callback(0); }
          });
        }
        await Promise.all([done, Promise.resolve(setup)]);
        if (destroyed || token !== epoch) return;
        current.animations.forEach(animation => animation.cancel());
        clock = null;
        controller = null;
        endpoint(next);
        if (playing) dwell();
      } catch (error) { if (error.name !== 'AbortError' && token === epoch) fail(error); }
    }
    function go(next) {
      if (destroyed) return;
      if (!Number.isInteger(next) || next < 0 || next >= steps.length) throw new RangeError('Invalid original figure step.');
      playing = false;
      interrupted = false;
      clearAction();
      endpoint(next);
    }
    function next() {
      if (destroyed) return;
      playing = false;
      interrupted = false;
      if (target != null) { const next = target; clearAction(); endpoint(next); return; }
      clearAction();
      if (index < steps.length - 1) move(index + 1);
      else syncControls();
    }
    function back() { go(target != null ? index : Math.max(0, index - 1)); }
    function restart() { go(0); }
    function play() {
      if (destroyed || blocked() || steps.length === 1) return;
      playing = true;
      interrupted = false;
      if (clock) resumeClock();
      else if (index === steps.length - 1) { clearAction(); endpoint(0); move(1); }
      else move(index + 1);
      syncControls();
    }
    function toggle() { if (playing || (clock && clock.running)) pause(); else play(); }
    // Compact dock geometry. The dock (control bar) sits right under the diagram. When its
    // natural place is below the fold it pins to the viewport bottom only together with its
    // one-line step title (the cue), and only if bar + cue cover at most COVER_CAP of the visible
    // diagram and no protected region (PROTECTED: the step's focus and its keep-visible readouts).
    // A bar that cannot show its title does not pin; it flows, and a small pill (step count +
    // title + "go to explanation") rides the bottom edge instead, under the same cap, lifted above
    // any protected region it would cover. Scrolled past the diagram, the dock sticks under the
    // header; its cue rides below only once the whole narration has scrolled under the dock.
    // Re-run on scroll, resize, state changes, and every animation frame while a move runs.
    function scheduleDock() {
      if (!compact || destroyed || dockFrame) return;
      dockFrame = requestAnimationFrame(updateDock);
    }
    function viewBounds() {
      if (dialog) { const b = dialogBody.getBoundingClientRect(); return { top: b.top, bottom: b.bottom }; }
      const offset = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--mlsys-original-header-offset')) || 0;
      return { top: offset, bottom: window.innerHeight };
    }
    function updateDock() {
      dockFrame = 0;
      if (destroyed || !root.isConnected || !root.getClientRects().length) return;
      const view = viewBounds(), vt = view.top, vb = view.bottom;
      const s = stage.getBoundingClientRect(), d = dock.getBoundingClientRect(), r = root.getBoundingClientRect();
      const n = narration.getBoundingClientRect();
      const barH = d.height, cueH = cue.offsetHeight, gapAbove = parseFloat(getComputedStyle(dock).marginTop) || 0;
      const visTop = Math.max(s.top, vt), visBot = Math.min(s.bottom, vb), visH = Math.max(0, visBot - visTop);
      const guarded = [...stage.querySelectorAll(PROTECTED)].filter(node => getComputedStyle(node).visibility !== 'hidden')
        .map(node => node.getBoundingClientRect()).filter(b => b.width || b.height);
      // Area of [left,right]x[top,bottom] over the visible diagram must stay within the cap
      // and must not touch a protected region.
      const fits = (top, bottom, left, right) => {
        if (visH < 1 || s.width < 1) return false;
        const hh = Math.max(0, Math.min(bottom, visBot) - Math.max(top, visTop)), ww = Math.max(0, Math.min(right, s.right) - Math.max(left, s.left));
        // 5% margin under the cap absorbs sub-pixel rounding between this estimate and the painted box.
        if (hh * ww > COVER_CAP * 0.95 * visH * s.width) return false;
        return !guarded.some(f => f.bottom > top && f.top < bottom && f.right > left && f.left < right && f.bottom > visTop && f.top < visBot);
      };
      // Natural (unpinned) bar position follows the diagram, so pinning cannot oscillate.
      const naturalTop = s.bottom + gapAbove, naturalBottom = naturalTop + barH;
      const belowFold = naturalBottom > vb + 1;
      const pin = belowFold && fits(vb - barH - cueH, vb, d.left, d.right);
      const hold = belowFold && !pin ? 'natural' : 'sticky';
      // A hold change moves the dock; re-measure now (one forced layout) instead of using stale
      // rects. The second pass computes the same hold from the diagram's position, so it ends.
      if (dock.dataset.hold !== hold) { dock.dataset.hold = hold; updateDock(); return; }
      const gap = n.top - d.bottom;
      const stuck = hold === 'sticky' && belowFold && gap > 1.5 ? 'bottom' : gap < -1.5 ? 'top' : 'none';
      const head = (narration.firstElementChild || narration).getBoundingClientRect();
      let mode = 'none';
      if (stuck === 'bottom') mode = 'above';
      else if (stuck === 'top') mode = n.bottom <= d.bottom + 1 && d.bottom + cueH <= r.bottom + 1 ? 'below' : 'none';
      else if (d.bottom <= vb + 1 && d.top >= vt - 1 && head.top > vb - 6 && fits(d.top - cueH, d.top, d.left, d.right)) mode = 'above';
      if (dock.dataset.stuck !== stuck) dock.dataset.stuck = stuck;
      if (dock.dataset.cue !== mode) dock.dataset.cue = mode;
      // Jump pill: the bar is not (fully) on screen, the diagram is, and the pill fits the cap.
      // Try the bottom edge first, then just above each protected region it would cover. At each
      // height try, in order: the diagram-wide pill when the compact one would truncate the title
      // (narrow players), the compact pill (count + title) right then left, the count-only pill.
      let side = 'none', size = 'full', lift = 0;
      const barShown = stuck === 'bottom' || (d.top >= vt - 1 && d.bottom <= vb + 1);
      if (!barShown && visH > 0 && s.bottom > vt && naturalTop > vt) {
        const ph = jump.offsetHeight, base = Math.min(vb, s.bottom) - 6;
        // Measure both pill sizes (two synchronous layout reads, only while the pill is needed).
        jumpAnchor.dataset.size = 'short';
        const shortW = jump.getBoundingClientRect().width;
        jumpAnchor.dataset.size = 'full';
        const widths = { full: jump.getBoundingClientRect().width + 1, short: shortW + 1, wide: s.width };
        const truncated = jumpText.scrollWidth > jumpText.clientWidth + 1;
        const tries = (truncated ? [['wide', 'right']] : []).concat([['full', 'right'], ['full', 'left'], ['short', 'right'], ['short', 'left']]);
        const heights = [...new Set([base, ...guarded.filter(f => f.top < base && f.bottom > visTop).map(f => Math.floor(f.top - 4))])]
          .filter(y => y <= base && y - ph >= visTop).sort((a, b) => b - a);
        search: for (const bottom of heights) {
          for (const [w, where] of tries) {
            const pw = widths[w], ok = where === 'right' ? fits(bottom - ph, bottom, s.right - pw, s.right) : fits(bottom - ph, bottom, s.left, s.left + pw);
            if (ok) { side = where; size = w; lift = base - bottom; break search; }
          }
        }
      }
      // Never hide the pill while it holds keyboard focus.
      if (side === 'none' && document.activeElement === jump && jumpAnchor.dataset.show && jumpAnchor.dataset.show !== 'none') return;
      if (jumpAnchor.dataset.show !== side) jumpAnchor.dataset.show = side;
      if (jumpAnchor.dataset.size !== size) jumpAnchor.dataset.size = size;
      const lifted = lift > 0 ? (6 + Math.round(lift)) + 'px' : '';
      if (jump.style.bottom !== lifted) jump.style.bottom = lifted;
      if (!popover.hidden) placePopover();
    }
    // Popover: open toward the larger free space, bounded to it with internal scrolling.
    function placePopover() {
      const d = dock.getBoundingClientRect(), view = viewBounds();
      const above = d.top - view.top - 8, below = view.bottom - d.bottom - 8, need = popover.scrollHeight;
      const side = below >= need || below >= above ? 'below' : 'above';
      if (dock.dataset.popover !== side) dock.dataset.popover = side;
      // Bounded to the free space on that side, however small; the popover scrolls internally.
      popover.style.maxHeight = Math.max(0, Math.floor(side === 'below' ? below : above)) + 'px';
    }
    function openPopover() {
      if (!popover.hidden) return;
      popover.hidden = false;
      placePopover();
      keysButton.setAttribute('aria-expanded', 'true');
      (speedInputs.find(input => input.checked) || speedInputs[0]).focus({ preventScroll: true });
    }
    function closePopover(returnFocus) {
      if (!popover || popover.hidden) return;
      popover.hidden = true;
      keysButton.setAttribute('aria-expanded', 'false');
      delete dock.dataset.popover;
      popover.style.removeProperty('max-height');
      if (returnFocus) keysButton.focus({ preventScroll: true });
    }
    let dialogBody = null, onEscape = null;
    function closeExpanded() {
      if (!dialog) return;
      pause();
      const current = dialog;
      dialog = null;
      dialogBody = null;
      expandButton.hidden = false;
      if (viewSlot && viewSlot.parentNode !== root) stage.before(viewSlot);
      if (placeholder?.parentNode) placeholder.replaceWith(root);
      placeholder = null;
      if (current.open) current.close();
      current.remove();
      const focusTarget = restoreFocus;
      restoreFocus = null;
      if (focusTarget?.isConnected) {
        focusTarget.focus({ preventScroll: true });
        // WebKit may restore the dialog's previous focus after cancel dispatch.
        requestAnimationFrame(() => { if (!dialog && focusTarget.isConnected && !destroyed) focusTarget.focus({ preventScroll: true }); });
      }
      if (compact) scheduleDock();
    }
    function expand(event) {
      if (destroyed || dialog) return;
      pause();
      if (compact) closePopover(false);
      restoreFocus = event?.currentTarget === expandButton ? expandButton : document.activeElement;
      expandButton.hidden = true;
      placeholder = document.createComment('Original figure position');
      root.before(placeholder);
      dialog = make('dialog', 'mlsys-original-dialog');
      dialog.setAttribute('aria-labelledby', heading.id);
      const close = make('button', 'mlsys-original-dialog-close', 'Close expanded view');
      close.type = 'button';
      close.autofocus = true;
      close.addEventListener('click', closeExpanded);
      dialog.addEventListener('cancel', event => { event.preventDefault(); closeExpanded(); });
      dialog.addEventListener('close', closeExpanded);
      if (compact) {
        // Close sits in its own bar above a scrolling body, so it never floats over the diagram.
        dialog.classList.add('mlsys-original-dialog-compact');
        // Name stays "Close expanded view"; very narrow screens show only "Close".
        close.setAttribute('aria-label', 'Close expanded view');
        close.replaceChildren(icon('close'), document.createTextNode('Close'), make('span', 'mlsys-original-dialog-close-long', ' expanded view'));
        const bar = make('div', 'mlsys-original-dialog-bar');
        // The view switch rides in the bar while expanded, so the body never scrolls it under Close.
        if (viewSlot) bar.append(viewSlot);
        bar.append(close);
        // Focus may sit on Close (outside the figure): the same one-thing-per-Escape rule applies.
        dialog.addEventListener('keydown', onEscape);
        dialogBody = make('div', 'mlsys-original-dialog-body');
        dialogBody.append(root);
        dialog.append(bar, dialogBody);
      } else dialog.append(close, root);
      document.body.append(dialog);
      dialog.showModal();
      if (compact) scheduleDock();
    }
    listen(expandButton, 'click', expand);
    listen(playButton, 'click', toggle);
    listen(backButton, 'click', back);
    listen(nextButton, 'click', next);
    listen(restartButton, 'click', restart);
    if (speedSelect) listen(speedSelect, 'change', () => setSpeed(speedSelect.value));
    listen(sequence, 'change', () => go(Number(sequence.value)));
    if (compact) {
      speedInputs.forEach(input => {
        listen(input, 'change', () => { if (input.checked) setSpeed(input.value); });
        // Safari does not focus a radio on click (focus lands on the figure region); keep it
        // on the chosen radio so arrow keys continue to work.
        listen(input, 'click', () => { if (document.activeElement !== input) input.focus({ preventScroll: true }); });
      });
      listen(keysButton, 'click', () => { if (popover.hidden) openPopover(); else closePopover(true); });
      listen(popoverRestart, 'click', () => { restart(); closePopover(false); playButton.focus({ preventScroll: true }); });
      // The pill moves the reader (and keyboard focus) to the step explanation under the bar.
      listen(jump, 'click', () => {
        narration.focus({ preventScroll: true });
        narration.scrollIntoView({ block: 'end', behavior: reduce.matches ? 'auto' : 'smooth' });
      });
      // Tooltips (WCAG 1.4.13): hoverable, and Escape hides the visible one without moving focus.
      const tips = [...controls.querySelectorAll('.mlsys-original-tip')];
      tips.forEach(tip => listen(tip, 'click', event => { event.stopPropagation(); event.preventDefault(); }));
      const tipsBack = event => { if (tipsOff && !tipsOff.contains(event.target)) { tipsOff = null; delete root.dataset.tips; } };
      listen(controls, 'pointerover', tipsBack);
      listen(root, 'focusin', tipsBack);
      // One Escape does one thing: close the popover, else hide a visible tooltip in place, else
      // (inside Expand) close the dialog. Consuming the key stops the dialog's cancel as well.
      // A tooltip counts as shown only if it was up before this key press: hovered, or on a control
      // that received focus from the keyboard. (Any key press flips a mouse-clicked button into
      // :focus-visible, which would otherwise surface its tip just in time to swallow the Escape.)
      let lastKey = -Infinity, keyFocus = null, hovered = null;
      listen(document, 'keydown', () => { lastKey = performance.now(); }, { capture: true });
      listen(document, 'pointerdown', () => { lastKey = -Infinity; keyFocus = null; }, { capture: true });
      listen(root, 'focusin', event => { keyFocus = performance.now() - lastKey < 400 ? event.target : null; });
      // Hover counts only when the pointer actually moved onto the button; a bar that opens or pins
      // under a still pointer (Expand) gets a synthetic hover the reader never made.
      listen(document, 'pointermove', event => { const b = event.target.closest?.('.mlsys-original-btn'); hovered = b && controls.contains(b) ? b : null; }, { capture: true, passive: true });
      onEscape = event => {
        if (event.key !== 'Escape') return;
        if (!popover.hidden) { event.preventDefault(); event.stopPropagation(); closePopover(true); return; }
        const shown = tips.find(tip => getComputedStyle(tip).display !== 'none' && (tip.parentElement === hovered || tip.parentElement === keyFocus));
        if (shown) { event.preventDefault(); event.stopPropagation(); tipsOff = shown.parentElement; root.dataset.tips = 'off'; }
      };
      listen(root, 'keydown', onEscape);
      listen(popover, 'focusout', event => {
        const next = event.relatedTarget;
        // WebKit focuses the region root when a radio is clicked; that is not leaving the popover.
        if (next && next !== root && !popover.contains(next) && next !== keysButton) closePopover(false);
      });
      listen(document, 'pointerdown', event => {
        if (!popover.hidden && !popover.contains(event.target) && !keysButton.contains(event.target)) closePopover(false);
      }, true);
      listen(document, 'scroll', scheduleDock, { capture: true, passive: true });
      listen(window, 'resize', scheduleDock, { passive: true });
      resize = typeof ResizeObserver === 'function' ? new ResizeObserver(scheduleDock) : null;
      resize?.observe(root);
    }
    listen(root, 'keydown', event => {
      if (event.defaultPrevented || event.repeat || event.ctrlKey || event.altKey || event.metaKey ||
          event.target.closest('input, textarea, select, a, [contenteditable]:not([contenteditable="false"])')) return;
      if (['-', '_', '+', '='].includes(event.key)) {
        event.preventDefault(); changeSpeed(event.key === '-' || event.key === '_' ? -1 : 1); return;
      }
      const action = { n: next, p: back, k: toggle, r: restart }[event.key.toLowerCase()];
      if (action) { event.preventDefault(); action(); }
    });
    listen(document, 'visibilitychange', () => { if (document.hidden) pause(); });
    listen(reduce, 'change', () => { if (reduce.matches && target != null) { const next = target; const auto = playing; clearAction(); endpoint(next); playing = auto; if (playing) dwell(); } });
    const observer = typeof IntersectionObserver === 'function' ? new IntersectionObserver(entries => {
      visible = entries[0].isIntersecting;
      if (!visible && !dialog?.open) pause();
    }) : null;
    if (observer) observer.observe(root);
    const mutation = new MutationObserver(() => { if (blocked()) pause(); });
    mutation.observe(outer, { attributes: true, subtree: true, attributeFilter: ['hidden', 'aria-hidden', 'class', 'style'] });
    let ancestor = outer.parentElement;
    while (ancestor && ancestor !== document.body) {
      mutation.observe(ancestor, { attributes: true, attributeFilter: ['hidden', 'aria-hidden', 'style', 'class'] });
      ancestor = ancestor.parentElement;
    }
    function destroy() {
      if (destroyed) return;
      playing = false;
      closeExpanded();
      clearAction();
      destroyed = true;
      observer?.disconnect();
      mutation.disconnect();
      stopHeaderWatch();
      owningFigure?.classList.remove('mlsys-original-sticky-frame');
      listeners.forEach(remove => remove());
      root.replaceChildren();
      root.classList.remove('mlsys-original-player');
      root.removeAttribute('role');
      root.removeAttribute('aria-labelledby');
      root.removeAttribute('aria-busy');
      root.removeAttribute('tabindex');
      if (compact) {
        cancelAnimationFrame(dockFrame); dockFrame = 0;
        resize?.disconnect();
        root.classList.remove('mlsys-original-compact');
        delete root.dataset.layout;
        delete root.dataset.status;
        delete root.dataset.tips;
        delete root.dataset.numberedLabels;
      }
    }
    setSpeed(1);
    endpoint(0);
    if (compact) {
      announceSteps = true;
      // Lint hook (check_original_player.py): labels should not start with a bare stage number.
      const numbered = steps.filter(step => NUMBERED.test(String(step.label))).length;
      if (numbered) root.dataset.numberedLabels = String(numbered);
    }
    owningFigure?.classList.add('mlsys-original-sticky-frame');
    stopHeaderWatch = watchHeader(root);
    return { go, next, back, play, pause, restart, expand, closeExpanded, destroy, setSpeed,
      get state() { return { index, target, playing, phase, paused: !!clock && !clock.running, speed, destroyed }; } };
  }
  window.MLSysOriginalPlayer = Object.freeze({ create });
}());
