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
    assumptions.append(make('summary', 'mlsys-original-assumptions-summary', 'Example and assumptions'),
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
    for (const [control, label, symbol] of [[restartButton,'Restart','↺'],[expandButton,'Expand','↗']]) {
      control.setAttribute('aria-label', label);
      const full = make('span', 'mlsys-original-control-full', label);
      const short = make('span', 'mlsys-original-control-short', symbol);
      short.setAttribute('aria-hidden', 'true');
      control.replaceChildren(full, short);
    }
    const speedLabel = make('label', 'mlsys-original-speed', 'Speed ');
    const speedSelect = make('select');
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
    const cueStep = make('span', 'mlsys-original-cue-step');
    const cueState = make('span', 'mlsys-original-cue-state');
    currentCue.append(cueStep, cueState); controls.append(currentCue);
    const owningFigure = outer.matches('figure') ? outer : outer.querySelector(':scope > figure');
    let stopHeaderWatch = () => {};
    const sequenceLabel = make('label', 'mlsys-original-sequence', 'Sequence ');
    const sequence = make('select');
    sequence.id = id + '-sequence';
    sequenceLabel.htmlFor = sequence.id;
    steps.forEach((step, index) => {
      const option = make('option', '', (index + 1) + '. ' + step.label);
      option.value = String(index);
      sequence.append(option);
    });
    sequenceLabel.append(sequence);
    const hints = make('p', 'mlsys-original-shortcuts', 'Keyboard: N next · P back · K play/pause · R restart · −/+ speed.');
    const narration = make('div', 'mlsys-original-narration');
    narration.setAttribute('aria-live', 'polite');
    narration.setAttribute('aria-atomic', 'true');
    const stage = make('div', 'mlsys-original-stage');
    const details = make('div', 'mlsys-original-details');
    root.append(heading, assumptions, sequenceLabel, narration, controls, hints, stage, details);

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
    function syncControls() {
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
      narration.replaceChildren(
        make('strong', '', prefix + (next + 1) + ' / ' + steps.length + ' · ' + steps[next].label),
        make('p', '', steps[next].description || '')
      );
    }
    function endpoint(next) {
      index = next;
      target = null;
      phase = 'idle';
      render(index, { stage, details, index, step: steps[index] });
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
      speedSelect.value = String(speed);
      root.dataset.speed = String(speed);
    }
    function changeSpeed(direction) {
      setSpeed(speeds[Math.max(0, Math.min(speeds.length - 1, speeds.indexOf(speed) + direction))]);
    }
    function pause() {
      playing = false;
      if (clock) clock.running = false;
      stopClock();
      syncControls();
    }
    function blocked() {
      return document.hidden || (!visible && !dialog?.open) || !!outer.closest('[hidden], [aria-hidden="true"]') || !!root.closest('[hidden], [aria-hidden="true"]') ||
        root.getClientRects().length === 0;
    }
    async function dwell() {
      if (!playing || destroyed) return;
      if (index === steps.length - 1) { playing = false; syncControls(); return; }
      phase = 'reading';
      const token = epoch;
      try {
        await startClock(readingMs);
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
      clearAction();
      endpoint(next);
    }
    function next() {
      if (destroyed) return;
      playing = false;
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
      if (clock) resumeClock();
      else if (index === steps.length - 1) { clearAction(); endpoint(0); move(1); }
      else move(index + 1);
      syncControls();
    }
    function toggle() { if (playing || (clock && clock.running)) pause(); else play(); }
    function closeExpanded() {
      if (!dialog) return;
      pause();
      const current = dialog;
      dialog = null;
      expandButton.hidden = false;
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
    }
    function expand(event) {
      if (destroyed || dialog) return;
      pause();
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
      dialog.append(close, root);
      document.body.append(dialog);
      dialog.showModal();
    }
    listen(expandButton, 'click', expand);
    listen(playButton, 'click', toggle);
    listen(backButton, 'click', back);
    listen(nextButton, 'click', next);
    listen(restartButton, 'click', restart);
    listen(speedSelect, 'change', () => setSpeed(speedSelect.value));
    listen(sequence, 'change', () => go(Number(sequence.value)));
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
    }
    setSpeed(1);
    endpoint(0);
    owningFigure?.classList.add('mlsys-original-sticky-frame');
    stopHeaderWatch = watchHeader(root);
    return { go, next, back, play, pause, restart, expand, closeExpanded, destroy, setSpeed,
      get state() { return { index, target, playing, phase, paused: !!clock && !clock.running, speed, destroyed }; } };
  }
  window.MLSysOriginalPlayer = Object.freeze({ create });
}());
