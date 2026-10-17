/* Original-art guided annotation. Geometry and narration are explicitly authored. */
(() => {
  'use strict';
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons['guided-figure'] = (root, context) => {
    const lesson = context?.lesson;
    const images = new Set();
    for (const node of context?.staticNodes || []) {
      if (node.nodeType !== 1) continue;
      if (node.matches('img')) images.add(node);
      node.querySelectorAll('img').forEach(image => images.add(image));
    }
    if (!root.id || images.size !== 1) throw new Error('Guided figure needs one canonical image and a unique root ID.');
    const entries = Object.entries(lesson?.regions || {});
    if (!entries.length || !Array.isArray(lesson?.steps) || !lesson.steps.length) throw new Error('Missing authored regions or states.');
    for (const [, region] of entries) {
      if (![region.x, region.y, region.w, region.h].every(Number.isFinite) ||
          region.x < 0 || region.y < 0 || region.w <= 0 || region.h <= 0 ||
          region.x + region.w > 1.000001 || region.y + region.h > 1.000001 || !region.label) {
        throw new Error('Invalid normalized image region.');
      }
    }
    for (const step of lesson.steps) {
      if (!step.title || !step.description || !Array.isArray(step.focus) || !step.focus.length ||
          step.focus.some(id => !Object.hasOwn(lesson.regions, id))) throw new Error('Every state needs explicit valid visual focus.');
    }
    const abort = new AbortController();
    const on = (target, event, fn) => target.addEventListener(event, fn, {signal:abort.signal});
    const make = (tag, cls, text) => {
      const element = document.createElement(tag);
      if (cls) element.className = cls;
      if (text !== undefined) element.textContent = text;
      return element;
    };
    let stage = 0, running = false, timer = null, destroyed = false;
    root.classList.add('mlsys-guided');
    const intro = make('p', 'mlsys-guided-question', lesson.question);
    const scope = make('p', 'mlsys-guided-scope', lesson.scope || 'Guided annotation of the original figure; highlights do not simulate execution or timing.');
    const controls = make('div', 'mlsys-guided-controls');
    const buttons = {};
    for (const [key, label] of [['play','Play'],['back','Back'],['next','Next'],['restart','Restart']]) {
      const button = make('button', '', label); button.type = 'button'; buttons[key] = button; controls.append(button);
    }
    const label = make('label', '', 'Step ');
    const menu = make('select'); menu.id = root.id + '-sequence'; label.htmlFor = menu.id;
    lesson.steps.forEach((step, i) => {
      const option = make('option', '', `${i + 1}. ${step.title}`); option.value = i; menu.append(option);
    });
    label.append(menu); controls.append(label);
    const current = make('div', 'mlsys-guided-current'); current.setAttribute('aria-live','polite'); current.setAttribute('aria-atomic','true');
    const heading = make('p', 'mlsys-guided-step');
    const description = make('p', 'mlsys-guided-description'); current.append(heading, description);
    const art = make('div', 'mlsys-guided-art');
    // Deliberately copy only image data: no source IDs, classes, links or lightbox metadata.
    const image = make('img');
    const original = [...images][0];
    image.src = original.currentSrc || original.src;
    image.alt = original.alt || lesson.title || 'Original figure'; image.decoding = 'async';
    const overlay = document.createElementNS('http://www.w3.org/2000/svg','svg');
    overlay.setAttribute('viewBox','0 0 1000 1000'); overlay.setAttribute('preserveAspectRatio','none');
    overlay.setAttribute('aria-hidden','true'); overlay.classList.add('mlsys-guided-overlay');
    const outlines = new Map();
    const key = make('div','mlsys-guided-key'); key.setAttribute('role','group'); key.setAttribute('aria-label','Figure regions; select to inspect');
    const regionButtons = new Map();
    entries.forEach(([id, region], i) => {
      const rect = document.createElementNS(overlay.namespaceURI,'rect');
      for (const [name,value] of Object.entries({x:region.x*1000,y:region.y*1000,width:region.w*1000,height:region.h*1000,rx:3})) rect.setAttribute(name,value);
      overlay.append(rect); outlines.set(id,rect);
      // Overlay buttons are transparent and never obscure existing labels.
      const hit = make('button','mlsys-guided-hit'); hit.type = 'button'; hit.setAttribute('aria-label',region.label);
      hit.style.left = `${region.x*100}%`; hit.style.top = `${region.y*100}%`;
      hit.style.width = `${region.w*100}%`; hit.style.height = `${region.h*100}%`;
      hit.tabIndex = -1; art.append(hit);
      const item = make('button','',`${i+1}. ${region.label}`); item.type = 'button'; key.append(item); regionButtons.set(id,item);
      const inspect = () => { pause(); stage = lesson.steps.findIndex(step => step.focus.includes(id)); render(); };
      if (!lesson.steps.some(step => step.focus.includes(id))) throw new Error('Unused region cannot be inspected.');
      on(hit,'click',inspect); on(item,'click',inspect);
    });
    art.prepend(image); art.append(overlay);
    const player = make('div', 'mlsys-guided-player');
    const viewport = make('div', 'mlsys-guided-viewport');
    viewport.tabIndex = 0; viewport.setAttribute('role','region');
    viewport.setAttribute('aria-label','Original figure; scroll horizontally in enlarged view');
    viewport.append(art);
    const panCue = make('p', 'mlsys-guided-pan', 'Pan horizontally to inspect the complete figure. The step controls and explanation remain above it.');
    player.append(controls, current, panCue, viewport, key);
    const enlarge = make('button', 'mlsys-guided-enlarge', 'Enlarge figure'); enlarge.type = 'button';
    const dialog = make('dialog', 'mlsys-guided mlsys-guided-dialog');
    const dialogHeading = make('h2', 'mlsys-guided-dialog-title', lesson.question || lesson.title);
    dialogHeading.id = root.id + '-dialog-title'; dialog.setAttribute('aria-labelledby',dialogHeading.id);
    const close = make('button', 'mlsys-guided-close', 'Close enlarged figure'); close.type = 'button';
    dialog.append(close, dialogHeading);
    root.append(intro, scope, enlarge, player, dialog);
    const restore = () => {
      pause(); root.insertBefore(player,dialog);
      if (!destroyed && enlarge.isConnected) enlarge.focus({preventScroll:true});
    };
    on(enlarge,'click',() => {
      pause(); dialog.append(player);
      try { dialog.showModal(); render(); close.focus(); }
      catch (error) { restore(); throw error; }
    });
    on(close,'click',() => { pause(); dialog.close(); });
    on(dialog,'keydown',event => {
      if (event.key !== 'Tab') return;
      const focusable = [...dialog.querySelectorAll('button:not(:disabled),select:not(:disabled),[tabindex="0"]')]
        .filter(element => element.tabIndex >= 0 && element.getClientRects().length);
      const first = focusable[0], last = focusable[focusable.length-1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    });
    on(dialog,'cancel',pause);
    on(dialog,'close',restore);
    function render() {
      const step = lesson.steps[stage]; menu.value = stage;
      heading.textContent = `${stage+1} / ${lesson.steps.length} · ${step.title}`;
      description.textContent = step.description;
      buttons.play.textContent = running ? 'Pause' : stage === lesson.steps.length-1 ? 'Replay' : 'Play';
      buttons.back.disabled = stage === 0; buttons.next.disabled = stage === lesson.steps.length-1;
      for (const [id,rect] of outlines) {
        rect.classList.toggle('is-focused',step.focus.includes(id));
        regionButtons.get(id).setAttribute('aria-pressed',String(step.focus.includes(id)));
      }
      if (dialog.open) {
        const focused = step.focus.map(id => lesson.regions[id]);
        const left = Math.min(...focused.map(region => region.x));
        const right = Math.max(...focused.map(region => region.x + region.w));
        viewport.scrollLeft = (left + right) / 2 * art.offsetWidth - viewport.clientWidth / 2;
      }
    }
    function pause() { running = false; clearTimeout(timer); timer = null; if (!destroyed) render(); }
    function schedule() {
      // Reading time, never a physical performance model; no interpolation required.
      const delay = Math.max(5000, lesson.steps[stage].description.split(/\s+/).length*280);
      timer = setTimeout(() => {
        if (!running || destroyed) return;
        if (stage >= lesson.steps.length-1) { pause(); return; }
        stage++; render(); schedule();
      }, delay);
    }
    on(buttons.play,'click',() => {
      if (running) { pause(); return; }
      if (stage === lesson.steps.length-1) stage = 0;
      running = true; render(); schedule();
    });
    on(buttons.back,'click',() => { pause(); stage=Math.max(0,stage-1); render(); });
    on(buttons.next,'click',() => { pause(); stage=Math.min(lesson.steps.length-1,stage+1); render(); });
    on(buttons.restart,'click',() => { pause(); stage=0; render(); });
    on(menu,'change',() => { const selected=Number(menu.value); pause(); stage=selected; render(); });
    on(document,'visibilitychange',() => { if (document.hidden) pause(); });
    on(window,'beforeprint',pause);
    const observer = typeof IntersectionObserver === 'function' ? new IntersectionObserver(records => {
      if (!dialog.open && records.some(record => !record.isIntersecting)) pause();
    }) : null;
    observer?.observe(root);
    render();
    return {pause, destroy() { pause(); destroyed=true; abort.abort(); observer?.disconnect(); if (dialog.open) dialog.close(); root.replaceChildren(); }};
  };
})();
