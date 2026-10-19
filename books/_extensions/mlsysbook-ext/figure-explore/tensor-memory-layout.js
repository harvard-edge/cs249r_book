/* Figure: fig-tensor-memory-layout, Volume I / Frameworks.
 * Same 2×3 values, panel order, and element-stride convention as Diagram.
 * Runtime calls pause() when hiding the panel; local visibility observers also
 * pause without resetting. No movement/interpolation or autoplay is used.
 */
(() => {
  'use strict';
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons['tensor-memory-layout'] = root => {
    const prefix = root.id;
    if (!prefix) throw new Error('Tensor lesson needs a unique root ID.');
    const abort = new AbortController();
    const on = (target, event, listener) => target.addEventListener(event, listener, {signal: abort.signal});
    const make = (tag, className, text) => {
      const element = document.createElement(tag);
      if (className) element.className = className;
      if (text !== undefined) element.textContent = text;
      return element;
    };
    let stage = 0, selected = 4, traversal = 'row', running = false, timer = null, destroyed = false;
    const stages = ['Read indices', 'Read strides', 'Compute offsets', 'Locate in storage', 'Compare locality'];
    const controls = make('div', 'mlsys-tensor-controls');
    const buttons = {};
    for (const [name, text] of [['play','Play'], ['back','Back'], ['next','Next'], ['restart','Restart']]) {
      const button = make('button', '', text);
      button.type = 'button'; buttons[name] = button; controls.append(button);
    }
    const label = make('label', '', 'Step ');
    const menu = make('select'); menu.id = prefix + '-sequence'; label.htmlFor = menu.id;
    stages.forEach((text, index) => {
      const option = make('option', '', `${index + 1}. ${text}`); option.value = index; menu.append(option);
    });
    label.append(menu); controls.append(label);
    const intro = make('p', 'mlsys-tensor-intro', 'Select a tensor cell to find the same value in both storage layouts. All indices and offsets start at zero.');
    const panels = make('div', 'mlsys-tensor-panels');
    const tensor = make('section', 'mlsys-tensor-source');
    const sourceTitle = make('h4', 'mlsys-tensor-heading', '2D Tensor (2 × 3)');
    sourceTitle.id = prefix + '-tensor-title'; tensor.setAttribute('aria-labelledby', sourceTitle.id);
    const grid = make('div', 'mlsys-tensor-grid');
    const cells = [];
    for (let index = 0; index < 6; index++) {
      const cell = make('button', 'mlsys-tensor-cell', index + 1); cell.type = 'button';
      cell.setAttribute('aria-label', `Value ${index + 1}, row ${Math.floor(index / 3)}, column ${index % 3}`);
      on(cell, 'click', () => { pause(); selected = index; render(); });
      cells.push(cell); grid.append(cell);
    }
    const identity = make('p', 'mlsys-tensor-identity');
    tensor.append(sourceTitle, grid, identity);
    const storage = make('div', 'mlsys-tensor-storage');
    const layouts = [
      {name:'Row-Major Layout', className:'row', order:[1,2,3,4,5,6], strides:[3,1]},
      {name:'Column-Major Layout', className:'column', order:[1,4,2,5,3,6], strides:[1,2]}
    ];
    for (const [index, layout] of layouts.entries()) {
      const section = make('section', 'mlsys-tensor-layout mlsys-tensor-' + layout.className);
      const heading = make('h4', 'mlsys-tensor-heading', layout.name);
      heading.id = prefix + '-layout-' + index; section.setAttribute('aria-labelledby', heading.id);
      const strip = make('div', 'mlsys-tensor-strip');
      layout.slots = layout.order.map((value, offset) => {
        const slot = make('div', 'mlsys-tensor-slot');
        const number = make('div', 'mlsys-tensor-value', value);
        const address = make('div', 'mlsys-tensor-address', `[${offset}]`);
        slot.append(number, address); strip.append(slot); return number;
      });
      layout.detail = make('p', 'mlsys-tensor-detail');
      section.append(heading, strip, layout.detail); storage.append(section);
    }
    panels.append(tensor, storage);
    const status = make('div', 'mlsys-tensor-status'); status.setAttribute('aria-live','polite'); status.setAttribute('aria-atomic','true');
    const explanation = make('p', 'mlsys-tensor-explanation');
    const equations = make('div', 'mlsys-tensor-equations');
    status.append(explanation, equations);
    const traversalControls = make('div', 'mlsys-tensor-traversal-controls');
    const traversalLabel = make('label', '', 'Traversal ');
    const traversalMenu = make('select'); traversalMenu.id = prefix + '-traversal'; traversalLabel.htmlFor = traversalMenu.id;
    for (const [value, text] of [['row', 'Across the selected row'], ['column', 'Down the selected column']]) {
      const option = make('option', '', text); option.value = value; traversalMenu.append(option);
    }
    traversalLabel.append(traversalMenu);
    const traversalLegend = make('p', 'mlsys-tensor-footnote', 'A bottom bar marks traversal members; the outline marks the single selected value.');
    const traversalSequence = make('div', 'mlsys-tensor-walk');
    traversalControls.append(traversalLabel, traversalLegend, traversalSequence);
    const bytes = make('p', 'mlsys-tensor-footnote', 'Strides and bracketed storage offsets count elements, not bytes. If each element occupies B bytes, byte offset = element offset × B, measured from the storage origin.');
    root.append(intro, controls, panels, traversalControls, status, bytes);
    function render() {
      const row = Math.floor(selected / 3), col = selected % 3;
      const walk = traversal === 'row' ? [row * 3, row * 3 + 1, row * 3 + 2] : [col, 3 + col];
      traversalControls.hidden = stage !== 4; traversalMenu.value = traversal;
      traversalSequence.replaceChildren();
      if (stage === 4) {
        traversalSequence.append(make('p', '', `Tensor values: ${walk.map(index => index + 1).join(' → ')}`));
        for (const layout of layouts) {
          const addresses = walk.map(index => Math.floor(index / 3) * layout.strides[0] + (index % 3) * layout.strides[1]);
          traversalSequence.append(make('p', '', `${layout.name} element offsets: ${addresses.join(' → ')}`));
        }
      }
      identity.textContent = `tensor[${row}, ${col}] = ${selected + 1}`;
      cells.forEach((cell,index) => {
        cell.classList.toggle('mlsys-tensor-selected', index === selected);
        cell.classList.toggle('mlsys-tensor-traversed', stage === 4 && walk.includes(index));
        cell.setAttribute('aria-pressed', String(index === selected));
      });
      const offsets = layouts.map(layout => row * layout.strides[0] + col * layout.strides[1]);
      layouts.forEach((layout,index) => {
        layout.slots.forEach((slot,offset) => {
          slot.classList.toggle('mlsys-tensor-selected', stage >= 3 && offset === offsets[index]);
          slot.classList.toggle('mlsys-tensor-traversed', stage === 4 && walk.includes(layout.order[offset] - 1));
        });
        layout.detail.textContent = stage === 0 ? 'Brackets identify element offsets.' : `Element strides: [${layout.strides.join(', ')}]`;
      });
      explanation.textContent = [
        `Value ${selected + 1} has row index ${row} and column index ${col}. Its tensor identity is unchanged by storage order.`,
        'Row-major strides [3, 1] advance three elements for one row and one for one column. Column-major strides [1, 2] advance one element for one row and two for one column.',
        'Multiply each index by its element stride and add. Compute both layouts from the same selected indices.',
        `The outlined slots both contain value ${selected + 1}: row-major offset ${offsets[0]}, column-major offset ${offsets[1]}. ${offsets[0] === offsets[1] ? 'This cell happens to have the same offset in both layouts.' : 'Different positions, same tensor element.'}`,
        'Row-major places neighboring columns consecutively; column-major places neighboring rows consecutively. Neither layout is universally faster: the operation’s traversal and supported layouts determine which is suitable.'
      ][stage];
      equations.replaceChildren();
      if (stage >= 2) layouts.forEach((layout,index) => {
        const offset = offsets[index];
        equations.append(make('p', '', `${layout.name}: ${row} × ${layout.strides[0]} + ${col} × ${layout.strides[1]} = ${offset} elements; byte offset = ${offset} × B.`));
      });
      menu.value = stage; buttons.back.disabled = stage === 0; buttons.next.disabled = stage === stages.length - 1;
      buttons.play.textContent = running ? 'Pause' : stage === stages.length - 1 ? 'Replay' : 'Play';
      buttons.play.setAttribute('aria-label', running ? 'Pause sequence' : stage === stages.length - 1 ? 'Replay sequence' : 'Play sequence');
    }
    function pause() { running = false; clearTimeout(timer); timer = null; if (!destroyed) render(); }
    function schedule() {
      timer = setTimeout(() => {
        if (!running || destroyed) return;
        stage++;
        if (stage === stages.length - 1) { traversal = 'row'; running = false; }
        render();
        if (running) schedule();
      }, 6500);
    }
    on(buttons.play, 'click', () => {
      if (running) { pause(); return; }
      if (stage === stages.length - 1) stage = 0;
      running = true; render(); schedule();
    });
    on(buttons.back, 'click', () => { pause(); stage = Math.max(0, stage - 1); render(); });
    on(buttons.next, 'click', () => { pause(); stage = Math.min(stages.length - 1, stage + 1); if (stage === 4) traversal = 'row'; render(); });
    on(buttons.restart, 'click', () => { pause(); stage = 0; selected = 4; traversal = 'row'; render(); });
    on(menu, 'change', () => { const requested = Number(menu.value); pause(); stage = requested; render(); });
    on(document, 'visibilitychange', () => { if (document.hidden) pause(); });
    on(traversalMenu, 'change', () => { const requested = traversalMenu.value; pause(); traversal = requested; render(); });
    const observer = typeof IntersectionObserver === 'undefined' ? null : new IntersectionObserver(entries => {
      if (!entries[0].isIntersecting && running) pause();
    }, {threshold:0.15});
    observer?.observe(root);
    render();
    return {
      pause,
      destroy() { if (destroyed) return; pause(); destroyed = true; observer?.disconnect(); abort.abort(); root.replaceChildren(); }
    };
  };
})();
