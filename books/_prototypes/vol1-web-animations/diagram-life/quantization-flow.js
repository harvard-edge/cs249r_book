/* Inspectable teaching scenes. Motion illustrates dependencies, never hardware timing. */
(() => {
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons['quantized-matmul'] = root => {
    const abort = new AbortController();
    const on = (node, event, fn) => node.addEventListener(event, fn, {signal: abort.signal});
    const el = (tag, cls, value) => { const node = document.createElement(tag); if (cls) node.className = cls; if (value !== undefined) node.textContent = value; return node; };
    const NS = 'http://www.w3.org/2000/svg';
    const shape = (tag, attrs = {}, value) => { const node = document.createElementNS(NS, tag); for (const [key, val] of Object.entries(attrs)) node.setAttribute(key, val); if (value !== undefined) node.textContent = value; return node; };
    const x = [.31, -.52, .77], w = [.42, -.18, .24];
    const round = n => Math.sign(n) * Math.floor(Math.abs(n) + .5);
    const clamp = n => Math.max(-128, Math.min(127, n));
    const fmt = n => String(Number(n.toFixed(6)));
    const signed = n => (n > 0 ? '+' : '') + fmt(n);
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    const phone = () => matchMedia('(max-width:600px)').matches;
    let step = 0, spacing = .1, selected = 0, playing = false, animating = false, paused = false;
    let selectedScene = true, dead = false, version = 0, animations = [], holdTimer = null, scene, particles;
    const stages = ['Start: two float arrays', 'Quantize activations', 'Quantize weights', 'Accumulate pair 1', 'Accumulate pair 2', 'Accumulate pair 3', 'Requantize the sum', 'Activate and write output'];
    root.classList.add('mlsys-flow');
    const intro = el('p', 'flow-intro', 'Follow values as they lose precision and build one retained sum.');
    const route = el('ol', 'flow-route');
    ['Float inputs', 'Integer grid', 'Multiply & retain', 'INT8 output'].forEach(label => route.append(el('li', '', label)));
    const controls = el('div', 'flow-controls'), buttons = {};
    for (const [key, label] of [['play', 'Play'], ['back', 'Back'], ['next', 'Next'], ['restart', 'Restart']]) { const b = el('button', '', label); b.type = 'button'; buttons[key] = b; controls.append(b); }
    const menuLabel = el('label', '', 'Sequence '), menu = el('select');
    stages.forEach((label, i) => { const option = el('option', '', `${i + 1}. ${label}`); option.value = i; menu.append(option); });
    menuLabel.append(menu); controls.append(menuLabel);
    const settings = el('label', 'flow-settings', 'Activation spacing '), scaleMenu = el('select');
    for (const [value, label] of [['0.1', '0.10 — coarser grid'], ['0.05', '0.05 — finer grid']]) { const option = el('option', '', label); option.value = value; scaleMenu.append(option); }
    settings.append(scaleMenu);
    const narration = el('p', 'flow-narration'); narration.setAttribute('aria-live', 'polite');
    const board = shape('svg', {class: 'flow-scene', role: 'group', 'aria-label': 'A worked quantized dot product, showing approximation and retained accumulation.'});
    const defs = shape('defs'), marker = shape('marker', {id: root.id + '-flow-arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto'});
    marker.append(shape('path', {d: 'M0 0L10 5L0 10Z', fill: 'currentColor'})); defs.append(marker);
    const note = el('p', 'flow-scene-note');
    const detail = el('details', 'flow-calculation'), summary = el('summary', '', 'Inspect the arithmetic'), equation = el('p', 'flow-equation');
    const assumptions = el('p', 'flow-assumptions', 'One 1 × 3 by 3 × 1 dot product; no bias. Signed INT8, nearest rounding (ties away from zero), then clamping. Zero-points: x = 3, w = 0, output = −4. Weight spacing 0.05; output spacing 0.02. ReLU uses stored zero −4. Offsets show equivalent logical arithmetic, not required hardware instructions. Weight conversion may happen offline.');
    detail.append(summary, equation, assumptions);
    const scope = el('p', 'flow-scope', 'Moving values are reads: source arrays stay available. Seconds and distances are illustrative. Product trails are a visual history, not extra storage. Finer spacing narrows the real range; it is not universally more accurate.');
    const clock = el('span', 'flow-clock'); clock.setAttribute('aria-hidden', 'true');
    root.append(intro, route, controls, settings, narration, board, note, detail, scope, clock);

    function data() {
      const qx = x.map(v => clamp(round(v / spacing) + 3));
      const qw = w.map(v => clamp(round(v / .05)));
      const terms = qx.map((v, i) => (v - 3) * qw[i]);
      const total = terms.reduce((a, b) => a + b, 0);
      const rawReal = total * spacing * .05;
      const requantized = clamp(round(rawReal / .02) - 4);
      const output = Math.max(-4, requantized), real = (output + 4) * .02;
      return {qx, qw, terms, total, rawReal, requantized, output, real};
    }
    const sumThrough = (d, n) => d.terms.slice(0, n).reduce((a, b) => a + b, 0);
    function setState(accumulator, moving) {
      const d = data();
      Object.assign(root.dataset, {stage: String(step), accumulator: String(accumulator), finalAccumulator: String(d.total), output: String(d.output), reconstructed: fmt(d.real), moving: String(moving), selectedValue: String(selected)});
      menu.value = step; scaleMenu.value = spacing;
      const phase = step === 0 ? 0 : step <= 2 ? 1 : step <= 5 ? 2 : 3;
      [...route.children].forEach((node, i) => { node.classList.toggle('is-current', i === phase); node.classList.toggle('is-done', i < phase); if (i === phase) node.setAttribute('aria-current', 'step'); else node.removeAttribute('aria-current'); });
      buttons.back.disabled = step === 0; buttons.next.disabled = step === 7;
      buttons.play.textContent = playing || animating && !paused ? 'Pause' : step === 7 && !selectedScene ? 'Replay' : 'Play';
      buttons.play.setAttribute('aria-pressed', String(playing || animating && !paused));
    }
    function resetScene() {
      board.setAttribute('viewBox', phone() ? '0 0 360 530' : '0 0 760 420');
      scene = shape('g'); particles = shape('g'); board.replaceChildren(defs, scene, particles);
    }
    function text(x, y, value, cls = '', anchor = 'middle', host = scene) { const n = shape('text', {x, y, class: cls, 'text-anchor': anchor}, value); host.append(n); return n; }
    function box(x, y, width, height, kind, host = scene) { const n = shape('rect', {x, y, width, height, rx: 7, class: `scene-box scene-${kind}`}); host.append(n); return n; }
    function line(points, cls = '', arrow = false, host = scene) { const n = shape('path', {d: points, class: `scene-line ${cls}`, fill: 'none'}); if (arrow) n.setAttribute('marker-end', `url(#${marker.id})`); host.append(n); return n; }
    function interactive(group, label, action) { group.setAttribute('role', 'button'); group.setAttribute('tabindex', '0'); group.setAttribute('aria-label', label); on(group, 'click', action); on(group, 'keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); action(); } }); }
    function inspect() { pause(); detail.open = true; summary.focus({preventScroll: true}); }
    function card(x, y, width, height, title, value, sub, kind, active = false, action = null) {
      const g = shape('g', {class: active ? 'scene-card is-selected' : 'scene-card'}); scene.append(g);
      box(x, y, width, height, kind, g); text(x + width / 2, y + 20, title, 'scene-small', 'middle', g);
      const val = text(x + width / 2, y + 47, value, 'scene-card-value', 'middle', g);
      const subNode = sub ? text(x + width / 2, y + height - 10, sub, 'scene-tiny', 'middle', g) : null;
      if (action) interactive(g, title + ': ' + value, action);
      return {g, value: val, sub: subNode, x: x + width / 2, y: y + height / 2};
    }
    function grid(value, scale, zero, gridY, heading) {
      const center = round(value / scale), represented = center * scale;
      const left = phone() ? 50 : 250, right = phone() ? 310 : 510, mid = (left + right) / 2, unit = (right - left) / 2;
      const originalX = mid + (value / scale - center) * unit;
      text(mid, gridY - 115, heading, 'scene-heading');
      line(`M${left - 12} ${gridY}H${right + 12}`, '', true);
      text(left - 25, gridY + 29, 'real', 'scene-tiny', 'end'); text(left - 25, gridY + 64, 'code', 'scene-tiny', 'end');
      for (let offset = -1; offset <= 1; offset++) {
        const px = mid + offset * unit, chosen = offset === 0;
        line(`M${px} ${gridY - 7}V${gridY + 8}`);
        text(px, gridY + 29, fmt((center + offset) * scale), chosen ? 'scene-tick chosen' : 'scene-tick');
        text(px, gridY + 64, String(clamp(center + offset + zero)), chosen ? 'scene-tick chosen' : 'scene-tick');
      }
      line(`M${originalX} ${gridY - 68}V${gridY - 9}`, 'scene-ghost');
      text(originalX, gridY - 78, 'original ' + fmt(value), 'scene-small');
      const dot = shape('g'); scene.append(dot); dot.append(shape('circle', {cx: originalX, cy: gridY, r: 8, class: 'scene-dot'}));
      const label = text(originalX, gridY - 25, fmt(value), 'scene-heading', 'middle', dot);
      const error = text(mid, gridY + 105, 'rounding change ' + signed(represented - value), 'scene-small');
      return {dot, label, error, originalX, mid, represented, y: gridY, codeY: gridY + 64};
    }
    function drawStart() {
      resetScene(); setState(0, false);
      const pw = phone(), centers = pw ? [70, 180, 290] : [170, 380, 590], width = pw ? 98 : 150;
      text(pw ? 180 : 380, 28, 'Activation array x · FP32', 'scene-heading');
      x.forEach((v, i) => card(centers[i] - width / 2, 48, width, 80, `x[${i}]`, fmt(v), '', 'quant'));
      text(pw ? 180 : 380, 178, 'Weight array w · FP32', 'scene-heading');
      w.forEach((v, i) => card(centers[i] - width / 2, 198, width, 80, `w[${i}]`, fmt(v), '', 'quant'));
      text(pw ? 180 : 380, 342, 'Two arrays → one dot product', 'scene-heading');
      text(pw ? 180 : 380, 385, '6 FP32 values · 24 operand bytes', 'scene-small');
      narration.textContent = 'Start with six floating-point values. Keep their identities while changing representation.';
      note.textContent = 'Play follows the whole story. Next illustrates one operation; the sequence menu jumps to an inspectable result.';
    }
    function selectValue(index) { clearMotion(); playing = false; selected = index; selectedScene = true; drawSettled(); }
    function drawQuant(index, completed, settled) {
      resetScene(); selected = index; setState(0, !settled);
      const d = data(), activation = step === 1, values = activation ? x : w, codes = activation ? d.qx : d.qw, scale = activation ? spacing : .05, zero = activation ? 3 : 0, name = activation ? 'x' : 'w';
      const src = [], dst = [], pw = phone();
      text(pw ? 180 : 80, pw ? 24 : 35, `${name} · FP32`, 'scene-heading');
      text(pw ? 180 : 680, pw ? 416 : 35, `${name} · INT8 codes`, 'scene-heading');
      values.forEach((v, i) => {
        const sx = pw ? 20 + i * 110 : 15, sy = pw ? 42 : 65 + i * 110, dx = pw ? sx : 615, dy = pw ? 432 : sy, width = pw ? 100 : 130;
        src.push(card(sx, sy, width, 76, `${name}[${i}]`, fmt(v), 'original retained', 'quant', i === index, () => selectValue(i)));
        dst.push(card(dx, dy, width, 76, `${name}[${i}]`, i < completed ? String(codes[i]) : '—', i < completed ? 'represents ' + fmt((codes[i] - zero) * scale) : 'not written', 'quant', i === index, () => selectValue(i)));
      });
      const qgrid = grid(values[index], scale, zero, pw ? 270 : 190, `Nearest level for ${name}[${index}]`);
      if (settled) { qgrid.dot.style.transform = `translate(${qgrid.mid - qgrid.originalX}px,0)`; qgrid.label.textContent = fmt(qgrid.represented); }
      else qgrid.error.style.opacity = 0;
      narration.textContent = settled ? `${name}[${index}]: ${fmt(values[index])} is represented as ${fmt(qgrid.represented)}, stored as code ${codes[index]}.` : `Follow ${name}[${index}] to its nearest real level, then store that level’s integer code.`;
      note.textContent = (activation ? '' : 'Weights may be quantized offline. ') + 'Select an input to inspect its rounding. Stored codes label real levels.';
      return {grid: qgrid, src, dst, values, codes, scale, zero, name};
    }
    function drawReduction(done, settled) {
      resetScene(); const d = data(), i = step - 3, pw = phone(), previous = sumThrough(d, i), total = sumThrough(d, i + 1);
      selected = i; setState(settled ? total : previous, !settled);
      const centers = pw ? [70, 180, 290] : [75, 180, 285], width = pw ? 96 : 90;
      text(pw ? 180 : 180, 24, `Pair ${i + 1} of 3 · stored INT8 operands`, 'scene-heading');
      const sourceX = [], sourceW = [];
      for (let j = 0; j < 3; j++) {
        sourceX.push(card(centers[j] - width / 2, 43, width, 68, `x[${j}]`, String(d.qx[j]), '', 'quant', j === i, () => seek(j + 3)));
        sourceW.push(card(centers[j] - width / 2, pw ? 128 : 143, width, 68, `w[${j}]`, String(d.qw[j]), '', 'quant', j === i, () => seek(j + 3)));
      }
      if (pw) {
        text(180, 227, `x: ${d.qx[i]} − 3 = ${d.qx[i] - 3}`, 'scene-small');
        text(180, 249, `w: ${d.qw[i]} − 0 = ${d.qw[i]}`, 'scene-small');
      } else text(180, 243, `x: ${d.qx[i]} − 3 = ${d.qx[i] - 3}   ·   w: ${d.qw[i]} − 0 = ${d.qw[i]}`, 'scene-small');
      const compute = pw ? {x: 20, y: 267, w: 140, h: 118} : {x: 365, y: 72, w: 160, h: 145};
      const register = pw ? {x: 200, y: 267, w: 140, h: 118} : {x: 570, y: 72, w: 170, h: 145};
      const cg = shape('g'); scene.append(cg); box(compute.x, compute.y, compute.w, compute.h, 'compute', cg);
      text(compute.x + compute.w / 2, compute.y + 26, 'Multiply', 'scene-heading', 'middle', cg);
      const centered = text(compute.x + compute.w / 2, compute.y + 67, settled ? `${d.qx[i] - 3} × ${d.qw[i]}` : 'read pair', 'scene-operation', 'middle', cg);
      const product = text(compute.x + compute.w / 2, compute.y + 100, settled ? `product ${d.terms[i]}` : 'remove zero-points', 'scene-small', 'middle', cg);
      interactive(cg, 'Inspect zero-point correction and multiplication', inspect);
      const rg = shape('g'); scene.append(rg); box(register.x, register.y, register.w, register.h, 'acc', rg);
      text(register.x + register.w / 2, register.y + 26, 'INT32 register', 'scene-heading', 'middle', rg);
      const regValue = text(register.x + register.w / 2, register.y + 73, String(settled ? total : previous), 'scene-register', 'middle', rg);
      const regStatus = text(register.x + register.w / 2, register.y + register.h - 16, 'retained sum', 'scene-small', 'middle', rg);
      interactive(rg, 'Inspect the retained sum', inspect);
      const y = compute.y + compute.h / 2;
      line(`M${compute.x + compute.w} ${y}H${register.x}`, '', true);
      const trailY = pw ? 448 : 300;
      text(pw ? 180 : 380, trailY - 24, 'Added products · visual trace', 'scene-small');
      const history = [];
      for (let j = 0; j < 3; j++) history.push(card((pw ? [70, 180, 290] : [180, 380, 580])[j] - (pw ? 48 : 75), trailY, pw ? 96 : 150, 68, `pair ${j + 1}`, j < done ? String(d.terms[j]) : '—', '', 'acc', j === i));
      narration.textContent = settled ? `The register now holds ${total}.${i === 2 ? ' The reduction is complete.' : ' Keep it for the next pair.'}` : `Pair ${i + 1}: the register retains ${previous} while the next product is formed.`;
      note.textContent = total > 127 && settled ? `${total} exceeds signed INT8’s maximum, 127. The sum stays wide until the reduction finishes.` : 'This small sum may fit INT8; longer reductions may not. The accumulator stays wide until the reduction finishes.';
      return {sourceX, sourceW, compute, register, centered, product, regValue, regStatus, history, previous, total, i};
    }
    function drawOutput(settled) {
      resetScene(); const d = data(), pw = phone(), cx = pw ? 180 : 380;
      setState(d.total, !settled);
      if (step === 6) {
        card(pw ? 15 : 15, pw ? 25 : 100, pw ? 145 : 140, 96, 'INT32 · retained', String(d.total), '', 'acc', false, inspect);
        card(pw ? 200 : 605, pw ? 25 : 100, pw ? 145 : 140, 96, 'Real sum', fmt(d.rawReal), `unit ${fmt(spacing * .05)}`, 'acc', false, inspect);
        const qgrid = grid(d.rawReal, .02, -4, pw ? 280 : 220, 'Round onto the output grid');
        if (settled) { qgrid.dot.style.transform = `translate(${qgrid.mid - qgrid.originalX}px,0)`; qgrid.label.textContent = fmt(d.real); qgrid.error.textContent = Math.abs(d.rawReal - d.real) < 1e-9 ? 'Already on an output level · no new rounding error' : 'output rounding change ' + signed(d.real - d.rawReal); }
        else qgrid.error.style.opacity = 0;
        text(cx, pw ? 465 : 385, settled ? `Stored output code ${d.requantized} · real value ${fmt(d.real)}` : 'Output code is written after rounding', 'scene-heading');
        narration.textContent = settled ? `Sum ${d.total} represents ${fmt(d.rawReal)}; output code ${d.requantized} represents ${fmt(d.real)}.` : 'Interpret the wide sum in real units, then round onto the output grid.';
        note.textContent = 'The accumulator is retained while requantization reads it. The output uses a different spacing and zero-point.';
        return {grid: qgrid};
      }
      const positions = pw ? [{x: 20, y: 40, w: 145}, {x: 200, y: 40, w: 140}, {x: 95, y: 225, w: 170}] : [{x: 25, y: 90, w: 175}, {x: 290, y: 90, w: 180}, {x: 560, y: 90, w: 175}];
      const a = card(positions[0].x, positions[0].y, positions[0].w, 110, 'Requantized INT8', String(d.requantized), '', 'output');
      const gate = card(positions[1].x, positions[1].y, positions[1].w, 110, 'ReLU', settled ? String(d.output) : '—', 'stored zero: −4', 'activation', false, inspect);
      const out = card(positions[2].x, positions[2].y, positions[2].w, 110, 'Final INT8 output', settled ? String(d.output) : '—', settled ? 'represents ' + fmt(d.real) : 'not written', 'output', false, inspect);
      line(`M${positions[0].x + positions[0].w} ${a.y}H${positions[1].x}`, '', true);
      if (pw) line(`M${gate.x} ${positions[1].y + 110}V185H${out.x}V${positions[2].y}`, '', true);
      else line(`M${positions[1].x + positions[1].w} ${gate.y}H${positions[2].x}`, '', true);
      const factsY = pw ? 385 : 290;
      text(cx, factsY, `Float reference ${fmt(.4086)} → reconstructed ${fmt(d.real)}`, 'scene-heading');
      text(cx, factsY + 36, `Absolute error ${fmt(Math.abs(d.real - .4086))}`, 'scene-small');
      text(cx, factsY + 90, 'Operand values: 24 bytes FP32 → 6 bytes INT8', 'scene-small');
      text(cx, factsY + 118, 'INT32 register: 4 bytes · metadata and workspace excluded', 'scene-tiny');
      narration.textContent = settled ? `ReLU preserves this positive value. Output code ${d.output} represents ${fmt(d.real)}.` : `Read code ${d.requantized}, pass ReLU, then write the final output.`;
      note.textContent = 'Change activation spacing to compare the same inputs. Smaller error here does not establish a speedup or a universally better scale.';
      return {a, gate, out};
    }
    function updateMath() {
      const d = data();
      if (step === 0) equation.textContent = 'x = [0.31, −0.52, 0.77]; w = [0.42, −0.18, 0.24]. Float dot product = 0.4086.';
      else if (step <= 2) { const sx = step === 1 ? spacing : .05, z = step === 1 ? 3 : 0, v = (step === 1 ? x : w)[selected], q = (step === 1 ? d.qx : d.qw)[selected]; equation.textContent = `Original ${fmt(v)} ÷ spacing ${sx} = ${fmt(v / sx)}; round to ${round(v / sx)}, add zero-point ${z}, and clamp → code ${q}. Reconstruction: ${sx} × (${q} − ${z}) = ${fmt(sx * (q - z))}.`; }
      else if (step <= 5) { const i = step - 3, prev = sumThrough(d, i); equation.textContent = `Pair ${i + 1}: (${d.qx[i]} − 3) × (${d.qw[i]} − 0) = ${d.terms[i]}. Previous sum ${prev}; after arrival ${prev} + ${d.terms[i]} = ${sumThrough(d, i + 1)}. One accumulator unit represents ${fmt(spacing * .05)} real units.`; }
      else equation.textContent = `Wide sum ${d.total} × ${fmt(spacing * .05)} = ${fmt(d.rawReal)}. Output code: clamp(round(${fmt(d.rawReal)} / 0.02) − 4) = ${d.requantized}. ReLU: max(−4, ${d.requantized}) = ${d.output}. Reconstruction: (${d.output} + 4) × 0.02 = ${fmt(d.real)}. Float-reference error ${fmt(Math.abs(d.real - .4086))}.`;
    }
    function drawSettled() {
      if (step === 0) drawStart(); else if (step <= 2) drawQuant(selected, 3, true); else if (step <= 5) drawReduction(step - 2, true); else drawOutput(true);
      updateMath();
    }
    function clearMotion() { version++; clearTimeout(holdTimer); holdTimer = null; animations.forEach(a => a.cancel()); animations = []; animating = false; paused = false; }
    function track(animation) { animations.push(animation); animation.finished.then(() => { animations = animations.filter(a => a !== animation); }).catch(() => {}); if (paused) animation.pause(); return animation; }
    function cue(ms, fn) { const v = version; if (ms === 0) { fn(); return; } const a = track(clock.animate([{opacity: 0}, {opacity: 0}], {duration: ms})); a.finished.then(() => { if (!dead && v === version) fn(); }).catch(() => {}); }
    function tween(node, frames, duration, delay = 0) { return track(node.animate(frames, {duration, delay, easing: 'ease-in-out', fill: 'forwards'})); }
    function packet(value, tag, points, duration) {
      const g = shape('g', {class: 'scene-packet'}); particles.append(g);
      g.append(shape('rect', {x: -35, y: -24, width: 70, height: 48, rx: 8})); text(0, -6, tag, 'scene-tiny', 'middle', g); text(0, 17, String(value), 'scene-packet-value', 'middle', g);
      tween(g, points.map(([x, y]) => ({transform: `translate(${x}px,${y}px)`})), duration);
      return g;
    }
    function hold() { holdTimer = setTimeout(() => { if (!playing || dead) return; if (step === 7) { playing = false; setState(data().total, false); } else startStep(step + 1); }, 1200); }
    function endStage() { animations.forEach(a => a.cancel()); animations = []; animating = false; paused = false; selectedScene = false; drawSettled(); if (playing) hold(); }
    function startStep(next) {
      const keepPlaying = playing; clearMotion(); playing = keepPlaying; step = next; animating = true; selectedScene = false;
      if (reduced.matches) { endStage(); return; }
      const d = data();
      if (step <= 2) {
        for (let index = 0; index < 3; index++) cue(index * 2600, () => {
          const view = drawQuant(index, index, false); updateMath();
          view.grid.dot.style.opacity = 0;
          const read = packet(fmt(view.values[index]), view.name + '[' + index + '] FP32', [[view.src[index].x, view.src[index].y], [view.grid.originalX, view.grid.y]], 250);
          cue(300, () => { read.remove(); view.grid.dot.style.opacity = 1; });
          tween(view.grid.dot, [{transform: 'translate(0px,0px)'}, {transform: `translate(${view.grid.mid - view.grid.originalX}px,0px)`}], 650, 350);
          cue(1050, () => { view.grid.label.textContent = fmt(view.grid.represented); view.grid.error.style.opacity = 1; });
          cue(1300, () => { const p = packet(view.codes[index], view.name + '[' + index + '] INT8', [[view.grid.mid, view.grid.codeY], [view.dst[index].x, view.dst[index].y]], 650); cue(700, () => { p.remove(); view.dst[index].value.textContent = String(view.codes[index]); view.dst[index].sub.textContent = 'represents ' + fmt(view.grid.represented); }); });
        });
        cue(7800, endStage);
      } else if (step <= 5) {
        const view = drawReduction(step - 3, false); updateMath();
        const center = [view.compute.x + view.compute.w / 2, view.compute.y + view.compute.h / 2], p1 = packet(d.qx[view.i], `x[${view.i}] code`, [[view.sourceX[view.i].x, view.sourceX[view.i].y], [view.compute.x, center[1] - 25]], 550);
        cue(600, () => { p1.remove(); view.centered.textContent = `${d.qx[view.i]} − 3 → ${d.qx[view.i] - 3}`; const p2 = packet(d.qw[view.i], `w[${view.i}] code`, [[view.sourceW[view.i].x, view.sourceW[view.i].y], [view.compute.x, center[1] + 25]], 500); cue(550, () => { p2.remove(); view.centered.textContent = `${d.qx[view.i] - 3} × ${d.qw[view.i]}`; view.product.textContent = `product ${d.terms[view.i]}`; }); });
        cue(1300, () => { const p = packet(d.terms[view.i], 'product', [[view.compute.x + view.compute.w, center[1]], [view.register.x, center[1]]], 650); cue(700, () => { p.remove(); view.regValue.textContent = `${view.previous} + ${d.terms[view.i]}`; view.regValue.classList.add('is-adding'); view.regStatus.textContent = 'adding product'; narration.textContent = `Product ${d.terms[view.i]} arrived. Add it to the retained ${view.previous}.`; }); });
        cue(2700, endStage);
      } else if (step === 6) {
        const view = drawOutput(false); updateMath();
        tween(view.grid.dot, [{transform: 'translate(0px,0px)'}, {transform: `translate(${view.grid.mid - view.grid.originalX}px,0px)`}], 850, 350);
        cue(1250, () => { view.grid.label.textContent = fmt(d.real); view.grid.error.style.opacity = 1; });
        cue(2400, endStage);
      } else {
        const view = drawOutput(false); updateMath();
        const points = phone() ? [[view.a.x, view.a.y], [view.gate.x, view.gate.y], [view.gate.x, view.gate.y], [view.gate.x, 185], [view.out.x, 185], [view.out.x, view.out.y]] : [[view.a.x, view.a.y], [view.gate.x, view.gate.y], [view.gate.x, view.gate.y], [view.out.x, view.out.y]];
        packet(d.output, 'INT8', points, 1800);
        cue(750, () => { view.gate.value.textContent = String(d.output); });
        cue(2000, endStage);
      }
    }
    function seek(next) { clearMotion(); playing = false; step = next; selectedScene = true; drawSettled(); }
    function pause() { playing = false; paused = true; clearTimeout(holdTimer); animations.forEach(a => a.pause()); setState(Number(root.dataset.accumulator), animating); }
    on(buttons.play, 'click', () => {
      if (playing || animating && !paused) { pause(); return; }
      playing = true;
      if (animating) { paused = false; animations.forEach(a => a.play()); setState(Number(root.dataset.accumulator), true); }
      else if (selectedScene && step > 0) startStep(step);
      else startStep(step === 7 ? 1 : step + 1);
    });
    on(buttons.next, 'click', () => { playing = false; startStep(Math.min(7, step + 1)); });
    on(buttons.back, 'click', () => seek(Math.max(0, step - 1)));
    on(buttons.restart, 'click', () => { selected = 0; seek(0); });
    on(menu, 'change', () => seek(Number(menu.value)));
    on(scaleMenu, 'change', () => { spacing = Number(scaleMenu.value); seek(step); });
    on(root, 'mlsys-view-change', e => { if (e.detail?.view !== 'explore') pause(); });
    on(document, 'visibilitychange', () => { if (document.hidden) pause(); }); on(window, 'beforeprint', pause);
    on(window, 'resize', () => seek(step)); on(reduced, 'change', () => seek(step));
    const observer = new IntersectionObserver(entries => { if (!entries[0].isIntersecting) pause(); }); observer.observe(root);
    drawSettled();
    return {pause, destroy() { dead = true; playing = false; clearMotion(); observer.disconnect(); abort.abort(); root.replaceChildren(); }};
  };
})();
