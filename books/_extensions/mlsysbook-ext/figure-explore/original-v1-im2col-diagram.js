(() => {
  'use strict';
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons['original-v1-im2col-diagram'] = async (root, { outer, verifiedArtwork }) => {
    if (!verifiedArtwork || !window.MLSysOriginalPlayer) {
      throw new Error('Verified original artwork and shared player required');
    }
    root.classList.add('original-v1-im2col-diagram');

    const NS = 'http://www.w3.org/2000/svg';
    const svgNode = (tag, attrs, txt) => {
      const n = document.createElementNS(NS, tag);
      if (attrs) Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, String(v)));
      if (txt != null) n.textContent = txt;
      return n;
    };
    const htmlNode = (tag, cls, txt) => {
      const n = document.createElement(tag);
      if (cls) n.className = cls;
      if (txt != null) n.textContent = txt;
      return n;
    };

    const patchIndices = r => {
      const py = Math.floor(r / 2);
      const px = r % 2;
      const out = [];
      for (let c = 0; c < 2; c += 1) {
        for (let dy = 0; dy < 2; dy += 1) {
          for (let dx = 0; dx < 2; dx += 1) {
            out.push(c * 9 + (py + dy) * 3 + (px + dx));
          }
        }
      }
      return out;
    };
    const rows = [0, 1, 2, 3].map(r => patchIndices(r).map(k => k + 1));
    const weights = [1, 2, 3, 4, 5, 6, 7, 8];
    const outputs = rows.map(vals => vals.reduce((acc, v, k) => acc + v * weights[k], 0)); // [356, 392, 464, 500]

    const steps = [
      {
        label: 'Inspect the 2-channel input and stacked kernel',
        description: 'Two 3 × 3 input channels hold 18 scalars (1–9 and 10–18) and two 2 × 2 filter channels stack into an 8 × 1 weight column (1–8). The 4 × 8 im2col buffer reserves 32 slots.',
        evidence: { rows: 0, copies: 0, outputs: 0 }
      }
    ];
    for (let r = 0; r < 4; r += 1) {
      const py = Math.floor(r / 2);
      const px = r % 2;
      steps.push({
        label: `Unfold patch (${py}, ${px}) into row ${r}`,
        description: `Copy the 2 × 2 window at (${py}, ${px}) from channel 0 ([${rows[r].slice(0, 4).join(', ')}]) and channel 1 ([${rows[r].slice(4, 8).join(', ')}]) into row ${r}: [${rows[r].join(', ')}] (${(r + 1) * 8} of 32 slots materialized).`,
        evidence: { rows: r + 1, copies: (r + 1) * 8, outputs: 0 }
      });
    }
    for (let r = 0; r < 4; r += 1) {
      steps.push({
        label: `Multiply row ${r} by W → output ${outputs[r]}`,
        description: `Dot-product row ${r} ([${rows[r].join(', ')}]) with stacked weights [1, 2, 3, 4, 5, 6, 7, 8] to obtain output[${r}] = ${outputs[r]} (${r + 1} of 4 outputs complete).`,
        evidence: { rows: 4, copies: 32, outputs: r + 1 }
      });
    }

    // Exact viewBox-space centers on verifiedArtwork (0 0 481.959 201.628)
    const CELL = 22.636;
    const inPos = k => {
      const ch = Math.floor(k / 9);
      const rem = k % 9;
      const r = Math.floor(rem / 3);
      const c = rem % 3;
      return [3.85 + (c + 0.5) * CELL, (ch === 0 ? 22.93 : 121.54) + (r + 0.5) * CELL];
    };
    const matPos = (r, c) => [226.57 + (c + 0.5) * CELL, 61.90 + (r + 0.5) * CELL];
    const wPos = k => [455.61 + 0.5 * CELL, 8.77 + (k + 0.5) * CELL];

    const svg = verifiedArtwork.cloneNode(true);
    svg.removeAttribute('width');
    svg.removeAttribute('height');
    svg.setAttribute('viewBox', '-14 -12 510 226');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'im2col patch unfolding into a 4 by 8 GEMM matrix');
    svg.prepend(svgNode('rect', { x: -14, y: -12, width: 510, height: 226, fill: '#ffffff' }));

    const overlay = svgNode('g', { class: 'im2col-overlay' });
    // Mask uninitialized rows of the 4x8 matrix cleanly until unfolded
    const rowMasks = [0, 1, 2, 3].map(r => svgNode('rect', {
      x: (226.57 + 0.8).toFixed(2),
      y: (61.90 + r * CELL + 0.8).toFixed(2),
      width: (8 * CELL - 1.6).toFixed(2),
      height: (CELL - 1.6).toFixed(2),
      fill: '#f8fafc',
      'fill-opacity': '0.86'
    }));
    const winCh0 = svgNode('rect', { width: (2 * CELL).toFixed(2), height: (2 * CELL).toFixed(2), rx: 2, fill: 'none', stroke: '#006395', 'stroke-width': 2.6 });
    const winCh1 = svgNode('rect', { width: (2 * CELL).toFixed(2), height: (2 * CELL).toFixed(2), rx: 2, fill: 'none', stroke: '#E67817', 'stroke-width': 2.6 });
    const activeRowRect = svgNode('rect', { x: 225.2, width: (8 * CELL + 2.8).toFixed(2), height: (CELL + 2.4).toFixed(2), rx: 2.5, fill: 'none', stroke: '#006395', 'stroke-width': 2.6 });
    const weightColRect = svgNode('rect', { x: 454.2, y: 7.4, width: (CELL + 2.8).toFixed(2), height: (8 * CELL + 2.8).toFixed(2), rx: 2.5, fill: 'none', stroke: '#008F45', 'stroke-width': 2.6 });
    const outBadge = svgNode('g', {});
    const outBadgeBg = svgNode('rect', { x: 318, y: 162, width: 126, height: 24, rx: 4, fill: '#D9F2E4', stroke: '#008F45', 'stroke-width': 1.6 });
    const outBadgeTxt = svgNode('text', { x: 381, y: 174.5, 'text-anchor': 'middle', 'dominant-baseline': 'middle', 'font-family': 'system-ui, sans-serif', 'font-size': 9.5, 'font-weight': '700', fill: '#0b4f2c' });
    outBadge.append(outBadgeBg, outBadgeTxt);
    const flights = svgNode('g', { 'aria-hidden': 'true' });
    const focusBox = svgNode('rect', { fill: 'none', stroke: 'none', 'pointer-events': 'none', 'data-mlsys-focus': '' });
    overlay.append(...rowMasks, winCh0, winCh1, activeRowRect, weightColRect, outBadge, flights, focusBox);
    svg.append(overlay);

    const surface = htmlNode('div', 'v1-art-surface');
    surface.append(svg);
    const status = htmlNode('p', 'v1-status');
    const eq = htmlNode('p', 'v1-eq');
    const outStrip = htmlNode('div', 'v1-out-strip');
    const outCards = [0, 1, 2, 3].map(r => {
      const c = htmlNode('div', 'v1-out-cell is-empty');
      outStrip.append(c);
      return c;
    });

    function paintEndpoint(i) {
      flights.replaceChildren();
      const ev = steps[i].evidence;
      rowMasks.forEach((m, r) => m.setAttribute('visibility', r >= ev.rows ? 'visible' : 'hidden'));

      if (i >= 1 && i <= 4) {
        const r = i - 1;
        const py = Math.floor(r / 2);
        const px = r % 2;
        winCh0.setAttribute('visibility', 'visible');
        winCh1.setAttribute('visibility', 'visible');
        winCh0.setAttribute('x', (3.85 + px * CELL).toFixed(2));
        winCh0.setAttribute('y', (22.93 + py * CELL).toFixed(2));
        winCh1.setAttribute('x', (3.85 + px * CELL).toFixed(2));
        winCh1.setAttribute('y', (121.54 + py * CELL).toFixed(2));
        activeRowRect.setAttribute('visibility', 'visible');
        activeRowRect.setAttribute('y', (61.90 + r * CELL - 1.2).toFixed(2));
        activeRowRect.setAttribute('stroke', '#006395');
        weightColRect.setAttribute('visibility', 'hidden');
        outBadge.setAttribute('visibility', 'hidden');
        focusBox.setAttribute('x', '2');
        focusBox.setAttribute('y', '20');
        focusBox.setAttribute('width', '410');
        focusBox.setAttribute('height', '170');
      } else if (i >= 5) {
        const r = i - 5;
        winCh0.setAttribute('visibility', 'hidden');
        winCh1.setAttribute('visibility', 'hidden');
        activeRowRect.setAttribute('visibility', 'visible');
        activeRowRect.setAttribute('y', (61.90 + r * CELL - 1.2).toFixed(2));
        activeRowRect.setAttribute('stroke', '#008F45');
        weightColRect.setAttribute('visibility', 'visible');
        outBadge.setAttribute('visibility', 'visible');
        outBadgeTxt.textContent = `row ${r} · W = ${outputs[r]}`;
        focusBox.setAttribute('x', '222');
        focusBox.setAttribute('y', '6');
        focusBox.setAttribute('width', '260');
        focusBox.setAttribute('height', '186');
      } else {
        winCh0.setAttribute('visibility', 'hidden');
        winCh1.setAttribute('visibility', 'hidden');
        activeRowRect.setAttribute('visibility', 'hidden');
        weightColRect.setAttribute('visibility', 'hidden');
        outBadge.setAttribute('visibility', 'hidden');
        focusBox.setAttribute('x', '4');
        focusBox.setAttribute('y', '8');
        focusBox.setAttribute('width', '476');
        focusBox.setAttribute('height', '184');
      }

      status.textContent = `Input: 18 scalars · Unfolded buffer: ${ev.copies}/32 slots (${(ev.copies / 18).toFixed(2)}× input size) · Completed outputs: ${ev.outputs}/4.`;
      eq.textContent = i >= 5
        ? `Dot product (row ${i - 5}): ` + rows[i - 5].map((v, k) => `${v}×${k + 1}`).join(' + ') + ` = ${outputs[i - 5]}`
        : i >= 1
          ? `Row ${i - 1} unfolded values: [${rows[i - 1].join(', ')}] (center scalars 5 and 14 appear in all 4 rows).`
          : 'Unfolding duplicates overlapping 2 × 2 spatial patches so all 4 outputs execute as one 4 × 8 by 8 × 1 GEMM.';

      outCards.forEach((c, r) => {
        const done = r < ev.outputs;
        const active = i >= 5 && r === i - 5;
        c.className = 'v1-out-cell' + (done ? '' : ' is-empty') + (active ? ' is-active' : '');
        c.textContent = done ? `out[${r}] = ${outputs[r]}` : `out[${r}] = —`;
      });

      root.dataset.lessonState = String(i);
      root.dataset.stateEvidence = JSON.stringify(ev);
    }

    return window.MLSysOriginalPlayer.create({
      root,
      outer,
      layout: 'compact',
      intro: 'Unfold 4 overlapping 2 × 2 patches from 18 input scalars into a 4 × 8 matrix and multiply by 8 weights.',
      title: 'Which input values are copied into each unfolded row?',
      scope: 'Exact source values: channels 1–9 and 10–18; two 2 × 2 kernels stack into weights 1–8. Valid stride-one lowering creates four rows of eight values. Explicit im2col materializes the 32-slot buffer; implicit GEMM streams addresses on the fly.',
      steps,
      render(i, { stage, details }) {
        if (!surface.isConnected) stage.replaceChildren(surface);
        if (details && !status.isConnected) details.replaceChildren(status, eq, outStrip);
        paintEndpoint(i);
      },
      transition({ to, onFrame }) {
        flights.replaceChildren();
        if (to >= 1 && to <= 4) {
          const r = to - 1;
          const idx = patchIndices(r);
          const dots = idx.map((k, col) => {
            const g = svgNode('g', {});
            const c = svgNode('circle', { r: 6.2, fill: col < 4 ? '#006395' : '#E67817', stroke: '#fff', 'stroke-width': 1.2 });
            const t = svgNode('text', { 'text-anchor': 'middle', 'dominant-baseline': 'central', 'font-family': 'system-ui, sans-serif', 'font-size': 6.6, 'font-weight': '700', fill: '#fff' }, String(k + 1));
            g.append(c, t);
            flights.append(g);
            return { g, a: inPos(k), b: matPos(r, col) };
          });
          onFrame(p => {
            dots.forEach(({ g, a, b }) => {
              const x = a[0] + (b[0] - a[0]) * p;
              const y = a[1] + (b[1] - a[1]) * p;
              g.setAttribute('transform', `translate(${x.toFixed(2)} ${y.toFixed(2)})`);
            });
          });
        } else if (to >= 5) {
          const r = to - 5;
          const dots = weights.map((w, col) => {
            const g = svgNode('g', {});
            const c = svgNode('circle', { r: 5.5, fill: '#008F45', stroke: '#fff', 'stroke-width': 1.1 });
            g.append(c);
            flights.append(g);
            return { g, a: matPos(r, col), b: wPos(col) };
          });
          onFrame(p => {
            dots.forEach(({ g, a, b }) => {
              const x = a[0] + (b[0] - a[0]) * p;
              const y = a[1] + (b[1] - a[1]) * p;
              g.setAttribute('transform', `translate(${x.toFixed(2)} ${y.toFixed(2)})`);
            });
          });
        }
      }
    });
  };
})();
