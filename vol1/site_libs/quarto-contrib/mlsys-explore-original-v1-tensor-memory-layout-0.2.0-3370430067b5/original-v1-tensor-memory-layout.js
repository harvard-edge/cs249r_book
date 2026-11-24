(() => {
  'use strict';
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons['original-v1-tensor-memory-layout'] = async (root, { outer, verifiedArtwork }) => {
    if (!verifiedArtwork || !window.MLSysOriginalPlayer) {
      throw new Error('Verified original artwork and shared player required');
    }
    root.classList.add('original-v1-tensor-memory-layout');

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

    // Exact coordinates on verifiedArtwork (0 0 425.994 130.37)
    const logBox = (i, j) => [1.22 + j * 47.96, 15.32 + i * 33.85, 42.32, 28.21];
    const rowBox = k => [238.58 + k * 31.032, 15.32, 31.03, 22.57];
    const colBox = k => [238.58 + k * 31.032, 71.24, 31.03, 22.57];
    const center = b => [b[0] + b[2] / 2, b[1] + b[3] / 2];

    const steps = [
      {
        label: 'Two 1D layouts for one 2 × 3 tensor',
        description: 'The 2 × 3 logical tensor holds values 1–6. Row-major storage [1, 2, 3, 4, 5, 6] uses strides [3, 1], whereas column-major storage [1, 4, 2, 5, 3, 6] uses strides [1, 2].',
        mode: 'overview',
        evidence: { shape: [2, 3], rowStrides: [3, 1], colStrides: [1, 2] }
      },
      {
        label: 'Index (0, 1) → scalar 2',
        description: 'At (i, j) = (0, 1), row-major offset = 0×3 + 1×1 = 1, while column-major offset = 0×1 + 1×2 = 2. Different 1D addresses return the same logical element 2.',
        mode: 'point', i: 0, j: 1, val: 2, rOff: 1, cOff: 2,
        evidence: { i: 0, j: 1, rowOffset: 1, columnOffset: 2, value: 2 }
      },
      {
        label: 'Index (1, 0) → scalar 4',
        description: 'At (i, j) = (1, 0), row-major jumps one full 3-element row to offset 1×3 + 0×1 = 3, while column-major steps +1 to offset 1×1 + 0×2 = 1; both return 4.',
        mode: 'point', i: 1, j: 0, val: 4, rOff: 3, cOff: 1,
        evidence: { i: 1, j: 0, rowOffset: 3, columnOffset: 1, value: 4 }
      },
      {
        label: 'Index (1, 2) → scalar 6',
        description: 'At the corner (1, 2), row-major offset = 1×3 + 2×1 = 5 and column-major offset = 1×1 + 2×2 = 5; both layouts converge at the final slot holding 6.',
        mode: 'point', i: 1, j: 2, val: 6, rOff: 5, cOff: 5,
        evidence: { i: 1, j: 2, rowOffset: 5, columnOffset: 5, value: 6 }
      },
      {
        label: 'Sweep row 0: contiguous vs stride-2',
        description: 'Scanning row 0 ((0,0) → (0,1) → (0,2)) walks contiguous offsets [0, 1, 2] (step +1) in row-major memory, but hops across offsets [0, 2, 4] (step +2) in column-major memory.',
        mode: 'rowSweep',
        evidence: { rowOffsets: [0, 1, 2], colOffsets: [0, 2, 4], rowStep: 1, colStep: 2 }
      },
      {
        label: 'Sweep column 1: stride-3 vs contiguous',
        description: 'Scanning column 1 ((0,1) → (1,1)) jumps across offsets [1, 4] (step +3) in row-major memory, whereas column-major memory reads adjacent offsets [2, 3] (step +1).',
        mode: 'colSweep',
        evidence: { rowOffsets: [1, 4], colOffsets: [2, 3], rowStep: 3, colStep: 1 }
      },
      {
        label: 'Zero-copy transpose swaps strides [3, 1] → [1, 3]',
        description: 'Calling .transpose() on the row-major tensor changes metadata to shape (3, 2) and strides [1, 3] without copying a single element: transposed index (2, 1) resolves to 2×1 + 1×3 = offset 5 (value 6).',
        mode: 'transpose',
        evidence: { shape: [3, 2], transposedStrides: [1, 3], copiedElements: 0 }
      }
    ];

    const svg = verifiedArtwork.cloneNode(true);
    svg.removeAttribute('width');
    svg.removeAttribute('height');
    svg.setAttribute('viewBox', '-12 -10 450 162');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', '2 by 3 logical tensor mapped to row-major and column-major 1D memory layouts');
    svg.prepend(svgNode('rect', { x: -12, y: -10, width: 450, height: 162, fill: '#ffffff' }));

    const overlay = svgNode('g', { class: 'tml-overlay' });
    const dynamic = svgNode('g', {});
    const flights = svgNode('g', { 'aria-hidden': 'true' });
    const focusBox = svgNode('rect', { fill: 'none', stroke: 'none', 'pointer-events': 'none', 'data-mlsys-focus': '' });
    overlay.append(dynamic, flights, focusBox);
    svg.append(overlay);

    const surface = htmlNode('div', 'v1-art-surface');
    surface.append(svg);
    const status = htmlNode('p', 'v1-status');
    const ledger = htmlNode('div', 'v1-ledger');
    const mkCard = (kind, title) => {
      const c = htmlNode('div', 'v1-card');
      c.dataset.kind = kind;
      const h = htmlNode('strong', '', title);
      const b = htmlNode('span', '');
      c.append(h, b);
      ledger.append(c);
      return { h, b };
    };
    const cards = {
      logical: mkCard('log', 'Logical View Metadata'),
      row: mkCard('row', 'Row-Major (Strides [3, 1])'),
      col: mkCard('col', 'Column-Major (Strides [1, 2])')
    };

    const drawRect = (b, stroke, fill = 'none', dash = '') => {
      const attrs = {
        x: (b[0] - 1.2).toFixed(2),
        y: (b[1] - 1.2).toFixed(2),
        width: (b[2] + 2.4).toFixed(2),
        height: (b[3] + 2.4).toFixed(2),
        rx: 2.5,
        fill,
        stroke,
        'stroke-width': 2.5
      };
      if (dash) attrs['stroke-dasharray'] = dash;
      return svgNode('rect', attrs);
    };

    function paintEndpoint(idx) {
      dynamic.replaceChildren();
      flights.replaceChildren();
      const s = steps[idx];

      if (s.mode === 'point') {
        const lb = logBox(s.i, s.j);
        const rb = rowBox(s.rOff);
        const cb = colBox(s.cOff);
        dynamic.append(
          drawRect(lb, '#006395', 'rgba(0,99,149,0.10)'),
          drawRect(rb, '#CB202D', 'rgba(203,32,45,0.10)'),
          drawRect(cb, '#008F45', 'rgba(0,143,69,0.10)')
        );
        const lc = center(lb);
        const rc = center(rb);
        const cc = center(cb);
        const sx = (s.j === 2 ? lb[0] + lb[2] + 1.8 : lc[0]).toFixed(1);
        const syR = (s.j === 2 ? lc[1] : (s.i === 0 ? lb[1] + lb[3] + 1.8 : lb[1] - 1.8)).toFixed(1);
        const syC = (s.j === 2 ? lc[1] : lb[1] + lb[3] + 1.8).toFixed(1);
        dynamic.append(
          svgNode('path', { d: `M ${sx} ${syR} C 185 48, ${rc[0].toFixed(1)} 51, ${rc[0].toFixed(1)} ${(rb[1] + rb[3] + 2.2).toFixed(1)}`, fill: 'none', stroke: '#CB202D', 'stroke-width': 1.8, 'stroke-dasharray': '4 2.5' }),
          svgNode('path', { d: `M ${sx} ${syC} C 185 101, ${cc[0].toFixed(1)} 104, ${cc[0].toFixed(1)} ${(cb[1] + cb[3] + 2.2).toFixed(1)}`, fill: 'none', stroke: '#008F45', 'stroke-width': 1.8, 'stroke-dasharray': '4 2.5' })
        );
        focusBox.setAttribute('x', '0');
        focusBox.setAttribute('y', '12');
        focusBox.setAttribute('width', '426');
        focusBox.setAttribute('height', '92');
        status.textContent = `Index (${s.i}, ${s.j}) = ${s.val}: row-major offset ${s.rOff} vs column-major offset ${s.cOff}.`;
        cards.logical.b.textContent = `shape (2, 3) · query (${s.i}, ${s.j}) = ${s.val}`;
        cards.row.b.textContent = `${s.i}×3 + ${s.j}×1 = offset ${s.rOff}`;
        cards.col.b.textContent = `${s.i}×1 + ${s.j}×2 = offset ${s.cOff}`;
      } else if (s.mode === 'rowSweep') {
        [0, 1, 2].forEach(j => dynamic.append(drawRect(logBox(0, j), '#006395')));
        [0, 1, 2].forEach(k => dynamic.append(drawRect(rowBox(k), '#008F45', 'rgba(0,143,69,0.12)')));
        [0, 2, 4].forEach(k => dynamic.append(drawRect(colBox(k), '#CB202D', 'rgba(203,32,45,0.10)', '3.5 2')));
        dynamic.append(
          svgNode('path', { d: 'M 254 41.5 H 316', fill: 'none', stroke: '#008F45', 'stroke-width': 2.2 }),
          svgNode('text', { x: 285, y: 49.5, 'text-anchor': 'middle', 'font-family': 'system-ui, sans-serif', 'font-size': 7.5, 'font-weight': '700', fill: '#008F45' }, 'contiguous (+1)'),
          svgNode('path', { d: 'M 254 97.5 Q 285 108 316 97.5 Q 347 108 378 97.5', fill: 'none', stroke: '#CB202D', 'stroke-width': 2, 'stroke-dasharray': '3 2' }),
          svgNode('text', { x: 316, y: 115, 'text-anchor': 'middle', 'font-family': 'system-ui, sans-serif', 'font-size': 7.5, 'font-weight': '700', fill: '#CB202D' }, 'stride +2 jumps')
        );
        focusBox.setAttribute('x', '234');
        focusBox.setAttribute('y', '12');
        focusBox.setAttribute('width', '192');
        focusBox.setAttribute('height', '108');
        status.textContent = 'Row 0 traversal [1, 2, 3]: unit-stride contiguous in row-major ([0, 1, 2]), strided (+2) in column-major ([0, 2, 4]).';
        cards.logical.b.textContent = 'row 0 scan: (0,0) → (0,1) → (0,2)';
        cards.row.b.textContent = 'offsets [0, 1, 2] · unit stride +1 (coalesced)';
        cards.col.b.textContent = 'offsets [0, 2, 4] · stride +2 (skips 4, 5)';
      } else if (s.mode === 'colSweep') {
        [0, 1].forEach(i => dynamic.append(drawRect(logBox(i, 1), '#006395')));
        [1, 4].forEach(k => dynamic.append(drawRect(rowBox(k), '#CB202D', 'rgba(203,32,45,0.10)', '3.5 2')));
        [2, 3].forEach(k => dynamic.append(drawRect(colBox(k), '#008F45', 'rgba(0,143,69,0.12)')));
        dynamic.append(
          svgNode('path', { d: 'M 285 41.5 Q 331.5 53 378 41.5', fill: 'none', stroke: '#CB202D', 'stroke-width': 2, 'stroke-dasharray': '3 2' }),
          svgNode('text', { x: 331.5, y: 54.5, 'text-anchor': 'middle', 'font-family': 'system-ui, sans-serif', 'font-size': 7.5, 'font-weight': '700', fill: '#CB202D' }, 'stride +3 jump'),
          svgNode('path', { d: 'M 316 97.5 H 347', fill: 'none', stroke: '#008F45', 'stroke-width': 2.2 }),
          svgNode('text', { x: 331.5, y: 108, 'text-anchor': 'middle', 'font-family': 'system-ui, sans-serif', 'font-size': 7.5, 'font-weight': '700', fill: '#008F45' }, 'contiguous (+1)')
        );
        focusBox.setAttribute('x', '234');
        focusBox.setAttribute('y', '12');
        focusBox.setAttribute('width', '192');
        focusBox.setAttribute('height', '104');
        status.textContent = 'Column 1 traversal [2, 5]: stride +3 jump in row-major ([1, 4]), unit-stride contiguous in column-major ([2, 3]).';
        cards.logical.b.textContent = 'col 1 scan: (0,1) → (1,1)';
        cards.row.b.textContent = 'offsets [1, 4] · stride +3 jump';
        cards.col.b.textContent = 'offsets [2, 3] · unit stride +1 (coalesced)';
      } else if (s.mode === 'transpose') {
        dynamic.append(
          svgNode('rect', { x: 236.5, y: 13.5, width: 190, height: 26.2, rx: 3, fill: 'none', stroke: '#7E317B', 'stroke-width': 2.6 }),
          svgNode('rect', { x: 236.5, y: 43, width: 190, height: 20, rx: 3.5, fill: '#EFE0EF', stroke: '#7E317B', 'stroke-width': 1.5 }),
          svgNode('text', { x: 331.5, y: 53.5, 'text-anchor': 'middle', 'dominant-baseline': 'middle', 'font-family': 'system-ui, sans-serif', 'font-size': 7.8, 'font-weight': '700', fill: '#5b1f59' }, 'Zero-copy .T view: shape (3, 2), strides [1, 3]')
        );
        focusBox.setAttribute('x', '234');
        focusBox.setAttribute('y', '11');
        focusBox.setAttribute('width', '194');
        focusBox.setAttribute('height', '56');
        status.textContent = 'Metadata-only transpose: 0 bytes copied; row-major backing buffer [1, 2, 3, 4, 5, 6] stays untouched under strides [1, 3].';
        cards.logical.b.textContent = 'transposed view: shape (3, 2), strides [1, 3]';
        cards.row.b.textContent = 'same buffer [1, 2, 3, 4, 5, 6] · 0 copies';
        cards.col.b.textContent = 'contiguous .T would require physical copy';
      } else {
        focusBox.setAttribute('x', '1');
        focusBox.setAttribute('y', '10');
        focusBox.setAttribute('width', '424');
        focusBox.setAttribute('height', '96');
        status.textContent = 'One logical 2 × 3 tensor backed by either row-major strides [3, 1] or column-major strides [1, 2].';
        cards.logical.b.textContent = 'shape (2, 3) · 6 scalar elements';
        cards.row.b.textContent = 'order [1, 2, 3, 4, 5, 6] · offset = 3i + j';
        cards.col.b.textContent = 'order [1, 4, 2, 5, 3, 6] · offset = i + 2j';
      }

      root.dataset.lessonState = String(idx);
      root.dataset.stateEvidence = JSON.stringify(s.evidence);
    }

    return window.MLSysOriginalPlayer.create({
      root,
      outer,
      layout: 'compact',
      intro: 'Map each (i, j) index of the 2 × 3 tensor to 1D storage using strides [3, 1] and [1, 2], then swap strides for a zero-copy transpose.',
      title: 'How do strides map 2D indices, memory coalescing, and zero-copy views?',
      scope: 'Exact chapter 2 × 3 tensor (values 1–6) and its row-major [3, 1] and column-major [1, 2] 1D storage layouts. Offsets count scalar elements; byte offsets multiply by dtype width.',
      steps,
      render(i, { stage, details }) {
        if (!surface.isConnected) stage.replaceChildren(surface);
        if (details && !status.isConnected) details.replaceChildren(status, ledger);
        paintEndpoint(i);
      },
      transition({ to, onFrame }) {
        flights.replaceChildren();
        const s = steps[to];
        if (s.mode === 'point') {
          const a = center(logBox(s.i, s.j));
          const b1 = center(rowBox(s.rOff));
          const b2 = center(colBox(s.cOff));
          const d1 = svgNode('circle', { r: 5.5, fill: '#CB202D', stroke: '#fff', 'stroke-width': 1.2 });
          const d2 = svgNode('circle', { r: 5.5, fill: '#008F45', stroke: '#fff', 'stroke-width': 1.2 });
          flights.append(d1, d2);
          onFrame(p => {
            d1.setAttribute('cx', (a[0] + (b1[0] - a[0]) * p).toFixed(2));
            d1.setAttribute('cy', (a[1] + (b1[1] - a[1]) * p).toFixed(2));
            d2.setAttribute('cx', (a[0] + (b2[0] - a[0]) * p).toFixed(2));
            d2.setAttribute('cy', (a[1] + (b2[1] - a[1]) * p).toFixed(2));
          });
        } else if (s.mode === 'rowSweep' || s.mode === 'colSweep') {
          const rSeq = s.mode === 'rowSweep' ? [0, 1, 2] : [1, 4];
          const cSeq = s.mode === 'rowSweep' ? [0, 2, 4] : [2, 3];
          const d1 = svgNode('circle', { r: 5.5, fill: s.mode === 'rowSweep' ? '#008F45' : '#CB202D', stroke: '#fff', 'stroke-width': 1.2 });
          const d2 = svgNode('circle', { r: 5.5, fill: s.mode === 'rowSweep' ? '#CB202D' : '#008F45', stroke: '#fff', 'stroke-width': 1.2 });
          flights.append(d1, d2);
          const interp = (seq, boxFn, p) => {
            const segs = seq.length - 1;
            const pos = Math.min(segs, p * segs);
            const k = Math.min(segs - 1, Math.floor(pos));
            const u = pos - k;
            const p0 = center(boxFn(seq[k]));
            const p1 = center(boxFn(seq[k + 1]));
            return [p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u];
          };
          onFrame(p => {
            const q1 = interp(rSeq, rowBox, p);
            const q2 = interp(cSeq, colBox, p);
            d1.setAttribute('cx', q1[0].toFixed(2));
            d1.setAttribute('cy', q1[1].toFixed(2));
            d2.setAttribute('cx', q2[0].toFixed(2));
            d2.setAttribute('cy', q2[1].toFixed(2));
          });
        }
      }
    });
  };
})();
