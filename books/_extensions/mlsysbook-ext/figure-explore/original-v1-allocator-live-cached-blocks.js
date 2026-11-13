(() => {
  'use strict';
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons['original-v1-allocator-live-cached-blocks'] = async (root, { outer, verifiedArtwork }) => {
    if (!verifiedArtwork || !window.MLSysOriginalPlayer) {
      throw new Error('Verified artwork and shared player required');
    }
    root.classList.add('original-v1-allocator-live-cached-blocks');

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

    const steps = [
      {
        label: 'Snapshot (a): all three tensors live',
        description: 'Tensors A, B, and C each occupy 4 contiguous units in the 12-unit virtual-address pool. Reserved: 12 units, Live: 12 units, Cached reusable: 0 units (largest free block: 0).',
        live: 12, free: 0, largest: 0
      },
      {
        label: 'Snapshot (b): release B into the cache',
        description: 'Tensor B finishes its last consumer, so its 4 middle units become cached and reusable inside the allocator pool rather than returning to the OS/driver. Live: 8, Cached: 4 (largest free block: 4).',
        live: 8, free: 4, largest: 4
      },
      {
        label: 'Carve 2 units from cached B for Tensor D',
        description: 'Before A is released, Tensor D allocates 2 units at the front of B’s 4-unit cached block ([4..6]), leaving a 2-unit cached remainder ([6..8]). Live: 10, Cached: 2 (largest free block: 2).',
        live: 10, free: 2, largest: 2
      },
      {
        label: 'Snapshot (c): release A into a split pool',
        description: 'Tensor A finishes and returns its 4 front units ([0..4]) to the cache. Total reusable memory is now 4 + 2 = 6 units, but live Tensor D ([4..6]) splits the two free blocks (largest free block: 4).',
        live: 6, free: 6, largest: 4
      },
      {
        label: 'Fail the 6-unit contiguous request',
        description: 'A new tensor requests 6 contiguous units. Sliding the 6-unit span hits live Tensor D after 4 units at [0..6] and hits live Tensor C after 2 units at [6..12], causing external fragmentation OOM.',
        live: 6, free: 6, largest: 4
      }
    ];

    // Canonical SVG (0 0 340.954 246.548):
    // Row (a): y=16.79, h=19.81; Row (b): y=83.29, h=19.81; Row (c): y=149.78, h=19.80; Request: y=200.71, h=19.80
    // Unit width = 339.54 / 12 = 28.295, x0 = 0.71
    const X0 = 0.71;
    const UW = 28.295;

    const svg = verifiedArtwork.cloneNode(true);
    svg.removeAttribute('width');
    svg.removeAttribute('height');
    svg.setAttribute('viewBox', '-12 -10 365 266');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Caching allocator pool fragmentation across snapshots a, b, and c');
    svg.prepend(svgNode('rect', { x: -12, y: -10, width: 365, height: 266, fill: '#ffffff' }));

    const overlay = svgNode('g', { class: 'alloc-overlay' });
    const dimBands = [
      svgNode('rect', { x: -2, y: 1, width: 345, height: 42, fill: '#ffffff', 'fill-opacity': '0.68' }),
      svgNode('rect', { x: -2, y: 67, width: 345, height: 42, fill: '#ffffff', 'fill-opacity': '0.68' }),
      svgNode('rect', { x: -2, y: 134, width: 345, height: 42, fill: '#ffffff', 'fill-opacity': '0.68' }),
      svgNode('rect', { x: -2, y: 184, width: 345, height: 42, fill: '#ffffff', 'fill-opacity': '0.74' })
    ];
    // Step 2 intermediate overlay on Row (c) showing A still live before release
    const liveAPatch = svgNode('g', {});
    liveAPatch.append(
      svgNode('rect', { x: X0, y: 149.78, width: (4 * UW).toFixed(2), height: 19.8, fill: '#d1f3ff', stroke: '#006395', 'stroke-width': 1.2 }),
      svgNode('text', { x: (X0 + 2 * UW).toFixed(2), y: 159.7, 'text-anchor': 'middle', 'dominant-baseline': 'central', 'font-family': 'system-ui, sans-serif', 'font-size': 8.2, 'font-weight': '700', fill: '#00486e' }, 'Live A (4) — still in use')
    );
    const activeFrame = svgNode('rect', { rx: 3, fill: 'none', stroke: '#006395', 'stroke-width': 2.4 });
    const caliper = svgNode('g', {});
    const caliperRect = svgNode('rect', { y: 147.8, width: (6 * UW).toFixed(2), height: 23.8, rx: 2, fill: 'rgba(203,32,45,0.12)', stroke: '#CB202D', 'stroke-width': 2.2, 'stroke-dasharray': '5 3' });
    const caliperTag = svgNode('text', { y: 181.5, 'text-anchor': 'middle', 'font-family': 'system-ui, sans-serif', 'font-size': 7.8, 'font-weight': '700', fill: '#CB202D', stroke: '#ffffff', 'stroke-width': 2.6, 'paint-order': 'stroke' }, 'Need 6 contiguous → blocked by D!');
    caliper.append(caliperRect, caliperTag);

    const focusBox = svgNode('rect', { fill: 'none', stroke: 'none', 'pointer-events': 'none', 'data-mlsys-focus': '' });
    overlay.append(...dimBands, liveAPatch, activeFrame, caliper, focusBox);
    svg.append(overlay);

    const surface = htmlNode('div', 'vb-surface');
    surface.append(svg);
    const readout = htmlNode('p', 'vb-readout');
    const metrics = htmlNode('div', 'vb-metrics');
    const mkMetric = (kind, label) => {
      const m = htmlNode('div', 'vb-metric');
      m.dataset.kind = kind;
      const h = htmlNode('strong', '', label);
      const v = htmlNode('span', '');
      m.append(h, v);
      metrics.append(m);
      return v;
    };
    const vals = {
      reserved: mkMetric('res', 'Reserved Pool'),
      live: mkMetric('live', 'Live Tensors'),
      free: mkMetric('free', 'Cached Free'),
      gap: mkMetric('gap', 'Max Contiguous')
    };

    function setCaliperX(x) {
      caliperRect.setAttribute('x', x.toFixed(2));
      caliperTag.setAttribute('x', (x + 3 * UW).toFixed(2));
    }

    function paintEndpoint(i) {
      const s = steps[i];
      const activeRows = i === 0 ? [0] : i === 1 ? [1] : i === 4 ? [2, 3] : [2];
      dimBands.forEach((b, idx) => b.setAttribute('visibility', activeRows.includes(idx) ? 'hidden' : 'visible'));
      liveAPatch.setAttribute('visibility', i === 2 ? 'visible' : 'hidden');
      caliper.setAttribute('visibility', i === 4 ? 'visible' : 'hidden');
      if (i === 4) setCaliperX(X0);

      const fy = i === 0 ? 14.5 : i === 1 ? 81.0 : 147.5;
      const fh = i === 4 ? 75.5 : 24.4;
      activeFrame.setAttribute('x', '-1.5');
      activeFrame.setAttribute('y', String(fy));
      activeFrame.setAttribute('width', '344');
      activeFrame.setAttribute('height', String(fh));
      activeFrame.setAttribute('stroke', i === 4 ? '#CB202D' : (i === 2 ? '#E67817' : '#006395'));

      focusBox.setAttribute('x', '0');
      focusBox.setAttribute('y', String(Math.max(4, fy - 4)));
      focusBox.setAttribute('width', '341');
      focusBox.setAttribute('height', String(fh + 8));

      vals.reserved.textContent = '12 units';
      vals.live.textContent = `${s.live} units`;
      vals.free.textContent = `${s.free} units`;
      vals.gap.textContent = `${s.largest} units${i === 4 ? ' < 6 needed' : ''}`;
      readout.textContent = `Reserved: 12 · Live: ${s.live} · Cached reusable: ${s.free} · Largest contiguous gap: ${s.largest} units.`;

      root.dataset.live = String(s.live);
      root.dataset.reserved = '12';
      root.dataset.reusable = String(s.free);
      root.dataset.largest = String(s.largest);
      root.dataset.endpoint = String(i);
    }

    return window.MLSysOriginalPlayer.create({
      root,
      outer,
      layout: 'compact',
      intro: 'Trace 12 reserved pool units through tensor releases and partial reuse to see why 6 cached units split into blocks of 4 and 2 fail a 6-unit request.',
      title: 'Why can six free units fail a six-unit request?',
      scope: 'Original fixed 12-unit virtual-address pool across snapshots (a), (b), and (c) without compaction or virtual-memory remapping. Cached blocks remain reserved inside the framework allocator.',
      steps,
      render(i, { stage, details }) {
        if (!surface.isConnected) stage.replaceChildren(surface);
        if (details && !readout.isConnected) details.replaceChildren(readout, metrics);
        paintEndpoint(i);
      },
      transition({ to, onFrame }) {
        if (to === 4) {
          caliper.setAttribute('visibility', 'visible');
          onFrame(p => {
            // Probe [0..6] then [6..12] then settle back on [0..6]
            const x = p < 0.5
              ? X0 + (6 * UW) * (p * 2)
              : X0 + (6 * UW) * (1 - (p - 0.5) * 2);
            setCaliperX(x);
          });
        }
      }
    });
  };
})();
