/* Scoped stateful extension of the inspected original artwork across both sequential (a) and pipelined (b) timelines. */
(() => {
'use strict';
const L = {
  title: 'How do batches overlap without violating dependencies?',
  intro: 'Step through 4 batches across Storage, CPU, and GPU lanes to see how pipelined prefetching overlaps 3 batches in flight after 1 initial startup stall.',
  scope: 'Two shared-axis timelines from the chapter diagram: (a) sequential loading completing 2 batches with repeated GPU idle stalls, and (b) pipelined loading with prefetch completing 4 batches (B1–B4) in roughly the same wall-clock span. Every batch obeys the strict Read → Prep → Train dependency order across the 3 hardware lanes (Storage, CPU, GPU).',
  lanes: ['Storage reads', 'CPU preparation', 'GPU training', 'Completed batches'],
  steps: [
    {
      label: 'Start reading',
      description: 'In both panel (a) and panel (b), batch 1 begins in the green Storage lane (Read 1). Because B1 has not been read or prepared on the CPU yet, the GPU sits in its 1 initial red dashed GPU idle block.',
      cards: [[0, 'Read B1', 'available'], [2, 'GPU idle · input not prepared', 'warning']]
    },
    {
      label: 'Prepare batch 1',
      description: 'Read 1 completes (✓) and hands B1 to the CPU lane for Prep 1 while the GPU remains in its initial idle window. Unlike sequential panel (a), pipelined panel (b) immediately starts prefetching Read 2 on Storage in parallel with Prep 1.',
      cards: [[0, 'Read B2 · pipelined only', 'available'], [1, 'Prep B1', 'queue'], [2, 'GPU idle', 'warning']]
    },
    {
      label: 'Overlap three stages',
      description: 'At the shaded 3 in flight column of panel (b), B1 is already trained (1 batch complete) while 3 different batches run concurrently across all 3 lanes: GPU runs Train 2 (B2), CPU runs Prep 3 (B3), and Storage runs Read 4 (B4). Meanwhile sequential panel (a) enters its second red GPU idle stall waiting on Read 2.',
      cards: [[0, 'Read B4 (panel b) vs Read B2 (panel a)', 'available'], [1, 'Prep B3 · B3 read complete', 'queue'], [2, 'Train B2 (panel b) vs GPU idle (panel a)', 'compute'], [3, 'B1 trained', 'available']]
    },
    {
      label: 'Drain prepared work',
      description: 'In panel (b), all 4 storage reads (Read 1–Read 4 ✓) and 3 CPU preps (Prep 1–Prep 3 ✓) are complete: the GPU executes Train 3 while Prep 4 finishes, avoiding the second GPU idle stall that still blocks sequential panel (a) during Prep 2.',
      cards: [[0, 'Reads B1–B4 complete', 'available'], [1, 'B4 prepared · queued', 'queue'], [2, 'Train B3 · 0 second GPU stall', 'compute'], [3, 'B1 B2 trained', 'available']]
    },
    {
      label: 'Complete the finite pipeline',
      description: 'Train 4 finishes on the GPU (all 4 batches B1–B4 ✓ complete in panel (b)) right near the dashed vertical line where sequential panel (a) finishes only 2 batches. Pipelined prefetching hides storage and CPU latency behind GPU compute as long as delivery keeps pace.',
      cards: [[0, 'B1 B2 B3 B4 read', 'available'], [1, 'B1 B2 B3 B4 prepared', 'queue'], [2, 'B4 training complete', 'compute'], [3, 'B1 B2 B3 B4 trained (vs B1 B2 sequential)', 'available']]
    }
  ],
  guard: 'No invented speedup; logical snapshots synchronize panel (a) sequential stalls against panel (b) pipelined overlap on the shared time axis.'
};
window.MLSysFigureLessons = window.MLSysFigureLessons || {};
window.MLSysFigureLessons['original-v2-data-pipeline-vol2'] = async (root, context) => {
  if (!context.verifiedArtwork || !window.MLSysOriginalPlayer) throw Error('Verified artwork/shared player required');
  root.classList.add('original-v2-data-pipeline-vol2');
  const make = (tag, cls, text) => {
    const n = document.createElement(tag);
    n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };
  const transfers = [
    [0, 1, 'B1 read → preparation + Read 2'],
    [1, 2, '3 in flight (B4/B3/B2) vs idle'],
    [1, 2, 'Train 3 + Prep 4 vs GPU stall'],
    [2, 3, '4 batches done vs 2 sequential']
  ];
  const board = make('div', 'v2b-board');
  const lanes = L.lanes.map((label, i) => {
    const lane = make('section', 'v2b-lane');
    lane.dataset.lane = i;
    lane.append(make('h5', '', label));
    const content = make('div', 'v2b-content');
    lane.append(content);
    board.append(lane);
    return content;
  });
  const invariant = make('p', 'v2b-invariant', L.guard);
  const art = context.verifiedArtwork.cloneNode(true);
  const NS = 'http://www.w3.org/2000/svg';
  const source = make('div', 'v2b-source');
  source.tabIndex = 0;
  source.setAttribute('aria-label', 'Original diagram with changing values; scroll to inspect on small screens');
  source.append(art);
  const overlay = document.createElementNS(NS, 'g');
  const focusGuard = document.createElementNS(NS, 'rect');
  Object.entries({ x: 24, y: 24, width: 632, height: 424, fill: 'none', stroke: 'none', 'data-mlsys-focus': '' }).forEach(([k, v]) => focusGuard.setAttribute(k, v));
  art.append(overlay, focusGuard);
  let mounted = false;

  const cursorXs = [156, 233, 409, 486, 585];

  function addHi(x, y, w, h, stroke = '#d97706', dash = '') {
    const hi = document.createElementNS(NS, 'rect');
    Object.entries({ x: x - 2, y: y - 2, width: w + 4, height: h + 4, rx: 4, fill: 'none', stroke, 'stroke-width': 2.5, ...(dash ? { 'stroke-dasharray': dash } : {}) }).forEach(([k, v]) => hi.setAttribute(k, v));
    overlay.append(hi);
  }

  function liveArt(index) {
    overlay.replaceChildren();
    art.dataset.liveState = String(index);
    const cx = cursorXs[index];

    // Shared vertical time playhead spanning both Panel (a) and Panel (b)
    const line = document.createElementNS(NS, 'line');
    Object.entries({ x1: cx, y1: 32, x2: cx, y2: 408, stroke: '#0284c7', 'stroke-width': 1.8, 'stroke-dasharray': '4 3' }).forEach(([k, v]) => line.setAttribute(k, v));
    overlay.append(line);

    for (const text of art.querySelectorAll('text')) {
      const y = +text.getAttribute('y');
      const value = text.dataset.originalLabel || text.textContent.trim();
      if (/^(Read|Prep|Train) [1-4]$/.test(value)) {
        if (!text.dataset.originalLabel) text.dataset.originalLabel = value;
        const original = text.dataset.originalLabel;
        const [stage, bStr] = original.split(' ');
        const batch = +bStr;
        const isPanelB = y > 280;
        let done = false, active = false;
        if (isPanelB) {
          done = index === 4 ||
            (index === 1 && stage === 'Read' && batch === 1) ||
            (index === 2 && (batch === 1 || (stage === 'Read' && batch < 4) || (stage === 'Prep' && batch < 3))) ||
            (index === 3 && (batch < 3 || stage === 'Read' || (stage === 'Prep' && batch === 3)));
          active = (index === 0 && stage === 'Read' && batch === 1) ||
            (index === 1 && ((stage === 'Prep' && batch === 1) || (stage === 'Read' && batch === 2))) ||
            (index === 2 && ((stage === 'Train' && batch === 2) || (stage === 'Prep' && batch === 3) || (stage === 'Read' && batch === 4))) ||
            (index === 3 && ((stage === 'Train' && batch === 3) || (stage === 'Prep' && batch === 4))) ||
            (index === 4 && stage === 'Train' && batch === 4);
        } else {
          // Sequential Panel (a) on the exact same shared time axis cx
          done = (index >= 1 && stage === 'Read' && batch === 1) ||
            (index >= 2 && (stage === 'Prep' || stage === 'Train') && batch === 1) ||
            (index >= 3 && stage === 'Read' && batch === 2) ||
            (index === 4 && stage === 'Prep' && batch === 2);
          active = (index === 0 && stage === 'Read' && batch === 1) ||
            (index === 1 && stage === 'Prep' && batch === 1) ||
            (index === 2 && stage === 'Read' && batch === 2) ||
            (index === 3 && stage === 'Prep' && batch === 2) ||
            (index === 4 && stage === 'Train' && batch === 2);
        }
        text.style.opacity = (done || active) ? '1' : '.35';
        text.dataset.complete = String(done);
        text.textContent = original + (done ? ' ✓' : '');
        text.setAttribute('font-size', '10');
        const box = text.previousElementSibling;
        if (box && box.tagName === 'rect') {
          box.style.opacity = (done || active) ? '1' : '.35';
          if (active) addHi(+box.getAttribute('x'), +box.getAttribute('y'), +box.getAttribute('width'), +box.getAttribute('height'), isPanelB ? '#d97706' : '#0284c7');
        }
      } else if (value === 'GPU idle' || text.dataset.originalIdle) {
        if (!text.dataset.originalIdle) text.dataset.originalIdle = 'GPU idle';
        const box = text.previousElementSibling;
        if (box && box.tagName === 'rect') {
          const bx = +box.getAttribute('x'), by = +box.getAttribute('y');
          const stallActive = (bx < 200 && index <= 1) || (by < 200 && bx > 300 && (index === 2 || index === 3));
          const stallPast = (bx < 200 && index >= 2) || (by < 200 && bx > 300 && index === 4);
          box.style.opacity = (stallActive || stallPast) ? '1' : '.35';
          text.style.opacity = (stallActive || stallPast) ? '1' : '.35';
          if (stallActive) addHi(bx, by, +box.getAttribute('width'), +box.getAttribute('height'), '#dc2626', '4 2');
        }
      }
    }
  }

  function endpoint(index) {
    board.querySelectorAll('.v2b-packet').forEach(n => n.remove());
    liveArt(index);
    root.dataset.state = String(index);
    lanes.forEach(l => l.replaceChildren());
    for (const [lane, text, kind] of L.steps[index].cards) {
      const card = make('div', 'v2b-card v2b-' + kind, text);
      card.dataset.motionObject = 'true';
      card.dataset.identity = text;
      lanes[lane].append(card);
    }
    board.style.opacity = '1';
    board.style.transform = 'none';
    root.dataset.animationProgress = '1';
  }

  const player = window.MLSysOriginalPlayer.create({
    root,
    outer: context.outer,
    layout: 'compact',
    title: L.title,
    intro: L.intro,
    scope: L.scope,
    steps: L.steps,
    render(index, { stage, details }) {
      if (!mounted) {
        stage.append(source);
        details.append(board, invariant);
        mounted = true;
      }
      endpoint(index);
      const scale = art.getBoundingClientRect().width / 680;
      source.scrollLeft = Math.max(0, cursorXs[index] * scale - source.clientWidth / 2);
    },
    transition({ from, to, onFrame }) {
      if (to !== from + 1) return;
      root.dataset.pending = String(to);
      const transfer = transfers[Math.max(0, to - 1)];
      const x0 = cursorXs[from], x1 = cursorXs[to];
      overlay.replaceChildren();
      const cursor = document.createElementNS(NS, 'line');
      Object.entries({ x1: x0, y1: 32, x2: x0, y2: 408, stroke: '#d97706', 'stroke-width': 2.2, 'stroke-dasharray': '4 3' }).forEach(([k, v]) => cursor.setAttribute(k, v));
      const packet = document.createElementNS(NS, 'g');
      packet.dataset.motionObject = 'true';
      const rect = document.createElementNS(NS, 'rect');
      Object.entries({ x: -84, y: -14, width: 168, height: 28, rx: 4, fill: '#fff7ed', stroke: '#d97706', 'stroke-width': 1.8 }).forEach(([k, v]) => rect.setAttribute(k, v));
      const label = document.createElementNS(NS, 'text');
      Object.entries({ x: 0, y: 4, 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 700, fill: '#7c2d12' }).forEach(([k, v]) => label.setAttribute(k, v));
      label.textContent = transfer[2];
      packet.append(rect, label);
      overlay.append(cursor, packet);
      onFrame(p => {
        root.dataset.animationProgress = String(p);
        const cx = x0 + (x1 - x0) * p;
        cursor.setAttribute('x1', String(cx));
        cursor.setAttribute('x2', String(cx));
        packet.setAttribute('transform', `translate(${cx}, 222)`);
        source.scrollLeft = Math.max(0, cx * art.getBoundingClientRect().width / 680 - source.clientWidth / 2);
      });
    }
  });
  return { pause: () => player.pause(), destroy: () => player.destroy() };
};
})();
