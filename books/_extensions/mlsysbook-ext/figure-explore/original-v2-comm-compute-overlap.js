/* Verified chapter artwork with live backward-pass compute, overhead o, and AllReduce overlap. */
(() => {
'use strict';
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, a = {}, text) => {
  const n = document.createElementNS(NS, tag);
  Object.entries(a).forEach(([k, v]) => n.setAttribute(k, v));
  if (text !== undefined) n.textContent = text;
  return n;
};
window.MLSysFigureLessons = window.MLSysFigureLessons || {};
window.MLSysFigureLessons['original-v2-comm-compute-overlap'] = async (root, context) => {
  if (!context.verifiedArtwork || !window.MLSysOriginalPlayer) throw Error('Verified source artwork and shared player required');
  if (context.verifiedArtwork.getAttribute('viewBox') !== '0 0 832 286') throw Error('Source geometry changed; review required');
  root.classList.add('v2c-lesson');
  const art = context.verifiedArtwork.cloneNode(true);
  art.removeAttribute('width');
  art.removeAttribute('height');
  art.setAttribute('viewBox', '0 0 832 326');
  art.setAttribute('role', 'img');
  art.setAttribute('aria-label', 'When is a gradient ready for communication and for the optimizer? Original chapter timeline with live gradient state.');
  art.insertBefore(el('rect', { x: 0, y: 0, width: 832, height: 326, fill: '#ffffff' }), art.firstChild);
  const overlay = el('g', { 'data-v2c-overlay': '' });
  const moving = el('g', { 'aria-hidden': 'true' });
  const focusGuard = el('rect', { x: 28, y: 24, width: 776, height: 284, fill: 'none', stroke: 'none', 'data-mlsys-focus': '' });
  art.append(overlay, moving, focusGuard);

  const chip = (x, y, label, fill = '#fef3c7', stroke = '#d97706', textFill = '#14294e', attrs = {}) => {
    const g = el('g', { 'data-object': label, ...attrs });
    const w = Math.max(30, label.length * 6.8 + 12);
    g.append(
      el('rect', { x: x - w / 2, y: y - 10, width: w, height: 20, rx: 3, fill, stroke, 'stroke-width': 1.5 }),
      el('text', { x, y: y + 4, 'text-anchor': 'middle', 'font-size': 10.5, 'font-weight': 700, 'font-family': 'sans-serif', fill: textFill }, label)
    );
    return g;
  };

  // Exact source SVG coordinates:
  // Backward L4..L1: x = 120 + i*124, y = 56, w = 110, h = 44
  // Overhead o4..o1: x = 230 + i*124, y = 56, w = 14, h = 44
  // AllReduce L4..L1: x = 244 + i*124, y = 124, w = 110, h = 44
  // GPU idle: rect(616, 56, 110, 44); Optimizer: rect(726, 56, 86, 44)
  const layers = [4, 3, 2, 1];
  const bx = i => 120 + i * 124;
  const gradientState = (index, i) => {
    const ready = 1 + i * 2, launch = ready + 1;
    return index === 10 ? 'consumed' : index < ready ? 'not ready' : index === ready ? 'local' : index === launch ? 'in flight' : 'reduced';
  };
  const playheadX = [175, 237, 299, 361, 423, 485, 547, 609, 671, 726, 769];

  const status = document.createElement('p');
  status.className = 'v2c-status';
  status.setAttribute('aria-live', 'polite');
  const note = document.createElement('p');
  note.className = 'v2c-reading';
  note.textContent = 'Local gℓ and globally reduced Σgℓ are distinct states of the same gradient on the verified chapter timeline. Amber o occupies the GPU before each collective launches.';
  let mounted = false, stageNode = null;
  function camera(x) {
    if (!stageNode || stageNode.clientWidth >= 540) { if (stageNode) stageNode.scrollLeft = 0; return; }
    const scale = art.getBoundingClientRect().width / 832;
    stageNode.scrollLeft = Math.max(0, x * scale - stageNode.clientWidth / 2);
  }

  const steps = [
    { label: 'Backward L4', description: 'Backpropagation starts at output layer 4 on the Compute lane (Backward L4). None of the 4 layer gradients (g4, g3, g2, g1) are ready yet, so the Network lane is idle and the Optimizer is blocked.' },
    { label: 'L4 gradient ready', description: 'Backward L4 finishes and writes local gradient g4 into the GPU gradient L4 buffer. Before network transfer can begin, the GPU must spend nonoverlappable processor overhead o (amber slice o4) on the Compute lane to initiate the collective.' },
    { label: 'AllReduce L4 / backward L3', description: 'Once amber overhead slice o4 finishes, AllReduce L4 launches g4 onto the Network lane while the GPU immediately begins Backward L3 on the Compute lane (1 of 3 overlapped transfers now in flight).' },
    { label: 'L4 reduced / L3 ready', description: 'AllReduce L4 completes beneath Backward L3, delivering globally reduced gradient Σg4 (green) while Backward L3 produces local gradient g3. The GPU now pays amber overhead slice o3 on the Compute lane before launching L3’s collective.' },
    { label: 'AllReduce L3 / backward L2', description: 'After o3 completes, AllReduce L3 communicates g3 on the Network lane concurrently with Backward L2 on the Compute lane (2 of 3 overlapped transfers). Reduced Σg4 remains stored in GPU memory awaiting the end-of-step optimizer update.' },
    { label: 'L3 reduced / L2 ready', description: 'AllReduce L3 finishes during Backward L2, so the GPU now holds 2 globally reduced gradients (Σg4 and Σg3) plus newly computed local gradient g2. Amber overhead slice o2 occupies the GPU before AllReduce L2 can launch.' },
    { label: 'AllReduce L2 / backward L1', description: 'AllReduce L2 runs g2 on the Network lane beneath Backward L1 on the Compute lane, completing the 3 transfers that fit behind later backward computation (AllReduce L4, L3, and L2).' },
    { label: 'L2 reduced / L1 ready', description: 'AllReduce L2 delivers Σg2 and Backward L1 finishes local gradient g1. All 4 backward compute blocks are now done, leaving 0 remaining backward layers to hide amber overhead slice o1 or AllReduce L1.' },
    { label: 'Exposed AllReduce L1', description: 'After exposed overhead slice o1, AllReduce L1 runs g1 on the Network lane while the Compute lane sits in the GPU idle box. This 1 final transfer (o1 + AllReduce L1) forms the exposed tail before the optimizer step.' },
    { label: 'All reductions complete', description: 'AllReduce L1 completes and delivers Σg1. All 4 globally reduced gradients (Σg4, Σg3, Σg2, Σg1) are now resident on the GPU, unblocking the Optimizer.' },
    { label: 'Optimizer step', description: 'The Optimizer block on the Compute lane consumes all 4 reduced gradients (Σg4, Σg3, Σg2, Σg1) and updates the model weights. Pipelining hid 3 of the 4 network transfers behind backward compute, leaving the 4 processor overhead slices (o) and 1 final transfer (AllReduce L1) exposed.' }
  ];

  const controller = window.MLSysOriginalPlayer.create({
    root,
    outer: context.outer,
    layout: 'compact',
    title: 'When is a gradient ready for communication and for the optimizer?',
    intro: 'Step through a 4-layer backward pass from L4 to L1 to see how 3 gradient AllReduces hide behind later compute while 1 final transfer stays exposed.',
    scope: 'A 4-layer backward pass (Backward L4 → L3 → L2 → L1) across 2 timeline lanes (Compute and Network). Each layer produces local gradient gℓ, pays nonoverlappable processor overhead o on the GPU, and launches AllReduce Lℓ on the network to produce globally reduced gradient Σgℓ before 1 final Optimizer step.',
    steps,
    render(index, { stage, details }) {
      if (!mounted) {
        stageNode = stage;
        stage.classList.add('v2c-stage');
        stage.tabIndex = 0;
        stage.setAttribute('aria-label', 'Mechanism diagram; swipe horizontally on a narrow screen or use Expand');
        stage.append(art);
        details.append(status, note);
        mounted = true;
      }
      overlay.replaceChildren();
      moving.replaceChildren();
      root.dataset.modelState = String(index);
      root.dataset.gradients = JSON.stringify(layers.map((_, i) => gradientState(index, i)));
      root.dataset.optimizer = index === 10 ? 'updated' : index >= 9 ? 'ready' : 'blocked';
      root.dataset.compute = index === 0 ? 'backward L4' : index === 8 ? 'idle' : index >= 9 ? 'optimizer' : index % 2 === 0 ? `backward L${4 - index / 2}` : `overhead o${4 - (index - 1) / 2}`;

      const px = playheadX[index];
      // Dim future timeline region lightly so completed + active work stands out while full artwork stays legible
      if (index < 10) {
        const veilX = index === 0 ? 230 : index % 2 === 1 ? bx((index - 1) / 2) + 124 : bx(index / 2 - 1) + 234;
        if (veilX < 812) {
          overlay.append(el('rect', { x: veilX, y: 50, width: 816 - veilX, height: 124, fill: '#ffffff', 'fill-opacity': '0.58' }));
        }
      }
      // Active highlights on verified SVG blocks
      layers.forEach((l, i) => {
        const st = gradientState(index, i);
        if (index === i * 2) {
          overlay.append(el('rect', { x: bx(i) - 2, y: 54, width: 114, height: 48, rx: 3, fill: 'none', stroke: '#1F407A', 'stroke-width': 2.6 }));
        }
        if (index === i * 2 + 1) {
          overlay.append(el('rect', { x: bx(i) + 108, y: 52, width: 18, height: 52, rx: 2, fill: 'none', stroke: '#d97706', 'stroke-width': 2.6 }));
        }
        if (st === 'in flight') {
          overlay.append(
            el('rect', { x: bx(i) + 122, y: 122, width: 114, height: 48, rx: 3, fill: 'none', stroke: '#d97706', 'stroke-width': 2.6 }),
            chip(bx(i) + 179, 146, `g${l} in flight`, '#fef3c7', '#d97706', '#7c2d12', { 'data-network-gradient': l })
          );
        }
        // Bottom ledger strip at y=282..314
        const gx = 120 + i * 174;
        const fill = st === 'reduced' ? '#dcfce7' : st === 'consumed' ? '#f3e8ff' : st === 'local' ? '#EAF1F8' : st === 'in flight' ? '#fef3c7' : '#f8fafc';
        const stroke = st === 'reduced' ? '#16a34a' : st === 'consumed' ? '#7c3aed' : st === 'local' ? '#1F407A' : st === 'in flight' ? '#d97706' : '#cbd5e1';
        const txt = st === 'reduced' ? `L${l}: Σg${l} ✓` : st === 'consumed' ? `L${l}: applied` : st === 'local' ? `L${l}: local g${l}` : st === 'in flight' ? `L${l}: AllReduce` : `L${l}: pending`;
        overlay.append(
          el('rect', { x: gx, y: 280, width: 162, height: 32, rx: 4, fill, stroke, 'stroke-width': 1.5, 'data-gradient': l, 'data-state': st }),
          el('text', { x: gx + 81, y: 300, 'text-anchor': 'middle', 'font-size': 11, 'font-weight': 700, 'font-family': 'sans-serif', fill: '#14294e' }, txt)
        );
      });
      if (index === 8) {
        overlay.append(el('rect', { x: 614, y: 54, width: 114, height: 48, rx: 3, fill: 'none', stroke: '#dc2626', 'stroke-width': 2.4, 'stroke-dasharray': '4 3' }));
      }
      if (index >= 9) {
        overlay.append(el('rect', { x: 724, y: 54, width: 90, height: 48, rx: 3, fill: index === 10 ? '#dcfce7' : 'none', 'fill-opacity': index === 10 ? '0.35' : '0', stroke: index === 10 ? '#16a34a' : '#1F407A', 'stroke-width': 2.6 }));
      }
      overlay.append(el('line', { x1: px, y1: 44, x2: px, y2: 180, stroke: '#d97706', 'stroke-width': 2, 'stroke-dasharray': '4 3' }));

      status.textContent = `GPU: ${root.dataset.compute}. Gradients L4 → L1: ${layers.map((l, i) => `g${l} ${gradientState(index, i)}`).join(' · ')}. Optimizer: ${root.dataset.optimizer}.`;
      camera(px);
    },
    transition({ from, to, onFrame }) {
      if (to !== from + 1) return;
      const motions = to === 10
        ? layers.map((l, i) => ({ label: `Σg${l}`, fill: '#dcfce7', stroke: '#16a34a', path: [[201 + i * 174, 280], [769, 78]] }))
        : to === 9
        ? [{ label: 'Σg1 ✓', fill: '#dcfce7', stroke: '#16a34a', path: [[671, 146], [723, 280]] }]
        : to % 2 === 0
        ? [{ label: `g${layers[to / 2 - 1]}`, fill: '#fef3c7', stroke: '#d97706', path: [[bx(to / 2 - 1) + 117, 78], [bx(to / 2 - 1) + 179, 146]] }]
        : [
            { label: `g${layers[(to - 1) / 2]}`, fill: '#EAF1F8', stroke: '#1F407A', path: [[bx((to - 1) / 2) + 55, 78], [201 + ((to - 1) / 2) * 174, 280]] },
            ...(to >= 3 ? [{ label: `Σg${layers[(to - 3) / 2]} ✓`, fill: '#dcfce7', stroke: '#16a34a', path: [[bx((to - 3) / 2) + 179, 146], [201 + ((to - 3) / 2) * 174, 280]] }] : [])
          ];
      status.textContent = 'Transition in progress; destination state is pending. ' + status.textContent;
      const packets = motions.map(m => {
        const g = chip(0, 0, m.label, m.fill, m.stroke, '#14294e', { 'data-motion-object': 'true' });
        moving.append(g);
        const lengths = m.path.slice(1).map((p, i) => Math.hypot(p[0] - m.path[i][0], p[1] - m.path[i][1]));
        return { ...m, g, lengths, total: lengths.reduce((a, b) => a + b, 0) };
      });
      onFrame(p => {
        root.dataset.animationProgress = String(p);
        packets.forEach(m => {
          let d = p * m.total, i = 0;
          while (i < m.lengths.length - 1 && d > m.lengths[i]) d -= m.lengths[i++];
          const a = m.path[i], b = m.path[i + 1], f = m.lengths[i] ? d / m.lengths[i] : 1;
          const px = a[0] + (b[0] - a[0]) * f, py = a[1] + (b[1] - a[1]) * f;
          m.g.setAttribute('transform', `translate(${px},${py})`);
          if (m === packets[0]) camera(px);
        });
      });
    }
  });
  return controller;
};
})();
