/* Verified chapter artwork with live 16-token Dispatch & Return AllToAll shuffles. */
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
window.MLSysFigureLessons['original-v2-moe-all-to-all-routing'] = async (root, context) => {
  if (!context.verifiedArtwork || !window.MLSysOriginalPlayer) throw Error('Verified source artwork and shared player required');
  if (context.verifiedArtwork.getAttribute('viewBox') !== '0 0 760 438') throw Error('Source geometry changed; review required');
  root.classList.add('v2c-lesson');
  const art = context.verifiedArtwork.cloneNode(true);
  art.removeAttribute('width');
  art.removeAttribute('height');
  art.setAttribute('role', 'img');
  art.setAttribute('aria-label', 'How does each expert result find its original token? Original chapter diagram with live token routing.');
  const overlay = el('g', { 'data-v2c-overlay': '' });
  const moving = el('g', { 'aria-hidden': 'true' });
  const focusGuard = el('rect', { x: 28, y: 26, width: 704, height: 388, fill: 'none', stroke: 'none', 'data-mlsys-focus': '' });
  art.append(overlay, moving, focusGuard);

  // Exact slot centers in moe-all-to-all-routing.svg:
  // Origin Rank r (0..3), slot s (0..3): rect(42+s*32, 104+r*90, 24, 24) -> center (54+s*32, 116+r*90)
  // Expert r (0..3), slot s (0..3): rect(317+s*32, 104+r*90, 24, 24) -> center (329+s*32, 116+r*90)
  // Return Rank r (0..3), slot s (0..3): rect(592+s*32, 104+r*90, 24, 24) -> center (604+s*32, 116+r*90)
  const labels = 'abcdefghijklmnop'.split('');
  const assignment = [2, 0, 3, 1, 1, 3, 0, 2, 0, 2, 1, 3, 3, 1, 2, 0];
  const expertRows = Array.from({ length: 4 }, (_, r) => labels.filter((_, i) => assignment[i] === r));
  const origin = i => [54 + (i % 4) * 32, 116 + Math.floor(i / 4) * 90];
  const expert = i => [329 + expertRows[assignment[i]].indexOf(labels[i]) * 32, 116 + assignment[i] * 90];
  const returned = i => [604 + (i % 4) * 32, 116 + Math.floor(i / 4) * 90];

  const tile = (x, y, txt, isAmber = false, isDone = false, attrs = {}) => {
    const g = el('g', attrs);
    const fill = isAmber ? '#fef3c7' : isDone ? '#dcfce7' : '#EAF1F8';
    const stroke = isAmber ? '#d97706' : isDone ? '#16a34a' : '#1F407A';
    g.append(
      el('rect', { x: x - 12, y: y - 12, width: 24, height: 24, rx: 2, fill, stroke, 'stroke-width': isAmber ? 2.2 : 1.3 }),
      el('text', { x, y: y + 4, 'text-anchor': 'middle', 'font-size': txt.length > 2 ? 8.5 : 10.5, 'font-weight': 700, 'font-family': 'sans-serif', fill: isAmber ? '#7c2d12' : '#14294e' }, txt)
    );
    return g;
  };

  const status = document.createElement('p');
  status.className = 'v2c-status';
  status.setAttribute('aria-live', 'polite');
  const note = document.createElement('p');
  note.className = 'v2c-reading';
  note.textContent = 'Amber highlights token a on the verified chapter diagram while all 16 tokens conserve their origin metadata across Dispatch and Return AllToAll.';
  let mounted = false, stageNode = null;
  function camera(x) {
    if (!stageNode || stageNode.clientWidth >= 540) { if (stageNode) stageNode.scrollLeft = 0; return; }
    const scale = art.getBoundingClientRect().width / 760;
    stageNode.scrollLeft = Math.max(0, x * scale - stageNode.clientWidth / 2);
  }

  const steps = [
    { label: 'Tokens at their owners', description: 'In column 1 (Tokens / top-1 gate), each of the 4 ranks holds 4 sequence tokens (16 tokens a–p total). Each rank’s top-1 gate selects 1 destination expert per token and records its routing metadata: on Rank 0, amber-highlighted token a in slot 0 selects Expert 2.' },
    { label: 'Dispatch to expert ranks', description: 'The first AllToAll exchange (Dispatch) shuffles all 16 tokens across the fabric to the ranks hosting their selected experts: Expert 0 receives (b, g, i, p), Expert 1 receives (d, e, k, n), Expert 2 receives (a, h, j, o) including amber token a from Rank 0, and Expert 3 receives (c, f, l, m).' },
    { label: 'Expert computation', description: 'In the center column (Expert FFNs), each of the 4 experts runs its feed-forward network locally on its 4 dispatched tokens (a–p → a′–p′). At Expert 2, input token a is transformed into expert activation a′ while retaining its origin metadata (Rank 0, slot a).' },
    { label: 'Inverse return', description: 'The second AllToAll exchange (Return) reverses the dispatch permutation across the fabric, sending all 16 expert outputs (a′–p′) back to their originating ranks and sequence slots. Amber output a′ travels from Expert 2 back to Rank 0, slot a.' },
    { label: 'Combine at origin', description: 'In the right column (Combine / gate-weighted sum), each of the 4 ranks multiplies its 4 returned expert outputs by their saved gating weights (on Rank 0, y_a = gate₂(a) × a′), restoring original sequence order after 2 AllToAll collectives.' }
  ];

  const controller = window.MLSysOriginalPlayer.create({
    root,
    outer: context.outer,
    layout: 'compact',
    title: 'How does each expert result find its original token?',
    intro: 'Follow token a from Rank 0 to Expert 2 and back as 2 AllToAll shuffles route 16 tokens across 4 ranks and 4 experts.',
    scope: 'Expert parallelism across 4 ranks (Rank 0–Rank 3), each owning 4 input tokens (16 tokens total: a–d on Rank 0, e–h on Rank 1, i–l on Rank 2, m–p on Rank 3) and hosting 1 expert FFN (Expert 0–Expert 3). Under top-1 gating, 1 forward MoE layer requires 2 AllToAll collectives (Dispatch and Return) before the gate-weighted Combine step.',
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
      root.dataset.liveCount = String(labels.length);
      root.dataset.tokenA = index === 0 ? 'rank0:a' : index <= 2 ? 'expert2' : 'rank0:a';
      root.dataset.valueA = index < 2 ? 'a' : index === 2 || index === 3 ? 'a-prime' : 'gate2-times-a-prime';

      // Soft veil on inactive columns so active column pops while original diagram remains visible
      if (index === 0) overlay.append(el('rect', { x: 300, y: 68, width: 435, height: 352, fill: '#ffffff', 'fill-opacity': '0.52' }));
      if (index === 1 || index === 2) {
        overlay.append(
          el('rect', { x: 26, y: 68, width: 158, height: 352, fill: '#ffffff', 'fill-opacity': '0.45' }),
          el('rect', { x: 576, y: 68, width: 158, height: 352, fill: '#ffffff', 'fill-opacity': '0.52' })
        );
      }
      if (index >= 3) overlay.append(el('rect', { x: 26, y: 68, width: 434, height: 352, fill: '#ffffff', 'fill-opacity': '0.45' }));

      // Highlight token a's route across the AllToAll bands
      const a0 = origin(0), a1 = expert(0), a2 = returned(0);
      if (index >= 1) {
        overlay.append(el('line', { x1: 180, y1: a0[1], x2: 305, y2: a1[1], stroke: '#d97706', 'stroke-width': 2.5 }));
      }
      if (index >= 3) {
        overlay.append(el('line', { x1: 455, y1: a1[1], x2: 580, y2: a2[1], stroke: '#d97706', 'stroke-width': 2.5 }));
      }

      // Render live 16 tokens in their exact verified SVG slots
      labels.forEach((lbl, i) => {
        const pos = index === 0 ? origin(i) : index <= 2 ? expert(i) : returned(i);
        const txt = index === 4 ? 'y' + lbl : lbl + (index >= 2 ? '′' : '');
        overlay.append(tile(pos[0], pos[1], txt, i === 0, index === 4 && i > 0, {
          'data-live-token': 'true',
          'data-origin': Math.floor(i / 4),
          'data-slot': i % 4,
          'data-expert': assignment[i],
          'data-location': index === 0 ? 'origin' : index <= 2 ? 'expert' : 'returned'
        }));
      });

      // Column focus frame
      const colX = index === 0 ? 28 : index <= 2 ? 303 : 578;
      overlay.append(el('rect', { x: colX, y: 72, width: 154, height: 346, rx: 5, fill: 'none', stroke: index === 4 ? '#16a34a' : '#d97706', 'stroke-width': 2.2 }));

      status.textContent = `Token a · ${index === 0 ? 'Rank 0, slot a → gate selects Expert 2' : index === 1 ? 'dispatched to Expert 2' : index === 2 ? 'a′ = Expert 2(a), retaining origin Rank 0 / slot a' : index === 3 ? 'a′ returned to Rank 0 / slot a' : 'Rank 0 combines y_a = gate₂(a) × a′'}. All ${labels.length} token identities conserved.`;
      camera(index === 0 ? 105 : index <= 2 ? 380 : 655);
    },
    transition({ from, to, onFrame }) {
      if (to !== from + 1) return;
      const motions = to === 1 || to === 3
        ? labels.map((lbl, i) => ({
            label: lbl + (to === 3 ? '′' : ''),
            amber: i === 0,
            path: to === 1 ? [origin(i), expert(i)] : [expert(i), returned(i)]
          }))
        : to === 2
        ? [{ label: 'a → a′', amber: true, path: [expert(0), [expert(0)[0], expert(0)[1] - 16], expert(0)] }]
        : to === 4
        ? [{ label: 'y_a', amber: true, path: [returned(0), [returned(0)[0], returned(0)[1] - 16], returned(0)] }]
        : [];
      if (motions.length === 16) overlay.querySelectorAll('[data-live-token]').forEach(n => n.setAttribute('visibility', 'hidden'));
      status.textContent = 'Transition in progress; destination state is pending. ' + status.textContent;
      const packets = motions.map(m => {
        const g = tile(0, 0, m.label, m.amber, false, { 'data-motion-object': 'true' });
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
