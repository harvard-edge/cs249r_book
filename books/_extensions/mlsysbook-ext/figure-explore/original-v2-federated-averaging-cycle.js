/* Verified chapter artwork with live cohort broadcast, local training, round deadline enforcement, and example-weighted FedAvg aggregation. */
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
window.MLSysFigureLessons['original-v2-federated-averaging-cycle'] = async (root, context) => {
  if (!context.verifiedArtwork || !window.MLSysOriginalPlayer) throw Error('Verified source artwork and shared player required');
  if (context.verifiedArtwork.getAttribute('viewBox') !== '0 0 920 450') throw Error('Source geometry changed; review required');
  root.classList.add('v2c-lesson');
  const art = context.verifiedArtwork.cloneNode(true);
  art.removeAttribute('width');
  art.removeAttribute('height');
  art.setAttribute('role', 'img');
  art.setAttribute('aria-label', 'Whose model enters this FedAvg round? Original chapter diagram with live client uploads and deadline aggregation.');
  const overlay = el('g', { 'data-v2c-overlay': '' });
  const moving = el('g', { 'aria-hidden': 'true' });
  const focusGuard = el('rect', { x: 36, y: 30, width: 848, height: 390, fill: 'none', stroke: 'none', 'data-mlsys-focus': '' });
  art.append(overlay, moving, focusGuard);

  const chip = (x, y, label, fill = '#fef3c7', stroke = '#d97706', textFill = '#7c2d12', attrs = {}) => {
    const g = el('g', { 'data-object': label, ...attrs });
    const w = Math.max(30, label.length * 6.6 + 10);
    g.append(
      el('rect', { x: x - w / 2, y: y - 10, width: w, height: 20, rx: 3, fill, stroke, 'stroke-width': 1.5 }),
      el('text', { x, y: y + 4, 'text-anchor': 'middle', 'font-size': 10.5, 'font-weight': 700, 'font-family': 'sans-serif', fill: textFill }, label)
    );
    return g;
  };
  const hi = (x, y, w, h, stroke = '#7c3aed', sw = 2.4, dash = '') =>
    el('rect', { x: x - 2, y: y - 2, width: w + 4, height: h + 4, rx: 4, fill: 'none', stroke, 'stroke-width': sw, ...(dash ? { 'stroke-dasharray': dash } : {}) });

  const status = document.createElement('p');
  status.className = 'v2c-status';
  status.setAttribute('aria-live', 'polite');
  const note = document.createElement('p');
  note.className = 'v2c-reading';
  note.textContent = 'The coordinator accepts completed model uploads before the round deadline and never transfers raw client datasets D1–D5. The scalar models θk and local counts nk illustrate example-weighted FedAvg.';
  let mounted = false, stageNode = null, stageResize = null, cameraTarget = 0;
  function camera(x) {
    cameraTarget = x;
    if (!stageNode || stageNode.clientWidth >= 540) { if (stageNode) stageNode.scrollLeft = 0; return; }
    const scale = art.getBoundingClientRect().width / 920;
    stageNode.scrollLeft = Math.max(0, x * scale - stageNode.clientWidth / 2);
  }

  const rowYs = [146, 196, 246, 296, 346];
  const barWidths = [216, 276, 156, 306, 360];
  const ends = [496, 556, 436, 586, 640];
  const dWidths = [52, 80, 36, 66, 58];
  const counts = [1, 2, 1, 2, 3];
  const models = [1, 3, 5, 7, 9];
  const accepted = [0, 1, 2, 3];
  const n = accepted.reduce((s, k) => s + counts[k], 0);
  const sum = accepted.reduce((s, k) => s + counts[k] * models[k], 0);
  const mean = sum / n;

  const steps = [
    { label: 'Select and broadcast', description: 'The coordinator selects a cohort of 5 clients (Client 1 to Client 5, holding D1–D5 with nk = 1, 2, 1, 2, 3 local examples) and broadcasts initial global model θ = 0 while private datasets stay on each device.' },
    { label: 'Local work', description: 'Each client runs E local epochs on its private data to produce local updates θ1 = 1, θ2 = 3, θ3 = 5, θ4 = 7, and θ5 = 9. Clients 1–4 finish before the red dashed round deadline, whereas straggler Client 5 is still running.' },
    { label: 'Timely uploads', description: 'The 4 timely clients (Clients 1–4) upload their completed models (θ1 = 1, θ2 = 3, θ3 = 5, θ4 = 7) and example counts (n1 = 1, n2 = 2, n3 = 1, n4 = 2) to the coordinator before the deadline.' },
    { label: 'Close the round', description: 'When the round deadline expires, straggler Client 5 (θ5 = 9, n5 = 3) is dropped. Only the 4 timely clients enter the round, fixing the accepted example denominator at n_round = 1 + 2 + 1 + 2 = 6 examples.' },
    { label: 'Example-weighted aggregate', description: 'The coordinator computes the example-weighted sum 1×1 + 2×3 + 1×5 + 2×7 = 26 and divides by the 6 accepted examples to publish the next global model: Next θ = 26 / 6 ≈ 4.33.' }
  ];

  const controller = window.MLSysOriginalPlayer.create({
    root,
    outer: context.outer,
    layout: 'compact',
    title: 'Whose model enters this FedAvg round?',
    intro: 'Follow 5 selected clients through local training and a round deadline that aggregates 4 timely uploads across 6 accepted examples.',
    scope: 'Coordinator and 5-client FedAvg round with toy scalar models θk = 1, 3, 5, 7, 9 and local dataset sizes nk = 1, 2, 1, 2, 3 examples (D1–D5). Illustrates over-selection, round deadline enforcement, and example-weighted aggregation Σ(nk / n_round) θk.',
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
        if (typeof ResizeObserver === 'function') {
          stageResize = new ResizeObserver(() => camera(cameraTarget));
          stageResize.observe(stage);
        }
      }
      overlay.replaceChildren();
      moving.replaceChildren();
      root.dataset.modelState = String(index);
      root.dataset.accepted = index >= 2 ? '1,2,3,4' : '';
      root.dataset.denominator = String(index >= 3 ? n : 0);
      root.dataset.weightedSum = String(index === 4 ? sum : 0);
      root.dataset.aggregate = index === 4 ? String(mean) : '';

      // Dataset size tags on D1..D5
      counts.forEach((cnt, k) => {
        const dy = 135 + k * 50;
        if (index === 0) overlay.append(hi(110, dy, dWidths[k], 22, '#16a34a', 2));
        overlay.append(
          el('text', { x: 110 + dWidths[k] / 2, y: dy - 4, 'text-anchor': 'middle', 'font-size': 9.5, 'font-weight': 700, 'font-family': 'sans-serif', fill: '#15803d' }, `n${k + 1}=${cnt}`)
        );
      });

      if (index === 0) {
        overlay.append(
          hi(240, 40, 220, 58, '#7c3aed', 2.6),
          el('rect', { x: 654, y: 38, width: 238, height: 62, rx: 4, fill: '#ffffff', 'fill-opacity': 0.65 })
        );
      }
      if (index >= 1) {
        models.forEach((m, k) => {
          const y = 135 + k * 50;
          const active = index === 1;
          if (active) overlay.append(hi(280, y, barWidths[k], 22, k === 4 ? '#dc2626' : '#0284c7', 2));
          const lbl = k === 4 && index < 3 ? 'running…' : `θ${k + 1}=${m}`;
          overlay.append(chip(280 + Math.min(barWidths[k] - 32, 128), rowYs[k], lbl, k === 4 && index >= 3 ? '#fee2e2' : '#e0f2fe', k === 4 && index >= 3 ? '#dc2626' : '#0284c7', '#0f172a'));
        });
      }
      if (index >= 2) {
        accepted.forEach(k => {
          overlay.append(
            hi(ends[k], 135 + k * 50, 26, 22, '#d97706', 2.2),
            chip(ends[k] + 13, rowYs[k] - 20, `θ${k + 1}=${models[k]}`, '#fef3c7', '#d97706', '#7c2d12', { 'data-accepted-client': String(k + 1) })
          );
        });
      }
      if (index >= 3) {
        overlay.append(
          hi(640, 335, 80, 22, '#dc2626', 2.4, '4 2'),
          chip(680, 320, 'excluded (n5=3)', '#fee2e2', '#dc2626', '#991b1b', { 'data-late-region': 'true', 'data-excluded-client': '5' })
        );
      }
      if (index === 4) {
        overlay.append(hi(656, 40, 234, 58, '#7c3aed', 2.8));
      }

      const headerFormula = index === 4
        ? `Σ n_k θ_k = 1×1 + 2×3 + 1×5 + 2×7 = ${sum} · Next θ = ${sum}/${n} ≈ ${mean.toFixed(2)}`
        : index === 3
        ? `Deadline closed · Clients 1–4 accepted (n_round = 1+2+1+2 = ${n}); Client 5 dropped`
        : index === 2
        ? 'Timely uploads received: C1(θ=1,n=1), C2(θ=3,n=2), C3(θ=5,n=1), C4(θ=7,n=2)'
        : index === 1
        ? 'Local training in progress across 5 clients; deadline set at x = round cutoff'
        : 'Broadcast global θ = 0 to 5 selected clients; raw datasets D1–D5 remain local';
      overlay.append(
        el('rect', { x: 240, y: 8, width: 650, height: 24, rx: 4, fill: index === 4 ? '#f3e8ff' : '#f8fafc', stroke: index === 4 ? '#7c3aed' : '#64748b', 'stroke-width': 1.4 }),
        el('text', { x: 565, y: 24, 'text-anchor': 'middle', 'font-size': 10.8, 'font-weight': 700, 'font-family': 'sans-serif', fill: '#1e1b4b', 'data-fed-header': index === 4 ? 'result' : 'formula' }, headerFormula)
      );

      status.textContent = index === 4
        ? `Accepted examples: ${counts.slice(0, 4).join(' + ')} = ${n}. Weighted model sum: ${sum}. Next θ (rounded): ${mean.toFixed(2)}. Client 5 contributes neither model nor examples.`
        : index >= 3
        ? `Deadline closed: clients 1–4 accepted; client 5 excluded. Denominator: ${n} accepted examples.`
        : index === 2
        ? 'Coordinator holds 4 timely uploads; client 5 has not completed.'
        : 'Private data D1–D5 remains on the clients; only model copies cross the network.';
      camera(index === 0 ? 350 : index === 1 ? 420 : index === 4 ? 720 : 620);
    },
    transition({ from, to, onFrame }) {
      if (to !== from + 1) return;
      const motions = to === 1
        ? counts.map((_, k) => ({ label: k === 4 ? 'epoch…' : `θ${k + 1}`, fill: '#e0f2fe', stroke: '#0284c7', path: [[295, rowYs[k]], [Math.min(ends[k] - 18, 620), rowYs[k]]] }))
        : to === 2
        ? accepted.map(k => ({ label: `θ${k + 1}`, fill: '#fef3c7', stroke: '#d97706', path: [[ends[k] + 13, rowYs[k]], [630, rowYs[k]], [630, 69], [656, 69]] }))
        : to === 3
        ? [{ label: 'late θ5', fill: '#fee2e2', stroke: '#dc2626', path: [[630, 346], [690, 346]] }]
        : to === 4
        ? accepted.map(k => ({ label: `${counts[k]}×${models[k]}`, fill: '#f3e8ff', stroke: '#7c3aed', path: [[656, 69], [773, 69]] }))
        : [];
      status.textContent = 'Transition in progress; destination state is pending. ' + status.textContent;
      const packets = motions.map(m => {
        const g = chip(0, 0, m.label, m.fill, m.stroke, '#0f172a', { 'data-motion-object': 'true' });
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
  const destroy = controller.destroy;
  controller.destroy = () => { stageResize?.disconnect(); destroy(); };
  return controller;
};
})();
