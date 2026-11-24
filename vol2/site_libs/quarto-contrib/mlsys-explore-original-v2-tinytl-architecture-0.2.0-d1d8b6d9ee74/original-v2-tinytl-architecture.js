/* Verified chapter artwork with live forward saved-input caching and backward gradient propagation across Full Fine-Tuning vs TinyTL. */
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
window.MLSysFigureLessons['original-v2-tinytl-architecture'] = async (root, context) => {
  if (!context.verifiedArtwork || !window.MLSysOriginalPlayer) throw Error('Verified source artwork and shared player required');
  if (context.verifiedArtwork.getAttribute('viewBox') !== '0 0 900 420') throw Error('Source geometry changed; review required');
  root.classList.add('v2c-lesson');
  const art = context.verifiedArtwork.cloneNode(true);
  art.removeAttribute('width');
  art.removeAttribute('height');
  art.setAttribute('role', 'img');
  art.setAttribute('aria-label', 'Why does a frozen backbone still need a backward pass? Original chapter diagram with live saved-input tiles and backward gradients.');
  const overlay = el('g', { 'data-v2c-overlay': '' });
  const moving = el('g', { 'aria-hidden': 'true' });
  const focusGuard = el('rect', { x: 36, y: 30, width: 828, height: 360, fill: 'none', stroke: 'none', 'data-mlsys-focus': '' });
  art.append(overlay, moving, focusGuard);

  const chip = (x, y, label, fill = '#fef3c7', stroke = '#d97706', textFill = '#7c2d12', attrs = {}) => {
    const g = el('g', { 'data-object': label, ...attrs });
    const w = Math.max(28, label.length * 6.8 + 10);
    g.append(
      el('rect', { x: x - w / 2, y: y - 11, width: w, height: 22, rx: 3, fill, stroke, 'stroke-width': 1.5 }),
      el('text', { x, y: y + 4, 'text-anchor': 'middle', 'font-size': 11, 'font-weight': 700, 'font-family': 'sans-serif', fill: textFill }, label)
    );
    return g;
  };
  const hi = (x, y, w, h, stroke = '#1F407A', sw = 2.4, dash = '') =>
    el('rect', { x: x - 2, y: y - 2, width: w + 4, height: h + 4, rx: 4, fill: 'none', stroke, 'stroke-width': sw, ...(dash ? { 'stroke-dasharray': dash } : {}) });

  const status = document.createElement('p');
  status.className = 'v2c-status';
  status.setAttribute('aria-live', 'polite');
  const note = document.createElement('p');
  note.className = 'v2c-reading';
  note.textContent = 'Each saved-input tile on the verified chapter diagram is released as backpropagation consumes its stage. Freezing backbone weights W shrinks saved inputs from full feature maps to compact lite-residual inputs without shortening the 3-block backward path.';
  let mounted = false, stageNode = null;
  function camera(x) {
    if (!stageNode || stageNode.clientWidth >= 540) { if (stageNode) stageNode.scrollLeft = 0; return; }
    const scale = art.getBoundingClientRect().width / 900;
    stageNode.scrollLeft = Math.max(0, x * scale - stageNode.clientWidth / 2);
  }

  const blockYs = [118, 170, 222];
  const centerYs = [136, 188, 240, 308];
  const saved = index => [0, 1, 2, 3].map(i => index >= i + 1 && (index < 5 || i < 8 - index));

  const steps = [
    { label: 'Before forward', description: 'Both 3-block architectures hold persistent weights before the forward pass begins (0 live saved-input tiles). Panel A marks backbone weights W trainable, whereas Panel B (TinyTL) freezes backbone weights W and trains only bias b, lite residual side-modules, and the 1 Classifier.' },
    { label: 'Forward · block 1', description: 'As the forward pass traverses Block 1, Panel A caches full-resolution input tile A1 (needed later for weight gradient ∂L/∂W = (∂L/∂y)xᵀ), whereas Panel B freezes W and caches only the downsampled lite-residual input s1 (1 saved-input record live per model).' },
    { label: 'Forward · block 2', description: 'Forward execution advances through Block 2: Panel A retains full input tile A2 alongside A1, while Panel B retains only small lite-residual input s2 alongside s1 (2 saved-input records live per model).' },
    { label: 'Forward · block 3', description: 'Block 3 executes in both panels, adding full input tile A3 in Panel A versus compact lite-residual input s3 in Panel B. All 3 block-input records (A1–A3 vs s1–s3) remain live in memory awaiting backpropagation.' },
    { label: 'Classifier forward', description: 'The 1 Classifier stage runs at the bottom of both panels and caches its input tile A4. Peak saved-input occupancy reaches 4 live records per model: wide tiles A1, A2, A3, A4 in Panel A versus compact s1, s2, s3 plus A4 in Panel B.' },
    { label: 'Classifier backward', description: 'Backpropagation starts at the Classifier: both panels consume and release Classifier input tile A4 to compute Classifier gradients and send activation gradient ∂L upward to Block 3 (3 block-input records remain live).' },
    { label: 'Backward · block 3', description: 'Activation gradient ∂L traverses Block 3 in both models. Panel A consumes wide tile A3 to form backbone weight gradients, while Panel B computes bias gradient ∂L/∂b = ∂L/∂y and consumes small tile s3 for the lite residual without updating frozen W (2 records remain live).' },
    { label: 'Backward · block 2', description: 'The dashed backward pass continues through frozen Block 2 so ∂L can reach earlier trainable biases and lite residuals. Panel A consumes and releases A2 while Panel B consumes and releases s2 (1 record remains live).' },
    { label: 'Backward · block 1', description: 'Activation gradient ∂L reaches Block 1 at the top of the network, consuming and releasing A1 in Panel A and s1 in Panel B (0 live saved-input records remain). Freezing backbone weights shrunk saved inputs without shortening the 3-block backward path.' },
    { label: 'Optimizer update', description: 'The optimizer commits parameter updates: Panel A updates every backbone weight W and the Classifier, whereas Panel B leaves frozen backbone weights W unchanged and updates only biases b′, lite residuals, and the Classifier ✓.' }
  ];

  const controller = window.MLSysOriginalPlayer.create({
    root,
    outer: context.outer,
    layout: 'compact',
    title: 'Why does a frozen backbone still need a backward pass?',
    intro: 'Compare saved activation tiles across 3 blocks and 1 classifier as gradients propagate backward through full fine-tuning and TinyTL.',
    scope: 'Three-block backbone plus 1 classifier comparing Panel A (Full fine-tuning, updating all weights W) against Panel B (TinyTL, freezing backbone W and training biases b, lite residual modules, and the classifier). Saved-input tile widths (A1–A3 vs s1–s3) are schematic.',
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
      const live = saved(index);
      root.dataset.saved = JSON.stringify(live);
      root.dataset.tinyBackbone = 'W0,W0,W0';
      root.dataset.backwardBlock = index >= 6 && index <= 8 ? String(9 - index) : '';

      // Saved-input tile status across Panel A and Panel B on the verified chapter diagram
      for (let panel = 0; panel < 2; panel++) {
        for (let b = 0; b < 4; b++) {
          const isLive = live[b];
          const tx = b < 3 ? (panel ? 440 : 30) : (panel ? 440 : 30);
          const ty = b < 3 ? 125 + b * 52 : 295;
          const tw = b < 3 ? (panel ? 20 : 80) : 40;
          const th = b < 3 ? 22 : 26;
          const maskW = b < 3 ? (panel ? 106 : 106) : 40;
          const g = el('g', { 'data-saved-panel': String(panel), 'data-saved-block': String(b + 1), 'data-live': String(isLive) });
          if (isLive) {
            g.append(hi(tx, ty, tw, th, '#0284c7', 2.2));
          } else {
            g.append(
              el('rect', { x: tx - 1, y: ty - 1, width: maskW + 2, height: th + 2, rx: 2, fill: '#ffffff', 'fill-opacity': 0.84, stroke: '#cbd5e1', 'stroke-width': 1.2, 'stroke-dasharray': '3 2' })
            );
          }
          overlay.append(g);
        }
      }

      // Active stage highlights (Forward 1..4, Backward 5..8, Update 9)
      if (index >= 1 && index <= 4) {
        const s = index - 1;
        const by = s < 3 ? blockYs[s] : 288;
        const bh = s < 3 ? 36 : 40;
        overlay.append(hi(180, by, 150, bh, '#0284c7', 2.5), hi(590, by, 150, bh, '#0284c7', 2.5));
        if (s < 3) overlay.append(hi(760, 122 + s * 52, 110, 28, '#0284c7', 2.2));
      } else if (index >= 5 && index <= 8) {
        const s = 8 - index; // 3=Classifier, 2=Block3, 1=Block2, 0=Block1
        const by = s < 3 ? blockYs[s] : 288;
        const bh = s < 3 ? 36 : 40;
        const cy = centerYs[s];
        overlay.append(hi(180, by, 150, bh, '#d97706', 2.5), hi(590, by, 150, bh, '#d97706', 2.5));
        if (s < 3) overlay.append(hi(700, 124 + s * 52, 32, 24, '#d97706', 2.2), hi(760, 122 + s * 52, 110, 28, '#d97706', 2.2));
        overlay.append(
          chip(162, cy, '∂L', '#fef3c7', '#d97706', '#7c2d12', { 'data-gradient-panel': '0', 'data-gradient-at': index === 5 ? 'classifier' : String(9 - index) }),
          chip(572, cy, '∂L', '#fef3c7', '#d97706', '#7c2d12', { 'data-gradient-panel': '1', 'data-gradient-at': index === 5 ? 'classifier' : String(9 - index) })
        );
      } else if (index === 9) {
        // Panel A updates all 3 backbone blocks + classifier; Panel B updates only bias, lite residual, and classifier
        blockYs.forEach((by, k) => {
          overlay.append(
            hi(180, by, 150, 36, '#16a34a', 2.6),
            hi(700, 124 + k * 52, 32, 24, '#16a34a', 2.4),
            hi(760, 122 + k * 52, 110, 28, '#16a34a', 2.4)
          );
        });
        overlay.append(
          hi(180, 288, 150, 40, '#16a34a', 2.6),
          hi(590, 288, 150, 40, '#16a34a', 2.6),
          chip(255, 344, 'W + Classifier updated ✓', '#dcfce7', '#16a34a', '#14532d', { 'data-weight': 'trainable', 'data-version': '1' }),
          chip(715, 344, 'W₀ frozen · b′ + lite + cls ✓', '#dcfce7', '#16a34a', '#14532d', { 'data-weight': 'frozen', 'data-version': '0' })
        );
      }

      const liveCount = live.filter(Boolean).length;
      const badgeLabel = index === 9
        ? 'Optimizer commit · TinyTL W₀ frozen'
        : `${index >= 5 ? 'Backward' : index === 0 ? 'Idle' : 'Forward'} · Live saved inputs: ${liveCount} / 4 per panel`;
      overlay.append(
        el('rect', { x: 316, y: 6, width: 268, height: 22, rx: 4, fill: index === 9 ? '#dcfce7' : index >= 5 ? '#fef3c7' : '#e0f2fe', stroke: index === 9 ? '#16a34a' : index >= 5 ? '#d97706' : '#0284c7', 'stroke-width': 1.4 }),
        el('text', { x: 450, y: 21, 'text-anchor': 'middle', 'font-size': 10.5, 'font-weight': 700, 'font-family': 'sans-serif', fill: '#0f172a' }, badgeLabel)
      );

      status.textContent = index === 9
        ? 'Update committed: full backbone weights change; TinyTL backbone W₀ stays frozen. Biases, lite residuals, and classifier train.'
        : index >= 5
        ? `Backward reaches ${index === 5 ? 'the classifier' : `block ${9 - index}`} in both models. Live saved-input records per model: ${liveCount}.`
        : `Forward progress: ${Math.min(index, 4)} stages complete; live saved-input records per model: ${liveCount}.`;
      camera(index >= 5 ? 640 : 450);
    },
    transition({ from, to, onFrame }) {
      if (to !== from + 1) return;
      const motions = to <= 4
        ? [0, 1].map(panel => ({
            label: to === 1 ? 'x₀' : `x${to - 1}`,
            fill: '#e0f2fe',
            stroke: '#0284c7',
            path: [[panel ? 665 : 255, to === 1 ? 81 : centerYs[to - 2]], [panel ? 665 : 255, centerYs[to - 1]]]
          }))
        : to <= 8
        ? [0, 1].map(panel => ({
            label: '∂L',
            fill: '#fef3c7',
            stroke: '#d97706',
            path: [[panel ? 572 : 162, to === 5 ? 334 : centerYs[9 - to]], [panel ? 572 : 162, to === 5 ? 308 : centerYs[8 - to]]]
          }))
        : to === 9
        ? [{ label: 'step ✓', fill: '#dcfce7', stroke: '#16a34a', path: [[715, 308], [715, 136]] }]
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
          if (m === packets[Math.min(1, packets.length - 1)]) camera(px);
        });
      });
    }
  });
  return controller;
};
})();
