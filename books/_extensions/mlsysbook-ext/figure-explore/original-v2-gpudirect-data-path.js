/* Verified chapter artwork with live payload transfers across host-staged vs GPUDirect RDMA paths. */
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
window.MLSysFigureLessons['original-v2-gpudirect-data-path'] = async (root, context) => {
  if (!context.verifiedArtwork || !window.MLSysOriginalPlayer) throw Error('Verified source artwork and shared player required');
  if (context.verifiedArtwork.getAttribute('viewBox') !== '-12 -12 704 380') throw Error('Source geometry changed; review required');
  root.classList.add('v2c-lesson');
  const art = context.verifiedArtwork.cloneNode(true);
  art.removeAttribute('width');
  art.removeAttribute('height');
  art.setAttribute('role', 'img');
  art.setAttribute('aria-label', 'Which copies disappear with GPUDirect RDMA? Original chapter diagram with live payload transfers.');
  const overlay = el('g', { 'data-v2c-overlay': '' });
  const moving = el('g', { 'aria-hidden': 'true' });
  const focusGuard = el('rect', { x: 24, y: 24, width: 636, height: 310, fill: 'none', stroke: 'none', 'data-mlsys-focus': '' });
  art.append(overlay, moving, focusGuard);

  const chip = (x, y, label, fill = '#EAF1F8', stroke = '#1F407A', textFill = '#14294e', attrs = {}) => {
    const g = el('g', { 'data-object': label, ...attrs });
    const w = Math.max(30, label.length * 6.8 + 12);
    g.append(
      el('rect', { x: x - w / 2, y: y - 11, width: w, height: 22, rx: 3, fill, stroke, 'stroke-width': 1.6 }),
      el('text', { x, y: y + 4, 'text-anchor': 'middle', 'font-size': 11, 'font-weight': 700, 'font-family': 'sans-serif', fill: textFill }, label)
    );
    return g;
  };
  const hi = (x, y, w, h, stroke = '#d97706', dash = '') =>
    el('rect', { x: x - 2, y: y - 2, width: w + 4, height: h + 4, rx: 4, fill: 'none', stroke, 'stroke-width': 2.4, ...(dash ? { 'stroke-dasharray': dash } : {}) });

  const status = document.createElement('p');
  status.className = 'v2c-status';
  status.setAttribute('aria-live', 'polite');
  const note = document.createElement('p');
  note.className = 'v2c-reading';
  note.textContent = 'The dashed host-staged route and solid green GPUDirect RDMA route are alternatives on the verified chapter diagram. A read or send copies payload G without erasing its source.';
  let mounted = false, stageNode = null;
  function camera(x) {
    if (!stageNode || stageNode.clientWidth >= 540) { if (stageNode) stageNode.scrollLeft = 0; return; }
    const scale = art.getBoundingClientRect().width / 704;
    stageNode.scrollLeft = Math.max(0, (x + 12) * scale - stageNode.clientWidth / 2);
  }

  const steps = [
    { label: 'Staged · GPU source', description: 'Start the staged send with the n-byte payload G resident in GPU memory. The registered host-memory buffer is empty (0 host writes, 0 host reads, 0 × n bytes of host bus traffic), and nothing has reached the RDMA NIC or network fabric.' },
    { label: 'Staged · host write', description: 'Without GPUDirect RDMA, DMA copies the n-byte payload G along the left dashed path through the PCIe switch and CPU root complex into the registered host-memory buffer (1 × n write, 0 × n reads, 1 × n bytes of host bus traffic so far).' },
    { label: 'Staged · NIC read', description: 'The RDMA NIC DMA engine reads the n-byte staged copy back out of host memory through the CPU root complex and PCIe switch (1 × n write + 1 × n read = 2 × n bytes of total host-memory bus traffic for a single n-byte payload).' },
    { label: 'Staged · network send', description: 'The RDMA NIC transmits the n-byte payload G onto the fabric link. Host staging consumed 2 host-memory transfers (2 × n bytes of host bus traffic) before a single payload byte left the node.' },
    { label: 'Direct · independent send', description: 'Reset to an independent send of the same n-byte payload G using GPUDirect RDMA. Payload G starts in GPU memory and the registered host-memory buffer stays empty (0 host writes, 0 host reads).' },
    { label: 'Direct · NIC reads GPU', description: 'The RDMA NIC DMA engine reads the n-byte payload G directly from GPU memory across the solid green PCIe switch peer-to-peer path. The CPU root complex and host memory incur 0 writes and 0 reads (0 × n bytes of host bus traffic).' },
    { label: 'Direct · network send', description: 'The RDMA NIC transmits the same n-byte payload G to the fabric. GPUDirect RDMA eliminates all 2 × n bytes of host-memory bounce traffic while preserving the required PCIe peer-to-peer read and fabric transmission.' }
  ];

  const controller = window.MLSysOriginalPlayer.create({
    root,
    outer: context.outer,
    layout: 'compact',
    title: 'Which copies disappear with GPUDirect RDMA?',
    intro: 'Compare the host-staged send path (2 host-memory transfers) against the GPUDirect RDMA peer-to-peer PCIe path (0 host-memory transfers) for one n-byte GPU payload.',
    scope: 'One sending node containing GPU memory (holding an n-byte payload G), a PCIe switch, a CPU root complex with a registered host-memory buffer, and an RDMA NIC. Steps 1–4 trace the dashed host-staged path (2 × n bytes of host-memory traffic); Steps 5–7 trace the solid green GPUDirect RDMA peer-to-peer path (0 × n bytes of host-memory traffic).',
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
      const direct = index >= 4, phase = direct ? index - 4 : index;
      const writes = !direct && phase >= 1 ? 1 : 0;
      const reads = !direct && phase >= 2 ? 1 : 0;
      const sent = phase >= 3 || (direct && phase >= 2);
      root.dataset.path = direct ? 'direct' : 'staged';
      root.dataset.hostWrites = String(writes);
      root.dataset.hostReads = String(reads);
      root.dataset.networkSent = String(sent);

      // Highlight active hardware blocks on the verified SVG
      overlay.append(chip(306, 261, 'G', '#EAF1F8', '#1F407A', '#14294e', { 'data-location': 'gpu' }));
      if (index === 0 || index === 4) {
        overlay.append(hi(170, 230, 160, 62, direct ? '#16a34a' : '#1F407A'));
      }
      if (!direct && phase >= 1) {
        overlay.append(chip(178, 77, 'G', '#fef3c7', '#d97706', '#7c2d12', { 'data-location': 'host' }));
        if (phase === 1) overlay.append(hi(40, 46, 160, 62, '#d97706'), hi(300, 46, 140, 62, '#d97706', '4 3'));
      }
      if (direct && phase >= 1) {
        overlay.append(
          el('rect', { x: 46, y: 108, width: 148, height: 19, rx: 3, fill: '#dcfce7', stroke: '#16a34a', 'stroke-width': 1.2 }),
          el('text', { x: 120, y: 121, 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 700, 'font-family': 'sans-serif', fill: '#14532d' }, '0 × n host bounce')
        );
      }
      if ((!direct && phase >= 2) || (direct && phase >= 1)) {
        overlay.append(chip(460, 294, 'G', '#fef3c7', '#d97706', '#7c2d12', { 'data-location': 'nic' }));
        if ((!direct && phase === 2) || (direct && phase === 1)) {
          overlay.append(hi(390, 230, 140, 62, direct ? '#16a34a' : '#d97706'), hi(100, 146, 440, 44, direct ? '#16a34a' : '#d97706'));
        }
      }
      if (sent) {
        overlay.append(chip(624, 274, 'G sent ✓', '#dcfce7', '#16a34a', '#14532d', { 'data-location': 'fabric' }));
      }
      const badgeText = direct
        ? `GPUDirect RDMA · Host traffic: 0 × n`
        : `Host-staged · Host traffic: ${writes + reads} × n (${writes}W + ${reads}R)`;
      overlay.append(
        el('rect', { x: 422, y: 16, width: 252, height: 26, rx: 4, fill: direct ? '#dcfce7' : '#fef3c7', stroke: direct ? '#16a34a' : '#d97706', 'stroke-width': 1.5 }),
        el('text', { x: 548, y: 33, 'text-anchor': 'middle', 'font-size': 11, 'font-weight': 700, 'font-family': 'sans-serif', fill: direct ? '#14532d' : '#7c2d12' }, badgeText)
      );

      status.textContent = `${direct ? 'Direct' : 'Staged'} send · Host payload writes: ${writes} × n; reads: ${reads} × n; total host bus traffic: ${writes + reads} × n. Network: ${sent ? 'G sent' : 'not yet sent'}.`;
      camera([250, 160, 440, 610, 250, 440, 610][index]);
    },
    transition({ from, to, onFrame }) {
      if (to !== from + 1) return;
      const motions = to === 1
        ? [{ label: 'G (write n)', fill: '#fef3c7', stroke: '#d97706', path: [[306, 261], [306, 168], [370, 168], [370, 77], [178, 77]] }]
        : to === 2
        ? [{ label: 'G (read n)', fill: '#fef3c7', stroke: '#d97706', path: [[178, 77], [370, 77], [370, 168], [460, 168], [460, 261]] }]
        : to === 3 || to === 6
        ? [{ label: 'G → fabric', fill: '#dcfce7', stroke: '#16a34a', path: [[530, 261], [624, 261]] }]
        : to === 5
        ? [{ label: 'G (P2P)', fill: '#dcfce7', stroke: '#16a34a', path: [[306, 261], [306, 168], [460, 168], [460, 261]] }]
        : [];
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
