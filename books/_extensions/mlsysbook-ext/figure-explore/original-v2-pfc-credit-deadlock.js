/* Verified chapter artwork with live cyclic PFC buffer occupancy and watchdog recovery. */
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
window.MLSysFigureLessons['original-v2-pfc-credit-deadlock'] = async (root, context) => {
  if (!context.verifiedArtwork || !window.MLSysOriginalPlayer) throw Error('Verified source artwork and shared player required');
  if (context.verifiedArtwork.getAttribute('viewBox') !== '0 -4 680 420') throw Error('Source geometry changed; review required');
  root.classList.add('v2c-lesson');
  const art = context.verifiedArtwork.cloneNode(true);
  art.removeAttribute('width');
  art.removeAttribute('height');
  art.setAttribute('role', 'img');
  art.setAttribute('aria-label', 'What must change before a deadlocked queue can forward? Original chapter diagram with live buffer slots.');
  const overlay = el('g', { 'data-v2c-overlay': '' });
  const moving = el('g', { 'aria-hidden': 'true' });
  const focusGuard = el('rect', { x: 45, y: 18, width: 590, height: 376, fill: 'none', stroke: 'none', 'data-mlsys-focus': '' });
  art.append(overlay, moving, focusGuard);

  const chip = (x, y, label, fill = '#fef3c7', stroke = '#d97706', textFill = '#14294e', attrs = {}) => {
    const g = el('g', { 'data-object': label, ...attrs });
    const w = Math.max(28, label.length * 6.5 + 10);
    g.append(
      el('rect', { x: x - w / 2, y: y - 10, width: w, height: 20, rx: 3, fill, stroke, 'stroke-width': 1.5 }),
      el('text', { x, y: y + 4, 'text-anchor': 'middle', 'font-size': 10.5, 'font-weight': 700, 'font-family': 'sans-serif', fill: textFill }, label)
    );
    return g;
  };

  // Exact switch boxes in pfc-credit-deadlock.svg:
  // Switch 1 (b1): rect(265,20,150,80), slots x=285+i*22, y=50, w=22, h=20, status y=88
  // Switch 2 (b2): rect(475,165,150,80), slots x=495+i*22, y=195, w=22, h=20, status y=233
  // Switch 3 (b3): rect(265,310,150,80), slots x=285+i*22, y=340, w=22, h=20, status y=378
  // Switch 4 (b4): rect(55,165,150,80), slots x=75+i*22, y=195, w=22, h=20, status y=233
  const switches = [
    { box: [265, 20, 150, 80], sx: 285, sy: 50, cx: 340, cy: 60, statusY: 88 },
    { box: [475, 165, 150, 80], sx: 495, sy: 195, cx: 550, cy: 205, statusY: 233 },
    { box: [265, 310, 150, 80], sx: 285, sy: 340, cx: 340, cy: 350, statusY: 378 },
    { box: [55, 165, 150, 80], sx: 75, sy: 195, cx: 130, cy: 205, statusY: 233 }
  ];
  const queues = [Array.from({ length: 4 }, (_, q) => Array.from({ length: 4 }, (_, i) => 'ABCD'[q] + i))];
  for (let i = 1; i < 8; i++) {
    const q = queues.at(-1).map(a => [...a]);
    if (i === 3) q[0].shift();
    if (i >= 4) {
      const sender = [3, 2, 1, 0][i - 4], receiver = (sender + 1) % 4;
      q[receiver].push(q[sender].shift());
    }
    queues.push(q);
  }
  const cwPaths = [
    [[415, 60], [530, 80], [550, 165]],
    [[550, 245], [530, 330], [415, 350]],
    [[265, 350], [150, 330], [130, 245]],
    [[130, 165], [150, 80], [265, 60]]
  ];
  const pausePaths = [
    [[265, 80], [190, 165]],
    [[490, 165], [415, 80]],
    [[415, 330], [490, 245]],
    [[190, 245], [265, 330]]
  ];

  const status = document.createElement('p');
  status.className = 'v2c-status';
  status.setAttribute('aria-live', 'polite');
  const note = document.createElement('p');
  note.className = 'v2c-reading';
  note.textContent = 'Queue contents and the free slot are overlaid directly on the verified 4-switch diagram. Clockwise arrows represent packet forwarding; counterclockwise red dashed arrows represent PFC PAUSE backpressure.';
  let mounted = false, stageNode = null;
  function camera(x) {
    if (!stageNode || stageNode.clientWidth >= 540) { if (stageNode) stageNode.scrollLeft = 0; return; }
    const scale = art.getBoundingClientRect().width / 680;
    stageNode.scrollLeft = Math.max(0, x * scale - stageNode.clientWidth / 2);
  }

  const steps = [
    { label: 'All buffers full', description: 'Each of the 4 switch queues (Q1–Q4, buffers b1–b4) holds 4 packets (16 packets total: A0–A3 in Q1, B0–B3 in Q2, C0–C3 in Q3, D0–D3 in Q4). Every head packet needs 1 free slot in the next queue clockwise, leaving 0 free slots across the ring.' },
    { label: 'PAUSE travels upstream', description: 'Because all 4 slots in each queue are occupied, every switch sends a dashed red PFC PAUSE frame counterclockwise (Q1 pauses Q4, Q2 pauses Q1, Q3 pauses Q2, and Q4 pauses Q3) while solid forwarding dependencies still point clockwise.' },
    { label: 'Deadlock', description: 'The 4 queues lock in a cyclic buffer dependency deadlock: all 16 packets wait clockwise for a full downstream queue while all 4 senders are paused counterclockwise. Lossless queues cannot drop packets on their own, so waiting cannot create space.' },
    { label: 'Watchdog drops A0', description: 'A hardware watchdog timer expires and drops 1 packet (head packet A0) from Switch 1 / Q1, reducing queued packets from 16 to 15. Dropping A0 opens 1 green free slot in Q1, allowing Q1 to de-assert PAUSE toward upstream Q4.' },
    { label: 'Queue 4 forwards D0', description: 'With PAUSE cleared on link Q4 → Q1, Switch 4 / Q4 forwards head packet D0 clockwise into the free slot in Q1. Now Q1 is full again (A1, A2, A3, D0) and the 1 free slot shifts counterclockwise to Q4.' },
    { label: 'Queue 3 forwards C0', description: 'The green free slot in Q4 clears PAUSE toward upstream Switch 3 / Q3, which forwards head packet C0 clockwise into Q4. All 15 surviving packets remain in the ring as the 1 free slot moves counterclockwise to Q3.' },
    { label: 'Queue 2 forwards B0', description: 'With 1 free slot now in Q3, Switch 2 / Q2 resumes and forwards head packet B0 clockwise into Q3. The free slot steps counterclockwise into Q2, clearing backpressure toward Switch 1 / Q1.' },
    { label: 'Queue 1 forwards A1', description: 'Switch 1 / Q1 forwards its next packet A1 clockwise into Q2, returning the 1 free slot to Q1. One full rotation of 4 clockwise hops has completed after the watchdog drop, though dropped packet A0 must still be recovered by end-to-end RDMA transport retransmission.' }
  ];

  // Dim static PAUSE elements in step 0 so step 1 visibly asserts them
  const staticPauses = [...art.querySelectorAll('text,path,line')].filter(n => n.textContent.trim() === 'PAUSE' || n.getAttribute('stroke') === '#dc2626');

  const controller = window.MLSysOriginalPlayer.create({
    root,
    outer: context.outer,
    layout: 'compact',
    title: 'What must change before a deadlocked queue can forward?',
    intro: 'Trace how 4 full switch queues (16 packets total) lock in a cyclic PFC pause deadlock until a watchdog drops 1 packet to open a free slot.',
    scope: 'A 4-switch cyclic buffer dependency ring (Switch 1 / Q1 through Switch 4 / Q4, corresponding to buffers b1–b4 in the chapter text) with 4 packet slots per queue (16 packets initially: A0–A3, B0–B3, C0–C3, D0–D3). A full 4-slot queue asserts PFC PAUSE counterclockwise to its upstream neighbor, and 1 freed slot allows the upstream queue to resume forwarding clockwise.',
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
      const q = queues[index];
      const freeIdx = q.findIndex(a => a.length < 4);
      root.dataset.queues = JSON.stringify(q);
      root.dataset.dropped = index >= 3 ? 'A0' : '';
      root.dataset.packetCount = String(q.reduce((n, a) => n + a.length, 0));
      root.dataset.deadlocked = String(index <= 2);

      staticPauses.forEach(n => { n.style.opacity = index === 0 ? '0.18' : index >= 3 ? '0.45' : '1'; });

      switches.forEach((sw, s) => {
        const full = q[s].length === 4;
        if (!full) {
          overlay.append(el('rect', { x: sw.box[0] - 2, y: sw.box[1] - 2, width: sw.box[2] + 4, height: sw.box[3] + 4, rx: 4, fill: 'none', stroke: '#16a34a', 'stroke-width': 2.5 }));
        } else if (index === 2) {
          overlay.append(el('rect', { x: sw.box[0] - 2, y: sw.box[1] - 2, width: sw.box[2] + 4, height: sw.box[3] + 4, rx: 4, fill: 'none', stroke: '#dc2626', 'stroke-width': 2.2 }));
        }
        for (let slot = 0; slot < 4; slot++) {
          const label = q[s][slot] || 'free';
          const isFree = label === 'free';
          overlay.append(
            el('rect', { x: sw.sx + slot * 22, y: sw.sy, width: 22, height: 20, fill: isFree ? '#dcfce7' : '#fee2e2', stroke: isFree ? '#16a34a' : '#dc2626', 'stroke-width': isFree ? 1.6 : 1.1 }),
            el('text', { x: sw.sx + slot * 22 + 11, y: sw.sy + 13.5, 'text-anchor': 'middle', 'font-size': isFree ? 8.2 : 9, 'font-weight': 700, 'font-family': 'sans-serif', fill: isFree ? '#14532d' : '#7f1d1d', 'data-packet': label }, label)
          );
        }
        overlay.append(
          el('rect', { x: sw.cx - 64, y: sw.statusY - 12, width: 128, height: 16, rx: 2, fill: full ? '#fef3c7' : '#dcfce7' }),
          el('text', { x: sw.cx, y: sw.statusY - 1, 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 700, 'font-family': 'sans-serif', fill: full ? '#991b1b' : '#14532d' }, full ? 'queue full · PAUSE' : '1 free slot · open')
        );
      });

      // Center state badge cleanly over the center word 'deadlock' (340, 210)
      const cTitle = index === 0 ? '16 / 16 slots occupied'
        : index === 1 ? '4 PAUSE frames upstream'
        : index === 2 ? 'Cyclic PFC deadlock'
        : index === 3 ? 'Watchdog dropped A0'
        : `Forwarding resumed (15 pkts)`;
      const cSub = index <= 2 ? '0 free slots in ring' : `Free slot now at Switch ${freeIdx + 1} (b${freeIdx + 1})`;
      const cBad = index <= 2;
      overlay.append(
        el('rect', { x: 244, y: 184, width: 192, height: 46, rx: 6, fill: cBad ? '#fff1f2' : '#f0fdf4', stroke: cBad ? '#dc2626' : '#16a34a', 'stroke-width': 2 }),
        el('text', { x: 340, y: 203, 'text-anchor': 'middle', 'font-size': 11.5, 'font-weight': 700, 'font-family': 'sans-serif', fill: cBad ? '#991b1b' : '#14532d' }, cTitle),
        el('text', { x: 340, y: 220, 'text-anchor': 'middle', 'font-size': 10.5, 'font-family': 'sans-serif', fill: '#334155' }, cSub)
      );

      status.textContent = index <= 2
        ? 'Clockwise: packets wait for the next queue. Counterclockwise: PAUSE blocks the previous sender. All 16 packets remain queued.'
        : `A0 dropped by watchdog. ${q.reduce((n, a) => n + a.length, 0)} packets remain queued; free slot is in Switch ${freeIdx + 1} (b${freeIdx + 1}).`;
      camera(index < 3 ? 340 : switches[[0, 3, 2, 1, 0][index - 3]].cx);
    },
    transition({ from, to, onFrame }) {
      if (to !== from + 1) return;
      const motions = to === 1
        ? pausePaths.map(path => ({ label: 'PAUSE', fill: '#fee2e2', stroke: '#dc2626', path }))
        : to === 3
        ? [{ label: 'A0 × drop', fill: '#fee2e2', stroke: '#dc2626', path: [[296, 60], [296, 132]] }]
        : to >= 4
        ? [{ label: ['D0', 'C0', 'B0', 'A1'][to - 4], fill: '#dcfce7', stroke: '#16a34a', path: cwPaths[[3, 2, 1, 0][to - 4]] }]
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
