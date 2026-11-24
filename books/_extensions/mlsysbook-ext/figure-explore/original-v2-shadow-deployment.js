/* Verified chapter artwork comparing Panel A (Shadow traffic mirroring) against Panel B (5% Canary traffic splitting and gated rollout). */
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
window.MLSysFigureLessons['original-v2-shadow-deployment'] = async (root, context) => {
  if (!context.verifiedArtwork || !window.MLSysOriginalPlayer) throw Error('Verified source artwork and shared player required');
  if (context.verifiedArtwork.getAttribute('viewBox') !== '0 0 712 362') throw Error('Source geometry changed; review required');
  root.classList.add('v2c-lesson');
  const art = context.verifiedArtwork.cloneNode(true);
  art.removeAttribute('width');
  art.removeAttribute('height');
  art.setAttribute('role', 'img');
  art.setAttribute('aria-label', 'When can a candidate response reach a user? Original chapter diagram comparing Shadow mirroring against Canary splitting.');
  const overlay = el('g', { 'data-v2c-overlay': '' });
  const moving = el('g', { 'aria-hidden': 'true' });
  const focusGuard = el('rect', { x: 28, y: 24, width: 656, height: 314, fill: 'none', stroke: 'none', 'data-mlsys-focus': '' });
  art.append(overlay, moving, focusGuard);

  const chip = (x, y, label, fill = '#fef3c7', stroke = '#d97706', textFill = '#7c2d12', attrs = {}) => {
    const g = el('g', { 'data-object': label, ...attrs });
    const w = Math.max(28, label.length * 6.5 + 10);
    g.append(
      el('rect', { x: x - w / 2, y: y - 10, width: w, height: 20, rx: 3, fill, stroke, 'stroke-width': 1.5 }),
      el('text', { x, y: y + 4, 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 700, 'font-family': 'sans-serif', fill: textFill }, label)
    );
    return g;
  };
  const hi = (x, y, w, h, stroke = '#d97706', sw = 2.3, dash = '') =>
    el('rect', { x: x - 2, y: y - 2, width: w + 4, height: h + 4, rx: 4, fill: 'none', stroke, 'stroke-width': sw, ...(dash ? { 'stroke-dasharray': dash } : {}) });

  const status = document.createElement('p');
  status.className = 'v2c-status';
  status.setAttribute('aria-live', 'polite');
  const note = document.createElement('p');
  note.className = 'v2c-reading';
  note.textContent = 'Request R19 is highlighted in amber on the verified chapter diagram. Shadow mode mirrors all 20 requests (40 executions, 0 candidate user responses); Canary mode assigns each request once (19 production, 1 candidate at the 5% stage).';
  let mounted = false, stageNode = null;
  function camera(x) {
    if (!stageNode || stageNode.clientWidth >= 540) { if (stageNode) stageNode.scrollLeft = 0; return; }
    const scale = art.getBoundingClientRect().width / 712;
    stageNode.scrollLeft = Math.max(0, x * scale - stageNode.clientWidth / 2);
  }

  const ids = Array.from({ length: 20 }, (_, i) => 'R' + i);
  const candidate = ids.slice(-1);
  const production = ids.slice(0, -1);
  const percent = (candidate.length / ids.length) * 100;

  const steps = [
    { label: 'Shadow · requests arrive', description: 'In Panel A, a batch of 20 live requests (R0–R19, with R19 highlighted in amber) arrives at the Traffic proxy (0 model executions scheduled yet; Production v1 remains the sole authority for user responses).' },
    { label: 'Shadow · copy every request', description: 'The proxy routes all 20 requests to Production v1 and mirrors a dashed copy of all 20 requests to isolated Candidate v2, scheduling 40 model executions in total (2× inference load).' },
    { label: 'Shadow · production answers', description: 'Production v1 returns all 20 user responses while Candidate v2 writes its 20 shadow predictions only to the log (0 candidate user responses reach live users).' },
    { label: 'Shadow · compare offline', description: 'Both sets of 20 outputs and latency traces meet in the violet Offline diff box. Shadow mode catches infrastructure and output regressions across 40 executions with 0 user blast radius, but cannot measure user reaction to Candidate v2.' },
    { label: 'Canary · separate batch', description: 'In Panel B, a separate batch of 20 live requests (R0–R19) arrives at the weighted Traffic splitter, which assigns each request to 1 model rather than mirroring.' },
    { label: 'Canary · weighted split', description: 'At the 5% canary stage, the splitter routes 19 of 20 requests (R0–R18, 95%) to Production v1 and routes 1 request (amber R19, 5%) to Candidate v2, keeping total inference load at 20 executions.' },
    { label: 'Canary · both may answer', description: 'Production v1 answers its 19 requests and Candidate v2 returns live response R19′ to the user (1 candidate user response), providing online user-outcome evidence within a bounded 5% blast radius.' },
    { label: 'Evidence gate', description: 'The deployment holds at the 5% stage along the 1%, 5%, 25%, 50%, and 100% progression until online metrics pass the release gate, or triggers an immediate rollback if latency SLO or error metrics drift.' }
  ];

  const controller = window.MLSysOriginalPlayer.create({
    root,
    outer: context.outer,
    layout: 'compact',
    title: 'When can a candidate response reach a user?',
    intro: 'Compare shadow mirroring at 40 model executions and 0 candidate user responses against a 5% canary split across 20 requests.',
    scope: 'Two-panel deployment comparison on a 20-request batch (R0–R19, with R19 highlighted in amber). Panel A (Shadow deployment) mirrors all 20 inputs across both models (40 executions, 0 candidate user responses); Panel B (Canary deployment) splits 19 requests (95%) to Production v1 and 1 request (5%) to Candidate v2.',
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
      root.dataset.shadowUserOwners = index >= 2 ? 'production' : '';
      root.dataset.shadowCandidateUsers = '0';
      root.dataset.canaryCandidateUsers = String(index >= 6 ? candidate.length : 0);
      root.dataset.canaryShare = String(percent);
      root.dataset.shadowExecutions = String(index >= 1 ? ids.length * 2 : 0);

      // Softly dim inactive panel so the narrated panel stands out while preserving full source context
      if (index < 4) {
        overlay.append(el('rect', { x: 8, y: 184, width: 696, height: 170, rx: 4, fill: '#ffffff', 'fill-opacity': 0.56 }));
      } else {
        overlay.append(el('rect', { x: 8, y: 8, width: 696, height: 166, rx: 4, fill: '#ffffff', 'fill-opacity': 0.56 }));
      }

      if (index === 0) {
        overlay.append(hi(16, 70, 72, 36, '#0284c7'), hi(102, 62, 90, 52, '#d97706'), chip(52, 54, '20 reqs (R19)', '#fef3c7', '#d97706', '#7c2d12', { 'data-batch': 'shadow-input' }));
      } else if (index === 1) {
        overlay.append(
          hi(102, 62, 90, 52, '#d97706'),
          hi(210, 48, 102, 16, '#0284c7', 1.8),
          hi(210, 112, 102, 16, '#7c3aed', 1.8, '3 2'),
          el('rect', { x: 307, y: 49, width: 4, height: 14, fill: '#d97706', 'data-request': 'R19', 'data-owner': 'shadow-production' }),
          el('rect', { x: 307, y: 113, width: 4, height: 14, fill: '#fef3c7', stroke: '#d97706', 'data-request': 'R19', 'data-owner': 'shadow-candidate' }),
          hi(328, 38, 120, 36, '#0284c7'),
          hi(328, 100, 120, 40, '#7c3aed', 2.2, '4 2')
        );
      } else if (index === 2) {
        overlay.append(
          hi(328, 38, 120, 36, '#16a34a'),
          hi(464, 38, 96, 36, '#16a34a'),
          hi(464, 100, 96, 40, '#7c3aed', 2.2, '4 2'),
          chip(512, 24, '20 user answers ✓', '#dcfce7', '#16a34a', '#14532d', { 'data-batch': 'shadow-user-production' }),
          chip(512, 154, '0 user answers', '#f3e8ff', '#7c3aed', '#4c1d95')
        );
      } else if (index === 3) {
        overlay.append(
          hi(584, 62, 92, 52, '#7c3aed', 2.8),
          chip(630, 46, '40 execs · 0 risk', '#f3e8ff', '#7c3aed', '#4c1d95', { 'data-batch': 'shadow-log-candidate' })
        );
      } else if (index === 4) {
        overlay.append(hi(16, 238, 72, 36, '#0284c7'), hi(102, 230, 90, 52, '#d97706'), chip(52, 222, '20 reqs (R19)', '#fef3c7', '#d97706', '#7c2d12', { 'data-batch': 'canary-input' }));
      } else if (index === 5) {
        overlay.append(
          hi(102, 230, 90, 52, '#d97706'),
          hi(210, 216, 102, 16, '#0284c7', 1.8),
          el('rect', { x: 256, y: 280, width: 5, height: 16, fill: '#d97706', 'data-request': 'R19', 'data-owner': 'canary-candidate' }),
          hi(328, 206, 120, 36, '#0284c7'),
          hi(328, 268, 120, 40, '#d97706'),
          chip(260, 307, 'R19 (5% canary)', '#fef3c7', '#d97706', '#7c2d12')
        );
      } else if (index === 6) {
        overlay.append(
          hi(328, 206, 120, 36, '#16a34a'),
          hi(328, 268, 120, 40, '#d97706'),
          chip(388, 192, '19 v1 answers', '#dcfce7', '#16a34a', '#14532d', { 'data-batch': 'canary-user-production' }),
          chip(388, 326, 'R19′ → user ✓', '#fef3c7', '#d97706', '#7c2d12', { 'data-request': 'R19', 'data-owner': 'canary-user-candidate' })
        );
      } else if (index === 7) {
        overlay.append(
          hi(512, 220, 150, 76, '#0284c7', 2.2),
          hi(516, 239, 140, 9, '#d97706', 2.2),
          hi(476, 304, 200, 24, '#dc2626', 2.4),
          chip(586, 193, 'Hold @ 5% gate', '#fef3c7', '#d97706', '#7c2d12', { 'data-rollout-share': '5' })
        );
      }

      status.textContent = index < 4
        ? `Shadow: ${index >= 1 ? ids.length * 2 : 0} model executions scheduled for ${ids.length} inputs. Candidate user responses: 0. Logs do not authorize candidate answers.`
        : `Canary: ${production.length} production / ${candidate.length} candidate requests (${percent}%). ${index >= 6 ? 'R19 receives the candidate response.' : 'R19 is assigned to the candidate only after the split.'} Further rollout is conditional on evidence.`;
      camera(index === 0 || index === 4 ? 140 : index === 1 || index === 5 ? 360 : 580);
    },
    transition({ from, to, onFrame }) {
      if (to !== from + 1) return;
      const motions = to === 1
        ? [
            { label: '20 → v1', fill: '#e0f2fe', stroke: '#0284c7', path: [[192, 88], [240, 56], [328, 56]] },
            { label: '20 copy → v2', fill: '#f3e8ff', stroke: '#7c3aed', path: [[192, 88], [240, 120], [328, 120]] }
          ]
        : to === 2
        ? [
            { label: 'v1 → user', fill: '#dcfce7', stroke: '#16a34a', path: [[448, 56], [512, 56]] },
            { label: 'v2 → log', fill: '#f3e8ff', stroke: '#7c3aed', path: [[448, 120], [512, 120]] }
          ]
        : to === 3
        ? [{ label: 'diff', fill: '#f3e8ff', stroke: '#7c3aed', path: [[560, 88], [630, 88]] }]
        : to === 5
        ? [
            { label: '19 → v1', fill: '#e0f2fe', stroke: '#0284c7', path: [[192, 256], [240, 224], [328, 224]] },
            { label: 'R19 → v2', fill: '#fef3c7', stroke: '#d97706', path: [[192, 256], [257, 288], [328, 288]] }
          ]
        : to === 6
        ? [
            { label: '19 → user', fill: '#dcfce7', stroke: '#16a34a', path: [[448, 224], [495, 224]] },
            { label: 'R19′ → user', fill: '#fef3c7', stroke: '#d97706', path: [[448, 288], [495, 288]] }
          ]
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
  return controller;
};
})();
