/* Scoped stateful extension of the inspected original Ulysses ownership-transpose artwork. */
(() => {
'use strict';
const L = {
  title: 'What changes ownership in Ulysses AllToAll?',
  intro: 'Trace how 2 AllToAll transposes switch 16 token-head tiles across 4 ranks from sequence-sharded rows to head-sharded columns and back.',
  scope: 'DeepSpeed Ulysses context parallelism with S = 4 tokens (t0–t3), h = 4 attention heads (h0–h3), and c = 4 ranks (R0–R3), forming 16 (t, h) tiles. AllToAll transposes per-rank tensor ownership from sequence-sharded [S/c, h, d_head] ([1, 4, d_head], 1 token × 4 heads per rank) to head-sharded [S, h/c, d_head] ([4, 1, d_head], 4 tokens × 1 head per rank) and back. Requires compatible head divisibility (h divisible by c, with no GQA KV-head replication shown).',
  lanes: ['Sequence-sharded input', 'Head-sharded attention', 'Sequence-sharded output'],
  steps: [
    {
      label: 'Token-row ownership',
      description: 'In the left grid (Sequence shards), the input Q/K/V tensor is partitioned along the sequence dimension as [S/c, h, d_head] = [1, 4, d_head] per rank. Each of the 4 ranks (Rt) owns 1 token row across all 4 heads (R0 owns t0, R1 owns t1, R2 owns t2, R3 owns t3; 16 tiles total).',
      cards: [[0, 't0: h0→R0 h1→R0 h2→R0 h3→R0 | t1: h0→R1 h1→R1 h2→R1 h3→R1 | t2: h0→R2 h1→R2 h2→R2 h3→R2 | t3: h0→R3 h1→R3 h2→R3 h3→R3', 'compute']]
    },
    {
      label: 'AllToAll transpose',
      description: 'Collective 1 of 2 (AllToAll) routes each (t, h) tile from token-owner rank Rt to head-owner rank Rh, transposing ownership in the center grid (Head shards) to [S, h/c, d_head] = [4, 1, d_head] per rank. Now each rank owns 1 complete head column across all 4 tokens (R0 owns h0, R1 owns h1, R2 owns h2, R3 owns h3; all 16 tiles conserved).',
      cards: [[1, 't0: h0→R0 h1→R1 h2→R2 h3→R3 | t1: h0→R0 h1→R1 h2→R2 h3→R3 | t2: h0→R0 h1→R1 h2→R2 h3→R3 | t3: h0→R0 h1→R1 h2→R2 h3→R3', 'available']]
    },
    {
      label: 'Local attention per head',
      description: 'In the bottom Local attention block, each of the 4 ranks independently runs standard single-GPU attention across the full 4-token sequence (t0–t3) for its 1 assigned head column (Rh computes head h). Because attention heads do not interact before the output projection, 0 inter-rank messages are needed during attention.',
      cards: [[1, 't0: h0→R0 h1→R1 h2→R2 h3→R3 | t1: h0→R0 h1→R1 h2→R2 h3→R3 | t2: h0→R0 h1→R1 h2→R2 h3→R3 | t3: h0→R0 h1→R1 h2→R2 h3→R3 · attention computed', 'compute']]
    },
    {
      label: 'Inverse AllToAll',
      description: 'Collective 2 of 2 (Inverse AllToAll) routes each computed attention output tile (t, h) from head-owner rank Rh back to token-owner rank Rt in the right grid (Sequence outputs). Ownership returns to [S/c, h, d_head] = [1, 4, d_head] token rows across all 16 tiles so each rank can run the output projection and MLP locally on its token.',
      cards: [[2, 't0: h0→R0 h1→R0 h2→R0 h3→R0 | t1: h0→R1 h1→R1 h2→R1 h3→R1 | t2: h0→R2 h1→R2 h2→R2 h3→R2 | t3: h0→R3 h1→R3 h2→R3 h3→R3 · output values', 'available']]
    }
  ],
  guard: 'Bijection and inverse preserve all sixteen identities; movement illustrates ownership transpose, not measured network latency.'
};
window.MLSysFigureLessons = window.MLSysFigureLessons || {};
window.MLSysFigureLessons['original-v2-ulysses-ownership-transpose'] = async (root, context) => {
  if (!context.verifiedArtwork || !window.MLSysOriginalPlayer) throw Error('Verified artwork/shared player required');
  root.classList.add('original-v2-ulysses-ownership-transpose');
  const make = (tag, cls, text) => {
    const n = document.createElement(tag);
    n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };
  const rankFills = ['#dcfce7', '#e0f2fe', '#f3e8ff', '#fef3c7'];
  const rankStrokes = ['#16a34a', '#0284c7', '#7c3aed', '#d97706'];
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
  Object.entries({ x: 24, y: 24, width: 632, height: 292, fill: 'none', stroke: 'none', 'data-mlsys-focus': '' }).forEach(([k, v]) => focusGuard.setAttribute(k, v));
  art.append(overlay, focusGuard);
  let mounted = false;

  function liveArt(index) {
    overlay.replaceChildren();
    art.dataset.liveState = String(index);
    const texts = [...art.querySelectorAll('text')].filter(t => /^R[0-3]$/.test(t.textContent));
    texts.forEach(t => {
      const x = +t.getAttribute('x');
      t.style.opacity = (index === 0 ? x < 200 : index === 3 ? x > 450 : x > 200 && x < 450) ? '1' : '.22';
    });
    art.querySelectorAll('rect[x="275"][y="230"],text[x="340"][y="255"],path[d="M340,195 V230"]').forEach(n => {
      n.style.opacity = index === 2 ? '1' : '.38';
    });
    const frame = (x, y, w, h, stroke = '#d97706') => {
      const r = document.createElementNS(NS, 'rect');
      Object.entries({ x, y, width: w, height: h, rx: 4, fill: 'none', stroke, 'stroke-width': 2.5 }).forEach(([k, v]) => r.setAttribute(k, v));
      overlay.append(r);
    };
    if (index === 0) frame(37, 72, 126, 126, '#0284c7');
    else if (index === 1) frame(277, 72, 126, 126, '#d97706');
    else if (index === 2) {
      frame(272, 227, 136, 46, '#16a34a');
      for (let h = 0; h < 4; h++) frame(281 + h * 30, 76, 28, 118, rankStrokes[h]);
      const pill = document.createElementNS(NS, 'rect');
      Object.entries({ x: 230, y: 282, width: 174, height: 20, rx: 4, fill: '#dcfce7', stroke: '#16a34a', 'stroke-width': 1.4 }).forEach(([k, v]) => pill.setAttribute(k, v));
      const msg = document.createElementNS(NS, 'text');
      Object.entries({ x: 317, y: 296, 'text-anchor': 'middle', 'font-size': 9.4, 'font-weight': 700, fill: '#14532d' }).forEach(([k, v]) => msg.setAttribute(k, v));
      msg.textContent = '0 network msgs · 4 local heads';
      overlay.append(pill, msg);
    } else if (index === 3) frame(517, 72, 126, 126, '#16a34a');
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
      if (text.includes('t0:')) {
        card.textContent = '';
        const table = make('table', 'v2b-grid');
        const caption = make('caption', '', index < 2 ? 'Input Q/K/V tiles · ownership' : 'Computed attention output tiles · ownership');
        table.append(caption);
        table.dataset.valueKind = index < 2 ? 'input' : 'output';
        table.setAttribute('aria-label', caption.textContent);
        for (let t = 0; t < 4; t++) {
          const row = make('tr', '');
          for (let h = 0; h < 4; h++) {
            const owner = index === 0 || index === 3 ? t : h;
            const cell = make('td', '', `t${t} h${h}\nR${owner}`);
            cell.dataset.tile = `${t},${h}`;
            cell.dataset.owner = String(owner);
            row.append(cell);
          }
          table.append(row);
        }
        card.append(table);
      }
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
      source.scrollLeft = Math.max(0, [100, 340, 340, 580][index] * scale - source.clientWidth / 2);
    },
    transition({ from, to, onFrame }) {
      if (to !== from + 1) return;
      root.dataset.pending = String(to);
      overlay.replaceChildren();
      if (to === 1 || to === 3) {
        const start = to === 1 ? 40 : 280, end = to === 1 ? 280 : 520;
        const tiles = [];
        for (let t = 0; t < 4; t++) {
          for (let h = 0; h < 4; h++) {
            const dstRank = to === 1 ? h : t;
            const srcRank = to === 1 ? t : h;
            const g = document.createElementNS(NS, 'g');
            g.dataset.motionObject = 'true';
            g.dataset.tileIdentity = `t${t},h${h}`;
            const r = document.createElementNS(NS, 'rect');
            Object.entries({ x: 0, y: 0, width: 30, height: 30, rx: 2, fill: rankFills[dstRank], stroke: rankStrokes[dstRank], 'stroke-width': 1.4 }).forEach(([k, v]) => r.setAttribute(k, v));
            const text = document.createElementNS(NS, 'text');
            Object.entries({ x: 15, y: 12, 'text-anchor': 'middle', 'font-size': 7.8, 'font-weight': 700, fill: '#0f172a' }).forEach(([k, v]) => text.setAttribute(k, v));
            text.textContent = `t${t}h${h}`;
            const owner = document.createElementNS(NS, 'text');
            Object.entries({ x: 15, y: 23, 'text-anchor': 'middle', 'font-size': 7.5, fill: '#334155' }).forEach(([k, v]) => owner.setAttribute(k, v));
            owner.textContent = srcRank === dstRank ? `R${dstRank}` : `R${srcRank}→R${dstRank}`;
            g.append(r, text, owner);
            overlay.append(g);
            tiles.push({ g, t, h, arc: (t - h) * 5 });
          }
        }
        onFrame(p => {
          root.dataset.animationProgress = String(p);
          source.scrollLeft = Math.max(0, (start + (end - start) * p + 60) * art.getBoundingClientRect().width / 680 - source.clientWidth / 2);
          const lift = Math.sin(Math.PI * p);
          for (const { g, t, h, arc } of tiles) {
            g.setAttribute('transform', `translate(${start + (end - start) * p + h * 30},${75 + t * 30 + arc * lift})`);
          }
        });
        return;
      }
      // Step 1 -> Step 2: 4 rank-local attention scans down each head column into Local attention (0 network messages)
      const bars = [0, 1, 2, 3].map(h => {
        const r = document.createElementNS(NS, 'rect');
        r.dataset.motionObject = 'true';
        Object.entries({ x: 282 + h * 30, y: 76, width: 26, height: 22, rx: 3, fill: rankFills[h], stroke: rankStrokes[h], 'stroke-width': 2 }).forEach(([k, v]) => r.setAttribute(k, v));
        overlay.append(r);
        return r;
      });
      const badge = document.createElementNS(NS, 'g');
      badge.dataset.motionObject = 'true';
      const br = document.createElementNS(NS, 'rect');
      Object.entries({ x: 244, y: 204, width: 192, height: 20, rx: 4, fill: '#dcfce7', stroke: '#16a34a', 'stroke-width': 1.5 }).forEach(([k, v]) => br.setAttribute(k, v));
      const bt = document.createElementNS(NS, 'text');
      Object.entries({ x: 340, y: 218, 'text-anchor': 'middle', 'font-size': 9.8, 'font-weight': 700, fill: '#14532d' }).forEach(([k, v]) => bt.setAttribute(k, v));
      bt.textContent = 'Local attention · 0 network msgs';
      badge.append(br, bt);
      overlay.append(badge);
      onFrame(p => {
        root.dataset.animationProgress = String(p);
        const y = 76 + 96 * p;
        bars.forEach(r => r.setAttribute('y', String(y)));
        source.scrollLeft = Math.max(0, 340 * art.getBoundingClientRect().width / 680 - source.clientWidth / 2);
      });
    }
  });
  return { pause: () => player.pause(), destroy: () => player.destroy() };
};
})();
