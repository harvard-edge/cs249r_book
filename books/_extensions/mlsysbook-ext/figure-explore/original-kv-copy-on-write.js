/* The original paged-KV drawing: table references and KV payload have different lifetimes. */
(() => {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons['original-kv-copy-on-write'] = async (root, context) => {
    if (!context.verifiedArtwork || !window.MLSysOriginalPlayer) throw Error('Verified paged-KV artwork and original player required');
    const svg = context.verifiedArtwork.cloneNode(true);
    if (svg.getAttribute('viewBox') !== '0 0 720 600') throw Error('Unreviewed paged-KV geometry');
    root.classList.add('mlsys-original-kv-cow');
    svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', 'Original two-panel paged KV block mapping and partial-tail copy-on-write diagram');
    const node = (tag, attrs = {}, text) => {
      const n = document.createElementNS(NS, tag);
      Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, String(v)));
      if (text !== undefined) n.textContent = text;
      return n;
    };
    const texts = [...svg.querySelectorAll('text')], rects = [...svg.querySelectorAll('rect')], paths = [...svg.querySelectorAll('path')];
    const at = (nodes, attrs) => {
      const found = nodes.filter(n => Object.entries(attrs).every(([key, value]) => Number(n.getAttribute(key)) === value));
      if (found.length !== 1) throw Error('Paged-KV artwork region changed: ' + JSON.stringify(attrs));
      return found[0];
    };
    const bTable = [260, 420, 580].map(x => at(texts, {x, y: 569}));
    const aTable = [260, 420, 580].map(x => at(texts, {x, y: 395}));
    if (aTable.map(n => n.textContent).join(',') !== '3,7,9') throw Error('Paged-KV prefix changed');
    const bCaption = at(texts, {x: 20, y: 578});
    const privateSlots = [532, 556, 580, 604].map(x => at(rects, {x, y: 492, width: 24}));
    const privateValues = [544, 568, 592].map(x => at(texts, {x, y: 509.5}));
    const privateOutline = at(rects, {x: 532, y: 492, width: 96});
    const privateLabel = at(texts, {x: 524, y: 510});
    const sourceValues = [544, 568].map(x => at(texts, {x, y: 449.5}));
    const path = value => {const p = paths.find(p => p.getAttribute('d') === value); if (!p) throw Error('Paged-KV connector changed: ' + value); return p;};
    const bRefs = ['M260,552 V458', 'M420,552 V458', 'M580,552 V518'].map(path);
    const copyArrow = path('M556,458 V492');
    const copyLabel = at(texts, {x: 568, y: 479});
    const overlay = node('g', {class: 'okv-overlay'}); svg.append(overlay);
    const inlet = node('g', {class: 'okv-inlet'});
    inlet.append(node('rect', {x: 638, y: 488, width: 70, height: 33, rx: 3, fill: '#dcfce7', stroke: '#16a34a'}), node('text', {x: 673, y: 501, 'text-anchor': 'middle'}, 'newly computed'), node('text', {x: 673, y: 514, 'text-anchor': 'middle'}, 'KV @ position 10'));
    overlay.append(inlet);
    overlay.append(node('text', {x: 625, y: 395, class: 'okv-local-label', 'text-anchor': 'middle'}, 'A table'), node('text', {x: 625, y: 569, class: 'okv-local-label', 'text-anchor': 'middle'}, 'B table'));
    const focus = node('rect', {class: 'okv-focus', fill: 'none', rx: 2, 'stroke-width': 2}); overlay.append(focus);
    const packet = node('g', {class: 'okv-packet', visibility: 'hidden'});
    packet.append(node('rect', {x: -19, y: -11, width: 38, height: 22, rx: 3}), node('text', {'text-anchor': 'middle', y: 4})); overlay.append(packet);
    const stages = [
      ['A retains a ten-token prefix', 'Panel (a) locates entries for the same prefix used in panel (b). A references physical blocks 3, 7, and 9. Its tail holds initialized KV for positions 8 and 9; the remaining slots are allocated but unwritten. B does not exist yet.'],
      ['Find logical block 2', 'Divide token position 9 by the four-position block size and take the floor. The result selects logical block 2 and its table entry. This is address metadata, not a copy of KV payload.'],
      ['Read physical block ID 9', 'Entry 2 of the block table contains physical block ID 9. Follow the original elbow connector to that scattered block. A physical block ID is a pool index, not a byte address.'],
      ['Select slot 1', 'The remainder of 9 divided by 4 is 1. The second slot in physical block 9 holds this position’s initialized KV. The matching numeral 9 in the table and in the cell has two different meanings: block ID and token position.'],
      ['Fork B by sharing references', 'B receives its own block table [3, 7, 9]. The full blocks and the partial tail now have two candidate references each. No KV payload is copied. B cannot privately append into the shared tail.'],
      ['Reserve private block 14', 'The allocator reserves block 14 for B. All four slots are still unwritten, and B’s active table continues to point to block 9. A reservation keeps this block unavailable to others even before a table references it.'],
      ['Copy initialized position 8', 'Copy the KV record for position 8 from shared block 9, slot 0, to private block 14, slot 0. The source remains intact. B still references the shared tail and cannot append yet.'],
      ['Copy initialized position 9', 'Copy the other initialized KV record, position 9, into private slot 1. Do not copy unwritten slots 2 and 3. Both candidate tables still reference block 9 until the replacement tail is ready.'],
      ['Remap only B’s tail entry', 'After both initialized records have been copied, change B’s table entry 2 from 9 to 14. Block 9 keeps A’s reference; block 14 gains B’s reference. Full blocks 3 and 7 remain shared.'],
      ['Append position 10 in B’s tail', 'B’s table entry 14 resolves the destination; the table stores only a block ID. Newly computed KV arrives from the separate inlet and is written to slot 2 of block 14. A’s table and block 9 are unchanged. Sharing avoids copying the full prefix; this private append requires one new tail block and copies only its initialized predecessors.']
    ].map(([label, description]) => ({label, description}));
    const model = index => ({
      a: [3, 7, 9], b: index >= 4 ? [3, 7, index >= 8 ? 14 : 9] : null,
      allocated: index >= 5, private: [index >= 6 ? 8 : null, index >= 7 ? 9 : null, index >= 9 ? 10 : null, null],
      refs: {3: index >= 4 ? 2 : 1, 7: index >= 4 ? 2 : 1, 9: index >= 4 && index < 8 ? 2 : 1, 14: index >= 8 ? 1 : 0}
    });
    const canvas = document.createElement('div'); canvas.className = 'okv-canvas'; canvas.tabIndex = 0;
    canvas.setAttribute('aria-label', 'Original block diagram. Scroll horizontally and vertically to inspect; the view follows the active mapping or tail.'); canvas.append(svg);
    const cue = document.createElement('p'); cue.className = 'okv-pan-cue'; cue.textContent = 'Swipe the diagram to inspect both panels. The view follows the active table entry or KV block.';
    const state = document.createElement('div'); state.className = 'okv-state'; state.setAttribute('aria-label', 'Exact mapping and ownership state');
    const rows = ['A block table', 'B block table', 'Block 14 slots', 'Table references · 3 / 7 / 9 / 14', 'B append gate'];
    const values = rows.map(label => {const row = document.createElement('div'), name = document.createElement('span'), value = document.createElement('strong'); name.textContent = label; row.append(name, value); state.append(row); return value;});
    let active = [404, 130];
    function follow(x, y) {
      active = [x, y];
      const scale = svg.getBoundingClientRect().width / 720;
      if (canvas.scrollWidth > canvas.clientWidth) canvas.scrollLeft = Math.max(0, x * scale - canvas.clientWidth / 2);
      if (canvas.scrollHeight > canvas.clientHeight) canvas.scrollTop = Math.max(0, y * scale - canvas.clientHeight / 2);
    }
    const camera = [[404, 140], [404, 100], [550, 205], [550, 249], [580, 477], [580, 477], [556, 477], [568, 477], [610, 477], [610, 477]];
    function render(i, {stage, details}) {
      if (!canvas.isConnected) stage.append(cue, canvas, state);
      const m = model(i);
      overlay.querySelectorAll('.okv-reference-copy').forEach(n => n.remove());
      packet.setAttribute('visibility', 'hidden');
      inlet.setAttribute('visibility', i >= 8 ? 'visible' : 'hidden');
      bTable.forEach((n, j) => {n.textContent = m.b ? m.b[j] : '—';});
      bCaption.textContent = m.b ? 'block table' : 'not forked';
      bRefs.forEach(n => {n.setAttribute('opacity', m.b ? 1 : 0.25); n.setAttribute('stroke-dasharray', m.b ? '' : '3,2');});
      bRefs[2].setAttribute('d', i >= 8 ? 'M580,552 V518' : 'M580,552 H656 V445 H628');
      bRefs[2].dataset.destination = i >= 8 ? '14' : '9';
      privateSlots.forEach((n, j) => {n.setAttribute('fill', m.private[j] === null ? '#ffffff' : j === 2 ? '#16a34a' : '#dcfce7'); n.setAttribute('stroke', m.allocated ? '#16a34a' : '#cbd5e1'); n.dataset.kvPosition = m.private[j] === null ? '' : String(m.private[j]);});
      privateValues.forEach((n, j) => {n.textContent = m.private[j] === null ? '' : String(m.private[j]);});
      privateOutline.setAttribute('stroke', m.allocated ? '#16a34a' : '#94a3b8'); privateOutline.setAttribute('stroke-dasharray', m.allocated ? '' : '3,2');
      privateLabel.textContent = m.allocated ? 'block 14' : 'block 14 · free';
      privateLabel.setAttribute('fill', m.allocated ? '#15803d' : '#475569');
      copyArrow.setAttribute('opacity', i >= 6 ? 1 : 0.25); copyLabel.setAttribute('opacity', i >= 6 ? 1 : 0.4);
      const boxes = [[140, 65, 313, 84], [377, 111, 54, 33], [529, 235, 54, 28], [543, 236, 13, 26], [556, 548, 49, 33], [530, 490, 100, 30], [532, 492, 24, 26], [556, 492, 24, 26], [556, 548, 49, 33], [580, 492, 24, 26]];
      ['x', 'y', 'width', 'height'].forEach((name, j) => focus.setAttribute(name, boxes[i][j]));
      values[0].textContent = `[${m.a.join(', ')}]`;
      values[1].textContent = m.b ? `[${m.b.join(', ')}]` : 'Not forked';
      values[2].textContent = m.allocated ? m.private.map(x => x === null ? 'unwritten' : `KV@${x}`).join(' | ') : 'Free · not allocated';
      values[3].textContent = Object.values(m.refs).join(' / ');
      values[4].textContent = i < 4 ? 'No B request' : i < 8 ? 'Closed · tail still shared' : i < 9 ? 'Open · private initialized tail' : 'Position 10 written only to B';
      root.dataset.cowStep = String(i); root.dataset.cowB = m.b ? m.b.join(',') : ''; root.dataset.cowPrivate = JSON.stringify(m.private);
      root.dataset.cowAllocated = String(m.allocated); root.dataset.cowReferences = JSON.stringify(m.refs);
      root.dataset.cowSource = sourceValues.map(n => n.textContent).join(',');
      follow(...camera[i]);
      if (!details.firstChild) {
        const p = document.createElement('p');
        p.textContent = 'Digits inside green cells label the token positions whose KV records are initialized; they are not scalar key or value contents. Table numbers are physical block IDs. Green-outlined empty cells are allocated but unwritten; gray dashed block 14 is free before reservation. Purple moving labels carry table metadata; green labels carry KV records. Reference counts here count A/B tables, not outstanding device operations. The example assumes no concurrent consumers or eviction.';
        details.append(p);
      }
    }
    function transition({from, to, onFrame, animate}) {
      if (to !== from + 1) return;
      let route, label, kind = 'metadata';
      if (to === 1) {route = [[392, 79], [404, 92], [404, 118]]; label = 'L2';}
      if (to === 2) {route = [[404, 142], [404, 170], [549.5, 170], [549.5, 236]]; label = 'ID 9';}
      if (to === 3) {route = [[536.5, 249], [549.5, 249]]; label = 'slot 1';}
      if (to === 4) {
        const ghosts = [260, 420, 580].map((x, j) => {const n = node('text', {x, y: 395, class: 'okv-reference-copy', 'text-anchor': 'middle'}, `ID ${[3, 7, 9][j]}`); overlay.append(n); return n;});
        animate(ghosts[0], [{opacity: 1}, {opacity: 1}], {});
        onFrame(p => {ghosts.forEach(n => n.setAttribute('y', 395 + (569 - 395) * p)); follow(260 + (580 - 260) * p, 477); if (p >= 1) ghosts.forEach(n => n.remove());});
        return;
      }
      if (to === 5) {animate(privateOutline, [{stroke: '#94a3b8'}, {stroke: '#16a34a'}], {}); return;}
      if (to === 6 || to === 7) {const x = to === 6 ? 544 : 568; route = [[x, 445], [556, 458], [556, 492], [x, 505]]; label = `KV@${to === 6 ? 8 : 9}`; kind = 'payload';}
      if (to === 8) {route = [[580, 505], [580, 552], [580, 564]]; label = 'ID 14';}
      if (to === 9) {route = [[673, 505], [628, 505], [592, 505]]; label = 'KV@10'; kind = 'payload';}
      if (!route) return;
      packet.dataset.kind = kind; packet.querySelector('text').textContent = label; packet.setAttribute('visibility', 'visible');
      const lengths = route.slice(1).map((p, i) => Math.hypot(p[0] - route[i][0], p[1] - route[i][1])); const total = lengths.reduce((a, b) => a + b, 0);
      onFrame(p => {let distance = total * p, j = 0; while (j < lengths.length - 1 && distance > lengths[j]) {distance -= lengths[j]; j++;} const t = lengths[j] ? distance / lengths[j] : 0, a = route[j], b = route[j + 1], x = a[0] + (b[0] - a[0]) * t, y = a[1] + (b[1] - a[1]) * t; packet.setAttribute('transform', `translate(${x} ${y})`); follow(x, to >= 4 ? 477 : y); if (p >= 1) packet.setAttribute('visibility', 'hidden');});
    }
    const player = window.MLSysOriginalPlayer.create({root, outer: context.outer, title: 'How can B append without changing A’s cached prefix?', scope: 'The original ten-position prefix, four-position blocks, and physical IDs [3, 7, 9]. Panel (a) explains the same prefix that panel (b) forks. B reserves block 14, copies initialized tail records, remaps its table, then appends. Pale B connectors are inactive before the fork.', steps: stages, render, transition});
    const resize = new ResizeObserver(() => follow(...active)); resize.observe(canvas);
    const destroy = player.destroy;
    player.destroy = () => {resize.disconnect(); destroy();};
    return player;
  };
})();
