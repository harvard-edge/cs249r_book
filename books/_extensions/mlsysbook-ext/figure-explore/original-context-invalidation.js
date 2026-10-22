/* Original serialized-zone diagram; token identity and derived KV identity stay distinct. */
(() => {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons['original-context-invalidation'] = async (root, context) => {
    if (!context.verifiedArtwork || !window.MLSysOriginalPlayer) throw Error('Verified context-zone artwork and original player required');
    const svg = context.verifiedArtwork.cloneNode(true);
    if (svg.getAttribute('viewBox') !== '0 0 680 392') throw Error('Unreviewed context-zone geometry');
    root.classList.add('mlsys-original-context');
    svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', 'Original three serialized request bars with prefix reuse, suffix recomputation, and an unchanged selected token T');
    const node = (tag, attrs = {}, text) => {const n = document.createElementNS(NS, tag); Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, String(v))); if (text !== undefined) n.textContent = text; return n;};
    const originals = [...svg.children], rects = [...svg.querySelectorAll('rect')], texts = [...svg.querySelectorAll('text')];
    const rows = [116, 196, 276], xs = [130, 240, 440], widths = [110, 200, 100];
    const zoneRects = rows.map(y => xs.map((x, z) => rects.filter(n => Number(n.getAttribute('x')) === x && Number(n.getAttribute('y')) === y && Number(n.getAttribute('width')) === widths[z] && Number(n.getAttribute('height')) === 34)));
    if (zoneRects.some(row => row.some(nodes => nodes.length !== 2))) throw Error('Context-zone rectangles changed');
    const tokenRulers = rows.map(y => xs.map((x, z) => rects.filter(n => Number(n.getAttribute('x')) === x && [y, y + 28].includes(Number(n.getAttribute('y'))) && Number(n.getAttribute('width')) === widths[z] && Number(n.getAttribute('height')) === 6)));
    const blueTicks = tokenRulers[0][0][0].getAttribute('fill'), greenTicks = tokenRulers[1][0][0].getAttribute('fill');
    const groups = rows.map(y => {const g = node('g'); originals.filter(n => ['rect', 'text'].includes(n.localName) && Number(n.getAttribute('y')) >= y && Number(n.getAttribute('y')) <= y + 34).forEach(n => g.append(n)); svg.append(g); return g;});
    const markers = [
      [rects.find(n => n.getAttribute('x') === '440' && n.getAttribute('y') === '196' && n.getAttribute('width') === '10'), svg.querySelector('line[x1="438"]'), texts.find(n => n.getAttribute('y') === '182')],
      [rects.find(n => n.getAttribute('x') === '240' && n.getAttribute('y') === '276' && n.getAttribute('width') === '10'), svg.querySelector('line[x1="238"]'), texts.find(n => n.getAttribute('y') === '262')]
    ];
    if (markers.flat().some(n => !n)) throw Error('Context prefix-boundary markers changed');
    const limitLabel = texts.find(n => n.getAttribute('x') === '540' && n.getAttribute('y') === '340');
    if (!limitLabel) throw Error('Context input-limit label changed');
    limitLabel.setAttribute('y', String(Number(limitLabel.getAttribute('y')) + 5));
    const overlay = node('g', {class: 'oci-overlay'}); svg.append(overlay);
    const tokenNodes = groups.map((g, r) => {const cell = node('g', {'data-token-row': r}); cell.append(node('rect', {x: 530, y: rows[r], width: 10, height: 34, fill: 'white', stroke: '#7c3aed', 'stroke-width': 1}), node('text', {x: 535, y: rows[r] + 22, class: 'oci-token', 'text-anchor': 'middle'}, 'T')); g.append(cell); return cell;});
    groups.forEach((g, r) => [[240, ['u', 'u', 'v'][r], 'trunk-first'], [440, ['a', 'b', 'c'][r], 'leaf-first']].forEach(([x, id, kind]) => {const cell = node('g', {'data-token-kind': kind, 'data-token-id': id}); cell.append(node('rect', {x: x + 1, y: rows[r] + 1, width: 8, height: 32, fill: 'white'}), node('text', {x: x + 5, y: rows[r] + 22, class: 'oci-selected-id', 'text-anchor': 'middle'}, id)); g.append(cell);}));
    const bands = rows.map((y, r) => xs.map((x, z) => {const g = node('g', {'data-kv-row': r, 'data-kv-zone': z}); const box = node('rect', {x, y: y + 40, width: widths[z], height: 13, rx: 2, stroke: '#94a3b8', 'stroke-width': .6}); const label = node('text', {x: x + widths[z] / 2, y: y + 49.5, class: 'oci-kv-label', 'text-anchor': 'middle'}); g.append(box, label); overlay.append(g); return {g, box, label};}));
    const front = node('rect', {class: 'oci-front', visibility: 'hidden', fill: '#38bdf8', 'fill-opacity': .2, stroke: '#0284c7', 'stroke-width': 1}); overlay.append(front);
    const steps = [
      ['Start with a cold request', 'Call n has no reusable KV in this example. Root, trunk at r17, and leaf are serialized in the original order. T marks one selected trailing input token; it is not part of the output reserve. Pale later rows are requests not yet staged.'],
      ['Process the cold input', 'Process all input zones for call n. When this pass finishes, the KV derived for T is labeled C0. This is a provenance label, not a numerical key or value. The hatched output reserve is not processed input.'],
      ['Stage a new observation', 'Call n+1 keeps the root and trunk token sequence exactly the same and changes the leaf. T itself stays identical at the same input position. Existing KV from call n remains visible in the historical row.'],
      ['Stop the match at the first changed token', 'Compare the final serialized token sequence, not the apparent similarity of the prose. The identical leading prefix ends at the first changed leaf token. A later match at T cannot restart prefix reuse.'],
      ['Reuse root and trunk KV', 'Bind call n+1 to the compatible resident KV for its identical root and trunk prefix. When this binding finishes, the new request can reuse that existing state. The green links represent reuse, not a physical copy. The changed leaf and later T still require processing.'],
      ['Recompute the leaf, including T', 'Process the changed leaf through its end. When this processing finishes, T’s newly derived KV is labeled C1. T’s token ID stays the same, but its preceding context differs. C0 remains in call n’s historical row; prefix invalidation itself does not erase bytes.'],
      ['Refresh the trunk to r18', 'Call n+2 stages the updated r18 trunk and its next leaf. The root remains exact, and the selected trailing token T is still identical. Correct source revision takes precedence over preserving a warm cache.'],
      ['Move the prefix cutoff to the trunk', 'The first changed token is now at the trunk boundary. Only the root remains an identical leading prefix. Matching tokens later in the trunk or leaf do not make the old suffix KV eligible.'],
      ['Reuse only root KV', 'Bind the compatible resident root state. When this binding finishes, call n+2 can reuse that state. The updated trunk and every later input position still require processing under the new prefix; unchanged T cannot inherit C1 merely because its own token ID matches.'],
      ['Recompute the updated trunk', 'Process the r18 trunk from the new cutoff. When this pass finishes, its newly derived KV provides context for later positions. The leaf still awaits processing; neither its old KV nor the output reserve is silently reused.'],
      ['Recompute the trailing leaf and T', 'Process the leaf with the updated preceding context. When this processing finishes, unchanged T’s newly derived state is labeled C2. Earlier C0 and C1 remain visible in prior request rows. Stable placement saves prefix work while current source and evidence remain correct.']
    ].map(([label, description]) => ({label, description}));
    const model = i => {
      const exists = [true, i >= 2, i >= 6], status = [['pending', 'pending', 'pending'], ['pending', 'pending', 'pending'], ['pending', 'pending', 'pending']];
      if (i >= 1) status[0] = ['new', 'new', 'new'];
      if (i >= 4) status[1] = ['reuse', 'reuse', i >= 5 ? 'new' : 'pending'];
      if (i >= 8) status[2] = ['reuse', i >= 9 ? 'new' : 'pending', i >= 10 ? 'new' : 'pending'];
      return {exists, status, t: [i >= 1 ? 'C0' : null, i >= 5 ? 'C1' : null, i >= 10 ? 'C2' : null], cutoff: i >= 7 ? 'trunk' : i >= 3 && i < 6 ? 'leaf' : null};
    };
    const canvas = document.createElement('div'); canvas.className = 'oci-canvas'; canvas.tabIndex = 0; canvas.setAttribute('aria-label', 'Original request diagram. Swipe to inspect; the view follows the active request and token.'); canvas.append(svg);
    const cue = document.createElement('p'); cue.className = 'oci-pan-cue'; cue.textContent = 'Swipe the diagram to inspect all zones. It follows the active prefix boundary or recomputed suffix.';
    const legend = document.createElement('p'); legend.className = 'oci-band-key'; legend.textContent = 'Letter chips label selected token identities; other tokens are elided. Added KV bands: green = reused; blue = newly processed; gray = pending. C0, C1, and C2 identify T’s derived state from different calls, not numerical K/V contents.';
    const readout = document.createElement('div'); readout.className = 'oci-readout';
    const values = ['Selected trailing input token', 'KV(T) by request · n / n+1 / n+2', 'Reusable leading prefix', 'Historical KV', 'First mismatch'].map(label => {const row = document.createElement('div'), name = document.createElement('span'), value = document.createElement('strong'); name.textContent = label; row.append(name, value); readout.append(row); return value;});
    let active = [340, 145];
    function follow(x, y) {active = [x, y]; const scale = svg.getBoundingClientRect().width / 680; if (canvas.scrollWidth > canvas.clientWidth) canvas.scrollLeft = Math.max(0, x * scale - canvas.clientWidth / 2); if (canvas.scrollHeight > canvas.clientHeight) canvas.scrollTop = Math.max(0, y * scale - canvas.clientHeight / 2);}
    const cameras = [[340, 145], [490, 145], [490, 220], [438, 220], [340, 220], [490, 220], [340, 300], [238, 300], [185, 300], [340, 300], [490, 300]];
    function render(i, {stage, details}) {
      if (!canvas.isConnected) stage.append(cue, canvas, legend, readout);
      overlay.querySelectorAll('.oci-reuse-link,.oci-reuse-tag,.oci-compare').forEach(n => n.remove()); front.setAttribute('visibility', 'hidden');
      const m = model(i);
      groups.forEach((g, r) => g.setAttribute('opacity', m.exists[r] ? 1 : .25));
      markers.forEach((nodes, r) => nodes.forEach(n => n.setAttribute('visibility', i >= [3, 7][r] ? 'visible' : 'hidden')));
      m.status.forEach((row, r) => row.forEach((s, z) => {
        const fill = s === 'reuse' ? '#dcfce7' : s === 'new' ? '#e0f2fe' : '#f1f5f9', stroke = s === 'reuse' ? '#16a34a' : s === 'new' ? '#0284c7' : '#94a3b8';
        zoneRects[r][z].forEach(n => {if (n.getAttribute('fill') !== 'none') n.setAttribute('fill', fill); else n.setAttribute('stroke', stroke);});
        tokenRulers[r][z].forEach(n => {n.setAttribute('fill', s === 'reuse' ? greenTicks : blueTicks); n.setAttribute('opacity', s === 'pending' ? .2 : 1);});
        const b = bands[r][z]; b.g.setAttribute('opacity', m.exists[r] ? 1 : .25); b.box.setAttribute('fill', fill); b.box.setAttribute('stroke', stroke); b.g.dataset.state = s;
        b.label.textContent = s === 'pending' ? 'KV pending' : z === 2 ? `KV(T): ${m.t[r]}` : s === 'reuse' ? 'KV reused from n' : `KV from ${['n', 'n+1', 'n+2'][r]}`;
      }));
      values[0].textContent = 'T → T → T · identical ID and position';
      values[1].textContent = m.t.map(t => t || 'pending').join(' / ');
      values[2].textContent = m.cutoff === 'trunk' ? 'n+2: root only' : m.cutoff === 'leaf' ? 'n+1: root + trunk' : i === 6 ? 'n+2: comparison pending' : i === 2 ? 'n+1: comparison pending' : 'Call n: cold start';
      values[3].textContent = 'Prior request history · no allocator action shown';
      values[4].textContent = m.cutoff === 'trunk' ? 'Trunk: u → v' : m.cutoff === 'leaf' ? 'Leaf: a → b' : 'Not compared yet';
      root.dataset.contextStep = String(i); root.dataset.contextStates = JSON.stringify(m.status); root.dataset.contextT = JSON.stringify(m.t); root.dataset.contextCutoff = m.cutoff || '';
      follow(...cameras[i]);
      if (!details.firstChild) {const p = document.createElement('p'); p.textContent = 'Scope: compatible resident cache, matching model weights, position offsets, and attention configuration are assumed. Actual block alignment and serving policy may shorten reuse. Token letters illustrate identity, not tokenization or token counts. Historical rows do not prescribe physical residency or cleanup. The output reserve remains future capacity. Motion duration is illustrative.'; details.append(p);}
    }
    function transition({from, to, onFrame, animate}) {
      if (to !== from + 1) return;
      if (to === 2 || to === 6) {const r = to === 2 ? 1 : 2; animate(groups[r], [{opacity: .25}, {opacity: 1}], {}); bands[r].forEach(b => animate(b.g, [{opacity: .25}, {opacity: 1}], {})); return;}
      if (to === 3 || to === 7) {const r = to === 3 ? 1 : 2, end = to === 3 ? 440 : 240; const cursor = node('line', {x1: 130, x2: 130, y1: rows[r], y2: rows[r] + 34, stroke: '#7c3aed', 'stroke-width': 2, class: 'oci-compare'}); overlay.append(cursor); onFrame(p => {const x = 130 + (end - 130) * p; cursor.setAttribute('x1', x); cursor.setAttribute('x2', x); follow(x, rows[r] + 17);}); return;}
      const passes = {1: [130, 540, 116], 5: [440, 540, 196], 9: [240, 440, 276], 10: [440, 540, 276]};
      if (passes[to]) {const [start, end, y] = passes[to]; front.setAttribute('x', start); front.setAttribute('y', y); front.setAttribute('height', 53); front.setAttribute('visibility', 'visible'); onFrame(p => {front.setAttribute('width', (end - start) * p); follow(start + (end - start) * p, y + 30); if (p >= 1) front.setAttribute('visibility', 'hidden');}); return;}
      if (to === 4 || to === 8) {
        const columns = to === 4 ? [185, 340] : [185], fromY = to === 4 ? 169 : 249, toY = to === 4 ? 236 : 316;
        const links = columns.map(x => {const link = node('line', {x1: x, x2: x, y1: fromY, y2: fromY, stroke: '#16a34a', 'stroke-width': 2, class: 'oci-reuse-link'}); const tag = node('text', {x: x + 4, y: fromY, class: 'oci-reuse-tag'}, 'reuse'); overlay.append(link, tag); return {link, tag};});
        onFrame(p => {links.forEach(({link, tag}) => {link.setAttribute('y2', fromY + (toY - fromY) * p); tag.setAttribute('y', fromY + (toY - fromY) * p);}); follow(columns[columns.length - 1], (fromY + toY) / 2);});
      }
    }
    const player = window.MLSysOriginalPlayer.create({root, outer: context.outer, title: 'Why must an unchanged trailing token get new KV state?', scope: 'The original root/trunk/leaf layout, with a selected token T kept identical at the same trailing position. C0/C1/C2 label its derived state across three calls. Exact serialized-prefix equality, compatible resident cache, and unchanged execution configuration are assumed; zone labels alone never establish reuse.', steps, render, transition});
    const resize = new ResizeObserver(() => follow(...active)); resize.observe(canvas); const destroy = player.destroy; player.destroy = () => {resize.disconnect(); destroy();}; return player;
  };
})();
