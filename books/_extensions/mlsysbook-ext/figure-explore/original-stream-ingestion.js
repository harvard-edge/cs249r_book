/* Animate the verified chapter SVG; the capture example is deliberately byte-sized. */
(() => {
  'use strict';
  let serial = 0;
  const NS = 'http://www.w3.org/2000/svg';
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons['original-stream-ingestion'] = async (root, context) => {
    if (!context.verifiedArtwork || !window.MLSysOriginalPlayer) throw Error('Verified stream artwork and original player required');
    const source = context.verifiedArtwork;
    if (source.getAttribute('viewBox') !== '0 0 854 360') throw Error('Unreviewed stream artwork geometry');
    const svg = source.cloneNode(true), prefix = `stream-life-${++serial}-`;
    for (const item of svg.querySelectorAll('[id]')) {
      const old = item.id, next = prefix + old;
      for (const n of svg.querySelectorAll('*')) for (const attr of [...n.attributes]) {
        if (attr.value.includes(`url(#${old})`)) n.setAttribute(attr.name, attr.value.replaceAll(`url(#${old})`, `url(#${next})`));
      }
      item.id = next;
    }
    root.classList.add('mlsys-original-stream-ingestion');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Original bounded capture diagram with bytes in its head buffer, tail ring, and staged observation');
    const node = (tag, attrs = {}, text) => {
      const n = document.createElementNS(NS, tag);
      Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, String(v)));
      if (text !== undefined) n.textContent = text;
      return n;
    };
    const textAt = (x, y, text, attrs = {}) => node('text', {x, y, 'text-anchor': 'middle', class: 'osi-byte', ...attrs}, text);
    const originalText = [...svg.querySelectorAll('text')];
    const exactText = value => {
      const matches = originalText.filter(n => n.textContent.trim() === value);
      if (matches.length !== 1) throw Error(`Stream artwork label changed: ${value}`);
      return matches[0];
    };
    const countLabel = exactText('+ bytes');
    const headRects = [...svg.querySelectorAll('rect')].filter(n => n.getAttribute('y') === '160' && n.getAttribute('width') === '17.5');
    if (headRects.length !== 4) throw Error('Stream artwork head capacity changed');
    const stageFields = ['status', 'head', 'notice: bytes omitted', 'tail', 'handle'].map(label => originalText.find(n => n.getAttribute('x') === '720' && n.textContent === label));
    if (stageFields.some(n => !n)) throw Error('Stream observation fields changed');
    const overlay = node('g', {class: 'osi-overlay'}); svg.append(overlay);
    const headValues = headRects.map(r => {
      const value = textAt(Number(r.getAttribute('x')) + Number(r.getAttribute('width')) / 2 + 26, 177, '·');
      overlay.append(value); return value;
    });
    // These labels sit in the eight existing ring sectors; no replacement diagram.
    const ringValues = Array.from({length: 8}, (_, i) => {
      const angle = -Math.PI / 2 + (i + 0.5) * Math.PI / 4;
      const value = textAt(586 + 17 * Math.cos(angle), 172 + 17 * Math.sin(angle) + 3.3, '·', {class: 'osi-ring-byte', 'data-ring-slot': i});
      overlay.append(value); return value;
    });
    const ringPointer = node('circle', {r: 4, fill: 'none', stroke: '#b45309', 'stroke-width': 1.5, 'data-ring-pointer': ''});
    overlay.append(ringPointer);
    const focus = node('rect', {class: 'osi-focus', fill: 'none', rx: 3, 'stroke-width': 2}); overlay.append(focus);
    const packet = node('g', {class: 'osi-packet', visibility: 'hidden'});
    packet.append(node('rect', {x: -9, y: -10, width: 18, height: 20, rx: 3}), textAt(0, 5, ''));
    overlay.append(packet);
    const HEAD = 4, TAIL = 8, BYTES = [...'ABCDEFGHIJKLMNOP'];
    const byteCount = n => `${n} ${n === 1 ? 'byte' : 'bytes'}`;
    const snapshots = [snapshot(0), ...BYTES.map((_, i) => snapshot(i + 1))];
    function snapshot(n) {
      const seen = BYTES.slice(0, n), head = seen.slice(0, HEAD), later = seen.slice(HEAD), ring = Array(TAIL).fill(null);
      later.forEach((byte, i) => {ring[i % TAIL] = byte;});
      const tail = later.slice(-TAIL), omitted = seen.length - head.length - tail.length;
      return {seen, head, ring, tail, omitted, write: later.length % TAIL, staged: false};
    }
    snapshots.push({...snapshot(BYTES.length), staged: true});
    const steps = [{label: 'Start with fixed capture memory', description: 'The child has separate stdout and stderr pipes. The adapter drains whichever pipe is ready. The head and tail have fixed capacities; the optional full-record path requires separate authorization.'}];
    for (let i = 0; i < BYTES.length; i++) {
      const n = i + 1, s = snapshots[n], channel = i % 2 ? 'stderr' : 'stdout';
      let description;
      if (n <= HEAD) description = `The readiness loop reads ${BYTES[i]} from ${channel} and fills the next head cell. Once full, these first ${HEAD} bytes remain unchanged.`;
      else if (n <= HEAD + TAIL) description = `The head is full. Read ${BYTES[i]} from ${channel} into tail slot ${(n - HEAD - 1) % TAIL}. The tail is still filling; no captured byte has been omitted.`;
      else description = `Read ${BYTES[i]} from ${channel} into tail slot ${(n - HEAD - 1) % TAIL}, replacing the oldest tail byte ${snapshots[n - 1].ring[(n - HEAD - 1) % TAIL]}. The omission count becomes ${s.omitted}; allocated capture capacity does not grow.`;
      steps.push({label: n <= HEAD ? `Read ${BYTES[i]}: fill the head` : n <= HEAD + TAIL ? `Read ${BYTES[i]}: fill the tail` : `Read ${BYTES[i]}: overwrite the oldest tail byte`, description});
    }
    steps.push({label: 'Stage the bounded observation', description: 'After the child exits successfully, serialize the head, an omission notice, and the tail in chronological order. No full record was authorized, so no handle is available. The next model call still needs a separate token-budget check.'});
    const canvas = document.createElement('div'); canvas.className = 'osi-canvas'; canvas.tabIndex = 0; canvas.setAttribute('aria-label', 'Original capture diagram. Scroll sideways to inspect. The view follows the current buffer.'); canvas.append(svg);
    const swipe = document.createElement('p'); swipe.className = 'osi-swipe'; swipe.textContent = 'Swipe the diagram sideways to inspect. The view follows each byte into its buffer.';
    function follow(x) {
      if (canvas.scrollWidth <= canvas.clientWidth) return;
      canvas.scrollLeft = Math.max(0, x * svg.getBoundingClientRect().width / 854 - canvas.clientWidth / 2);
    }
    const readout = document.createElement('div'); readout.className = 'osi-readout';
    readout.setAttribute('aria-label', 'Exact capture state');
    const labels = ['Ready byte', 'Head · fixed after filling', 'Tail · oldest to newest', 'Omitted middle', 'Capture occupancy', 'Outlined ring slot'];
    const values = labels.map(label => {
      const item = document.createElement('div'), name = document.createElement('span'), value = document.createElement('strong');
      name.textContent = label; item.append(name, value); readout.append(item); return value;
    });
    function render(i, {stage, details}) {
      if (!readout.isConnected) stage.append(swipe, canvas, readout);
      const s = snapshots[i], current = i > 0 && !s.staged ? BYTES[i - 1] : null;
      packet.setAttribute('visibility', 'hidden');
      headValues.forEach((n, j) => {n.textContent = s.head[j] || '·'; n.dataset.byte = s.head[j] || '';});
      ringValues.forEach((n, j) => {n.textContent = s.ring[j] || '·'; n.dataset.byte = s.ring[j] || '';});
      const pointer = ringValues[s.write];
      ringPointer.setAttribute('cx', pointer.getAttribute('x')); ringPointer.setAttribute('cy', Number(pointer.getAttribute('y')) - 3.3);
      ringPointer.setAttribute('visibility', s.seen.length >= HEAD && !s.staged ? 'visible' : 'hidden');
      countLabel.textContent = byteCount(s.omitted);
      const fieldText = s.staged ? ['status: exited 0', s.head.join(' '), `${byteCount(s.omitted)} omitted`, s.tail.join(' '), 'handle: none'] : ['status', 'head', 'notice: bytes omitted', 'tail', 'handle'];
      stageFields.forEach((n, j) => {n.textContent = fieldText[j]; n.setAttribute('font-size', j === 3 && s.staged ? '12' : '14');});
      const box = s.staged ? [673, 88, 162, 90] : i === 0 ? [215, 137, 432, 134] : i <= HEAD ? [225, 159, 72, 26] : [560, 146, 52, 52];
      ['x', 'y', 'width', 'height'].forEach((a, j) => focus.setAttribute(a, box[j]));
      values[0].textContent = current ? `${i % 2 ? 'stdout' : 'stderr'}: ${current}` : s.staged ? 'Capture complete' : 'Waiting';
      values[1].textContent = s.head.join(' ') || 'Empty'; values[2].textContent = s.tail.join(' ') || 'Empty';
      values[3].textContent = byteCount(s.omitted); values[4].textContent = `${s.head.length + s.tail.length} / ${HEAD + TAIL} byte slots`;
      values[5].textContent = s.staged ? 'Capture complete' : s.seen.length >= HEAD ? `${s.write}: next write` : 'Waiting for the head to fill';
      root.dataset.streamStep = String(i); root.dataset.streamOmitted = String(s.omitted); root.dataset.streamTail = s.tail.join('');
      root.dataset.streamSeen = String(s.seen.length); root.dataset.streamStaged = String(s.staged);
      follow(s.staged ? 754 : i === 0 ? 105 : i <= HEAD ? 261 : 586);
      if (!details.querySelector('.osi-accounting')) {
        const p = document.createElement('p'); p.className = 'osi-accounting';
        p.textContent = 'Capture accounting: bytes read = retained head + retained tail + omitted middle. This byte capacity excludes notice/status framing, pipe buffers, and any separately authorized retained record. Bytes and model tokens have different budgets.';
        details.append(p);
      }
    }
    function transition({from, to, onFrame}) {
      if (to !== from + 1) return;
      const staging = to === BYTES.length + 1;
      const head = to <= HEAD, channelX = to % 2 ? 65 : 145;
      const slot = head ? headValues[to - 1] : ringValues[(to - HEAD - 1) % TAIL];
      const end = staging ? [754, 160] : [Number(slot.getAttribute('x')), Number(slot.getAttribute('y')) - 5];
      const route = staging ? [[646, 160], [660, 160], end] : [[channelX, 139], [channelX, 172], [105, 209], [203, 209], [203, 78], [head ? 261 : 586, 78], [head ? 261 : 586, 145], end];
      const lengths = route.slice(1).map((p, i) => Math.hypot(p[0] - route[i][0], p[1] - route[i][1]));
      const total = lengths.reduce((a, b) => a + b, 0);
      packet.querySelector('text').textContent = staging ? 'copy' : BYTES[to - 1];
      packet.querySelector('rect').setAttribute('x', staging ? -22 : -9);
      packet.querySelector('rect').setAttribute('width', staging ? 44 : 18);
      packet.setAttribute('visibility', 'visible');
      onFrame(progress => {
        let distance = total * progress, index = 0;
        while (index < lengths.length - 1 && distance > lengths[index]) {distance -= lengths[index]; index++;}
        const f = lengths[index] ? distance / lengths[index] : 0, a = route[index], b = route[index + 1];
        const x = a[0] + (b[0] - a[0]) * f;
        packet.setAttribute('transform', `translate(${x} ${a[1] + (b[1] - a[1]) * f})`);
        follow(x);
        if (progress >= 1) packet.setAttribute('visibility', 'hidden');
      });
    }
    const player = window.MLSysOriginalPlayer.create({root, outer: context.outer, title: 'How can capture stay bounded while a child keeps writing?', scope: 'Illustrative ASCII bytes A–P: four head slots and eight tail slots, matching the drawn buffers. The alternating channels show one observed readiness order, not a causal order between stdout and stderr. No full-record retention is authorized in this example.', steps, render, transition});
    return player;
  };
})();
