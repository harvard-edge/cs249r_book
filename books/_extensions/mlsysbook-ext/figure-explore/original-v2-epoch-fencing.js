/* Keep the source timeline; animate enforcement at the checkpoint-store boundary. */
(() => {
  'use strict';
  const NAME = 'original-v2-epoch-fencing';
  const NS = 'http://www.w3.org/2000/svg';
  const svg = (tag, attributes = {}, text) => {
    const node = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const steps = [
    {label: 'The old worker holds token 7', description: 'The coordinator and checkpoint store recognize epoch 7. The old worker can reach storage with token 7; later timeline regions are shown faintly as a reference.'},
    {label: 'Lease expires; ownership changes', description: 'The coordinator grants epoch 8. The paused old worker still holds token 7, and the store still enforces epoch 7. Lease expiry alone has not fenced the old worker.'},
    {label: 'Advance the store fence', description: 'The checkpoint store now enforces epoch 8. This resource-side change must finish before replacement writes proceed. A write carrying token 7 will now be rejected.'},
    {label: 'Replacement receives token 8', description: 'The replacement holds token 8 after the store fence is advanced. It is ready to write; no replacement write has been accepted yet.'},
    {label: 'Accept the replacement write', description: 'The write arrives with token 8. The checkpoint store checks it against the enforced epoch and accepts it. The token travels with the write; the coordinator is not the enforcement point.'},
    {label: 'Old worker resumes with token 7', description: 'The old worker resumes without learning that its lease expired. Its local token remains 7. It can still send a write, but it cannot bypass the store fence.'},
    {label: 'Reject the stale write', description: 'The token-7 write reaches the protected boundary and is rejected. Ownership changed at the coordinator, but safety depends on enforcement at every protected resource; this figure shows the checkpoint store.'},
  ];
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons[NAME] = async (root, context) => {
    if (!context.verifiedArtwork || !window.MLSysOriginalPlayer) throw Error('Verified fencing artwork and player required');
    const art = context.verifiedArtwork.cloneNode(true);
    if (art.getAttribute('viewBox') !== '0 0 700 290') throw Error('Epoch-fencing geometry changed');
    art.classList.add(NAME + '-art');
    art.setAttribute('role', 'img');
    art.setAttribute('aria-label', 'Original four-lane epoch-fencing timeline; coordinator ownership and checkpoint-store enforcement advance separately.');
    const group = (name, selectors) => {
      const nodes = selectors.map(selector => art.querySelector(selector));
      if (nodes.some(node => !node)) throw Error('Fencing source mapping changed: ' + name);
      const result = svg('g', {'data-fencing-object': name});
      result.append(...nodes); art.append(result); return result;
    };
    const objects = {
      old: group('old-token', ['rect[x="120"][y="148"]', 'text[x="175"][y="164"]']),
      epoch: group('new-epoch', ['rect[x="350"][y="28"]', 'text[x="510"][y="44"]', 'text[x="300"][y="18"]', 'line[x1="300"][y1="21"]']),
      fence: group('fence-update', ['line[x1="356"][y1="52"]', 'text[x="348"][y="126"]', 'text[x="348"][y="139"]']),
      store: group('store-fence', ['rect[x="360"][y="224"]', 'text[x="515"][y="242"]']),
      replacement: group('replacement', ['rect[x="380"][y="88"]', 'text[x="530"][y="104"]']),
      accepted: group('accepted-write', ['line[x1="395"][y1="112"]', 'text[x="403"][y="196"]', 'text[x="403"][y="209"]']),
      resumed: group('old-resumes', ['rect[x="530"][y="148"]', 'text[x="570"][y="164"]']),
      rejected: group('rejected-write', ['line[x1="590"][y1="172"]', 'g[stroke="#dc2626"]', 'text[x="574"][y="196"]', 'text[x="574"][y="209"]']),
    };
    const moving = svg('g', {'aria-hidden': 'true'}); art.append(moving);
    // Keep the original lane names visible while Follow pans the timeline.
    const laneLabels = svg('g', {'aria-hidden': 'true'});
    laneLabels.append(svg('rect', {x: 0, y: 0, width: 116, height: 260, fill: '#fff'}));
    for (const label of art.querySelectorAll('text[x="12"]')) laneLabels.append(label.cloneNode(true));
    art.append(laneLabels);
    const viewControls = document.createElement('label');
    viewControls.className = NAME + '-views';
    viewControls.append(document.createTextNode('Diagram view '));
    const view = document.createElement('select'); view.setAttribute('aria-label', 'Diagram view');
    view.append(new Option('Follow', 'follow'), new Option('Whole', 'whole')); viewControls.append(view);

    const readout = document.createElement('p'); readout.className = NAME + '-readout';
    readout.setAttribute('data-mlsys-keep', '');
    const note = document.createElement('p'); note.className = NAME + '-note';
    note.textContent = 'Follow keeps lane names visible; scroll to inspect, or choose Whole. The original timeline and token identities are unchanged. Faint regions are future events. Motion illustrates ordering, not network latency; epoch numbers are membership counters, not training epochs.';
    let mounted = false, stageNode, observer, current = 0;
    const centers = [220, 510, 515, 530, 430, 570, 550];
    function pinLabels() {
      if (!stageNode) return;
      const scale = art.getBoundingClientRect().width / 700;
      laneLabels.setAttribute('transform', `translate(${scale ? stageNode.scrollLeft / scale : 0},0)`);
    }
    function center(index, focusX = centers[index]) {
      if (!stageNode) return;
      stageNode.dataset.view = view.value;
      if (view.value === 'whole') {stageNode.scrollLeft = 0; pinLabels(); return;}
      const scale = art.getBoundingClientRect().width / 700;
      stageNode.scrollLeft = Math.max(0, focusX * scale - (stageNode.clientWidth + 116 * scale) / 2);
      pinLabels();
    }
    function render(index) {
      moving.replaceChildren(); current = index;
      const reached = {old: 0, epoch: 1, fence: 2, store: 2, replacement: 3, accepted: 4, resumed: 5, rejected: 6};
      const focus = ['old', 'epoch', 'store', 'replacement', 'accepted', 'resumed', 'rejected'][index];
      for (const [key, object] of Object.entries(objects)) {
        object.style.opacity = index >= reached[key] ? '1' : '.14';
        object.toggleAttribute('data-mlsys-focus', key === focus);
      }
      const state = {coordinatorEpoch: index >= 1 ? 8 : 7, storeFence: index >= 2 ? 8 : 7,
        oldToken: 7, replacementToken: index >= 3 ? 8 : null,
        accepted: index >= 4, oldResumed: index >= 5, rejected: index >= 6};
      readout.textContent = `Coordinator: epoch ${state.coordinatorEpoch} · Store fence: ${state.storeFence} · Old token: ${state.oldToken}. ` +
        (index === 1 ? 'Fence update pending; replacement writes must wait.' :
         index >= 6 ? 'Token 8 accepted; token 7 rejected.' :
         index >= 4 ? 'Replacement write accepted; stale write not yet received.' :
         index >= 2 ? 'Token 7 is fenced; no write has been accepted.' : 'No recovery action completed.');
      root.dataset.originalState = JSON.stringify(state); center(index);
    }
    const player = window.MLSysOriginalPlayer.create({
      root, outer: context.outer, viewControls, title: 'Why does lease expiry need a storage fence?',
      intro: 'Follow tokens 7 and 8 to the checkpoint store and distinguish changed ownership from an enforced fence.',
      scope: 'Original four-lane recovery timeline. One protected checkpoint store; monotonically increasing membership epochs 7 and 8. Replacement writes wait for the store fence. No failure-detection or network timing model.',
      steps,
      render(index, {stage, details}) {
        if (!mounted) {
          stageNode = stage; stage.classList.add(NAME + '-stage'); stage.tabIndex = 0;
          stage.setAttribute('aria-label', 'Fencing timeline; scroll horizontally to inspect the full diagram');
          stage.append(art); details.append(readout, note);
          stage.addEventListener('scroll', pinLabels, {passive: true});
          observer = new ResizeObserver(() => center(current)); observer.observe(stage); mounted = true;
        }
        render(index);
      },
      transition({from, to, onFrame}) {
        if (to !== from + 1) return;
        center(to, to === 2 ? 430 : centers[to]);
        const paths = {2: [356, 60, 356, 214, '8', '#7c3aed'],
          4: [395, 120, 395, 214, '8', '#16a34a'],
          6: [590, 179, 590, 201, '7', '#dc2626']};
        const path = paths[to];
        if (!path) return;
        const [x1, y1, x2, y2, token, color] = path;
        const chip = svg('g', {'data-motion-object': 'true'});
        chip.append(svg('rect', {x: -10, y: -9, width: 20, height: 18, rx: 2, fill: '#fff', stroke: color}),
          svg('text', {x: 0, y: 4, 'text-anchor': 'middle', 'font-size': 11, 'font-weight': 700, fill: color}, token));
        moving.append(chip);
        onFrame(progress => chip.setAttribute('transform', `translate(${x1 + (x2-x1)*progress},${y1 + (y2-y1)*progress})`));
      },
    });
    view.addEventListener('change', () => {player.pause(); center(current);});
    return {pause: () => player.pause(), destroy: () => {stageNode?.removeEventListener('scroll', pinLabels); observer?.disconnect(); player.destroy();}};
  };
})();
