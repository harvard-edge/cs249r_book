/* Replay the authored schedule; compute activation lifetimes from its F/B cells. */
(() => {
  'use strict';
  const NAME = 'original-v2-pipeline-1f1b';
  const NS = 'http://www.w3.org/2000/svg';
  const element = (tag, attrs = {}, text) => {
    const node = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  };
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons[NAME] = async (root, context) => {
    if (!context.verifiedArtwork || !window.MLSysOriginalPlayer) throw Error('Verified 1F1B artwork and player required');
    const art = context.verifiedArtwork.cloneNode(true);
    if (art.getAttribute('viewBox') !== '0 0 1024 374') throw Error('1F1B source geometry changed');
    art.classList.add(NAME + '-art');
    art.setAttribute('role', 'img');
    art.setAttribute('aria-label', 'Original four-stage 1F1B schedule, with completed slots and retained forward activations');
    const rectangles = [...art.querySelectorAll('rect')], texts = [...art.querySelectorAll('text')];
    const stages = 4, slots = 22, microbatches = 8, cells = [];
    for (let rank = 0; rank < stages; rank++) {
      for (let slot = 0; slot < slots; slot++) {
        const x = 96 + slot * 34, y = 50 + rank * 56;
        const rect = rectangles.find(n => +n.getAttribute('x') === x && +n.getAttribute('y') === y && +n.getAttribute('width') === 32);
        const labels = texts.filter(n => +n.getAttribute('x') === x + 16 && [y + 20, y + 38].includes(+n.getAttribute('y')));
        if (!rect || (labels.length !== 0 && labels.length !== 2)) throw Error('1F1B cell mapping changed');
        const kind = labels.length ? labels[0].textContent.trim() : 'idle';
        const micro = labels.length ? Number(labels[1].textContent) : null;
        if (!['F', 'B', 'idle'].includes(kind) || (micro !== null && (!Number.isInteger(micro) || micro < 1 || micro > microbatches))) throw Error('Invalid 1F1B source operation');
        const group = element('g', {'data-rank': rank, 'data-slot': slot, 'data-operation': kind});
        group.append(rect, ...labels); art.append(group);
        const retained = element('rect', {x: x + 2, y: y + 2, width: 28, height: 42, rx: 1, fill: 'none', stroke: '#1F407A', 'stroke-width': 2.5, 'stroke-dasharray': '3 2', visibility: 'hidden'});
        group.append(retained);
        cells.push({rank, slot, kind, micro, group, retained});
      }
    }
    // Reconstruct every endpoint before mounting. Dependencies refer to the
    // previous slot, so simultaneous operations cannot consume future output.
    const live = Array.from({length: stages}, () => new Set());
    const forwards = Array.from({length: stages}, () => new Set());
    const backwards = Array.from({length: stages}, () => new Set());
    const peaks = Array(stages).fill(0), idle = Array(stages).fill(0), states = [];
    const snapshot = () => ({live: live.map(s => [...s].sort((a,b) => a-b)), peaks: [...peaks], idle: [...idle]});
    states.push(snapshot());
    for (let slot = 0; slot < slots; slot++) {
      const work = cells.filter(c => c.slot === slot);
      for (const c of work) {
        if (c.kind === 'F' && (forwards[c.rank].has(c.micro) || (c.rank > 0 && !forwards[c.rank - 1].has(c.micro)))) throw Error('1F1B forward dependency violation');
        if (c.kind === 'B' && (!live[c.rank].has(c.micro) || backwards[c.rank].has(c.micro) || (c.rank < stages - 1 && !backwards[c.rank + 1].has(c.micro)))) throw Error('1F1B backward dependency violation');
      }
      for (const c of work) {
        if (c.kind === 'F') { forwards[c.rank].add(c.micro); live[c.rank].add(c.micro); }
        else if (c.kind === 'B') { backwards[c.rank].add(c.micro); live[c.rank].delete(c.micro); }
        else idle[c.rank]++;
        peaks[c.rank] = Math.max(peaks[c.rank], live[c.rank].size);
      }
      states.push(snapshot());
    }
    for (let rank = 0; rank < stages; rank++) {
      const printedPeak = texts.find(n => +n.getAttribute('x') === 978 && +n.getAttribute('y') === 78 + rank * 56);
      const filledBars = rectangles.filter(n => +n.getAttribute('y') === 65 + rank * 56 && +n.getAttribute('width') === 12 && n.getAttribute('fill') === '#C7D7EB').length;
      if (!printedPeak || +printedPeak.textContent !== peaks[rank] || filledBars !== peaks[rank] || live[rank].size || forwards[rank].size !== microbatches || backwards[rank].size !== microbatches) throw Error('1F1B endpoints disagree with source peak bars');
    }
    const focus = element('rect', {x: 92, y: 44, width: 40, height: 226, rx: 3, fill: 'none', stroke: '#b45309', 'stroke-width': 2, 'data-mlsys-focus': '', 'aria-hidden': 'true'});
    art.append(focus);
    // Retain the source row labels while its wide schedule scrolls on phones.
    const rowLabels = element('g', {'data-mlsys-keep': '', 'aria-hidden': 'true'});
    rowLabels.append(element('rect', {x: 0, y: 42, width: 90, height: 230, fill: '#fff'}));
    texts.filter(n => +n.getAttribute('x') === 82).forEach(n => rowLabels.append(n.cloneNode(true)));
    art.append(rowLabels);
    const board = document.createElement('div'); board.className = NAME + '-board';
    const cards = Array.from({length: stages}, (_, rank) => {
      const card = document.createElement('section'), heading = document.createElement('strong');
      heading.textContent = `GPU ${rank} · saved activations`; card.append(heading);
      const value = document.createElement('p'), count = document.createElement('p'); card.append(value, count); board.append(card);
      return {card, value, count};
    });
    const proof = document.createElement('p'); proof.className = NAME + '-proof';
    const legend = document.createElement('p'); legend.className = NAME + '-legend';
    legend.textContent = 'F: forward; B: backward; red hatching: idle. Dashed blue outline on a completed F cell: its activation is still retained. The original right-hand bars show the full-schedule peak, not the current live count.';
    const operationText = slot => cells.filter(c => c.slot === slot).map(c => `GPU ${c.rank}: ${c.kind === 'idle' ? 'idle' : (c.kind === 'F' ? 'save' : 'release') + ' m' + c.micro}`).join('; ');
    const steps = [{label: 'Begin with no saved activations', description: 'Each row is a stage and each column is one schematic time slot. Forward completion saves an activation; its matching backward completion releases it. The original peak bars remain a reference.'}];
    for (let slot = 0; slot < slots; slot++) steps.push({label: `Complete slot ${slot + 1}`, description: operationText(slot) + '. ' + (slot === slots - 1 ? `All saved activations are released. Peak live counts were ${peaks.join(', ')} for GPU 0 through GPU 3, versus ${microbatches} for all-forwards-first GPipe. Each stage still had ${idle[0]} idle slots; lower activation residency does not remove the bubble.` : 'Retained activations survive idle slots until their own backward consumer completes.')});
    let stageNode, mounted = false, current = 0, observer;
    function pinLabels() {
      if (!stageNode) return;
      const scale = art.getBoundingClientRect().width / 1024;
      rowLabels.setAttribute('transform', `translate(${scale ? stageNode.scrollLeft / scale : 0},0)`);
    }
    function camera(index) {
      if (!stageNode) return;
      const scale = art.getBoundingClientRect().width / 1024;
      stageNode.scrollLeft = Math.max(0, (112 + 34 * Math.max(0, index - 1)) * scale - stageNode.clientWidth / 2);
      pinLabels();
    }
    const player = window.MLSysOriginalPlayer.create({root, outer: context.outer,
      title: 'Why does alternating forward and backward reduce saved activations?',
      intro: 'Follow each microbatch through forward and backward, and watch its saved activation remain live until its backward consumer completes.',
      scope: 'The original four-stage, eight-microbatch 1F1B schedule. Equal slots are schematic, not measured kernel durations. Live sets count retained microbatch activations, not bytes or total training memory. The optimizer update follows all microbatches; it is not a backward cell.',
      steps,
      render(index, {stage, details}) {
        current = index;
        if (!mounted) { stageNode = stage; stage.classList.add(NAME + '-stage'); stage.tabIndex = 0; stage.setAttribute('aria-label', 'Original schedule; scroll horizontally to inspect on narrow screens'); stage.append(art); stage.addEventListener('scroll', pinLabels, {passive: true}); details.append(legend, board, proof); observer = new ResizeObserver(() => camera(current)); observer.observe(stage); mounted = true; }
        const state = states[index];
        for (const c of cells) {
          c.group.style.opacity = c.slot < index ? '1' : '.2';
          c.group.dataset.executed = String(c.slot < index);
          c.retained.setAttribute('visibility', c.kind === 'F' && c.slot < index && state.live[c.rank].includes(c.micro) ? 'visible' : 'hidden');
        }
        focus.setAttribute('x', String(92 + 34 * Math.max(0, index - 1)));
        cards.forEach((item, rank) => {
          item.value.textContent = state.live[rank].length ? state.live[rank].map(m => 'm' + m).join(' · ') : 'None retained';
          item.count.textContent = `Live now: ${state.live[rank].length} · peak so far: ${state.peaks[rank]}`;
          item.card.dataset.live = JSON.stringify(state.live[rank]);
        });
        proof.textContent = index === slots ? `Completed schedule: ${idle.join(', ')} idle slots across GPU 0 through GPU 3. All activations released; the memory advantage comes from earlier release, not fewer pipeline bubbles.` : index ? 'Last completed slot: ' + operationText(index - 1) + '.' : 'No forward or backward operation has completed.';
        root.dataset.originalState = JSON.stringify({completedSlots: index, ...state});
        root.dataset.animationProgress = '0'; camera(index);
      },
      transition({from, to, onFrame}) {
        if (to !== from + 1) return;
        camera(to);
        const work = cells.filter(c => c.slot === to - 1);
        work.forEach(c => c.group.dataset.motionObject = 'true');
        focus.setAttribute('x', String(92 + 34 * (to - 1)));
        onFrame(progress => { root.dataset.animationProgress = String(progress); work.forEach(c => c.group.style.opacity = String(.2 + .8 * progress)); });
      }
    });
    return {pause: () => player.pause(), destroy: () => { observer?.disconnect(); stageNode?.removeEventListener('scroll', pinLabels); player.destroy(); }};
  };
})();
