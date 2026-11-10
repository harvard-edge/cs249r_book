/* The chapter's four-rank ring, with provenance instead of an unrelated toy. */
(() => {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const letters = 'ABCD';
  let instance=0;
  const make = (tag, attributes = {}, text) => {
    const element = document.createElementNS(NS, tag);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    if (text !== undefined) element.textContent = text;
    return element;
  };
  const copy = value => JSON.parse(JSON.stringify(value));
  // Every round reads the previous snapshot. A send never observes another
  // rank's receive from the same round; ReduceScatter combines disjoint sets.
  const snapshots = [Array.from({length: 4}, (_, rank) => Array.from({length: 4}, () => [rank]))];
  const transfers = [null];
  for (let round = 0; round < 3; round++) {
    const previous = snapshots.at(-1), next = copy(previous), sent = [];
    for (let rank = 0; rank < 4; rank++) {
      const chunk = (rank - round + 4) % 4, receiver = (rank + 1) % 4;
      const contributors = [...previous[rank][chunk]];
      if (contributors.some(item => previous[receiver][chunk].includes(item))) throw new Error('Repeated ring contribution');
      next[receiver][chunk] = [...previous[receiver][chunk], ...contributors].sort();
      sent.push({rank, receiver, chunk, contributors});
    }
    snapshots.push(next); transfers.push(sent);
  }
  // AllGather starts from only the finished shard on each rank. Partially
  // reduced scratch values are not inputs to this copying phase.
  const reduced = snapshots[3].map(row => row.map(set => set.length === 4 ? set : []));
  for (let round = 0; round < 3; round++) {
    const previous = round === 0 ? reduced : snapshots.at(-1), next = copy(previous), sent = [];
    for (let rank = 0; rank < 4; rank++) {
      const chunk = (rank + 1 - round + 4) % 4, receiver = (rank + 1) % 4;
      const contributors = [...previous[rank][chunk]];
      if (contributors.length !== 4) throw new Error('AllGather requires a finished shard');
      next[receiver][chunk] = contributors;
      sent.push({rank, receiver, chunk, contributors});
    }
    snapshots.push(next); transfers.push(sent);
  }
  const names = ['Start with local chunks', 'ReduceScatter · round 1', 'ReduceScatter · round 2', 'ReduceScatter · round 3', 'AllGather · round 1', 'AllGather · round 2', 'AllGather · round 3'];
  const explanations = [
    'Start on row (a) with 4 GPUs (GPU 0–GPU 3) holding their own local contributions to chunks A–D (1 of 4 rank contributions per tile, 0 of 6 ring rounds completed). Follow amber-outlined chunk A from GPU 0 while B, C, and D step in parallel on the other links.',
    'In ReduceScatter round 1 of 3, every GPU sends 1 chunk (size M/4) clockwise: GPU 0 sends A₀ to GPU 1 (forming partial sum A₀ + A₁, 2/4 fill), while GPU 1 sends B₁→GPU 2, GPU 2 sends C₂→GPU 3, and GPU 3 sends D₃→GPU 0 over the wrap-around link.',
    'In ReduceScatter round 2 of 3, each GPU forwards the partial sum it just updated (2 × M/4 bytes sent per GPU so far): GPU 1 sends A₀ + A₁ to GPU 2, which adds A₂ to hold A₀ + A₁ + A₂ (3 of 4 rank contributions, 3/4 gray fill).',
    'In ReduceScatter round 3 of 3 (3 × M/4 bytes sent per GPU), GPU 2 sends A₀ + A₁ + A₂ to GPU 3, which adds A₃ to complete green full sum ΣA. All 4 GPUs now own 1 distinct fully reduced shard: ΣB on GPU 0, ΣC on GPU 1, ΣD on GPU 2, and ΣA on GPU 3.',
    'Row (b) begins AllGather round 1 of 3 (round 4 of 6 overall, 4 × M/4 bytes sent per GPU): each GPU sends its finished green shard 1 hop clockwise without adding again. GPU 3 copies ΣA over the bottom wrap-around link to GPU 0, while ΣB→GPU 1, ΣC→GPU 2, and ΣD→GPU 3.',
    'In AllGather round 2 of 3 (round 5 of 6 overall, 5 × M/4 bytes sent per GPU), each GPU forwards the finished sum it received in the previous round: GPU 0 copies ΣA to GPU 1 (so GPUs 3, 0, and 1 now hold ΣA) while ΣB, ΣC, and ΣD also advance 1 hop.',
    'In AllGather round 3 of 3 (round 6 of 6 overall), GPU 1 copies ΣA to GPU 2. Every GPU now holds all 4 green full sums (ΣA, ΣB, ΣC, ΣD) after sending 2(N − 1)/N × M = (6/4)M = 1.5M bytes across the 6 rounds.'
  ];
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons['original-ring-allreduce'] = async (root, context) => {
    if (!context.verifiedArtwork || !window.MLSysOriginalPlayer) throw new Error('Verified ring artwork and shared player required');
    const art = context.verifiedArtwork.cloneNode(true);
    if (art.getAttribute('viewBox') !== '0 0 840 364') throw new Error('Ring geometry requires review');
    const prefix=`original-ring-${++instance}-`;
    for(const element of art.querySelectorAll('[id]')){const old=element.id;element.id=prefix+old;for(const reference of art.querySelectorAll('[marker-end],[marker-start],[fill],[clip-path]'))for(const attribute of ['marker-end','marker-start','fill','clip-path'])if(reference.getAttribute(attribute)===`url(#${old})`)reference.setAttribute(attribute,`url(#${prefix+old})`);}
    art.classList.add('original-ring-art');
    art.setAttribute('role', 'img'); art.setAttribute('aria-label', 'Original four-GPU ring; upper row adds contributions, lower row copies finished sums');
    const base = art.querySelector('g[transform="translate(0,6)"]');
    if (!base) throw new Error('Ring artwork group missing');
    const originalTexts = [...base.querySelectorAll('text')];
    const originalRects = [...base.querySelectorAll('rect')];
    const cells = [0, 1].map(panel => Array.from({length: 4}, (_, rank) => Array.from({length: 4}, (_, chunk) => {
      const x = 28 + rank * 224 + chunk * 28, y = 72 + panel * 160;
      const text = originalTexts.find(node => +node.getAttribute('x') === x + 12 && Math.abs(+node.getAttribute('y') - (y + 18.725)) < .01);
      const rects = originalRects.filter(node => +node.getAttribute('x') === x && +node.getAttribute('width') === 24 && +node.getAttribute('y') >= y && +node.getAttribute('y') < y + 28);
      if (!text || !rects.length) throw new Error('Ring chunk mapping changed');
      rects.filter(node => +node.getAttribute('height') !== 28).forEach(node => node.setAttribute('visibility', 'hidden'));
      const bar = make('rect', {x, y: y + 21, width: 24, height: 7, fill: '#d5dde8', 'pointer-events': 'none'});
      base.insertBefore(bar, text);
      const tooltip = make('title'); text.append(tooltip);
      return {x, y, text, rects: rects.filter(node => +node.getAttribute('height') === 28), bar};
    })));
    const headings = [originalTexts.find(t => t.textContent.startsWith('(a)')), originalTexts.find(t => t.textContent.startsWith('(b)'))];
    const carriers = originalRects.filter(node => +node.getAttribute('width') === 22).concat(originalTexts.filter(node => [80.375,142.375,240.375,302.375].some(y => Math.abs(+node.getAttribute('y') - y) < .01)));
    const moving = make('g', {'aria-hidden': 'true', class: 'original-ring-moving'}); base.append(moving);
    const cue = document.createElement('p'); cue.className = 'original-ring-provenance'; cue.setAttribute('aria-live', 'polite');
    const reading = document.createElement('p'); reading.className = 'original-ring-reading';
    reading.textContent = 'Tile fill records how many distinct GPUs contributed. The line below the diagram tracks A’s contributors. The lower row begins after reduction. On narrow screens, use Expand or scroll inside the diagram.';
    let mounted = false, currentIndex=0, mountedStage=null;
    function centerArrival(index) {
      if (!mountedStage || mountedStage.scrollWidth <= mountedStage.clientWidth) return;
      const rank = index <= 3 ? index : index - 4;
      const scale = art.getBoundingClientRect().width / 840;
      const center = (82 + rank * 224) * scale;
      // Center only at causal boundaries, not on scroll events or every frame:
      // the reader remains free to swipe elsewhere and inspect the whole ring.
      mountedStage.scrollLeft = Math.max(0, center - mountedStage.clientWidth / 2);
    }
    function updatePanel(panel, values, active) {
      for (let rank = 0; rank < 4; rank++) for (let chunk = 0; chunk < 4; chunk++) {
        const cell = cells[panel][rank][chunk], set = values[rank][chunk], full = set.length === 4;
        cell.rects.forEach(rect => { rect.setAttribute('fill', full ? '#dcfce7' : '#ffffff'); rect.setAttribute('stroke', full ? '#16a34a' : '#94a3b8'); });
        cell.bar.setAttribute('height', set.length * 7); cell.bar.setAttribute('y', cell.y + 28 - set.length * 7); cell.bar.setAttribute('visibility', full || !set.length ? 'hidden' : 'visible');
        cell.text.replaceChildren(document.createTextNode((full ? 'Σ' : '') + letters[chunk]), make('title', {}, set.length ? `${letters[chunk]}: contributions from GPU ${set.join(', ')}` : 'Not yet gathered'));
        cell.text.setAttribute('fill', full ? '#15803d' : set.length ? '#0f172a' : '#94a3b8');
        cell.text.setAttribute('font-weight', full ? '700' : '400');
        cell.rects.forEach(rect => { const followed=chunk===0 && (panel===0 ? rank===Math.min(currentIndex,3) : currentIndex>3 && [3,...Array.from({length:currentIndex-3},(_,i)=>i)].includes(rank)); rect.setAttribute('stroke-width', followed ? '2.3' : '1'); if(followed) rect.setAttribute('stroke','#b45309'); });
        cell.text.dataset.contributors = set.join(','); cell.text.dataset.panel = panel; cell.text.dataset.rank = rank; cell.text.dataset.chunk = letters[chunk];
        cell.bar.style.opacity = active ? '1' : '.5';
      }
    }
    const controller = window.MLSysOriginalPlayer.create({
      root, outer: context.outer, layout: 'compact',
      title: 'How does a partial chunk become a sum on every GPU?',
      intro: 'Follow chunk A around a 4-GPU ring across 3 ReduceScatter rounds and 3 AllGather rounds to see how 6 steps reduce and distribute all 4 chunks.',
      scope: 'A 4-GPU logical ring (GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 0) with a gradient of size M split into 4 equal chunks (A, B, C, D of size M/4). Row (a) executes N − 1 = 3 ReduceScatter rounds (adding disjoint contributions), and row (b) executes N − 1 = 3 AllGather rounds (copying finished sums ΣA–ΣD), totaling 6 rounds and 6 × (M/4) = 1.5M bytes sent per GPU.',
      steps: names.map((label, index) => ({label, get description() { return currentIndex === index ? explanations[index] : 'On completion: ' + explanations[index]; }})),
      render(index, {stage, details}) {
        currentIndex=index;
        if (!mounted) { mountedStage=stage; stage.classList.add('original-ring-stage'); stage.tabIndex=0; stage.setAttribute('aria-label','Ring diagram; horizontal scrolling is available on narrow screens'); stage.append(art); details.append(cue, reading); mounted = true; }
        moving.replaceChildren();
        carriers.forEach(node => {
          const y=+node.getAttribute('y'), lower=y>160;
          const x=+node.getAttribute('x');
          const first=lower ? 3 : 0;
          const position=x>690 ? 0 : x>=378 && x<=455 && (y%160)>110 ? Math.round((x-381)/26) : Math.round(((x-154)%224)/26);
          node.setAttribute('visibility', index>first+position ? 'visible' : 'hidden');
        });
        updatePanel(0, snapshots[Math.min(index, 3)], index <= 3);
        updatePanel(1, index < 3 ? reduced.map(row => row.map(() => [])) : index === 3 ? reduced : snapshots[index], index >= 3);
        centerArrival(index);
        headings[0].textContent = index < 3 ? `(a) ReduceScatter: ${index} of 3 rounds complete` : '(a) After ReduceScatter: N − 1 = 3 steps';
        headings[1].textContent = index < 3 ? '(b) AllGather follows ReduceScatter' : `(b) AllGather: ${index - 3} of 3 rounds complete`;
        if (index <= 3) {
          const rank = index, provenance = snapshots[index][rank][0].map(r => `A${String.fromCharCode(8320 + r)}`).join(' + ');
          cue.textContent = `Follow A · GPU ${rank}: ${provenance}${index === 3 ? ' = ΣA. All four contributions are now included.' : '. This partial sum is still waiting for other ranks.'} · Per-GPU bytes sent: ${index} × (M/4).`;
        } else cue.textContent = `Follow A · ΣA is now on GPUs ${[3, ...Array.from({length: index - 3}, (_, i) => i)].join(', ')}. The contributor set stays {0, 1, 2, 3}; this phase copies rather than adds. · Per-GPU bytes sent: ${index} × (M/4)${index === 6 ? ' = 1.5M' : ''}.`;
      },
      transition({from, to, onFrame}) {
        if (to !== from + 1 || !transfers[to]) return;
        centerArrival(to);
        const panel = to <= 3 ? 0 : 1, y = panel * 160;
        const packets = transfers[to].map(transfer => {
          const emphasized = transfer.chunk === 0;
          const group = make('g', {opacity: emphasized ? '1' : '.7'});
          group.append(make('rect', {x: -13, y: -12, width: 26, height: 24, rx: 2, fill: to <= 3 ? '#fef3c7' : '#dcfce7', stroke: emphasized ? '#b45309' : '#d97706', 'stroke-width': emphasized ? 2 : 1}), make('text', {'text-anchor': 'middle', y: 5, 'font-size': 13, 'font-weight': 700, fill: '#14294E'}, (to > 3 ? 'Σ' : '') + letters[transfer.chunk]));
          moving.append(group);
          const path = transfer.rank === 3 ? [[754,112+y],[754,138+y],[82,138+y],[82,112+y]] : [[144+transfer.rank*224,76+y],[244+transfer.rank*224,76+y]];
          const lengths = path.slice(1).map((point, i) => Math.hypot(point[0]-path[i][0],point[1]-path[i][1]));
          return {group, path, lengths, total: lengths.reduce((a,b)=>a+b,0)};
        });
        onFrame(progress => packets.forEach(({group,path,lengths,total}) => {
          let distance=progress*total, segment=0;
          while(segment<lengths.length-1 && distance>lengths[segment]) distance-=lengths[segment++];
          const fraction=Math.min(1,distance/lengths[segment]), a=path[segment], b=path[segment+1];
          group.setAttribute('transform',`translate(${a[0]+(b[0]-a[0])*fraction},${a[1]+(b[1]-a[1])*fraction})`);
        }));
      }
    });
    return {pause: () => controller.pause(), destroy: () => controller.destroy()};
  };
})();
