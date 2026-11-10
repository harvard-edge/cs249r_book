/* Animate the source lifetime grid without treating phase columns as memory. */
(() => {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const rows = ['Parameter shard', 'Optimizer shard', 'Gathered parameters', 'Saved activations', 'Gradient buffer', 'Next-unit prefetch'];
  const live = [
    ['present', 'present', 'live', '', '', ''],
    ['present', 'present', 'live', 'live', '', 'optional'],
    ['present', 'present', '', 'live', '', ''],
    ['present', 'present', 'live', 'live', '', ''],
    ['present', 'present', 'live', 'live', 'full', 'optional'],
    ['present', 'present', '', '', 'shard', ''],
  ];
  const steps = [
    {label: 'Begin with persistent shards', description: 'Before entering the 6 execution phase columns for this sharding unit, 1 rank permanently holds only 2 persistent allocations: its 1/N Parameter shard (row 1) and 1/N Optimizer shard (row 2). Gathered parameters, activations, and gradients are not yet allocated.'},
    {label: 'Gather forward parameters', description: 'In phase column 1 (Gather forward, collective 1 of 3), AllGather reconstructs the unit’s full parameters across ranks into row 3 (Gathered parameters: live). Total resident allocations rise to 3 rows (Parameter shard, Optimizer shard, and Gathered parameters).'},
    {label: 'Forward creates saved activations', description: 'In phase column 2 (Forward compute), the forward pass consumes the live Gathered parameters and writes row 4 (Saved activations: live) for backpropagation. Resident allocations reach 4 rows (plus row 6 dashed optional next-unit prefetch if overlap headroom permits).'},
    {label: 'Release gathered parameters', description: 'In phase column 3 (Release parameters), FSDP frees row 3 (Gathered parameters) immediately after forward compute to reclaim HBM for subsequent layers, dropping resident allocations to 3 rows. Row 4 (Saved activations) must remain live until backward compute consumes it.'},
    {label: 'Gather backward parameters', description: 'In phase column 4 (Gather backward, collective 2 of 3), a second AllGather reconstructs row 3 (Gathered parameters: live) just before this unit’s backward pass, bringing resident allocations back to 4 rows alongside the retained Saved activations.'},
    {label: 'Backward produces a full gradient', description: 'In phase column 5 (Backward compute), backward kernels read both row 3 (Gathered parameters) and row 4 (Saved activations) to compute row 5 (Gradient buffer: full). Simultaneously live allocations peak at 5 rows (or 6 if optional backward prefetch is active).'},
    {label: 'Release consumed backward state', description: 'As soon as backward compute finishes its final reads in column 5, FSDP releases both row 3 (Gathered parameters) and row 4 (Saved activations), leaving 3 resident rows: the 2 persistent shards and the unreduced full local gradient in row 5.'},
    {label: 'Reduce the gradient to its owned shard', description: 'In phase column 6 (Reduce / update, collective 3 of 3), ReduceScatter sums the full gradient across ranks and replaces row 5 with this rank’s 1/N reduced gradient shard (shard), ready for the local optimizer update on the 2 persistent shards.'},
  ];
  const phaseFor = index => Math.min(5, Math.max(0, index - 1));
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons['original-fsdp-lifetime'] = async (root, context) => {
    if (!context.verifiedArtwork || !window.MLSysOriginalPlayer) throw new Error('Verified lifetime artwork and shared player required');
    const art = context.verifiedArtwork.cloneNode(true);
    if (art.getAttribute('viewBox') !== '0 0 680 410') throw new Error('FSDP lifetime geometry requires review');
    art.classList.add('original-fsdp-art');
    art.setAttribute('role', 'img');
    art.setAttribute('aria-label', 'One-rank allocation lifetimes across six execution phases. Rows identify objects; columns are phases, not memory locations.');
    const texts = [...art.querySelectorAll('text')];
    const rectangles = [...art.querySelectorAll('rect')];
    const cells = Array.from({length: 6}, (_, column) => Array.from({length: 6}, (_, row) => {
      const x = 170 + column * 80, y = 111 + row * 42;
      const rects = rectangles.filter(node => +node.getAttribute('x') === x && +node.getAttribute('y') === y && +node.getAttribute('width') === 70);
      const value = texts.find(node => +node.getAttribute('x') === x + 35 && +node.getAttribute('y') === y + 19);
      if (rects.length !== (live[column][row] ? 2 : 1) || Boolean(value) !== Boolean(live[column][row])) throw new Error('FSDP source cell mapping changed');
      const group = document.createElementNS(NS, 'g');
      group.dataset.fsdpColumn = column; group.dataset.fsdpRow = row;
      // Retain the source cells and labels; moving them into groups changes no
      // positions and lets the original allocation appear as one coherent object.
      rects.forEach(node => group.append(node));
      if (value) group.append(value);
      art.append(group);
      return {group, rects, value};
    }));
    // Keep source row names readable while the same wide phase grid scrolls.
    // Only the labels and their white backing track the local scroll viewport.
    const labelLayer = document.createElementNS(NS, 'g');
    const backing = document.createElementNS(NS, 'rect');
    for (const [name, value] of Object.entries({x: 0, y: 100, width: 163, height: 255, fill: '#ffffff'})) backing.setAttribute(name, value);
    labelLayer.append(backing);
    for (const row of rows) {
      const label = texts.find(node => node.textContent === row);
      if (!label) throw new Error('FSDP row name missing');
      labelLayer.append(label);
    }
    art.append(labelLayer);
    const currentOutline = document.createElementNS(NS, 'rect');
    for (const [name, value] of Object.entries({x: 166, y: 104, width: 78, height: 210, rx: 3, fill: 'none', stroke: '#d97706', 'stroke-width': 2, 'pointer-events': 'none'})) currentOutline.setAttribute(name, value);
    art.append(currentOutline);
    const livePanel=document.createElement('section');livePanel.className='original-fsdp-live';livePanel.setAttribute('aria-label','Current allocations on this rank; a live state view, not another execution phase');
    const liveHeading=document.createElement('h5');liveHeading.textContent='Live now';livePanel.append(liveHeading);
    const liveSubheading=document.createElement('p');liveSubheading.className='original-fsdp-live-subtitle';liveSubheading.textContent='current state';livePanel.append(liveSubheading);
    const liveObjects=rows.map((label,row)=>{
      const line=document.createElement('div');line.className='original-fsdp-live-row';line.style.setProperty('--fsdp-row-y',String(111+42*row));
      const name=document.createElement('span');name.className='original-fsdp-live-label';name.textContent=label;
      const slot=document.createElement('div');slot.className='original-fsdp-live-slot';slot.dataset.liveRow=String(row);slot.setAttribute('role','img');
      const object=document.createElement('span');object.className='original-fsdp-live-object';object.dataset.kind=row===3?'activation':row===4?'gradient':'parameter';slot.append(object);line.append(name,slot);livePanel.append(line);
      return {slot,object};
    });
    const currentObjects=index=>[
      'Parameter shard','Optimizer shard',
      [1,2,4,5].includes(index)?'Full parameters':'',
      [2,3,4,5].includes(index)?'Saved activations':'',
      index===5||index===6?'Full gradient':index===7?'Reduced shard':'','',
    ];
    function updateLive(index){
      currentObjects(index).forEach((value,row)=>{const {slot,object}=liveObjects[row];slot.dataset.resident=String(Boolean(value));object.textContent=value;object.hidden=!value;object.style.opacity='1';slot.setAttribute('aria-label',`${rows[row]}: ${value|| (row===5?'optional prefetch not executed':'not resident')}`);});
      livePanel.dataset.fsdpStep=String(index);
    }
    const cue = document.createElement('p'); cue.className = 'original-fsdp-resident';
    const optional = document.createElement('p'); optional.className = 'original-fsdp-note';
    optional.textContent = 'Dashed “optional” cells show where another unit could be prefetched. This walkthrough does not execute prefetch. Past cells record the earlier phase; pale future cells are a preview.';
    let stageNode = null, mounted = false, currentIndex=0, resizeObserver=null, resizeFrame=0;
    function keepLabels() {
      if (!stageNode) return;
      const scale = art.getBoundingClientRect().width / 680;
      labelLayer.setAttribute('transform', `translate(${scale ? stageNode.scrollLeft / scale : 0},0)`);
    }
    function centerPhase(index) {
      if (!stageNode) return;
      const scale = art.getBoundingClientRect().width / 680;
      livePanel.style.setProperty('--fsdp-scale',String(scale));
      const labelWidth = 163 * scale;
      stageNode.scrollLeft = Math.max(0, (205 + 80 * phaseFor(index)) * scale - (stageNode.clientWidth + labelWidth) / 2);
      keepLabels();
    }
    // Resize delivery only schedules work: writing the sibling's scaled height
    // during ResizeObserver delivery can trigger WebKit's undelivered-loop error.
    function scheduleCenterPhase() {
      if (resizeFrame) return;
      resizeFrame = requestAnimationFrame(() => {
        resizeFrame = 0;
        centerPhase(currentIndex);
      });
    }
    function appearance(index, column, row) {
      if (row < 2) return {opacity: 1, state: 'present'};
      if (row === 5) return {opacity: .55, state: live[column][row] ? 'optional-not-executed' : 'empty'};
      const current = phaseFor(index), revealed = index > 0 && column <= current;
      const value = index === 6 && column === 5 && row === 4 ? 'full' : live[column][row];
      return {opacity: revealed ? (column === current ? 1 : .45) : .12, state: revealed ? (value || 'empty') : 'future'};
    }
    function applyCell(index, column, row) {
      const cell = cells[column][row], state = appearance(index, column, row);
      cell.group.style.opacity = String(state.opacity);
      cell.group.dataset.allocationState = state.state;
      cell.group.dataset.current = String(column === phaseFor(index));
      if(cell.value){cell.value.style.opacity='1';cell.value.textContent=index===6&&column===5&&row===4?'full':live[column][row];}
    }
    const controller = window.MLSysOriginalPlayer.create({
      root, outer: context.outer, layout: 'compact',
      title: 'Why can gathered weights disappear while activations remain?',
      intro: 'Step through 6 execution phases on 1 FSDP rank to see why gathered parameters are released between forward and backward while saved activations remain live.',
      scope: 'One rank executing 1 FSDP sharding unit (ZeRO-3 with reshard-after-forward) across 6 execution phase columns and 6 allocation rows. Each unit performs 3 collectives per step (forward AllGather, backward AllGather, and gradient ReduceScatter). Rows track logical buffer residency (from 2 persistent shards up to 5 live allocations during backward compute), not physical memory addresses or equal time durations.',
      steps: steps.map((step, index) => ({label: step.label, get description() { return currentIndex === index ? step.description : 'On completion: ' + step.description; }})),
      render(index, {stage, details}) {
        currentIndex=index;
        if (!mounted) {
          stage.classList.add('original-fsdp-workspace');stageNode=document.createElement('div');stageNode.className='original-fsdp-stage';stageNode.tabIndex=0;
          stageNode.setAttribute('aria-label','Original FSDP phase history; scroll horizontally to inspect phases');
          stageNode.append(art);stage.append(stageNode);details.append(cue,livePanel,optional);
          stageNode.addEventListener('scroll',keepLabels,{passive:true});
          resizeObserver=new ResizeObserver(scheduleCenterPhase);resizeObserver.observe(stageNode);mounted=true;
        }
        for (let column = 0; column < 6; column++) for (let row = 0; row < 6; row++) applyCell(index, column, row);
        updateLive(index);
        const column = phaseFor(index);
        currentOutline.setAttribute('x', 166 + column * 80);
        const resident = index === 0 ? [] : [live[column][2] && 'full parameters', live[column][3] && 'saved activations', live[column][4] && (index===6||live[column][4] === 'full' ? 'full local gradient' : 'reduced gradient shard')].filter(Boolean);
        cue.textContent = `Resident now (${2 + resident.length} of 6 rows active): parameter shard and optimizer shard${resident.length ? '; ' + resident.join('; ') : ''}.`;
        cue.dataset.fsdpStep = index;
        centerPhase(index);
      },
      transition({from, to, onFrame}) {
        if (to !== from + 1) return;
        centerPhase(to);
        const column = phaseFor(to), previousColumn = phaseFor(from);
        const before=currentObjects(from),after=currentObjects(to);
        const liveChanges=after.map((value,row)=>({value,row,old:before[row]})).filter(change=>change.value!==change.old);
        for(const change of liveChanges){const object=liveObjects[change.row].object;object.hidden=false;if(!change.old)object.textContent=change.value;}
        const changed = [];
        for(let c=0;c<6;c++)for(const row of [2,3,4])if(appearance(from,c,row).opacity!==appearance(to,c,row).opacity)changed.push({column:c,row});
        // Allocation fields in the arriving phase appear together. Their row
        // identities stay fixed: nothing is physically copied between columns.
        onFrame(progress => {
          for(const change of liveChanges){const object=liveObjects[change.row].object;if(change.old&&change.value)object.style.opacity=String(1-progress*.85);else object.style.opacity=String(change.value?progress:1-progress);}
          for (const {column:c,row} of changed) {
            const start = appearance(from, c, row).opacity;
            const end = appearance(to, c, row).opacity;
            cells[c][row].group.style.opacity = String(start + (end - start) * progress);
          }
          currentOutline.setAttribute('x', 166 + 80 * (previousColumn + (column - previousColumn) * progress));
          if(to===7)cells[5][4].value.style.opacity=String(1-progress*.85);
        });
      },
    });
    return {pause: () => controller.pause(), destroy: () => {stageNode?.removeEventListener('scroll', keepLabels); resizeObserver?.disconnect(); cancelAnimationFrame(resizeFrame); controller.destroy();}};
  };
})();
