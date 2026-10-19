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
    {label: 'Begin with persistent shards', description: 'The rank owns parameter and optimizer shards. Full parameters have not yet been gathered for this unit.'},
    {label: 'Gather forward parameters', description: 'AllGather reconstructs full parameters for this unit. Persistent shards remain owned by the rank.'},
    {label: 'Forward creates saved activations', description: 'Forward uses the gathered parameters and produces saved activations. Backward will need those activations later.'},
    {label: 'Release gathered parameters', description: 'After forward, the gathered full parameters are released. The saved activations stay live because backward has not consumed them.'},
    {label: 'Gather backward parameters', description: 'AllGather reconstructs the same unit’s full parameters again. The saved forward activations are still available.'},
    {label: 'Backward produces a full gradient', description: 'Backward reads the full parameters and saved activations and produces a full local gradient. They remain live while their final consumers execute.'},
    {label: 'Release consumed backward state', description: 'Backward has finished its last reads. Release the gathered full parameters and saved activations; the full local gradient remains ready for ReduceScatter.'},
    {label: 'Reduce the gradient to its owned shard', description: 'ReduceScatter combines gradient contributions and leaves this rank’s reduced gradient shard for the local optimizer update. Persistent parameter and optimizer shards remain.'},
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
    let stageNode = null, mounted = false, currentIndex=0, resizeObserver=null;
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
      root, outer: context.outer,
      title: 'Why can gathered weights disappear while activations remain?',
      scope: 'One rank, one sharding unit, with resharding after forward. Columns are execution phases, not memory locations or equal time intervals. Cells show logical lifetimes, not bytes; buffers may be shared or reused.',
      steps,
      render(index, {stage, details}) {
        currentIndex=index;
        if (!mounted) {
          stage.classList.add('original-fsdp-workspace');stageNode=document.createElement('div');stageNode.className='original-fsdp-stage';stageNode.tabIndex=0;
          stageNode.setAttribute('aria-label','Original FSDP phase history; scroll horizontally to inspect phases');
          stageNode.append(art);stage.append(stageNode,livePanel);details.append(cue,optional);
          stageNode.addEventListener('scroll',keepLabels,{passive:true});
          resizeObserver=new ResizeObserver(()=>centerPhase(currentIndex));resizeObserver.observe(stageNode);mounted=true;
        }
        for (let column = 0; column < 6; column++) for (let row = 0; row < 6; row++) applyCell(index, column, row);
        updateLive(index);
        const column = phaseFor(index);
        currentOutline.setAttribute('x', 166 + column * 80);
        const resident = index === 0 ? [] : [live[column][2] && 'full parameters', live[column][3] && 'saved activations', live[column][4] && (index===6||live[column][4] === 'full' ? 'full local gradient' : 'reduced gradient shard')].filter(Boolean);
        cue.textContent = `Resident now: parameter shard and optimizer shard${resident.length ? '; ' + resident.join('; ') : ''}.`;
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
    return {pause: () => controller.pause(), destroy: () => {stageNode?.removeEventListener('scroll', keepLabels); resizeObserver?.disconnect(); controller.destroy();}};
  };
})();
