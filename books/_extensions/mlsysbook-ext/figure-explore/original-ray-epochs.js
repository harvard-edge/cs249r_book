/* Qualified cell evidence, animated on the chapter's verified SVG geometry. */
(() => {
  'use strict';
  let serial = 0;
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons['original-ray-epochs'] = async (root, context) => {
    if (!context.verifiedArtwork || !window.MLSysOriginalPlayer) throw Error('Verified ray artwork and shared player required');
    const ns = 'http://www.w3.org/2000/svg', svg = context.verifiedArtwork.cloneNode(true);
    // Localize paint servers so multiple enhanced figures cannot share a hatch or marker.
    const prefix = 'ore-' + (++serial) + '-';
    for (const identified of [...svg.querySelectorAll('[id]')]) {
      const oldId = identified.id, newId = prefix + oldId;
      identified.id = newId;
      for (const n of [svg,...svg.querySelectorAll('*')]) {
        for (const attribute of [...n.attributes]) {
          if (attribute.value.includes('url(#' + oldId + ')')) n.setAttribute(attribute.name,attribute.value.replaceAll('url(#' + oldId + ')','url(#' + newId + ')'));
          else if (attribute.value === '#' + oldId) n.setAttribute(attribute.name,'#' + newId);
        }
      }
    }
    const cells = [...svg.querySelectorAll('rect')].filter(n => n.getAttribute('width') === '34' && n.getAttribute('height') === '34');
    const row = cells.filter(n => n.getAttribute('y') === '102').sort((a,b) => +a.getAttribute('x') - +b.getAttribute('x'));
    const ray = [...svg.querySelectorAll('line')].find(n => n.getAttribute('x1') === '74' && n.getAttribute('y1') === '128.0');
    const phantom = [...svg.querySelectorAll('rect')].find(n => n.getAttribute('x') === '284' && n.getAttribute('width') === '26');
    const texts = [...svg.querySelectorAll('text')];
    const labels = row.map(c => texts.find(t => +t.getAttribute('x') === +c.getAttribute('x') + 17 && +t.getAttribute('y') === 117));
    const header = texts.find(t => t.textContent.startsWith('ray crosses:'));
    const phantomLeader = texts.find(t => t.textContent === 'phantom retired');
    const returnLeader = texts.find(t => t.textContent.startsWith('return: occupied'));
    if (cells.length !== 70 || row.length !== 14 || !ray || !phantom || labels.some(x => !x) || !header || !phantomLeader || !returnLeader) throw Error('Ray artwork geometry changed; retain Diagram until reviewed');
    // These are source geometry coordinates, not a new grid or timing model.
    const rayStart = +ray.getAttribute('x1'), hitEdge = +row[9].getAttribute('x');
    const original = row.map((c,i) => ({fill:c.getAttribute('fill'),stroke:c.getAttribute('stroke'),label:labels[i].cloneNode(true)}));
    const headerFinal = header.cloneNode(true), returnFinal = returnLeader.cloneNode(true);
    root.classList.add('mlsys-original-ray-epochs');
    svg.setAttribute('role','img');svg.setAttribute('aria-label','Original occupancy grid: a qualified return updates only crossed cells and the hit; occluded and off-ray cells retain their evidence epochs');
    const canvas = document.createElement('div');canvas.className = 'ore-canvas';canvas.tabIndex = 0;canvas.setAttribute('aria-label','Original ray diagram. Scroll horizontally on narrow screens; the current cell stays visible.');canvas.append(svg);
    const note = document.createElement('p');note.className = 'ore-state';note.setAttribute('aria-live','off');
    const details = document.createElement('p');details.className = 'ore-invariant';
    details.textContent = 'The map may be published again without renewing untouched cells. Hatched cells behind the return remain at t₀; blank off-ray cells retain their own prior evidence.';
    const steps = [
      {label:'Inspect the old map',description:'The dashed phantom is an old occupied claim. No cell has received this return yet. Blank cells retain their own earlier evidence, which the figure does not specify.'},
      {label:'Receive one qualified return',description:'A range return captured at t₁ reaches the occupied cell. Follow the cells on this line; no evidence is supplied behind the hit.'},
      ...row.slice(0,9).map((_,i) => ({label:i===5?'Retire the phantom':`Update crossed cell ${i+1}`,description:i===5?'The qualified ray crosses the old phantom. Its cell becomes free with evidence epoch t₁; the dashed outline remains as a reminder of the retired claim.':`Crossed cell ${i+1} becomes free with evidence epoch t₁. Other cells keep their previous evidence until this return actually updates them.`})),
      {label:'Record the occupied return',description:'The hit cell becomes occupied with evidence epoch t₁. The update stops here: cells behind it are occluded, not newly observed free space.'},
      {label:'Publish without renewing untouched cells',description:'The updated map can now be published. Only the crossed cells and hit carry t₁; publication does not refresh the old t₀ evidence behind the return or any off-ray cell.'}
    ];
    function epoch(label,value,color) {
      label.replaceChildren(document.createTextNode('t'));
      const sub=document.createElementNS(ns,'tspan');sub.setAttribute('baseline-shift','sub');sub.setAttribute('font-size','7');sub.textContent=String(value);label.append(sub);label.setAttribute('fill',color);
    }
    function render(index,{stage,details:drawer}) {
      if(!canvas.isConnected)stage.append(canvas,note);if(!details.isConnected)drawer.append(details);
      const updated=Math.max(0,Math.min(9,index-1)), hit=index>=11;
      cells.forEach(c=>{c.dataset.evidenceEpoch=c.getAttribute('y')==='102'?'0':'unchanged';c.dataset.occupancy='unchanged';c.classList.remove('ore-current');});
      row.forEach((c,i)=>{
        const renewed=i<updated || (i===9 && hit);
        c.dataset.evidenceEpoch=renewed?'1':'0';c.dataset.occupancy=i<updated?'free':i===9&&hit?'occupied':i===5&&updated<=5?'old-occupied':'unchanged';
        c.setAttribute('fill',renewed?original[i].fill:(i>=10?original[i].fill:i===5?'#EDF2F7':'#F8FAFC'));
        c.setAttribute('stroke',renewed?original[i].stroke:(i>=10?original[i].stroke:'#A0AEC0'));
        epoch(labels[i],renewed?1:0,renewed?(i===9?'#FFFFFF':'#007A87'):'#4A5568');
        if ((index>=2&&index<=10&&i===updated-1)||(index===11&&i===9)) c.classList.add('ore-current');
      });
      phantom.dataset.retired=String(updated>5);phantom.setAttribute('opacity',updated>5?'0.45':'1');phantomLeader.textContent=updated>5?'phantom retired':'old phantom';
      returnLeader.replaceChildren(...(hit?[...returnFinal.childNodes].map(n=>n.cloneNode(true)):[document.createTextNode('return not yet recorded')]));
      header.replaceChildren(...(index>=11?[...headerFinal.childNodes].map(n=>n.cloneNode(true)):[document.createTextNode(index===0?'Old evidence: no cell renewed yet':'One qualified return captured at t₁')]));
      const end=index<=1?rayStart:updated<9?+row[updated-1].getAttribute('x')+34:hitEdge-1;
      ray.setAttribute('x2',String(end));ray.setAttribute('visibility',index<2?'hidden':'visible');
      root.dataset.rayStep=String(index);root.dataset.updatedFree=String(updated);root.dataset.hitRecorded=String(hit);root.dataset.mapPublished=String(index===12);
      note.textContent=index===0?'Before this return: old evidence is retained.':`${updated} crossed cells renewed · hit ${hit?'recorded at t₁':'pending'} · 4 cells behind the hit remain t₀`;
      if(canvas.scrollWidth>canvas.clientWidth){const focus=index===0?297:index<=1?rayStart:index===12?501:index===11?433:+row[Math.max(0,updated-1)].getAttribute('x')+17;canvas.scrollLeft=Math.max(0,focus/680*svg.getBoundingClientRect().width-canvas.clientWidth/2);}
    }
    function transition({from,to,onFrame}) {
      if(to!==from+1 || to<2 || to>11)return;
      const source=from<=1?rayStart:+ray.getAttribute('x2'), target=to===11?hitEdge-1:+row[to-2].getAttribute('x')+34;
      ray.setAttribute('visibility','visible');
      onFrame(progress=>ray.setAttribute('x2',String(source+(target-source)*progress))); 
    }
    return window.MLSysOriginalPlayer.create({root,outer:context.outer,title:'Which cells does one range return actually renew?',scope:'The original grid and qualified-return assumptions. Logical update order, not sensor flight time or measured processing speed. Blank off-ray cells keep their own unspecified prior evidence.',steps,render,transition});
  };
})();
