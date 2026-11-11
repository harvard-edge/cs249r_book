(() => {
 'use strict';const NAME='original-v2-paged-attention',NS='http://www.w3.org/2000/svg',MAP=[0,1,4,5],BASE=[null,null,'C','C',null,null,'D','D'];
 const svg=(tag,attrs={},text)=>{const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;};
 const steps=[
  {label:'Inspect reserved but unused blocks',description:'In panel (a), contiguous reservation preallocates the maximum length of 4 slots per sequence across the 8-slot pool. Seq A fills 2 of 4 slots and Seq B fills 3 of 4, leaving 3 hatched slots wasted to internal fragmentation.'},
  {label:'Switch to the fragmented admission snapshot',description:'In panel (b), a later occupancy snapshot shows live sequences C (slots 2–3) and D (slots 6–7), leaving 4 free physical slots split into two non-adjacent 2-slot gaps (slots 0–1 and slots 4–5).'},
  {label:'Test contiguous admission for E',description:'Arriving sequence E requires 4 slots. Although 4 of the 8 physical slots are free in total, the largest contiguous gap is only 2 slots, so contiguous allocation rejects Seq E due to external fragmentation.'},
  {label:'Map E page 0 to slot 0',description:'In panel (c), PagedAttention allocates on demand through a block table, mapping Seq E’s logical page 0 to free physical slot 0 (E0) while sequences C and D stay in place.'},
  {label:'Map E page 1 to slot 1',description:'The block table maps Seq E’s logical page 1 to free physical slot 1 (E1), filling the first 2-slot gap without needing 4 contiguous slots.'},
  {label:'Map E page 2 to slot 4',description:'Skipping over Seq C in physical slots 2 and 3, the block table maps Seq E’s logical page 2 to noncontiguous physical slot 4 (E2).'},
  {label:'Map E page 3 to slot 5',description:'The block table maps Seq E’s final logical page 3 to physical slot 5 (E3). All 4 pages of Seq E now fit in the 8-slot pool, and internal waste is bounded to the hatched unfilled tail of page 3.'},
  {label:'Read E page 2 through its table',description:'During attention, reading logical page 2 translates through the violet block table (page 2 → slot 4) to fetch E2 directly from physical slot 4 without relocating C or D.'},
  {label:'Release E’s request-owned blocks',description:'When Seq E finishes, the allocator clears its 4 block-table entries and returns physical slots 0, 1, 4, and 5 to the free pool while C (slots 2–3) and D (slots 6–7) remain undisturbed.'},
 ];
 window.MLSysFigureLessons=window.MLSysFigureLessons||{};
 window.MLSysFigureLessons[NAME]=async(root,context)=>{let visualCameraIndex=0;
  if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified paged artwork/player required');
  const art=context.verifiedArtwork.cloneNode(true);if(art.getAttribute('viewBox')!=='-9 -14 727 378')throw Error('Paged geometry changed');
  art.classList.add(NAME+'-art');art.setAttribute('role','img');art.setAttribute('aria-label','Original reservation, fragmented admission, and paged memory panels');
  const rects=[...art.querySelectorAll('rect')],texts=[...art.querySelectorAll('text')],cells=[];
  const request=svg('g',{'data-e-request':'true'});
  [...rects,...texts].filter(n=>n.getAttribute('stroke')==='#dc2626'||n.getAttribute('fill')==='#b91c1c').forEach(n=>request.append(n));art.append(request);
  const panels=[0,1,2].map(i=>svg('g',{'data-original-panel':i}));
  for(const n of [...art.children]){if(n.localName==='defs'||n.localName==='g')continue;const y=+(n.getAttribute('y')??n.getAttribute('y1')??-99),x=+(n.getAttribute('x')??n.getAttribute('x1')??0);if(y>=8&&y<170)panels[x<350?0:1].append(n);else if(y>=170&&y<322)panels[2].append(n);}
  panels.forEach(n=>art.append(n));
  for(let slot=0;slot<8;slot++){
   const x=368+41*slot,y=238,shapes=rects.filter(n=>+n.getAttribute('x')===x&&+n.getAttribute('y')===y),text=texts.find(n=>+n.getAttribute('y')===258.5&&+n.getAttribute('x')>=x&&+n.getAttribute('x')<x+38);
   if(!shapes.length||!text)throw Error('Paged physical slot mapping changed');
   const group=svg('g',{'data-slot':slot});shapes.forEach(n=>group.append(n));group.append(text);art.append(group);
   cells.push({group,text,shapes,original:shapes.map(n=>[...n.attributes].reduce((a,v)=>(a[v.name]=v.value,a),{})),label:text.textContent});
  }
  const tableValues=MAP.map((_,page)=>texts.find(n=>+n.getAttribute('x')===320&&+n.getAttribute('y')===251+18*page));if(tableValues.some(n=>!n))throw Error('Paged table rows changed');
  tableValues.forEach((n,p)=>n.dataset.pageMapping=String(p));
  const readGroup=svg('g',{'data-read-result':'true'});readGroup.append(svg('rect',{x:480,y:294,width:160,height:22,rx:2,fill:'#dcfce7',stroke:'#16a34a'}),svg('text',{x:560,y:309,'text-anchor':'middle','font-size':11,fill:'#15803d'},'Read result: E2'));art.append(readGroup);
  const moving=svg('g',{'aria-hidden':'true'});art.append(moving);
  const current=document.createElement('p');current.className=NAME+'-proof';
  let stageNode=null,now=0,observer;
  const allocated=i=>i>=8?0:Math.max(0,Math.min(4,i-2));
  function camera(i){visualCameraIndex=i;root.dataset.visualCameraIndex=String(i);if(!stageNode)return;const c=i===0?170:i<3?530:i>=3?530:530;stageNode.scrollLeft=Math.max(0,(c+9)*art.getBoundingClientRect().width/727-stageNode.clientWidth/2);}
  function apply(i){const count=allocated(i),owners=[...BASE];panels[0].style.opacity=i===0?'1':'.4';panels[1].style.opacity=i>=1?'1':'.12';panels[2].style.opacity=i>=3?'1':'.12';request.style.opacity=i>=2?'1':'.1';for(let page=0;page<count;page++)owners[MAP[page]]='E'+page;
   cells.forEach((cell,slot)=>{const page=MAP.indexOf(slot),occupied=BASE[slot]||page<count&&i<8;cell.group.dataset.owner=owners[slot]||'';cell.group.style.opacity=i>=3?'1':'.15';cell.text.textContent=occupied?cell.label:'';
    cell.shapes.forEach((shape,j)=>{for(const[k,v]of Object.entries(cell.original[j]))shape.setAttribute(k,v);shape.style.visibility=occupied?'visible':j===0?'visible':'hidden';if(!occupied){shape.setAttribute('fill','#fff');shape.setAttribute('stroke','#94a3b8');shape.setAttribute('stroke-dasharray','4,3');}});
   });
   tableValues.forEach((n,p)=>{n.style.opacity=p<count&&i<8?'1':'.12';n.dataset.physicalSlot=p<count&&i<8?String(MAP[p]):'';});
   readGroup.style.visibility=i===7?'visible':'hidden';readGroup.style.opacity='1';readGroup.dataset.value=i===7?'E2':'';
   const gaps=[];let run=0;for(const owner of BASE){if(owner===null)run++;else{gaps.push(run);run=0;}}gaps.push(run);const free=BASE.filter(n=>n===null).length,maxGap=Math.max(...gaps);
   current.textContent=i===0?'Panel (a) internal fragmentation: Seq A uses 2/4 reserved slots and Seq B uses 3/4 (3/8 slots reserved but unused).':i<3?`Panel (b) external fragmentation: ${free} of 8 slots are free (slots 0–1 and 4–5), but the largest contiguous gap is ${maxGap} while Seq E needs ${MAP.length}.`:i===8?'Seq E released: physical slots 0, 1, 4, and 5 are free again (4/8 free); C (2–3) and D (6–7) have not moved.':`Panel (c) PagedAttention block table (${count}/4 pages mapped): ${MAP.slice(0,count).map((slot,p)=>`page ${p} → slot ${slot}`).join('; ')}.${i===7?' Indirection lookup: page 2 → slot 4 → E2.':''}`;
   root.dataset.originalState=JSON.stringify({owners,mappings:i>=8?[]:MAP.slice(0,count),read:i===7?'E2':null,free,maxGap});root.dataset.animationProgress='0';moving.replaceChildren();camera(i);
  }
  let narrationIndex=0;const narratedSteps=steps.map((step,i)=>({label:step.label,get description(){return narrationIndex===i?step.description:'This action is still in progress. Expected after completion: '+step.description;}}));
const player=window.MLSysOriginalPlayer.create({root,outer:context.outer,layout:'compact',title:'How can free blocks admit E without a contiguous gap?',intro:'Map 4 logical pages of Seq E through a block table into noncontiguous physical slots 0, 1, 4, and 5 across an 8-slot pool.',scope:'Eight-slot KV-cache memory pool comparing contiguous reservation in panel (a), external fragmentation across two 2-slot gaps in panel (b), and noncontiguous block-table mapping for 4-page Seq E in panel (c). Each cell represents a fixed-size KV page block rather than a single token.',steps:narratedSteps,
   render(i,{stage,details}){narrationIndex=i;now=i;if(!stageNode){stageNode=stage;stage.classList.add(NAME+'-stage');stage.tabIndex=0;stage.append(art);details.append(current);observer=new ResizeObserver(()=>camera(visualCameraIndex));observer.observe(stage);}apply(i);},
   transition({from,to,onFrame}){if(to!==from+1)return;camera(to);const packet=svg('circle',{r:4,fill:'#d97706','data-motion-object':'true'});if(to>=3&&to<=7)moving.append(packet);if(to===1)panels[1].dataset.motionObject='true';if(to===2)request.dataset.motionObject='true';const page=to>=3&&to<=6?to-3:2,slot=MAP[page];
    if(to===3){panels[2].style.opacity='1';cells.forEach(cell=>cell.group.style.opacity='1');}if(to>=3&&to<=6){const outline=svg('rect',{x:368+41*slot,y:238,width:38,height:32,fill:'none',stroke:'#d97706','stroke-width':2,'stroke-dasharray':'3,2','data-pending-slot':slot,'data-motion-object':'true'});moving.append(outline);}
    if(to===7){const pending=svg('g',{'data-read-pending':'true','data-motion-object':'true'});pending.append(svg('rect',{x:480,y:294,width:160,height:22,rx:2,fill:'#fef3c7',stroke:'#d97706'}),svg('text',{x:560,y:309,'text-anchor':'middle','font-size':11,fill:'#92400e'},'Read pending…'));moving.append(pending);}
    onFrame(p=>{root.dataset.animationProgress=String(p);if(to===1){panels[0].style.opacity=String(1-.6*p);panels[1].style.opacity=String(.12+.88*p);}if(to===2)request.style.opacity=String(.1+.9*p);if(to>=3&&to<=7){const points=[[62+41*page,254],[210,254],[270,251+18*page],[334,251+18*page],[350,278],[387+41*slot,278],[387+41*slot,254]],lengths=points.slice(1).map((q,j)=>Math.hypot(q[0]-points[j][0],q[1]-points[j][1])),total=lengths.reduce((a,b)=>a+b,0);let d=p*total,j=0;while(j<lengths.length-1&&d>lengths[j]){d-=lengths[j];j++;}const f=d/lengths[j],q=points[j],r=points[j+1];packet.setAttribute('transform',`translate(${q[0]+(r[0]-q[0])*f},${q[1]+(r[1]-q[1])*f})`);}

     if(to===8)for(const s of MAP)cells[s].group.style.opacity=String(1-p);
    });
   }
  });return{pause:()=>player.pause(),destroy:()=>{observer?.disconnect();player.destroy();}};
 };
})();
