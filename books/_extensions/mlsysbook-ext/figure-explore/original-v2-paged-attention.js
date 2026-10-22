(() => {
 'use strict';const NAME='original-v2-paged-attention',NS='http://www.w3.org/2000/svg',MAP=[0,1,4,5],BASE=[null,null,'C','C',null,null,'D','D'];
 const svg=(tag,attrs={},text)=>{const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;};
 const steps=[
  {label:'Inspect reserved but unused blocks',description:'A and B reserve their maximum length. Their hatched, unused blocks belong to those reservations; another request cannot simply claim them.'},
  {label:'Switch to the fragmented admission snapshot',description:'Panel B is a different illustrative occupancy snapshot: C and D are other live sequences. They occupy slots 2–3 and 6–7; this is not an animation that renames A and B.'},
  {label:'Test contiguous admission for E',description:'E requires four contiguous blocks. The total free capacity is sufficient, but neither free run is large enough; contiguous admission must reject E.'},
  ...MAP.map((slot,page)=>({label:`Map E page ${page} to slot ${slot}`,description:`Use the block table to assign E’s logical page ${page} to free physical slot ${slot}. The new mapping and ownership appear only after this action completes; C and D stay in place.`})),
  {label:'Read E page 2 through its table',description:'Look up logical page 2, obtain physical slot 4, and read E2. Logical order is independent of physical adjacency; this read does not move the C/D blocks.'},
  {label:'Release E’s request-owned blocks',description:'After E finishes, remove its block-table entries and release only E’s blocks. C and D retain exactly their original slots.'},
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
   current.textContent=i===0?'Reserved capacity belongs to its sequence, even before every reserved block is filled.':i<3?`Different snapshot: ${free} free blocks; largest contiguous run ${maxGap}. E requires ${MAP.length} blocks.`:i===8?'E released. Slots 0,1,4,5 are free again; C and D have not moved.':`E block table: ${MAP.slice(0,count).map((slot,p)=>`page ${p} → slot ${slot}`).join('; ')}.${i===7?' Read: page 2 → slot 4 → E2.':''}`;
   root.dataset.originalState=JSON.stringify({owners,mappings:i>=8?[]:MAP.slice(0,count),read:i===7?'E2':null,free,maxGap});root.dataset.animationProgress='0';moving.replaceChildren();camera(i);
  }
  let narrationIndex=0;const narratedSteps=steps.map((step,i)=>({label:step.label,get description(){return narrationIndex===i?step.description:'This action is still in progress. Expected after completion: '+step.description;}}));
const player=window.MLSysOriginalPlayer.create({root,outer:context.outer,title:'How can free blocks admit E without a contiguous gap?',scope:'The source eight-block example. A cell is a page block, not one token. Panels A and B are different illustrative snapshots; B and C use the same C/D occupancy. The last E page retains its original partly used tail. Pale regions preview later stages.',steps:narratedSteps,
   render(i,{stage,details}){narrationIndex=i;now=i;if(!stageNode){stageNode=stage;stage.classList.add(NAME+'-stage');stage.tabIndex=0;stage.append(art);stage.before(current);observer=new ResizeObserver(()=>camera(visualCameraIndex));observer.observe(stage);}apply(i);},
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
