/* Original logical-page/block-table/HBM drawing; no compaction or payload reduction. */
(() => {
  'use strict';
  const NS='http://www.w3.org/2000/svg';
  const make=(tag,attrs={},text)=>{const n=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;return n;};
  const blocks=[0,1,4,5],values=['a b','c d','e f','g'],steps=[{label:'One request, scattered storage',description:'The upper pool is live: E owns blocks 0, 1, 4, and 5; C and D keep their blocks. The lower pool is the after-release reference, not a second allocation.',kind:'start',page:0}];
  for(let p=0;p<4;p++){
    steps.push({label:`Page ${p}: look up its block`,description:`Page ${p} contains K/V records ${values[p]}. Follow its existing arrow to the table entry naming physical block ${blocks[p]}. The records remain in HBM.`,kind:'lookup',page:p});
    steps.push({label:`Page ${p}: resolve block ${blocks[p]}`,description:`The table sends the read to block ${blocks[p]}, without moving any blocks. ${p===3?'Only g is stored in this toy last block; the hatched half is unused but still allocated to E.':'Logical token order is independent of physical placement.'}`,kind:'resolve',page:p});
    steps.push({label:`Page ${p}: read retained records`,description:`Read ${values[p]} from block ${blocks[p]}. The returning copy does not remove or change the resident records.`,kind:'read',page:p});
  }
  steps.push({label:'The final consumer finishes',description:'All reads have finished and E has completed. Its private blocks are still allocated at this endpoint; release is a separate action.',kind:'finish',page:3});
  steps.push({label:'Release E; preserve C and D',description:'E’s four private blocks return to the free pool only after the last consumer finishes. The lower reference is now the resulting pool. C and D retain the same blocks; no compaction occurs.',kind:'release',page:3});
  window.MLSysFigureLessons=window.MLSysFigureLessons||{};
  window.MLSysFigureLessons['original-kv-paging']=async(root,{outer,verifiedArtwork})=>{
    if(!verifiedArtwork||!window.MLSysOriginalPlayer)throw new Error('Verified paged KV artwork and original player required.');
    root.classList.add('mlsys-original-kv-paging');
    const svg=verifiedArtwork.cloneNode(true);svg.removeAttribute('width');svg.removeAttribute('height');svg.setAttribute('role','img');svg.setAttribute('aria-label','Original request E logical pages, block table, and eight-block physical pool before and after release.');
    const originalBox=svg.getAttribute('viewBox');
    if(originalBox!=='0 0 357.912 209.991')throw new Error('Paged KV source geometry changed.');
    const bounds=path=>{const a=path.getAttribute('d').match(/-?\d*\.?\d+(?:e[+-]?\d+)?/gi).map(Number),m=new DOMMatrix(path.getAttribute('transform')||undefined),pts=[0,2,4,6].map(i=>new DOMPoint(a[i],a[i+1]).matrixTransform(m));return{x:Math.min(...pts.map(p=>p.x)),y:Math.min(...pts.map(p=>p.y)),w:Math.max(...pts.map(p=>p.x))-Math.min(...pts.map(p=>p.x)),h:Math.max(...pts.map(p=>p.y))-Math.min(...pts.map(p=>p.y))};};
    const rects=[...svg.querySelectorAll('path[stroke]')].filter(p=>p.getAttribute('d')?.includes(' Z M ')).map(p=>({p,...bounds(p)}));
    const row=(y,w)=>{const seen=new Set();return rects.filter(b=>Math.abs(b.y-y)<1&&Math.abs(b.w-w)<1).filter(b=>{const key=Math.round(b.x);if(seen.has(key))return false;seen.add(key);return true;}).sort((a,b)=>a.x-b.x);};
    const pages=row(21.129,56.68),table=row(55.137,56.68),physical=row(120.32,39.68),after=row(189.75,39.68);
    if(pages.length!==4||table.length!==4||physical.length!==8||after.length!==8)throw new Error('Paged KV object mapping changed.');
    const center=b=>({x:b.x+b.w/2,y:b.y+b.h/2});
    const overlay=make('g',{'data-kv-overlay':''}),motion=make('g',{'aria-hidden':'true'});svg.append(overlay,motion);
    const labels=[];
    function label(b,text,fill,opts={}){const g=make('g'),width=opts.width||b.w;g.append(make('rect',{x:b.x+.8,y:b.y+.8,width:width-1.6,height:b.h-1.6,fill}));g.append(make('text',{x:b.x+width/2,y:center(b).y,'dominant-baseline':'middle','text-anchor':'middle',fill:'#17242b','font-family':'sans-serif','font-size':opts.size||6},text));overlay.append(g);return g;}
    pages.forEach((b,p)=>label(b,`Page ${p}: ${values[p]}`,'#eaf9ff',{size:6}));
    blocks.forEach((block,p)=>labels.push(label(physical[block],`E: ${values[p]}`,'#e1f7ff',{width:p===3?physical[block].w/2:physical[block].w,size:p===3?5.5:6.3})));
    const focus=make('rect',{fill:'none',stroke:'#b45100','stroke-width':1.4,rx:.6});overlay.append(focus);
    // The original lower panel remains a labelled comparison, never an initial-state transition.
    const reference=make('rect',{x:0,y:174,width:357.912,height:35.991,fill:'white',opacity:.63,'pointer-events':'none'});overlay.append(reference);
    const readout=document.createElement('p');readout.className='kv-paging-readout';readout.setAttribute('aria-live','off');
    const viewport=document.createElement('div');viewport.className='kv-paging-viewport';viewport.append(svg);
    const viewLabel=document.createElement('label');viewLabel.className='kv-paging-view';viewLabel.textContent='Diagram view ';const view=document.createElement('select');view.setAttribute('aria-label','Diagram view');for(const [value,text]of [['follow','Follow the active page'],['whole','Whole diagram']]){const option=document.createElement('option');option.value=value;option.textContent=text;view.append(option);}view.value=matchMedia('(max-width:600px)').matches?'follow':'whole';viewLabel.append(view);
    let index=0;
    function follow(target){const step=steps[target],p=step.page,poolDetail=view.value==='follow'&&matchMedia('(max-width:600px)').matches&&['finish','release'].includes(step.kind);viewport.classList.toggle('is-pool-detail',poolDetail);if(poolDetail){svg.setAttribute('viewBox','0 106 357.912 103.991');viewport.scrollLeft=245;return;}if(view.value==='whole'||['start','finish','release'].includes(step.kind)){svg.setAttribute('viewBox',originalBox);return;}const logical=pages[p],pool=physical[blocks[p]],left=Math.max(0,Math.min(logical.x,pool.x)-7),right=Math.min(357.912,Math.max(logical.x+logical.w,pool.x+pool.w)+7);svg.setAttribute('viewBox',`${left} 14 ${Math.max(92,right-left)} 148`);}
    view.addEventListener('change',()=>follow(index));
    function render(next,{stage}){index=next;const step=steps[next];if(!stage.contains(viewport))stage.append(viewLabel,viewport,readout);motion.replaceChildren();const released=step.kind==='release';labels.forEach(g=>g.style.visibility=released?'hidden':'visible');reference.style.opacity=released?'0':'.63';
      // Upper live pool changes to free only at release; the reference below then agrees.
      overlay.querySelectorAll('[data-released-block]').forEach(n=>n.remove());if(released){blocks.forEach(block=>{const g=label(physical[block],'Free','white',{size:6.3});g.dataset.releasedBlock='';});const stale=make('g',{'data-released-block':''});stale.append(make('rect',{x:0,y:15,width:330,height:94,fill:'white',opacity:.72}),make('rect',{x:224,y:141,width:55,height:25,fill:'white'}),make('text',{x:250,y:151,'text-anchor':'middle','font-size':6,'font-family':'sans-serif',fill:'#17242b'},'Tail released'));overlay.append(stale);}
      const b=step.kind==='lookup'?table[step.page]:['resolve','read'].includes(step.kind)?physical[blocks[step.page]]:null;focus.style.visibility=b?'visible':'hidden';if(b)for(const[k,v]of Object.entries({x:b.x-.5,y:b.y-.5,width:b.w+1,height:b.h+1}))focus.setAttribute(k,v);
      root.dataset.kvStep=String(next);root.dataset.kvReleased=String(released);root.dataset.kvPending=step.kind==='finish'||released?'0':step.kind==='read'?'1':'0';root.dataset.kvResident=released?'C C D D':'E E C C E E D D';
      readout.textContent=released?'Result: E’s blocks are free. C and D have not moved. Paging changed placement, not the number of KV bytes per token. In the close pool view, swipe horizontally to inspect all eight blocks.':`Upper row: current allocation. Lower row: after-release reference. ${step.kind==='read'?`Read copy: ${values[step.page]}; resident records unchanged.`:'Toy block capacity: two K/V records. Letters name records, not scalar values or byte sizes.'}`;follow(next);
    }
    function route(step){const p=step.page,a=center(pages[p]),b=center(table[p]),c=center(physical[blocks[p]]);if(step.kind==='lookup')return[a,b];if(step.kind==='resolve')return[{x:b.x,y:table[p].y+table[p].h},{x:b.x,y:86.69},{x:c.x,y:86.69},{x:c.x,y:physical[blocks[p]].y}];if(step.kind==='read')return[c,{x:c.x,y:104},{x:a.x,y:104},a];return[];}
    return window.MLSysOriginalPlayer.create({root,outer,title:'How does one ordered cache use scattered blocks?',scope:'Original eight-block pool, no prefix sharing or compaction. Two records per block is a declared toy capacity; g leaves an allocated unused tail. Motion shows logical lookup and read dependencies, not timing or physical transport.',steps:steps.map((step,i)=>({...step,get description(){return index===i?step.description:'On completion: '+step.description;}})),render,transition:({to,onFrame})=>{
      const step=steps[to],points=route(step);follow(to);if(!points.length)return;
      const packet=make('g');packet.append(make('rect',{x:-14,y:-6,width:28,height:12,rx:2,fill:step.kind==='read'?'#f6dfbc':'#e1f7ff',stroke:'#854600','stroke-width':.7}),make('text',{'text-anchor':'middle','dominant-baseline':'middle','font-size':6,'font-family':'sans-serif',fill:'#17242b'},step.kind==='read'?values[step.page]:step.kind==='lookup'?`P${step.page}`:`B${blocks[step.page]}`));motion.append(packet);
      const lengths=points.slice(1).map((b,i)=>Math.hypot(b.x-points[i].x,b.y-points[i].y)),total=lengths.reduce((a,b)=>a+b,0);
      onFrame(t=>{let distance=t*total,j=0;while(j<lengths.length-1&&distance>lengths[j])distance-=lengths[j++];const u=lengths[j]?distance/lengths[j]:0,a=points[j],b=points[j+1];packet.setAttribute('transform',`translate(${a.x+(b.x-a.x)*u} ${a.y+(b.y-a.y)*u})`);});
    }});
  };
})();
