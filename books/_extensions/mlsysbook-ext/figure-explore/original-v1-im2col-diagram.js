/* Authored mechanism derived from the verified chapter figure; endpoints are authoritative. */
(()=>{'use strict';
const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
const sum=a=>a.reduce((a,b)=>a+b,0),cell=(text,kind='')=>{const n=el('span','v1c-cell '+kind,text);return n;};
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v1-im2col-diagram"]=async(root,{outer,verifiedArtwork})=>{
 if(!verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 root.classList.add('mlsys-v1c');const diagram=el('div','v1c-diagram'),status=el('p','v1c-status'),inspection=el('details','v1c-original');inspection.append(el('summary','','Inspect the unchanged source drawing'));const art=verifiedArtwork.cloneNode(true);art.removeAttribute('width');art.removeAttribute('height');art.setAttribute('aria-label','Verified original chapter artwork');inspection.append(art);
 const region=(title,columns=1)=>{const box=el('section','v1c-region');box.append(el('strong','',title));const grid=el('div','v1c-grid');grid.style.setProperty('--columns',columns);box.append(grid);diagram.append(box);return grid;};
 const detail=el('details');detail.append(el('summary','','Inspect the operation'));const equation=el('p');detail.append(equation);
 let steps=[],tracks=[],current=0,clearFlightResize=()=>{};
 
 const input=region('Input channel 0 · original 3 × 3',3),input2=region('Input channel 1 · original 3 × 3',3),dest=region('Materialized patch matrix · 4 × 8',8),weights=region('W · 8×1',1),result=region('Output positions · supporting calculation',4); diagram.classList.add('v1c-im2col');const gemm=el('div','v1c-gemm');diagram.insertBefore(gemm,dest.parentElement);gemm.append(dest.parentElement,weights.parentElement);
 const ins=Array.from({length:18},(_,k)=>{const n=cell(String(k+1),k<9?'':'orange');(k<9?input:input2).append(n);return n;}),ds=Array.from({length:32},()=>{const n=cell('—','empty');dest.append(n);return n;}),ws=Array.from({length:8},(_,k)=>{const n=cell(String(k+1),'green');weights.append(n);return n;}),os=Array.from({length:4},()=>{const n=cell('—','empty');result.append(n);return n;});
 const patch=r=>{const y=Math.floor(r/2),x=r%2;return[0,1].flatMap(c=>[0,1].flatMap(dy=>[0,1].map(dx=>c*9+(y+dy)*3+x+dx)));},values=Array.from({length:4},(_,r)=>patch(r).map(k=>k+1)),outputs=values.map(v=>sum(v.map((x,k)=>x*(k+1))));
 steps=[{label:'Allocate the unfolded buffer',description:'All 18 input values and eight weights are retained. The materialized matrix has 32 reserved scalar slots, none initialized.'}];for(let r=0;r<4;r++)steps.push({label:`Copy patch (${Math.floor(r/2)}, ${r%2}) into row ${r}`,description:`Copy ${values[r].join(', ')} into row ${r}; every source value stays in the input. Overlap creates additional copies, not new input identities.`,evidence:{rows:r+1,copies:(r+1)*8}});for(let r=0;r<4;r++)steps.push({label:`Read row ${r} and weights → output ${r}`,description:`Eight products sum to ${outputs[r]}. Read both operands without consuming the input buffer or weights.`,evidence:{rows:4,copies:32,outputs:r+1}});
 function draw(i){diagram.querySelectorAll('.v1c-flight').forEach(n=>n.remove());const s=steps[i].evidence||{rows:0,copies:0,outputs:0};ins.forEach(n=>n.classList.remove('active'));if(i>0&&i<5)patch(i-1).forEach(k=>ins[k].classList.add('active'));ds.forEach((n,k)=>{const ready=Math.floor(k/8)<s.rows;n.textContent=ready?String(values[Math.floor(k/8)][k%8]):'—';n.className='v1c-cell '+(ready?(k%8<4?'':'orange'):'empty');});os.forEach((n,k)=>{n.textContent=k<(s.outputs||0)?String(outputs[k]):'—';n.className='v1c-cell '+(k<(s.outputs||0)?'green':'empty');});status.textContent=`Input: 18 retained scalars. Unfolded: ${s.copies}/32 initialized slots. Completed outputs: ${s.outputs||0}/4. Center values 5 and 14 each occur in every patch.`;equation.textContent=i>4?values[i-5].map((v,k)=>`${v} × ${k+1}`).join(' + ')+` = ${outputs[i-5]}.`:'Source has 2 × 3 × 3 elements; unfolded storage has 4 × (2 × 2 × 2) slots. These are scalar storage counts, not measured memory traffic.';}
 function prepare(i){ins.forEach(n=>n.classList.remove('active'));if(i>0&&i<5){const r=i-1;patch(r).forEach(k=>ins[k].classList.add('active'));tracks=patch(r).map((k,j)=>({source:ins[k],dest:ds[r*8+j],text:String(k+1)}));}else if(i>=5){const r=i-5;tracks=values[r].map((v,k)=>({source:ds[r*8+k],dest:os[r],text:`${v}×${k+1}`}));}}

 steps=steps.map((s,i)=>({...s,description:(i?'On completion: ':'Initial state: ')+s.description}));
 function render(i,{stage,details}){clearFlightResize();current=i;tracks=[];if(!stage.contains(diagram))stage.append(diagram,status);if(!details.contains(detail))details.append(detail,inspection);draw(i);root.dataset.lessonState=String(i);root.dataset.stateEvidence=JSON.stringify(steps[i].evidence||{});}
 const controller=window.MLSysOriginalPlayer.create({root,outer,title:"Which input values are copied into each unfolded row?",scope:"Exact source values: channels 1\u20139 and 10\u201318; two 2 \u00d7 2 kernels stack into weights 1\u20138. Valid stride-one lowering creates four rows of eight values. This version materializes the buffer; implicit GEMM need not.",steps,render,transition:({to,onFrame,signal})=>{
  clearFlightResize();tracks=[];prepare(to);
  const flight=tracks.map(({source,dest,text,kind})=>{const n=cell(text,kind||'moving');n.classList.add('v1c-flight');diagram.append(n);return{n,source,dest,release:kind==='release'};});
  let progress=0,disposed=false,resizeFrame=0;
  // Layout changes redraw the frozen visual fraction; only the player advances it.
  function paint(){if(disposed)return;const r=diagram.getBoundingClientRect(),routes=flight.map(f=>({...f,a:f.source.getBoundingClientRect(),b:f.dest.getBoundingClientRect()}));routes.forEach(({n,a,b,release})=>{n.style.left=(a.left-r.left)+'px';n.style.top=(a.top-r.top)+'px';n.style.width=Math.min(a.width,86)+'px';n.style.height=Math.min(a.height,38)+'px';n.style.transform=`translate(${(b.left-a.left)*progress}px,${(b.top-a.top)*progress}px)`;if(release)n.style.opacity=String(1-progress);});}
  function schedulePaint(){if(!disposed&&!resizeFrame)resizeFrame=requestAnimationFrame(()=>{resizeFrame=0;paint();});}
  const observer=typeof ResizeObserver==='function'?new ResizeObserver(schedulePaint):null;
  const relocation=new MutationObserver(schedulePaint);relocation.observe(outer,{childList:true,subtree:true});
  function cleanup(){if(disposed)return;disposed=true;observer?.disconnect();relocation.disconnect();cancelAnimationFrame(resizeFrame);resizeFrame=0;window.removeEventListener('resize',schedulePaint);signal.removeEventListener('abort',cleanup);if(clearFlightResize===cleanup)clearFlightResize=()=>{};}
  clearFlightResize=cleanup;signal.addEventListener('abort',cleanup,{once:true});window.addEventListener('resize',schedulePaint);
  new Set([diagram,...flight.flatMap(({source,dest})=>[source,dest])]).forEach(n=>observer?.observe(n));
  status.textContent='Pending: '+steps[to].label+'. Displayed storage and completed results remain at the preceding endpoint until this operation finishes.';
  onFrame(t=>{progress=t;paint();});
 }});const destroy=controller.destroy;controller.destroy=()=>{clearFlightResize();destroy();};return controller;
};})();
