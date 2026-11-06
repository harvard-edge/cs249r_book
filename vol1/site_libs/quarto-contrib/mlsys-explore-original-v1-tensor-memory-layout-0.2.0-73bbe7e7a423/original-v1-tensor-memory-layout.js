/* Authored mechanism derived from the verified chapter figure; endpoints are authoritative. */
(()=>{'use strict';
const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
const sum=a=>a.reduce((a,b)=>a+b,0),cell=(text,kind='')=>{const n=el('span','v1c-cell '+kind,text);return n;};
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v1-tensor-memory-layout"]=async(root,{outer,verifiedArtwork})=>{
 if(!verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 root.classList.add('mlsys-v1c','mlsys-v1c-tensor-layout');const diagram=el('div','v1c-diagram'),status=el('p','v1c-status'),inspection=el('details','v1c-original');inspection.append(el('summary','','Inspect the unchanged source drawing'));const art=verifiedArtwork.cloneNode(true);art.removeAttribute('width');art.removeAttribute('height');art.setAttribute('aria-label','Verified original chapter artwork');inspection.append(art);
 const region=(title,columns=1)=>{const box=el('section','v1c-region');box.append(el('strong','',title));const grid=el('div','v1c-grid');grid.style.setProperty('--columns',columns);box.append(grid);diagram.append(box);return grid;};
 const detail=el('details');detail.append(el('summary','','Inspect the operation'));const equation=el('p');detail.append(equation);
 let steps=[],tracks=[],current=0,clearResize=()=>{};
 
 const logical=region('Logical tensor: indices (i, j)',3),row=region('Row-major backing storage · strides [3, 1]',6),col=region('Column-major backing storage · strides [1, 2]',6),read=region('Read result · both alternatives',2);
 const ln=[1,2,3,4,5,6].map((v,k)=>{const n=cell(`${v} · (${Math.floor(k/3)},${k%3})`);logical.append(n);return n;}),rn=[1,2,3,4,5,6].map(v=>{const n=cell(String(v),'pink');row.append(n);return n;}),cn=[1,4,2,5,3,6].map(v=>{const n=cell(String(v),'green');col.append(n);return n;}),out=[cell('—','empty'),cell('—','empty')];read.append(...out);
 steps=[{label:'Two layouts, one logical tensor',description:'Storage already contains the source values. Indexing computes an address and reads a value; it does not rearrange the storage.'}];
 for(let k=0;k<6;k++){const i=Math.floor(k/3),j=k%3,r=i*3+j,c=i+j*2;steps.push({label:`Read (${i}, ${j}) → value ${k+1}`,description:`Row-major reads offset ${r}; column-major reads offset ${c}. Both return ${k+1}. Source storage remains unchanged.`,evidence:{i,j,rowOffset:r,columnOffset:c,value:k+1}});}
 function draw(i){diagram.querySelectorAll('.v1c-flight,.v1c-read-port').forEach(n=>n.remove());[...ln,...rn,...cn].forEach(n=>n.classList.remove('active','pending-read'));const s=steps[i].evidence;out.forEach(n=>{n.className='v1c-cell '+(s?'orange':'empty');n.textContent=s?String(s.value):'—';});if(s){ln[s.i*3+s.j].classList.add('active');rn[s.rowOffset].classList.add('active');cn[s.columnOffset].classList.add('active');}status.textContent=s?`Index (${s.i}, ${s.j}): row-major element offset ${s.rowOffset}; column-major element offset ${s.columnOffset}. Read result ${s.value}; backing arrays unchanged.`:'Choose Next to perform the first read. The two backing arrays are alternative layouts.';equation.textContent=s?`${s.i} × 3 + ${s.j} × 1 = ${s.rowOffset}; ${s.i} × 1 + ${s.j} × 2 = ${s.columnOffset}. Byte offset = element offset × element size, once a dtype is specified.`:'Every cell remains allocated throughout. Row-wise traversal visits column-major offsets 0, 2, 4, 1, 3, 5.';}
 function prepare(i){const s=steps[i].evidence;if(s)tracks=[{source:rn[s.rowOffset],dest:out[0],text:String(s.value)},{source:cn[s.columnOffset],dest:out[1],text:String(s.value)}];}

 steps=steps.map((s,i)=>({...s,description:(i?'On completion: ':'Initial state: ')+s.description}));
 function render(i,{stage,details}){clearResize();current=i;tracks=[];if(!stage.contains(diagram))stage.append(diagram,status);if(!details.contains(detail))details.append(detail,inspection);draw(i);root.dataset.lessonState=String(i);root.dataset.stateEvidence=JSON.stringify(steps[i].evidence||{});}
 const controller=window.MLSysOriginalPlayer.create({root,outer,title:"How does the same index reach different storage offsets?",scope:"The source 2 \u00d7 3 tensor and both alternative serializations. Each cell is one scalar element; offsets count elements. No byte width or cache model is assumed. The logical tensor is a view of either alternative storage, not a third allocation.",steps,render,transition:({to,onFrame,signal})=>{
  tracks=[];prepare(to);const size=28,gap=6;
  const flight=tracks.map(({source,dest,text},index)=>{
   source.classList.add('pending-read');
   const ports=[0,1].map(()=>{const port=el('span','v1c-read-port');port.setAttribute('aria-hidden','true');diagram.append(port);return port;});
   const n=cell(text,'moving');n.classList.add('v1c-flight');n.style.left='0';n.style.top='0';n.style.width=size+'px';n.style.height=size+'px';n.dataset.layout=index===0?'row-major':'column-major';diagram.append(n);
   return{n,source,dest,index,ports};
  });
  status.textContent='Pending: '+steps[to].label+'. Displayed storage and completed results remain at the preceding endpoint until this operation finishes.';
  let progress=0;
  const paint=()=>{const bounds=diagram.getBoundingClientRect();flight.forEach(({n,source,dest,index,ports})=>{
   const a=source.getBoundingClientRect(),b=dest.getBoundingClientRect(),panel=source.closest('.v1c-region').getBoundingClientRect(),result=dest.closest('.v1c-region').getBoundingClientRect();
   const gutter=index===0?panel.left-bounds.left-gap-size/2:panel.right-bounds.left+gap+size/2;
   const endY=result.bottom-bounds.top+gap+size/2,points=[[gutter,a.top-bounds.top+a.height/2],[gutter,endY],[b.left-bounds.left+b.width/2,endY]];
   [points[0],points[2]].forEach((point,k)=>{ports[k].style.left=(point[0]-3)+'px';ports[k].style.top=(point[1]-3)+'px';});
   const lengths=points.slice(1).map((b,k)=>Math.hypot(b[0]-points[k][0],b[1]-points[k][1])),total=lengths.reduce((a,b)=>a+b,0);let distance=progress*total,k=0;while(k<lengths.length-1&&distance>lengths[k])distance-=lengths[k++];const u=lengths[k]?distance/lengths[k]:0,a0=points[k],b0=points[k+1];n.style.transform=`translate(${a0[0]+(b0[0]-a0[0])*u-size/2}px,${a0[1]+(b0[1]-a0[1])*u-size/2}px)`;n.dataset.progress=String(progress);
  });};
  let resizeFrame=0;const resize=()=>{if(!resizeFrame)resizeFrame=requestAnimationFrame(()=>{resizeFrame=0;paint();});};
  const observer=new ResizeObserver(resize);observer.observe(diagram);
  const relocation=new MutationObserver(resize);relocation.observe(outer,{childList:true,subtree:true});
  const cleanup=()=>{observer.disconnect();relocation.disconnect();cancelAnimationFrame(resizeFrame);window.removeEventListener('resize',resize);signal.removeEventListener('abort',cleanup);if(clearResize===cleanup)clearResize=()=>{};};clearResize=cleanup;signal.addEventListener('abort',cleanup,{once:true});window.addEventListener('resize',resize,{passive:true});
  onFrame(t=>{progress=t;paint();});
 }});return controller;
};})();