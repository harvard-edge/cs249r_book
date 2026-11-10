(() => {
'use strict';
const NS='http://www.w3.org/2000/svg';
const el=(tag,a={},text)=>{const n=document.createElementNS(NS,tag);Object.entries(a).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;return n;};
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v2-gpudirect-data-path"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified source artwork and shared player required');
 if(context.verifiedArtwork.getAttribute('viewBox')!=="-12 -12 704 380")throw Error('Source geometry changed; review required');
 root.classList.add('v2c-lesson');
 const art=el('svg',{viewBox:"0 0 730 370",role:'img','aria-label':"Which copies disappear with GPUDirect RDMA?"});
 const base=el('g'),moving=el('g',{'aria-hidden':'true'});art.append(base,moving);
 const text=(x,y,t,a={})=>el('text',{x,y,'font-size':14,fill:'#14294e',...a},t);
 const box=(x,y,w,h,label,fill='#edf2fa',stroke='#45699b')=>{const g=el('g');g.append(el('rect',{x,y,width:w,height:h,rx:2,fill,stroke}),text(x+w/2,y+h/2+5,label,{'text-anchor':'middle','font-weight':600}));return g;};
 const line=(pts,color='#8b9bb0',dash='')=>{const g=el('g'),a=pts.at(-2),b=pts.at(-1),theta=Math.atan2(b[1]-a[1],b[0]-a[0]),tip=[b,[b[0]-8*Math.cos(theta)+4*Math.sin(theta),b[1]-8*Math.sin(theta)-4*Math.cos(theta)],[b[0]-8*Math.cos(theta)-4*Math.sin(theta),b[1]-8*Math.sin(theta)+4*Math.cos(theta)]];g.append(el('polyline',{points:pts.map(p=>p.join(',')).join(' '),fill:'none',stroke:color,'stroke-width':2,'stroke-dasharray':dash}),el('polygon',{points:tip.map(p=>p.join(',')).join(' '),fill:color}));return g;};
 const token=(x,y,label,color='#edf2fa',attrs={})=>{const g=el('g',{'data-object':label,...attrs}),w=Math.max(28,label.length*7+10);g.append(el('rect',{x:x-w/2,y:y-12,width:w,height:24,rx:2,fill:color,stroke:'#45699b'}),text(x,y+5,label,{'text-anchor':'middle','font-size':12,'font-weight':700}));return g;};
 const status=document.createElement('p');status.className='v2c-status';status.setAttribute('aria-live','polite');
 const note=document.createElement('p');note.className='v2c-reading';note.textContent="The gray and green paths are alternatives. Sources remain readable after copying. Swipe the diagram horizontally on a phone; Expand shows the full board.";
 note.textContent+=' On narrow screens, the view follows the active transfer; pause to swipe across the full diagram.';
 let mounted=false,stageNode=null;
 function camera(x){if(!stageNode)return;const scale=art.getBoundingClientRect().width/art.viewBox.baseVal.width;stageNode.scrollLeft=Math.max(0,x*scale-stageNode.clientWidth/2);}
 const coordinates={gpu:[235,252],host:[135,92],nic:[465,252],fabric:[660,252]};
 const controller=window.MLSysOriginalPlayer.create({root,outer:context.outer,layout:'compact',title:"Which copies disappear with GPUDirect RDMA?",intro:"Compare the host-staged send path (2 host-memory transfers) against the GPUDirect RDMA peer-to-peer PCIe path (0 host-memory transfers) for one n-byte GPU payload.",scope:"One sending node containing GPU memory (holding an n-byte payload G), a PCIe switch, a CPU root complex with a registered host-memory buffer, and an RDMA NIC. Steps 1–4 trace the dashed gray host-staged path (2 × n bytes of host-memory traffic); Steps 5–7 trace the solid green GPUDirect RDMA peer-to-peer path (0 × n bytes of host-memory traffic).",steps:[{"label": "Staged \u00b7 GPU source", "description": "Start the staged send with the n-byte payload G resident in GPU memory. The registered host-memory buffer is empty (0 host writes, 0 host reads, 0 × n bytes of host bus traffic), and nothing has reached the RDMA NIC or network fabric."}, {"label": "Staged \u00b7 host write", "description": "Without GPUDirect RDMA, DMA copies the n-byte payload G along the left dashed gray path through the PCIe switch and CPU root complex into the registered host-memory buffer (1 × n write, 0 × n reads, 1 × n bytes of host bus traffic so far)."}, {"label": "Staged \u00b7 NIC read", "description": "The RDMA NIC DMA engine reads the n-byte staged copy back out of host memory through the CPU root complex and PCIe switch (1 × n write + 1 × n read = 2 × n bytes of total host-memory bus traffic for a single n-byte payload)."}, {"label": "Staged \u00b7 network send", "description": "The RDMA NIC transmits the n-byte payload G onto the fabric link. Host staging consumed 2 host-memory transfers (2 × n bytes of host bus traffic) before a single payload byte left the node."}, {"label": "Direct \u00b7 independent send", "description": "Reset to an independent send of the same n-byte payload G using GPUDirect RDMA. Payload G starts in GPU memory and the registered host-memory buffer stays empty (0 host writes, 0 host reads)."}, {"label": "Direct \u00b7 NIC reads GPU", "description": "The RDMA NIC DMA engine reads the n-byte payload G directly from GPU memory across the solid green PCIe switch peer-to-peer path. The CPU root complex and host memory incur 0 writes and 0 reads (0 × n bytes of host bus traffic)."}, {"label": "Direct \u00b7 network send", "description": "The RDMA NIC transmits the same n-byte payload G to the fabric. GPUDirect RDMA eliminates all 2 × n bytes of host-memory bounce traffic while preserving the required PCIe peer-to-peer read and fabric transmission."}].map((step,i)=>({label:step.label,get description(){return Number(root.dataset.modelState)===i?step.description:"On arrival: "+step.description;}})),
 render(index,{stage,details}){
  if(!mounted){stageNode=stage;stage.classList.add('v2c-stage');stage.tabIndex=0;stage.setAttribute('aria-label','Mechanism diagram; swipe horizontally on a narrow screen or use Expand');stage.append(art);details.append(status,note);mounted=true;}
  base.replaceChildren();moving.replaceChildren();root.dataset.modelState=String(index);
  const direct=index>=4,phase=direct?index-4:index;root.dataset.path=direct?'direct':'staged';root.dataset.hostWrites=String(!direct&&phase>=1?1:0);root.dataset.hostReads=String(!direct&&phase>=2?1:0);root.dataset.networkSent=String(phase>=3||direct&&phase>=2);
base.append(box(20,10,580,292,'','#f8fafc','#bac6d6'),text(35,32,'Sending node',{'font-weight':700}),box(45,52,180,70,'','#fff'),box(320,52,150,70,'','#fff'),text(135,75,'Host memory',{'text-anchor':'middle'}),text(395,75,'CPU root complex',{'text-anchor':'middle'}),box(105,156,455,38,''),text(120,170,'PCIe switch',{'text-anchor':'start','font-size':12.5}),box(155,222,150,65,''),box(390,222,150,65,'','#fff3cd','#d97706'),text(230,241,'GPU memory',{'text-anchor':'middle'}),text(465,241,'RDMA NIC',{'text-anchor':'middle'}));
base.append(line([[290,222],[290,190],[350,190],[350,108],[225,108]],'#94a3b8','5 4'),line([[225,108],[395,108],[395,190],[522,190],[522,222]],'#94a3b8','5 4'),line([[290,222],[290,190],[522,190],[522,222]],'#16a34a'),line([[540,252],[700,252]],'#d97706'));
base.append(text(650,226,'to fabric',{'text-anchor':'middle'}),token(235,270,'G'));
if(!direct&&phase>=1)base.append(token(135,108,'G','#f1f5f9',{'data-location':'host'}));
if((!direct&&phase>=2)||(direct&&phase>=1))base.append(token(465,270,'G','#fff3cd',{'data-location':'nic'}));
if((!direct&&phase>=3)||(direct&&phase>=2))base.append(token(660,252,'G','#dcfce7',{'data-location':'fabric'}));
base.append(text(35,330,direct?'Solid green: GPU → PCIe peer path → NIC':'Dashed gray: GPU → host write → host read → NIC'),text(35,354,'G denotes the same payload; a read or send does not erase its source.'));
status.textContent=`${direct?'Direct':'Staged'} send · Host payload writes: ${root.dataset.hostWrites} × n; reads: ${root.dataset.hostReads} × n; total host bus traffic: ${Number(root.dataset.hostWrites)+Number(root.dataset.hostReads)} × n. Network: ${root.dataset.networkSent==='true'?'G sent':'not yet sent'}.`;
 camera([235,135,465,660,235,465,660][index]);
 },
 transition({from,to,onFrame}){
  if(to!==from+1)return;
  const motions=to===1?[{label:'G',path:[[290,270],[290,214],[235,214],[235,190],[350,190],[350,108],[135,108]]}]:to===2?[{label:'G',path:[[135,108],[395,108],[395,190],[522,190],[522,270],[465,270]]}]:to===3||to===6?[{label:'G',path:[[465,270],[560,270],[560,252],[660,252]]}]:to===5?[{label:'G',color:'#dcfce7',path:[[290,270],[290,214],[235,214],[235,190],[522,190],[522,270],[465,270]]}]:[];
  status.textContent="Transition in progress; destination state is pending. "+status.textContent;
  if(motions.length===16)base.querySelectorAll("[data-live-token]").forEach(n=>n.setAttribute("visibility","hidden"));
  const packets=motions.map(m=>{const g=token(0,0,m.label,m.color||'#fff3cd',{'data-motion-object':'true'});moving.append(g);const lengths=m.path.slice(1).map((p,i)=>Math.hypot(p[0]-m.path[i][0],p[1]-m.path[i][1]));return {...m,g,lengths,total:lengths.reduce((a,b)=>a+b,0)};});
  onFrame(p=>{root.dataset.animationProgress=String(p);packets.forEach(m=>{let d=p*m.total,i=0;while(i<m.lengths.length-1&&d>m.lengths[i])d-=m.lengths[i++];const a=m.path[i],b=m.path[i+1],f=m.lengths[i]?d/m.lengths[i]:1;const px=a[0]+(b[0]-a[0])*f;m.g.setAttribute('transform',`translate(${px},${a[1]+(b[1]-a[1])*f})`);if(m===packets[0])camera(px);});});
 }});
 return controller;
};
})();
