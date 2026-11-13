/* Original-artwork Explore mechanism. Shared player owns the pausable clock. */
(()=>{'use strict';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v2-node-pcie-numa"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),NS='http://www.w3.org/2000/svg';
 const vb=svg.viewBox.baseVal,W=vb.width,H=vb.height;
 if(!W||!H)throw Error('Original artwork viewBox required');
 root.classList.add("original-v2-node-pcie-numa");
 svg.setAttribute('role','img');svg.setAttribute('aria-label',"Which transfers bypass PCIe roots and which cross the inter-socket NUMA link?");
 svg.querySelectorAll('text').forEach(n=>{const f=n.getAttribute('fill');if(f)n.style.setProperty('fill',f,'important');});
 const overlay=document.createElementNS(NS,'g');overlay.setAttribute('data-explore-overlay','');svg.append(overlay);
 const mk=(tag,attrs={},text)=>{const el=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))el.setAttribute(k,String(v));if(text!=null)el.textContent=text;if(tag==='text'&&attrs.fill)el.style.setProperty('fill',String(attrs.fill),'important');return el;};
 const add=(tag,attrs={},text)=>{const el=mk(tag,attrs,text);overlay.append(el);return el;};
 add('path',{d:`M${(vb.x+W*.08).toFixed(1)},${(vb.y+H*.08).toFixed(1)}h${(W*.84).toFixed(1)}v${(H*.84).toFixed(1)}h-${(W*.84).toFixed(1)}z`,fill:'none',stroke:'none','data-mlsys-focus':''});
 const lerp=(a,b,t)=>a+(b-a)*t;
 const routePt=(pts,p)=>{if(pts.length===1)return pts[0];const lens=pts.slice(1).map((b,i)=>Math.hypot(b[0]-pts[i][0],b[1]-pts[i][1]));const tot=lens.reduce((s,x)=>s+x,0)||1;let rem=Math.max(0,Math.min(1,p))*tot,j=0;while(j<lens.length-1&&rem>lens[j]){rem-=lens[j];j++;}const q=lens[j]?rem/lens[j]:1;return[lerp(pts[j][0],pts[j+1][0],q),lerp(pts[j][1],pts[j+1][1],q)];};
 const canvas=document.createElement('div');canvas.className='mlsys-exp-canvas';canvas.tabIndex=0;canvas.setAttribute('aria-label','Original chapter diagram with synchronized state overlay.');canvas.append(svg);
 const readout=document.createElement('p');readout.className='mlsys-exp-readout';
 const invariant=document.createElement('p');invariant.className='mlsys-exp-invariant';
 const steps=[{"label": "Inspect the 2 NUMA domains", "description": "The 8-GPU server splits host I/O across 2 NUMA domains: CPU 0 roots GPUs G0–G3 and NICs N0–N3, while CPU 1 roots GPUs G4–G7 and NICs N4–N7."}, {"label": "Route GPU collectives over NVSwitch", "description": "Tensor-parallel traffic between G0 (domain 0) and G6 (domain 1) traverses the bottom NVSwitch NVLink fabric directly in 1 fabric hop without touching PCIe switches or CPUs."}, {"label": "Load pinned DRAM inside NUMA 0", "description": "A dataloader pinned to CPU 0 streams batches from Host DRAM 0 through CPU 0's root complex and local PCIe switch into G0, staying inside NUMA domain 0."}, {"label": "Stream GPUDirect RDMA under one switch", "description": "Inter-node AllReduce traffic entering HCA N0 routes peer-to-peer through the shared PCIe switch directly into G0, avoiding host DRAM copies and CPU root traversal."}, {"label": "Pay the cross-socket NUMA penalty", "description": "When a mis-placed worker stages G0's batch in Host DRAM 1, every read crosses CPU 1, the inter-socket link, CPU 0's root complex, and the PCIe switch—contending with host traffic."}, {"label": "Align placement to NUMA affinity", "description": "Binding workers, pinned DRAM buffers, HCAs (N0–N3 / N4–N7), and GPUs (G0–G3 / G4–G7) inside each NUMA domain keeps ingestion local while NVSwitch carries GPU-to-GPU collectives."}];

 const dom0=add('rect',{x:20,y:44,width:352,height:192,rx:5,fill:'#1F407A','fill-opacity':'.06',stroke:'#1F407A','stroke-width':2,visibility:'hidden'});
 const dom1=add('rect',{x:388,y:44,width:352,height:192,rx:5,fill:'#7C3AED','fill-opacity':'.06',stroke:'#7C3AED','stroke-width':2,visibility:'hidden'});
 const nvswHi=add('rect',{x:34,y:244,width:692,height:28,rx:4,fill:'#16A34A','fill-opacity':'.08',stroke:'#16A34A','stroke-width':2,visibility:'hidden'});
 const pathLine=add('path',{fill:'none',stroke:'#16A34A','stroke-width':3.5,'stroke-linecap':'round','stroke-linejoin':'round','stroke-dasharray':'7 4',visibility:'hidden'});
 const pkt=add('circle',{r:6.5,fill:'#16A34A',stroke:'#fff','stroke-width':2,'data-motion-object':'true',visibility:'hidden'});
 const callout=add('text',{x:385,y:351,'text-anchor':'middle','font-size':10.5,'font-weight':'700',fill:'#1F407A',stroke:'#fff','stroke-width':3.5,'paint-order':'stroke'},'');
 const routes={
  1:{pts:[[52.5,222],[52.5,256],[584.5,256],[584.5,222]],color:'#16A34A',label:'NVSwitch path: G0 ↔ NVLink fabric ↔ G6 (0 PCIe / 0 socket hops)'},
  2:{pts:[[84,82],[253,82],[114,154],[52.5,209]],color:'#1F407A',label:'Local NUMA 0 ingest: DRAM 0 → CPU 0 → PCIe switch → G0'},
  3:{pts:[[134.5,209],[114,154],[52.5,209]],color:'#D97706',label:'GPUDirect RDMA: HCA N0 → shared PCIe switch → G0 (bypasses CPU/DRAM)'},
  4:{pts:[[676,82],[507,82],[253,82],[114,154],[52.5,209]],color:'#DC2626',label:'Cross-socket anti-pattern: DRAM 1 → CPU 1 → Socket link → CPU 0 → PCIe → G0'}
 };
 const readouts=[
  'Topology: 2 NUMA domains × (1 CPU + Host DRAM + 2 PCIe switches + 4 GPUs + 4 HCAs) = 8 GPUs total',
  'GPU-to-GPU collective (G0 ↔ G6): uses NVSwitch full-bisection NVLink fabric; 0 PCIe root traversals',
  'NUMA-local host read (DRAM 0 → G0): stays inside NUMA domain 0; 0 inter-socket link crossings',
  'Peer-to-peer RDMA (N0 → G0): turns around inside local PCIe switch; 0 host DRAM copies',
  'Mis-pinned host read (DRAM 1 → G0): crosses inter-socket link between CPU 1 and CPU 0, throttling ingestion',
  'Optimal affinity: pin dataloader + DRAM + HCA + GPU per socket; reserve NVSwitch for 8-GPU collectives'
 ];
 function paint(from,to,p){
  const idx=p<1?from:to,moving=from!==to&&p<1;
  dom0.setAttribute('visibility',(idx===0||idx===2||idx===5)?'visible':'hidden');
  dom1.setAttribute('visibility',(idx===0||idx===5)?'visible':'hidden');
  nvswHi.setAttribute('visibility',(idx===1||idx===5)?'visible':'hidden');
  const r=routes[to]&&p>0?routes[to]:routes[idx];
  if(r&&(idx>=1&&idx<=4)){
   pathLine.setAttribute('visibility','visible');
   pathLine.setAttribute('d',r.pts.map((pt,k)=>(k?'L':'M')+pt.join(',')).join(' '));
   pathLine.setAttribute('stroke',r.color);
   pkt.setAttribute('visibility',moving?'visible':'hidden');pkt.setAttribute('fill',r.color);
   const q=from!==to&&routes[to]?p:1,[x,y]=routePt(r.pts,q);
   pkt.setAttribute('cx',x);pkt.setAttribute('cy',y);
   callout.setAttribute('fill',r.color);callout.textContent=r.label;
  }else{
   pathLine.setAttribute('visibility','hidden');pkt.setAttribute('visibility','hidden');
   callout.setAttribute('fill','#1F407A');
   callout.textContent=idx===0?'2 NUMA domains (4 GPUs + 4 HCAs each) above shared 8-GPU NVSwitch':'Affinity locked: local PCIe/DRAM per socket + NVSwitch across all 8 GPUs';
  }
  readout.textContent=readouts[idx];
  invariant.textContent='Placement invariant: NVSwitch unifies GPU-to-GPU bandwidth, but host DRAM and NIC paths remain partitioned by CPU socket and PCIe switch.';
  root.dataset.values=JSON.stringify({step:idx,numa_domains:2,gpus:8,cross_socket:idx===4});
 }

 return window.MLSysOriginalPlayer.create({
  root,outer:context.outer,layout:'compact',
  title:"Which transfers bypass PCIe roots and which cross the inter-socket NUMA link?",
  intro:"Compare NVSwitch GPU traffic against local PCIe paths and cross-socket NUMA transfers across the 2 CPU domains of an 8-GPU node.",
  scope:"Dual-socket 8-GPU node topology: 2 NUMA domains (CPU 0 owns G0\u2013G3 and N0\u2013N3; CPU 1 owns G4\u2013G7 and N4\u2013N7 across 2 schematic PCIe switches per socket) plus an all-to-all NVSwitch fabric across all 8 GPUs.",
  steps,
  render(i,{stage,details}){if(!canvas.isConnected)stage.append(canvas);if(details&&!readout.isConnected)details.append(readout,invariant);root.dataset.sceneStep=String(i);paint(i,i,1);},
  transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>{root.dataset.animationProgress=String(p);paint(from,to,p);});}
 });
};})();
