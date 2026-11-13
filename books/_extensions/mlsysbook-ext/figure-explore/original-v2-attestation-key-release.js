/* Original-artwork Explore mechanism. Shared player owns the pausable clock. */
(()=>{'use strict';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v2-attestation-key-release"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),NS='http://www.w3.org/2000/svg';
 const vb=svg.viewBox.baseVal,W=vb.width,H=vb.height;
 if(!W||!H)throw Error('Original artwork viewBox required');
 root.classList.add("original-v2-attestation-key-release");
 svg.setAttribute('role','img');svg.setAttribute('aria-label',"What gates checkpoint-key delivery before a restarted node rejoins training?");
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
 const steps=[{"label": "Challenge the restarted node", "description": "When a failed worker restarts outside the active collective, the Verifier issues a fresh cryptographic nonce at line 1 so replayed boot quotes from earlier runs cannot pass."}, {"label": "Return hardware-signed measurements", "description": "The restarted node's hardware root of trust hashes loaded firmware and runtime images and returns signed evidence bound to the fresh challenge nonce."}, {"label": "Appraise against admission policy", "description": "The Verifier checks hardware signatures, nonce freshness, and approved image digests—only a valid appraisal sends Policy approval to the Key service (stale or mismatched hashes halt here)."}, {"label": "Deliver the 2.45 TB checkpoint key", "description": "After Verifier authorization, the Key service releases the wrapped decryption key for the 2.45 TB (175B × 14 B/param) training checkpoint over a protected channel."}, {"label": "Restore state before collective admission", "description": "The node decrypts and loads its shard of the 2.45 TB checkpoint locally inside the green Restore stage while the Collective still marks the node Not admitted."}, {"label": "Rejoin the collective", "description": "Once local checkpoint restoration completes under valid custody, the node crosses the Rejoin barrier to resume synchronized distributed training."}];

 const cols=[15,185,355,525].map(x=>add('rect',{x:x-2,y:18,width:134,height:39,rx:5,fill:'#7C3AED','fill-opacity':'.10',stroke:'#7C3AED','stroke-width':2,visibility:'hidden'}));
 const restoreHi=add('rect',{x:18,y:323,width:124,height:44,rx:5,fill:'#16A34A','fill-opacity':'.14',stroke:'#16A34A','stroke-width':2.2,visibility:'hidden'});
 const gateBadge=add('text',{x:340,y:434,'text-anchor':'middle','font-size':11,'font-weight':'700',fill:'#7C3AED',stroke:'#fff','stroke-width':3.5,'paint-order':'stroke'},'');
 const msg=add('circle',{r:6.5,fill:'#7C3AED',stroke:'#fff','stroke-width':2,'data-motion-object':'true'});
 const segs=[
  [[250,92],[80,92]],
  [[80,152],[250,152]],
  [[250,212],[420,212]],
  [[420,272],[80,272]],
  [[80,272],[80,345]],
  [[80,392],[590,392]]
 ];
 const activeCols=[[1,0],[0,1],[1,2],[2,0],[0],[0,3]];
 const readouts=[
  'Stage 1 (Challenge): Verifier → Restarted node · fresh nonce prevents replay · Collective: Not admitted',
  'Stage 2 (Evidence): Restarted node → Verifier · hardware-signed boot + workload measurements',
  'Stage 3 (Appraisal): Policy check PASS → Verifier authorizes Key service (stale/tampered evidence aborts here)',
  'Stage 4 (Key release): Key service → Restarted node · wrapped key for 2.45 TB (175B × 14 B/param) checkpoint',
  'Stage 5 (Restore): Node decrypts & loads checkpoint shard locally · Collective still withholds admission',
  'Stage 6 (Rejoin): Restore verified → Restarted node admitted back into Collective synchronization'
 ];
 function paint(from,to,p){
  const idx=p<1?from:to,moving=from!==to&&p<1;
  cols.forEach((c,k)=>c.setAttribute('visibility',activeCols[idx].includes(k)?'visible':'hidden'));
  restoreHi.setAttribute('visibility',idx===4?'visible':'hidden');
  const s=from!==to?segs[to]:segs[idx],q=from!==to?p:1,[x,y]=routePt(s,q);
  msg.setAttribute('visibility',moving?'visible':'hidden');
  msg.setAttribute('cx',x);msg.setAttribute('cy',y);
  msg.setAttribute('fill',idx>=4?'#16A34A':idx===3?'#1F407A':'#7C3AED');
  gateBadge.setAttribute('fill',idx>=4?'#16A34A':'#7C3AED');
  gateBadge.textContent=['Nonce issued → awaiting quote','Signed quote → Verifier','Appraisal PASS → Key service authorized','Protected key delivered (2.45 TB ckpt)','Restoring state (collective gated)','Admitted to collective'][idx];
  readout.textContent=readouts[idx];
  invariant.textContent='Provenance gate: fresh appraisal must precede key release, and local restore must complete before collective rejoin.';
  root.dataset.values=JSON.stringify({step:idx,ckpt_tb:2.45,key_released:idx>=3,admitted:idx===5});
 }

 return window.MLSysOriginalPlayer.create({
  root,outer:context.outer,layout:'compact',
  title:"What gates checkpoint-key delivery before a restarted node rejoins training?",
  intro:"Follow a restarted node from challenge nonce and hardware measurement appraisal through protected release of the 2.45 TB checkpoint key and collective rejoin.",
  scope:"Four-actor recovery attestation ordering (Restarted node, Verifier, Key service, Collective) referencing the chapter's 175B-parameter, 14 B/param (2.45 TB) encrypted checkpoint. Vertical position shows causal order rather than measured duration.",
  steps,
  render(i,{stage,details}){if(!canvas.isConnected)stage.append(canvas);if(details&&!readout.isConnected)details.append(readout,invariant);root.dataset.sceneStep=String(i);paint(i,i,1);},
  transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>{root.dataset.animationProgress=String(p);paint(from,to,p);});}
 });
};})();
