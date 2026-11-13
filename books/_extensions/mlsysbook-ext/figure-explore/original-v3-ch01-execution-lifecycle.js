/* Original-artwork Explore mechanism. Shared player owns the pausable clock. */
(()=>{'use strict';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v3-ch01-execution-lifecycle"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),NS='http://www.w3.org/2000/svg';
 const vb=svg.viewBox.baseVal,W=vb.width,H=vb.height;
 if(!W||!H)throw Error('Original artwork viewBox required');
 root.classList.add("original-v3-ch01-execution-lifecycle");
 svg.setAttribute('role','img');svg.setAttribute('aria-label',"Why can a model proposal never execute or self-certify without runtime gates?");
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
 const steps=[{"label": "Assemble bounded context and invoke model", "description": "In phases 1 and 2, the runtime compiles state and budgets into a prompt and invokes the model, which emits an untrusted action proposal carrying zero execution authority."}, {"label": "Authorize and execute the tool", "description": "In phase 3, the runtime validates schema, policy, and grants before dispatching phase 4 tool execution; phase 5 records the external outcome (marking timed-out calls unknown, never assumed)."}, {"label": "Bypass phase 4 on rejected proposals", "description": "When phase 3 denies an out-of-policy proposal, the red dashed arc routes structured error feedback directly into phase 5 observation history without touching any external tool."}, {"label": "Pause in Wait without exiting", "description": "Proposals requiring human approval in phase 3 or pending background jobs in phase 6 enter the dashed Wait state and resume cleanly once settled rather than terminating."}, {"label": "Assess evidence for next turn or acceptance", "description": "In phase 6, the runtime checks phase 5 external evidence against the contract—looping back to phase 1 when incomplete, or entering green Accept deliverable once independent checks pass."}, {"label": "Stop without acceptance on exhausted budgets", "description": "If step, token, or wall-clock budgets expire—or a timed-out tool leaves state unknown—phase 6 exits to red Stop without acceptance (unresolved or escalate) rather than trusting model self-reports."}];

 const phases=[[30,30,220,64,'#7C3AED'],[320,30,220,64,'#0284C7'],[610,30,220,64,'#7C3AED'],[610,250,220,64,'#D97706'],[320,250,220,64,'#7C3AED'],[30,250,220,64,'#7C3AED'],[170,126,220,48,'#64748b'],[30,370,190,54,'#16A34A'],[270,370,210,54,'#DC2626']];
 const hi=phases.map(([x,y,w,h,c])=>add('rect',{x:x-2.5,y:y-2.5,width:w+5,height:h+5,rx:6,fill:c,'fill-opacity':'.11',stroke:c,'stroke-width':2.2,visibility:'hidden'}));
 const tok=add('circle',{r:7,fill:'#7C3AED',stroke:'#fff','stroke-width':2,'data-motion-object':'true'});
 const tag=add('text',{x:700,y:402,'text-anchor':'middle','font-size':11,'font-weight':'700',fill:'#7C3AED',stroke:'#fff','stroke-width':3.5,'paint-order':'stroke'},'');
 const activeSet=[[0,1],[2,3,4],[2,4],[2,6,5],[4,5,7],[5,8]];
 const segs=[
  [[140,62],[430,62]],
  [[430,62],[720,62],[720,282],[430,282]],
  [[720,62],[575,180],[430,282]],
  [[720,62],[280,150]],
  [[430,282],[140,282],[125,397]],
  [[140,282],[375,397]]
 ];
 const cols=['#0284C7','#D97706','#DC2626','#64748b','#16A34A','#DC2626'];
 const readouts=[
  'Phases 1 → 2: Runtime builds context → Model emits untrusted proposal (0 side effects)',
  'Phases 3 → 4 → 5: Runtime authorizes → Tool executes → Runtime logs outcome (timeout = unknown)',
  'Phase 3 → 5 exception: Rejected proposal skips Phase 4 completely; error recorded as observation',
  'Wait detour: Needs approval / async job pauses turn in Wait (resumes; not a terminal exit)',
  'Phase 6 → Accept: Independent phase 5 evidence satisfies contract → deliverable accepted',
  'Phase 6 → Stop without acceptance: Budget exhausted or outcome unknown → unresolved / escalate'
 ];
 function paint(from,to,p){
  const idx=p<1?from:to,moving=from!==to&&p<1;
  hi.forEach((el,k)=>el.setAttribute('visibility',activeSet[idx].includes(k)?'visible':'hidden'));
  const s=from!==to?segs[to]:segs[idx],q=from!==to?p:1,[x,y]=routePt(s,q);
  tok.setAttribute('visibility',moving?'visible':'hidden');
  tok.setAttribute('cx',x);tok.setAttribute('cy',y);tok.setAttribute('fill',cols[idx]);
  tag.setAttribute('fill',cols[idx]);
  tag.textContent=['Proposal ≠ execution','Authorized → Phase 4 → Phase 5','Rejected: Phase 4 untouched','Paused in Wait (resumable)','Contract met → Accepted','Unresolved → Escalate'][idx];
  readout.textContent=readouts[idx];
  invariant.textContent='Responsibility boundary: only Phase 3 authorizes Phase 4 side effects, and only Phase 5 external evidence can justify Phase 6 acceptance.';
  root.dataset.values=JSON.stringify({step:idx,phase_count:6,tool_touched:idx===1});
 }

 return window.MLSysOriginalPlayer.create({
  root,outer:context.outer,layout:'compact',
  title:"Why can a model proposal never execute or self-certify without runtime gates?",
  intro:"Trace a turn across all 6 lifecycle phases to contrast authorized tool execution, rejected proposals, approval waits, and contract-verified acceptance.",
  scope:"Six-phase agent turn responsibility model: Runtime (phases 1, 3, 5, 6), Model (phase 2 proposal only), and Tool (phase 4 authorized execution), plus exception arcs for rejected calls, approval/background waits, next-turn loops, and unresolved stops.",
  steps,
  render(i,{stage,details}){if(!canvas.isConnected)stage.append(canvas);if(details&&!readout.isConnected)details.append(readout,invariant);root.dataset.sceneStep=String(i);paint(i,i,1);},
  transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>{root.dataset.animationProgress=String(p);paint(from,to,p);});}
 });
};})();
