/* Original-artwork mechanism. Shared player owns the pausable clock. */
(() => {'use strict';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v4-14-authority-handshake-fsm"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),ns='http://www.w3.org/2000/svg';
 const originalView=svg.getAttribute('viewBox');if(originalView!=="0 0 680 392")throw Error('Original geometry changed; retain Diagram for review');
 root.classList.add("mlsys-original-v4-14-authority-handshake-fsm");
 const canvas=document.createElement('div');canvas.className='v4b-canvas';canvas.tabIndex=0;canvas.setAttribute('aria-label','Original artwork. Scroll to inspect; Expand opens a larger view.');canvas.append(svg);
 svg.setAttribute('role','img');svg.setAttribute('aria-label',"When does the proposal token change owner?");
 const overlay=document.createElementNS(ns,'g');svg.append(overlay);
 const add=(tag,attrs={},text)=>{const el=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))el.setAttribute(k,String(v));if(text!=null)el.textContent=text;overlay.append(el);return el;};
 const readout=document.createElement('p');readout.className='v4b-readout';readout.setAttribute('aria-live','off');
 const equation=document.createElement('p');equation.className='v4b-equation';
 const note=document.createElement('p');note.className='v4b-note';note.textContent='Scroll the original artwork to inspect labels, or use Expand. Playback duration is illustrative.';
 const width=+originalView.split(' ')[2],height=+originalView.split(' ')[3];
 function follow(x,y){const scale=svg.getBoundingClientRect().width/width;canvas.scrollLeft=Math.max(0,x*scale-canvas.clientWidth/2);canvas.scrollTop=Math.max(0,y*scale-canvas.clientHeight/2);}
 function mount(stage,details){if(!canvas.isConnected)stage.append(canvas,note);if(!readout.isConnected)details.append(readout,equation);}
 function dot(x,y,color='#007A87',r=5){return add('circle',{cx:x,cy:y,r,fill:color,stroke:'#fff','stroke-width':r/3,'data-motion-object':'true'});}
 function route(token,points,p){const lens=points.slice(1).map((b,i)=>Math.hypot(b[0]-points[i][0],b[1]-points[i][1]));let rem=p*lens.reduce((a,b)=>a+b,0),j=0;while(j<lens.length-1&&rem>lens[j]){rem-=lens[j];j++;}const q=lens[j]?rem/lens[j]:1,a=points[j],b=points[j+1];token.setAttribute('cx',a[0]+(b[0]-a[0])*q);token.setAttribute('cy',a[1]+(b[1]-a[1])*q);}
 
 const rects=[...svg.querySelectorAll('rect')];const box=(x,y)=>rects.find(e=>+e.getAttribute('x')===x&&+e.getAttribute('y')===y);
 const stateBoxes={AUTO:box(40,70),PEND:box(265,70),BLEND:box(490,70),MAN:box(490,190),FALLBACK:box(265,190),PROTECTIVE:box(265,318)};
 if(Object.values(stateBoxes).some(x=>!x))throw Error('Authority-state objects changed');
 const token=dot(115,99,'#1A4D3E',5);
 const coords={AUTO:[52,82],PEND:[277,82],BLEND:[502,82],MAN:[502,202],FALLBACK:[277,202],PROTECTIVE:[277,330]};
 const states=['AUTO','PEND','PEND','BLEND','BLEND','MAN','FALLBACK','FALLBACK','AUTO','PROTECTIVE'];
 const owners=['policy proposes','policy proposes','policy proposes','enforcer profile','enforcer profile','operator proposes','validated fallback','validated fallback','policy proposes','protective circuit'];
 const steps=[
 ['Autonomous proposal token','The policy proposes on this channel. The permission path still checks the candidate before actuation.'],
 ['Request handover; start the timer','At this endpoint the channel enters PEND. The policy retains the proposal token while the timer runs. A request alone does not transfer control.'],
 ['Acknowledge; await local commit','At this endpoint ACK has prepared the handover. Fresh state, a feasible target, and stopping reserve are checked. Before the commit tick the policy token remains active; the console can be uncertain.'],
 ['Commit on one control tick','When the local commit occurs at this endpoint, it revokes the policy proposal token and gives the enforcer sole ownership of the checked blend between latched endpoints. ACK and commit are distinct events.'],
 ['Evolve the enforcer-owned profile','An illustrative normalized profile progresses between fixed checked endpoints. Live operator proposals do not write this transition profile. Its physical feasibility is a separate prerequisite, not proved by the state machine.'],
 ['Complete with a live manual lease','At this endpoint, after the profile completes with the manual lease valid, MAN can admit operator proposals for screening. The policy token remains revoked.'],
 ['Manual lease lapses; fallback owns the channel','Follow lease lapse along the original MAN → FALLBACK edge. At the endpoint the validated holding or stopping response owns the channel without network consensus.'],
 ['Remain in fallback while moving','A reset request cannot bypass standstill, relevant fault handling, and a fresh authenticated re-admission handshake. No edge returns directly to manual operation.'],
 ['Reset from rest and re-admit','After standstill, cleared faults, and fresh authenticated re-admission, follow the dashed reset edge toward AUTO. At the endpoint the policy token is re-admitted. This endpoint asserts those prerequisites; it does not animate a moving reset.'],
 ['Hardwired input preempts every protocol state','The protective input is outside the handover protocol. This alternative preemption can occur from any state, including fallback; at the endpoint the independent circuit owns brake or hold.']
 ].map(([label,description],i)=>({label,get description(){return root.dataset.lessonCommittedIndex===String(i)?description:'Next endpoint, not yet completed: '+description;}}));
 const paths={1:[[190,90],[263,90]],3:[[415,90],[488,90]],5:[[610,128],[610,188]],6:[[490,222],[417,222]],8:[[265,238],[80,238],[80,130]],9:[[340,282],[340,316]]};
 const bar=add('rect',{x:500,y:122,width:0,height:3,fill:'#1A4D3E','data-motion-object':'true'});
 function frame(i){const state=states[i];for(const[k,b]of Object.entries(stateBoxes)){b.classList.toggle('v4b-active',k===state);b.style.opacity=k===state?'1':'.65';}
  token.setAttribute('cx',coords[state][0]);token.setAttribute('cy',coords[state][1]);bar.setAttribute('width',i>=5&&i<9?130:i===4?65:0);
  root.dataset.authority=state;root.dataset.proposalOwner=owners[i];root.dataset.acknowledged=String(i>=2&&i<=7);root.dataset.committed=String(i>=3&&i<=7);root.dataset.policyToken=String(i<=2||i===8);root.dataset.blend=i<3?'pending':i===3?'0':i===4?'0.5':'1';
  readout.textContent=`Channel state ${state} · ${owners[i]} · policy proposal token ${i<=2||i===8?'active':'revoked'}`;
  equation.textContent=i===2?'ACK received ≠ commit. Before the commit tick: policy token active.':i===3||i===4?'Commit → enforcer owns fixed-endpoint profile; normalized progress '+(i===3?'0.00':'0.50')+'.':i===7?'Reset guard false while moving: remain in FALLBACK.':i===8?'Reset guard: standstill ∧ faults cleared ∧ authenticated re-admission.':'One per-channel authority state; all proposals remain subject to permission checks.';
 }
 function render(i,{stage,details}){root.dataset.lessonCommittedIndex=String(i);mount(stage,details);frame(i);follow(...coords[states[i]]);}
 function transition({from,to,onFrame}){if(to!==from+1)return;const path=paths[to];onFrame(p=>{if(path)route(token,path,p);if(to===4){bar.setAttribute('width',65*p);readout.textContent=`BLEND · enforcer owns profile · normalized progress ${(p*.5).toFixed(2)}`;equation.textContent='Fixed checked endpoints; live manual proposals do not write the blend.';}root.dataset.animationProgress=String(p);});}

 return window.MLSysOriginalPlayer.create({root,outer:context.outer,title:"When does the proposal token change owner?",scope:"Original per-channel authority FSM. Ordered successful handover, manual lease loss, guarded reset, and an explicit protective-preemption alternative. Normalized blend progress is illustrative.",steps,render,transition});
};})();
