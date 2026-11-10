/* Original-artwork mechanism. Shared player owns the pausable clock. */
(() => {'use strict';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v4-05-action-taps"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),ns='http://www.w3.org/2000/svg';
 const originalView=svg.getAttribute('viewBox');if(originalView!=="0 0 680 340")throw Error('Original geometry changed; retain Diagram for review');
 root.classList.add("mlsys-original-v4-05-action-taps");
 const canvas=document.createElement('div');canvas.className='v4b-canvas';canvas.tabIndex=0;canvas.setAttribute('aria-label','Original artwork. Scroll to inspect; Expand opens a larger view.');canvas.append(svg);
 svg.setAttribute('role','img');svg.setAttribute('aria-label',"Why keep Asked, Mapped, Allowed, and Done?");
 const overlay=document.createElementNS(ns,'g');svg.append(overlay);
 const add=(tag,attrs={},text)=>{const el=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))el.setAttribute(k,String(v));if(text!=null)el.textContent=text;overlay.append(el);return el;};
 const readout=document.createElement('p');readout.className='v4b-readout';readout.setAttribute('aria-live','off');
 const equation=document.createElement('p');equation.className='v4b-equation';
 const note=document.createElement('p');note.className='v4b-note';note.textContent='Scroll the original artwork to inspect labels, or use Expand. Playback duration is illustrative.';
 const width=+originalView.split(' ')[2],height=+originalView.split(' ')[3];
 // Keep the endpoint in view when Expand or rotation changes the artwork scale.
 // Scrolling alone does not trigger this observer, so manual inspection persists.
 let focus=null,resizeFrame=0,disposed=false;
 function recenter(){if(!focus||disposed)return;const scale=svg.getBoundingClientRect().width/width;canvas.scrollLeft=Math.max(0,focus[0]*scale-canvas.clientWidth/2);canvas.scrollTop=Math.max(0,focus[1]*scale-canvas.clientHeight/2);}
 function follow(x,y){focus=[x,y];recenter();}
 const cameraResize=typeof ResizeObserver==='function'?new ResizeObserver(()=>{if(!disposed&&!resizeFrame)resizeFrame=requestAnimationFrame(()=>{resizeFrame=0;recenter();});}):null;
 cameraResize?.observe(canvas);cameraResize?.observe(svg);
 function mount(stage,details){if(!canvas.isConnected)stage.append(canvas,note);if(!readout.isConnected)details.append(readout,equation);}
 function dot(x,y,color='#007A87',r=5){return add('circle',{cx:x,cy:y,r,fill:color,stroke:'#fff','stroke-width':r/3,'data-motion-object':'true'});}
 function route(token,points,p){const lens=points.slice(1).map((b,i)=>Math.hypot(b[0]-points[i][0],b[1]-points[i][1]));let rem=p*lens.reduce((a,b)=>a+b,0),j=0;while(j<lens.length-1&&rem>lens[j]){rem-=lens[j];j++;}const q=lens[j]?rem/lens[j]:1,a=points[j],b=points[j+1];token.setAttribute('cx',a[0]+(b[0]-a[0])*q);token.setAttribute('cy',a[1]+(b[1]-a[1])*q);}
 
 const boxes=[...svg.querySelectorAll('rect')];const by=(x,y)=>boxes.find(e=>+e.getAttribute('x')===x&&+e.getAttribute('y')===y);
 const blocks=[by(28,72),by(152,72),by(364,72),by(528,160),by(528,222)];
 if(blocks.some(x=>!x))throw Error('Action-tap objects changed');
 const taps=[[131,95],[272,95],[506,95],[594,262]];
 const values=['request +1.0','mapped +0.8','allowed +0.4','measured 0.0'];
 const log=svg.querySelector('text[x="338"][y="318"]');if(!log)throw Error('Log geometry changed');log.setAttribute('y','309');log.setAttribute('font-size','7');
 const cells=[98,255,414,574].map(x=>add('text',{x,y:322,'text-anchor':'middle','font-size':8,fill:'#1A202C'},'—'));
 const moving=dot(131,95,'#B87333',5),copy=dot(131,95,'#007A87',4);
 const steps=[
 ['One admitted command, four distinct taps','On the application processor, the policy or operator block prepares a command before any of the 4 taps are written to the hardware-synchronized telemetry log.'],
 ['Record the request','Tap 1 logs the raw request a_req = +1.0 in native operator coordinates before kinematic mapping; this unvalidated proposal carries no actuation permission.'],
 ['Map the command','Kinematic mapping transforms the request into the joint command interface and logs Tap 2 as a_cmd = +0.8 before crossing the black-channel proposal boundary.'],
 ['Enforce a clamp','On the discrete safety MCU, the 1 kHz permission path applies velocity and barrier limits, clamping +0.8 down to Tap 3 (a_enf = +0.4) and logging the intervention.'],
 ['Apply the admitted command','The admitted command a_enf = +0.4 drives the motor current loops and applies effort to the physical plant, where the latch strike plate resists motion.'],
 ['Measure the stalled plant','Optical joint encoders measure displacement over Δt and log Tap 4 as a_meas = 0.0 because contact stalls the joint even while motor holding torque persists.'],
 ['Keep the full tuple','The synchronized log retains all 4 taps (+1.0, +0.8, +0.4, 0.0): training on +1.0 alone hides the safety clamp, whereas training on 0.0 mistakes a mechanical stall for a stop command.']
 ].map(([label,description],i)=>({label,get description(){return root.dataset.lessonCommittedIndex===String(i)?description:'Next endpoint, not yet completed: '+description;}}));
 const endpoints=[[131,95],[131,95],[272,95],[506,95],[594,180],[594,262],[594,262]];
 const routes={2:[[131,95],[272,95]],3:[[272,95],[506,95]],4:[[506,95],[594,95],[594,180]],5:[[594,180],[594,262]]};
 const stepReadouts=[
  'Log tuple [a_req, a_cmd, a_enf, a_meas] = [—, —, —, —] · Active block: Policy or operator (Application processor).',
  'Log tuple [a_req, a_cmd, a_enf, a_meas] = [+1.0, —, —, —] · Tap 1 (Asked) recorded in user space; no actuation authority.',
  'Log tuple [a_req, a_cmd, a_enf, a_meas] = [+1.0, +0.8, —, —] · Tap 2 (Mapped) recorded after inverse-kinematics retargeting.',
  'Log tuple [a_req, a_cmd, a_enf, a_meas] = [+1.0, +0.8, +0.4, —] · Tap 3 (Allowed) recorded on 1 kHz safety MCU after clamping +0.8 → +0.4.',
  'Log tuple [a_req, a_cmd, a_enf, a_meas] = [+1.0, +0.8, +0.4, —] · Admitted +0.4 delivered to drives and plant against latch contact.',
  'Log tuple [a_req, a_cmd, a_enf, a_meas] = [+1.0, +0.8, +0.4, 0.0] · Tap 4 (Done) logs 0.0 encoder motion while holding torque persists.',
  'Log tuple [a_req, a_cmd, a_enf, a_meas] = [+1.0, +0.8, +0.4, 0.0] · Complete PTP-synchronized record preserves intervention and stall attribution.'
 ];
 function frame(i,p=1){
  const known=[i>=1,i>=2,i>=3,i>=5];cells.forEach((el,j)=>el.textContent=known[j]?values[j]:'—');
  blocks.forEach(el=>el.classList.remove('v4b-active'));const b=[0,0,1,2,3,4,4][i];blocks[b].classList.add('v4b-active');
  moving.setAttribute('cx',endpoints[i][0]);moving.setAttribute('cy',endpoints[i][1]);copy.style.display='none';
  root.dataset.tapStep=String(i);root.dataset.logged=known.map((v,j)=>v?j+1:null).filter(Boolean).join(',');root.dataset.request='1';root.dataset.mapped='0.8';root.dataset.enforced='0.4';root.dataset.measured=i>=5?'0':'pending';
  readout.textContent=stepReadouts[i];
  equation.textContent='Declared normalized toy: request +1.0 → mapping +0.8 → clamp +0.4 → measured motion 0.0. These are interface values, not a physical unit conversion.';
 }
 function render(i,{stage,details}){root.dataset.lessonCommittedIndex=String(i);mount(stage,details);frame(i);follow(...endpoints[i]);}
 function transition({from,to,onFrame}){if(to!==from+1)return;const r=routes[to];const tap=to===1?0:to===2?1:to===3?2:to===5?3:null;
  onFrame(p=>{if(r)route(moving,r,p);if(tap!==null){copy.style.display='';route(copy,[taps[tap],[taps[tap][0],298]],p);}
   root.dataset.animationProgress=String(p);});
 }

 const player=window.MLSysOriginalPlayer.create({root,outer:context.outer,layout:'compact',intro:'Follow one command across 4 telemetry taps as mapping scales +1.0 to +0.8, the 1 kHz permission path clamps +0.4, and contact yields 0.0 motion.',title:"Why keep Asked, Mapped, Allowed, and Done?",scope:"Four-stream action tap tuple (a_req, a_cmd, a_enf, a_meas) across the unprivileged Linux application processor and the 1 kHz discrete safety MCU over a black-channel link. Normalized scalar values (+1.0 request, +0.8 mapped, +0.4 clamped, 0.0 measured velocity against a latch) illustrate causal stage order rather than a physical unit conversion.",steps,render,transition});
 const destroy=player.destroy;player.destroy=()=>{disposed=true;cameraResize?.disconnect();cancelAnimationFrame(resizeFrame);destroy();};return player;
};})();
