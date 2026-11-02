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
 function follow(x,y){const scale=svg.getBoundingClientRect().width/width;canvas.scrollLeft=Math.max(0,x*scale-canvas.clientWidth/2);canvas.scrollTop=Math.max(0,y*scale-canvas.clientHeight/2);}
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
 ['One admitted command, four distinct taps','A normalized toy scalar at each interface illustrates a clamped request followed by a stall. Values at different taps need not share a coordinate frame or denote the same physical quantity.'],
 ['Record the request','The operator asks for +1.0 in its native interface. The moving copy records the request before kinematic mapping; at the endpoint tap 1 is retained in the log. This proposal has no actuation permission.'],
 ['Map the command','Kinematic mapping produces +0.8 in the command interface. At the endpoint tap 2 retains this new representation, while tap 1 remains in the synchronized log.'],
 ['Enforce a clamp','Follow the proposal through the permission path; at the endpoint +0.8 has been modified to +0.4 in the same normalized command interface. The intervention must remain attributed; a permitted command is not automatically an expert target.'],
 ['Apply the admitted command','Follow the +0.4 admitted command toward the drives; at the endpoint it has reached the plant path. Command delivery is distinct from mechanical response; the plant can resist the requested motion.'],
 ['Measure the stalled plant','At this endpoint the encoders have measured zero motion over the observation interval while effort can persist against the latch. A zero measured velocity does not prove the operator requested a stop.'],
 ['Keep the full tuple','The hardware-synchronized log retains Asked, Mapped, Allowed, and Done. A dataset label must match the policy interface and retain intervention attribution; the measured response is proprioceptive state, not an interchangeable action label.']
 ].map(([label,description],i)=>({label,get description(){return root.dataset.lessonCommittedIndex===String(i)?description:'Next endpoint, not yet completed: '+description;}}));
 const endpoints=[[131,95],[131,95],[272,95],[506,95],[594,180],[594,262],[594,262]];
 const routes={2:[[131,95],[272,95]],3:[[272,95],[506,95]],4:[[506,95],[594,95],[594,180]],5:[[594,180],[594,262]]};
 function frame(i,p=1){
  const known=[i>=1,i>=2,i>=3,i>=5];cells.forEach((el,j)=>el.textContent=known[j]?values[j]:'—');
  blocks.forEach(el=>el.classList.remove('v4b-active'));const b=[0,0,1,2,3,4,4][i];blocks[b].classList.add('v4b-active');
  moving.setAttribute('cx',endpoints[i][0]);moving.setAttribute('cy',endpoints[i][1]);copy.style.display='none';
  root.dataset.tapStep=String(i);root.dataset.logged=known.map((v,j)=>v?j+1:null).filter(Boolean).join(',');root.dataset.request='1';root.dataset.mapped='0.8';root.dataset.enforced='0.4';root.dataset.measured=i>=5?'0':'pending';
  readout.textContent=i===0?'No command has crossed the permission path.':i<3?'Proposal values are recorded; actuation has not been admitted.':i<5?'Clamp attributed: mapped +0.8 → allowed +0.4.':'Measured motion 0.0; admitted effort can persist against contact.';
  equation.textContent='Declared normalized toy: request +1.0 → mapping +0.8 → clamp +0.4 → measured motion 0.0. These are interface values, not a physical unit conversion.';
 }
 function render(i,{stage,details}){root.dataset.lessonCommittedIndex=String(i);mount(stage,details);frame(i);follow(...endpoints[i]);}
 function transition({from,to,onFrame}){if(to!==from+1)return;const r=routes[to];const tap=to===1?0:to===2?1:to===3?2:to===5?3:null;
  onFrame(p=>{if(r)route(moving,r,p);if(tap!==null){copy.style.display='';route(copy,[taps[tap],[taps[tap][0],298]],p);}
   root.dataset.animationProgress=String(p);});
 }

 return window.MLSysOriginalPlayer.create({root,outer:context.outer,title:"Why keep Asked, Mapped, Allowed, and Done?",scope:"Original four-tap architecture. Declared normalized toy values illustrate a clamp and stall; logical causal order, not measured network timing.",steps,render,transition});
};})();
