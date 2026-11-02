/* Original-artwork mechanism. Shared player owns the pausable clock. */
(() => {'use strict';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v4-10-latch-force-budget"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),ns='http://www.w3.org/2000/svg';
 const originalView=svg.getAttribute('viewBox');if(originalView!=="0 0 3660 2100")throw Error('Original geometry changed; retain Diagram for review');
 root.classList.add("mlsys-original-v4-10-latch-force-budget");
 const canvas=document.createElement('div');canvas.className='v4b-canvas';canvas.tabIndex=0;canvas.setAttribute('aria-label','Original artwork. Scroll to inspect; Expand opens a larger view.');canvas.append(svg);
 svg.setAttribute('role','img');svg.setAttribute('aria-label',"Why size the stop from force at brake onset?");
 const overlay=document.createElementNS(ns,'g');svg.append(overlay);
 const add=(tag,attrs={},text)=>{const el=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))el.setAttribute(k,String(v));if(text!=null)el.textContent=text;overlay.append(el);return el;};
 const readout=document.createElement('p');readout.className='v4b-readout';readout.setAttribute('aria-live','off');
 const equation=document.createElement('p');equation.className='v4b-equation';
 const note=document.createElement('p');note.className='v4b-note';note.textContent='Scroll the original artwork to inspect labels, or use Expand. Playback duration is illustrative.';
 const width=+originalView.split(' ')[2],height=+originalView.split(' ')[3];
 function follow(x,y){const scale=svg.getBoundingClientRect().width/width;canvas.scrollLeft=Math.max(0,x*scale-canvas.clientWidth/2);canvas.scrollTop=Math.max(0,y*scale-canvas.clientHeight/2);}
 function mount(stage,details){if(!canvas.isConnected)stage.append(physicalScene,canvas,note);if(!readout.isConnected)details.append(readout,equation);}
 function dot(x,y,color='#007A87',r=5){return add('circle',{cx:x,cy:y,r,fill:color,stroke:'#fff','stroke-width':r/3,'data-motion-object':'true'});}
 function route(token,points,p){const lens=points.slice(1).map((b,i)=>Math.hypot(b[0]-points[i][0],b[1]-points[i][1]));let rem=p*lens.reduce((a,b)=>a+b,0),j=0;while(j<lens.length-1&&rem>lens[j]){rem-=lens[j];j++;}const q=lens[j]?rem/lens[j]:1,a=points[j],b=points[j+1];token.setAttribute('cx',a[0]+(b[0]-a[0])*q);token.setAttribute('cy',a[1]+(b[1]-a[1])*q);}
 const D={"k": 400000.0, "v": 0.03, "trip": 15.0, "limit": 100.0, "response": 2.0, "onset_force": 39.0, "a": 2.9508196721311477, "stop_ms": 10.166666666666666, "trip_ms": 1.25, "onset_ms": 3.25, "end_ms": 13.416666666666666, "unbraked_limit_ms": 8.333333333333334};

 const X=t=>354+(3515-354)*t/D.end_ms,Y=f=>1798-(1798-596)*f/100;
 const actual=dot(X(0),Y(0),'#1A4D3E',25),unbraked=dot(X(0),Y(0),'#A51C30',23);
 const trace=add('path',{d:'',fill:'none',stroke:'#1A4D3E','stroke-width':15,opacity:.55,'data-motion-object':'true'});
 const plate=add('line',{x1:3250,x2:3250,y1:1200,y2:1450,stroke:'#4A5568','stroke-width':12});
 const arm=add('rect',{x:2250,y:1270,width:180,height:95,rx:0,fill:'#E8F0EC',stroke:'#1A4D3E','stroke-width':8,'data-motion-object':'true'});
 const spring=add('path',{d:'',fill:'none',stroke:'#B87333','stroke-width':9,'data-motion-object':'true'});
 add('text',{x:2700,y:1430,'text-anchor':'middle','font-size':38,fill:'#4A5568'},'spring compression (amplified)');

 const physicalScene=document.createElementNS(ns,'svg');physicalScene.setAttribute('viewBox','0 0 500 100');physicalScene.classList.add('v4b-physical');physicalScene.setAttribute('role','img');physicalScene.setAttribute('aria-label','Same arm and spring state as the force plot; compression amplified');
 const pg=add('g');physicalScene.append(pg);
 const miniAdd=(tag,attrs,text)=>{const e=add(tag,attrs,text);pg.append(e);return e;};
 miniAdd('line',{x1:460,x2:460,y1:16,y2:72,stroke:'#718096','stroke-width':4});
 const miniArm=miniAdd('rect',{x:10,y:28,width:65,height:32,fill:'#E8F0EC',stroke:'#1A4D3E','stroke-width':3,'data-motion-object':'true'});
 const miniSpring=miniAdd('path',{fill:'none',stroke:'#B87333','stroke-width':3,'data-motion-object':'true'});
 const miniLabel=miniAdd('text',{x:250,y:92,'font-size':17,'text-anchor':'middle',fill:'currentColor'},'Same arm/plate state · compression amplified');
 const times=[0,D.trip_ms,D.onset_ms,6,D.unbraked_limit_ms,D.end_ms];
 const steps=[
 ['First contact: compression starts','The linear spring model begins at zero contact force. The arm initially approaches at the chapter’s guarded speed.'],
 ['Cross the contact tripwire','Follow force toward the 15 N tripwire. Crossing it starts the contact-loop response; the arm has not yet begun decelerating.'],
 ['Wait through response to brake onset','During the 2 ms detection, command, and torque-rise response, compression continues. At physical braking onset the force is 39 N. The stop must be sized from this force, not from the tripwire.'],
 ['Decelerate while the spring keeps compressing','The guarded braking curve bends as velocity falls. Contact force still rises because the arm continues moving into the plate; the brake-onset force is not the peak force.'],
 ['Compare the unbraked branch at the limit','At this endpoint the red unbraked continuation reaches 100 N while the sized braking curve remains lower. The two markers share time; they are alternative trajectories, not simultaneous forces on one arm.'],
 ['Stop at the modeled force limit','The constant-deceleration bound reaches zero velocity exactly at 100 N. No extra force margin remains in this idealized model; real qualification and uncertainty are outside this illustration.']
 ].map(([label,description],i)=>({label,get description(){return root.dataset.lessonCommittedIndex===String(i)?description:'Next endpoint, not yet completed: '+description;}}));
 const force=t=>{const tau=Math.max(0,(t-D.onset_ms)/1000);return t<=D.onset_ms?D.k*D.v*t/1000:D.onset_force+D.k*(D.v*tau-.5*D.a*tau*tau);};
 function frame(t){const f=force(t),tau=Math.max(0,(t-D.onset_ms)/1000),v=Math.max(0,D.v-D.a*tau),nb=D.k*D.v*t/1000;
  actual.setAttribute('cx',X(t));actual.setAttribute('cy',Y(f));unbraked.style.display=nb<=145?'':'none';unbraked.setAttribute('cx',X(t));unbraked.setAttribute('cy',Y(nb));
  const points=Array.from({length:81},(_,i)=>{const z=t*i/80;return `${i?'L':'M'} ${X(z)} ${Y(force(z))}`;});trace.setAttribute('d',points.join(' '));
  const armX=2250+600*f/D.limit;arm.setAttribute('x',armX);const left=armX+180,right=3250,dx=(right-left)/10;spring.setAttribute('d',`M ${left} 1317 `+Array.from({length:9},(_,i)=>`L ${left+(i+1)*dx} ${1317+(i%2?35:-35)}`).join(' ')+` L ${right} 1317`);
  const mx=10+250*f/D.limit;miniArm.setAttribute('x',mx);const ml=mx+65,md=(460-ml)/10;miniSpring.setAttribute('d',`M ${ml} 44 `+Array.from({length:9},(_,i)=>`L ${ml+(i+1)*md} ${44+(i%2?12:-12)}`).join(' ')+` L 460 44`);physicalScene.setAttribute('aria-label',`Same modeled arm/plate: force ${f.toFixed(1)} N, compression ${(1000*f/D.k).toFixed(4)} mm, speed ${v.toFixed(3)} m/s; compression amplified`);
  root.dataset.compressionMm=String(1000*f/D.k);root.dataset.time=String(t);root.dataset.force=String(f);root.dataset.velocity=String(v);root.dataset.brakeOnset=String(t>=D.onset_ms);
  readout.textContent=`Since contact ${t.toFixed(2)} ms · braking-path force ${f.toFixed(1)} N · arm speed ${v.toFixed(3)} m/s · compression ${(1000*f/D.k).toFixed(4)} mm`;
  equation.textContent=t<D.onset_ms?`F = k v t; ${t<D.trip_ms?'tripwire not yet crossed':'tripwire crossed; response pending'}; physical deceleration has not begun.`:`F = 39 N + k(vτ − aτ²/2); τ = ${((t-D.onset_ms)).toFixed(2)} ms; a = ${D.a.toFixed(3)} m/s². Unbraked alternative ${nb.toFixed(1)} N${nb>145?' (above plot range)':''}.`;
 }
 function render(i,{stage,details}){root.dataset.lessonCommittedIndex=String(i);mount(stage,details);frame(times[i]);follow(X(times[i]),Y(force(times[i])));}
 function transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>{frame(times[from]+(times[to]-times[from])*p);root.dataset.animationProgress=String(p);});}

 return window.MLSysOriginalPlayer.create({root,outer:context.outer,title:"Why size the stop from force at brake onset?",scope:"Original linear-spring/rigid-arm force plot and exact chapter profile. Braked and unbraked curves are alternative trajectories on one time axis; qualification and uncertainty are outside this model.",steps,render,transition});
};})();
