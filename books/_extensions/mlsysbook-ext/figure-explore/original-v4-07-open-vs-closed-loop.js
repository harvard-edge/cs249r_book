/* Original-artwork mechanism. Shared player owns the pausable clock. */
(() => {'use strict';
window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons["original-v4-07-open-vs-closed-loop"]=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
 const svg=context.verifiedArtwork.cloneNode(true),ns='http://www.w3.org/2000/svg';
 const originalView=svg.getAttribute('viewBox');if(originalView!=="0 0 3658 3901")throw Error('Original geometry changed; retain Diagram for review');
 root.classList.add("mlsys-original-v4-07-open-vs-closed-loop");
 const canvas=document.createElement('div');canvas.className='v4b-canvas';canvas.tabIndex=0;canvas.setAttribute('aria-label','Original artwork. Scroll to inspect; Expand opens a larger view.');canvas.append(svg);
 svg.setAttribute('role','img');svg.setAttribute('aria-label',"Why does the same heading error spend physical clearance?");
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
 const D={"theta": 0.017453292519943295, "speed": 1.5, "clearance": 0.15, "exhausted": 8.593494244613913, "exhausted_s": 5.729868849855017, "mse": 0.00030461741978670857};

 const X=x=>405+(3553-405)*x/10,YA=1465,YB=3397,Y=y=>YB-y*(3397-2527)/.15;
 const upper=dot(X(0),YA,'#007A87',27),lower=dot(X(0),Y(0),'#A51C30',27);
 const base=add('g',{'data-motion-object':'true'});const chassis=add('rect',{x:-48,y:-26,width:96,height:52,rx:6,fill:'#fff',stroke:'#A51C30','stroke-width':8});base.append(chassis);const heading=add('path',{d:'M 0 0 H 63 L 42 -15 M 63 0 L 42 15',fill:'none',stroke:'#A51C30','stroke-width':8});base.append(heading);
 const trace=add('path',{d:'',fill:'none',stroke:'#A51C30','stroke-width':15,opacity:.55,'data-motion-object':'true'});
 const steps=[
 ['Start from the recorded centerline','Both views start at zero lateral offset. The one-degree heading bias is scored on recorded inputs; this figure isolates its unmitigated physical consequence.'],
 ['Advance two meters','Follow both markers: replay returns its next input to the recorded centerline, while the physical state retains the lateral displacement caused by previous admitted motion.'],
 ['Advance four meters','The same angular score remains fixed. The physical offset grows with distance because the biased heading persists; replay does not feed the displaced state back into its next input.'],
 ['Advance six meters','Observe the remaining side clearance decrease. A real permission path must intervene before clearance exhaustion, reserving response time and stopping motion.'],
 ['Spend the side clearance','At this endpoint the unmitigated physical trajectory reaches the 0.15 m side-clearance line. This is a collision horizon, not a permissible braking trigger or a demonstration of a safe controller.']
 ].map(([label,description],i)=>({label,get description(){return root.dataset.lessonCommittedIndex===String(i)?description:'Next endpoint, not yet completed: '+description;}}));
 const positions=[0,2,4,6,D.exhausted];
 function frame(x){const y=x*Math.tan(D.theta),remaining=D.clearance-y;
  upper.setAttribute('cx',X(x));lower.setAttribute('cx',X(x));lower.setAttribute('cy',Y(y));trace.setAttribute('d',`M ${X(0)} ${Y(0)} L ${X(x)} ${Y(y)}`);
  base.setAttribute('transform',`translate(${X(x)} ${Y(y)}) rotate(-1)`);root.dataset.position=String(x);root.dataset.offset=String(y);root.dataset.remaining=String(remaining);
  readout.textContent=`Forward ${x.toFixed(2)} m · replay input offset 0.000 m · physical offset ${y.toFixed(3)} m · remaining side clearance ${Math.max(0,remaining).toFixed(3)} m`;
  equation.textContent=`y = x tan(1°) = ${y.toFixed(3)} m; t = x / (v cos 1°), v = ${D.speed.toFixed(1)} m/s = ${(x/(D.speed*Math.cos(D.theta))).toFixed(2)} s. Recorded-state angular MSE = ${D.mse.toExponential(3)} rad². ${remaining<1e-9?'Clearance exhausted; stopping would have to begin earlier.':'Stopping reserve is not modeled by this drift plot.'}`;
 }
 function render(i,{stage,details}){root.dataset.lessonCommittedIndex=String(i);mount(stage,details);frame(positions[i]);follow(X(positions[i]),Y(positions[i]*Math.tan(D.theta)));}
 function transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>{frame(positions[from]+(positions[to]-positions[from])*p);root.dataset.animationProgress=String(p);});}

 return window.MLSysOriginalPlayer.create({root,outer:context.outer,title:"Why does the same heading error spend physical clearance?",scope:"Original two-panel plot, exact fixed one-degree heading series and axes. Navigation-stack illustration; base icon position follows the plotted trajectory, icon size is schematic. No feedback correction or stopping intervention is simulated.",steps,render,transition});
};})();
