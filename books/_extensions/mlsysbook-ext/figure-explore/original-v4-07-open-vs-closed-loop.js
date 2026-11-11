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
 // Keep the endpoint in view when Expand or rotation changes the artwork scale.
 // Scrolling alone does not trigger this observer, so manual inspection persists.
 let focus=null,resizeFrame=0,disposed=false;
 function recenter(){if(!focus||disposed)return;if(canvas.clientWidth>=520){canvas.scrollLeft=0;canvas.scrollTop=0;return;}const scale=svg.getBoundingClientRect().width/width;canvas.scrollLeft=Math.max(0,focus[0]*scale-canvas.clientWidth/2);canvas.scrollTop=Math.max(0,focus[1]*scale-canvas.clientHeight/2);}
 function follow(x,y){focus=[x,y];recenter();}
 const cameraResize=typeof ResizeObserver==='function'?new ResizeObserver(()=>{if(!disposed&&!resizeFrame)resizeFrame=requestAnimationFrame(()=>{resizeFrame=0;recenter();});}):null;
 cameraResize?.observe(canvas);cameraResize?.observe(svg);
 function mount(stage,details){if(!canvas.isConnected)stage.append(canvas,note);if(!readout.isConnected)details.append(readout,equation);}
 function dot(x,y,color='#007A87',r=5){return add('circle',{cx:x,cy:y,r,fill:color,stroke:'#fff','stroke-width':r/3,'data-motion-object':'true'});}
 function route(token,points,p){const lens=points.slice(1).map((b,i)=>Math.hypot(b[0]-points[i][0],b[1]-points[i][1]));let rem=p*lens.reduce((a,b)=>a+b,0),j=0;while(j<lens.length-1&&rem>lens[j]){rem-=lens[j];j++;}const q=lens[j]?rem/lens[j]:1,a=points[j],b=points[j+1];token.setAttribute('cx',a[0]+(b[0]-a[0])*q);token.setAttribute('cy',a[1]+(b[1]-a[1])*q);}
 const D={"theta": 0.017453292519943295, "speed": 1.5, "clearance": 0.15, "exhausted": 8.593494244613913, "exhausted_s": 5.729868849855017, "mse": 0.00030461741978670857};

 const X=x=>405+(3553-405)*x/10,YA=1465,YB=3397,Y=y=>YB-y*(3397-2527)/.15;
 const upper=dot(X(0),YA,'#007A87',27),lower=dot(X(0),Y(0),'#A51C30',27);
 const base=add('g',{'data-motion-object':'true'});const chassis=add('rect',{x:-48,y:-26,width:96,height:52,rx:6,fill:'#fff',stroke:'#A51C30','stroke-width':8});base.append(chassis);const heading=add('path',{d:'M 0 0 H 63 L 42 -15 M 63 0 L 42 15',fill:'none',stroke:'#A51C30','stroke-width':8});base.append(heading);
 const trace=add('path',{d:'',fill:'none',stroke:'#A51C30','stroke-width':15,opacity:.55,'data-motion-object':'true'});
 const steps=[
 ['Start from the recorded centerline','At x = 0.00 m (t = 0.00 s), both the blue replay marker in panel (a) and the red base in panel (b) start on the centerline with 0.150 m of rack side clearance and a 1.0° heading bias.'],
 ['Advance two meters','At x = 2.00 m (t = 1.33 s at 1.5 m/s), panel (a) scores the 1.0° bias at 3.05 × 10⁻⁴ rad² MSE and resets to y = 0.000 m, whereas panel (b) integrates prior motion to y = 0.035 m (0.115 m clearance left).'],
 ['Advance four meters','At x = 4.00 m (t = 2.67 s), offline replay still reports the same 3.05 × 10⁻⁴ rad² per-step score on the centerline, while the physical base in panel (b) drifts to y = 0.070 m (0.080 m clearance left).'],
 ['Advance six meters','At x = 6.00 m (t = 4.00 s), physical lateral drift in panel (b) reaches y = 0.105 m, leaving only 0.045 m of side clearance—where a permission path must already intervene to reserve stopping distance.'],
 ['Spend the side clearance','At x = 8.59 m (t = 5.73 s), the unmitigated physical trajectory in panel (b) crosses the 0.15 m side-clearance line while panel (a) replay remains flat at 0.000 m.']
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

 const player=window.MLSysOriginalPlayer.create({root,outer:context.outer,layout:'compact',intro:'Compare recorded-state replay against physical rollout as a 1.0° heading bias at 1.5 m/s exhausts 0.15 m side clearance in 5.73 s.',title:"Why does the same heading error spend physical clearance?",scope:"Warehouse aisle navigation stack at the v = 1.5 m/s drive limit with a persistent Δθ = 1.0° (0.0175 rad) heading bias and 0.15 m (15 cm) rack side clearance. Panel (a) resets each step to the recorded centerline (constant angular MSE = 3.05 × 10⁻⁴ rad²), while panel (b) integrates unmitigated planar kinematics y = x tan(1°) without feedback correction or braking reserve.",steps,render,transition});
 const destroy=player.destroy;player.destroy=()=>{disposed=true;cameraResize?.disconnect();cancelAnimationFrame(resizeFrame);destroy();};return player;
};})();
