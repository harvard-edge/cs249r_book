(() => {
 'use strict';
 const NAME="original-v4-15-hardware-fault-injection", CFG={"question": "Can travel pass while reaction timing fails?", "scope": "Canonical model F1 witness timelines and speed curves, not empirical trials. All times share the last valid renewal as reference.", "purpose": "Trace lease expiry through physical onset, then evaluate onset and distance with independent oracles.", "steps": [["Last valid renewal", "The proposer renews once and then is killed. The permission-path lease and physical witnesses continue independently."], ["Lease expires", "The lease expires; a permission tick issues the stop and the fieldbus carries it. A proposal probed after expiry is refused."], ["Onset bound", "The green trace begins powered deceleration by the bound. A software stop command is not itself evidence of physical onset."], ["Late physical onset", "The dashed trace begins deceleration late. Its onset predicate has already failed, regardless of its eventual travel."], ["Evaluate both predicates", "The dashed trace retains a small positive distance margin but a negative onset margin. Both predicates must pass; distance slack cannot repair reaction timing."]], "guard": "Independent onset and travel oracles use one renewal reference. Tracking inset belongs in the travel allowance, not extra motion. No claim of measured hardware safety.", "key": "vol4:fig-15-hardware-fault-injection"}, NS='http://www.w3.org/2000/svg';
 const el=(tag,attrs={},text)=>{const n=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;return n;};
 window.MLSysFigureLessons=window.MLSysFigureLessons||{};
 window.MLSysFigureLessons[NAME]=async(root,context)=>{
  if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
  const art=context.verifiedArtwork.cloneNode(true),vb=art.viewBox.baseVal;
  if(!vb.width||!vb.height)throw Error('Original geometry missing');
  art.classList.add(NAME+'-art');art.setAttribute('role','img');art.setAttribute('aria-label',CFG.question+' Original artwork with synchronized mechanism overlay.');
  const layer=el('svg',{viewBox:'0 0 1000 1000',width:vb.width,height:vb.height,preserveAspectRatio:'none','aria-hidden':'true'});art.append(layer);
  const scene=el('svg',{viewBox:'0 0 680 185',role:'img','aria-label':'Synchronized mechanism assembly',class:NAME+'-scene'});const proof=document.createElement('p');proof.className=NAME+'-proof';
  let mounted=false,current=0,stageNode;
  const D={
  "v": 1.3,
  "a": 2.0,
  "j": 7.111111111111111,
  "stop_s": 0.93125,
  "lease_ms": 60.0,
  "command_ms": 61.0,
  "frame_ms": 62.0,
  "onset_ms": 82.0,
  "late_ms": 112.0,
  "rest_late_ms": 1043.25,
  "onset_margin_ms": -30.0,
  "travel_margin_mm": 1.0,
  "travel_late_mm": 750.9125,
  "travel_bound_mm": 751.9125
}
;const times=[0,D.frame_ms,D.onset_ms,D.late_ms,D.rest_late_ms];
  function draw(from,to,p){
   layer.replaceChildren();root.dataset.animationProgress=String(p);
   const t=times[from]+(times[to]-times[from])*p,topx=251+Math.min(t,130)*5.48,bottomx=236.13+(Math.min(t,1200)+3)/1253*(966.7-236.13);layer.append(el('line',{x1:topx,x2:topx,y1:75,y2:397,stroke:'#B87333','stroke-width':2,'stroke-dasharray':'6 4','data-motion-object':'true'}),el('line',{x1:bottomx,x2:bottomx,y1:645,y2:921,stroke:'#B87333','stroke-width':2,'stroke-dasharray':'6 4','data-motion-object':'true'}));function speed(t,onset){const s=Math.max(0,(t-onset)/1000),t1=D.a/D.j,t2=D.stop_s-2*t1;if(s<=t1)return D.v-.5*D.j*s*s;if(s<=t1+t2)return D.v-.5*D.j*t1*t1-D.a*(s-t1);if(s<=D.stop_s){const z=s-t1-t2;return Math.max(0,D.v-.5*D.j*t1*t1-D.a*t2-D.a*z+.5*D.j*z*z);}return 0;}for(const [onset,color]of[[D.onset_ms,'#009E73'],[D.late_ms,'#A51C30']])layer.append(el('circle',{cx:bottomx,cy:909.4-speed(t,onset)/(D.v*1.18)*(909.4-604.0),r:6,fill:color,stroke:'#fff','stroke-width':2,'data-motion-object':'true'}));const phase=p<1?from:to;proof.textContent=phase<4?`Time from renewal ${t.toFixed(1)} ms. Lease ${t>=D.lease_ms?'expired':'active'}; stop command ${t>=D.command_ms?'issued':'pending'}; drive frame ${t>=D.frame_ms?'delivered':'pending'}. Green physical onset ${t>=D.onset_ms?'modeled onset reached':'pending'}; dashed physical onset ${t>=D.late_ms?'modeled onset reached':'pending'}. The software channels do not substitute for the physical witness.`:`Dashed trace: onset margin ${D.onset_margin_ms.toFixed(0)} ms (FAIL); travel margin +${D.travel_margin_mm.toFixed(1)} mm (PASS). Nominal travel ${D.travel_late_mm.toFixed(1)} mm; independent travel oracle ${D.travel_bound_mm.toFixed(1)} mm. Combined requirement FAILS. Tracking allowance is not added to travel.`;root.dataset.timeMs=String(t);root.dataset.onsetMarginMs=String(D.onset_margin_ms);root.dataset.travelMarginMm=String(D.travel_margin_mm);
   scene.replaceChildren();function distance(ms,onset){const n=1000,dt=ms/n;let total=0;for(let z=0;z<n;z++)total+=(speed(z*dt,onset)+speed((z+1)*dt,onset))*.5*dt;return total;}scene.append(el('text',{x:12,y:18,'font-size':13,fill:'#1A202C'},'Powered base travel integrates the two original speed traces (model)'));for(const [j,onset,color]of[[0,D.onset_ms,'#009E73'],[1,D.late_ms,'#A51C30']]){const yy=55+j*65,d=distance(t,onset),xx=150+d*.45;scene.append(el('line',{x1:150,x2:510,y1:yy+24,y2:yy+24,stroke:'#4A5568'}),el('line',{x1:150+D.travel_bound_mm*.45,x2:150+D.travel_bound_mm*.45,y1:yy-17,y2:yy+24,stroke:'#A51C30','stroke-dasharray':'4 3'}),el('rect',{x:xx-15,y:yy-4,width:30,height:19,fill:color,'fill-opacity':.2,stroke:color,'data-motion-object':'true'}),el('circle',{cx:xx-8,cy:yy+18,r:4,fill:'#4A5568'}),el('circle',{cx:xx+8,cy:yy+18,r:4,fill:'#4A5568'}),el('text',{x:12,y:yy,'font-size':11,fill:color},j?'late onset':'on-time onset'),el('text',{x:535,y:yy,'font-size':11,fill:'#1A202C'},`${d.toFixed(1)} mm traveled`),el('text',{x:535,y:yy+17,'font-size':10,fill:color},t<onset?'powered coast':speed(t,onset)>0?'powered deceleration':'rest'));}scene.append(el('text',{x:12,y:174,'font-size':11,fill:'#1A202C'},`lease ${t>=D.lease_ms?'expired':'active'} → command ${t>=D.command_ms?'issued':'pending'} → bus ${t>=D.frame_ms?'delivered':'pending'}; late onset fails independently of the travel limit`));
   if(root.clientWidth<500){scene.setAttribute('viewBox','0 0 320 220');scene.replaceChildren();scene.append(el('text',{x:8,y:18,'font-size':13,fill:'#1A202C'},'Base travel integrates modeled speed'));for(const[j,onset,color]of[[0,D.onset_ms,'#009E73'],[1,D.late_ms,'#A51C30']]){const yy=62+j*78,d=distance(t,onset),xx=20+d*.30;scene.append(el('text',{x:8,y:yy-20,'font-size':12,fill:color},j?'late onset':'on-time onset'),el('line',{x1:20,x2:280,y1:yy+15,y2:yy+15,stroke:'#4A5568'}),el('line',{x1:20+D.travel_bound_mm*.30,x2:20+D.travel_bound_mm*.30,y1:yy-14,y2:yy+15,stroke:'#A51C30','stroke-dasharray':'4 3'}),el('rect',{x:xx-10,y:yy-6,width:20,height:15,fill:color,'fill-opacity':.2,stroke:color}),el('text',{x:8,y:yy+35,'font-size':12,fill:'#1A202C'},`${d.toFixed(1)} mm · ${t<onset?'coast':speed(t,onset)>0?'deceleration':'rest'}`));}scene.append(el('text',{x:8,y:212,'font-size':12,fill:'#1A202C'},t>=D.late_ms?'late onset FAIL is independent of travel':'physical onset witness remains separate'));}else scene.setAttribute('viewBox','0 0 680 185');
   if(stageNode)stageNode.scrollLeft=Math.max(0,art.getBoundingClientRect().width*((t<=130?topx:bottomx)/1000)-stageNode.clientWidth/2);
  }
  const steps=CFG.steps.map(([label,description],i)=>({label,get description(){return i!==Number(root.dataset.step)?'Next endpoint will show the following (not yet completed): '+description:description;}}));
  const player=window.MLSysOriginalPlayer.create({root,outer:context.outer,title:CFG.question,scope:CFG.scope,steps,
   render(index,{stage,details}){current=index;root.dataset.step=String(index);if(!mounted){const artScroll=document.createElement('div');artScroll.className=NAME+'-art-scroll';stageNode=artScroll;stage.tabIndex=0;stage.setAttribute('aria-label','Original diagram; scroll horizontally or use Expand for dense artwork');stage.classList.add(NAME+'-stage');artScroll.append(art);stage.append(artScroll,scene);details.append(proof);mounted=true;}draw(index,index,1);root.dataset.originalState=JSON.stringify({index,kind:CFG.key});},
   transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>draw(from,to,p));}
  });return{pause:()=>player.pause(),destroy:()=>player.destroy()};
 };
})();
