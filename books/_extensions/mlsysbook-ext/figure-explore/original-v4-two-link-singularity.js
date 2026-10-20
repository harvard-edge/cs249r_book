/* Canonical two-link geometry with a posture-dependent velocity demand. */
(()=>{'use strict';window.MLSysFigureLessons=window.MLSysFigureLessons||{};
window.MLSysFigureLessons['original-v4-two-link-singularity']=async(root,context)=>{
 if(!context.verifiedArtwork||!window.MLSysOriginalPlayer||!window.MLSysTwoLinkSingularityData)throw Error('Verified artwork, data, and shared player required');
 root.classList.add('original-v4-two-link-singularity');
 const D=window.MLSysTwoLinkSingularityData,NS='http://www.w3.org/2000/svg';
 const make=(tag,a={},text)=>{const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;};
 const wrap=document.createElement('div');wrap.className='two-link-layout';
 const svg=make('svg',{viewBox:'0 0 600 550',role:'img','aria-label':'Two-link arm and current tool-velocity ellipse. Green links, teal velocity ellipse, orange weakest-direction arrow.'});
 const px=p=>[70+p[0]*560,430-p[1]*560],path=points=>points.map((p,i)=>(i?'L':'M')+p.join(',')).join(' ');
 const arc=[];for(let a=-14;a<=80;a+=1){const t=a*Math.PI/180;arc.push(px([D.reach*Math.cos(t),D.reach*Math.sin(t)]));}
 svg.append(make('path',{d:path(arc),class:'two-link-reach'}));
 const ghost=make('path',{class:'two-link-ghost'}),ellipse=make('path',{class:'two-link-ellipse'}),arm=make('path',{class:'two-link-arm'}),arrow=make('path',{class:'two-link-arrow'});
 const joints=[0,1,2].map(()=>make('circle',{r:6,class:'two-link-joint'}));
 svg.append(ghost,ellipse,arm,...joints,arrow);
 const title=make('text',{x:20,y:30,class:'two-link-plot-title'}),shoulder=make('text',{x:75,y:462},'shoulder'),reach=make('text',{x:478,y:355,class:'two-link-reach-label'},`reach ${D.reach.toFixed(2)} m`),angle=make('text',{x:20,y:525});
 const toolLabel=make('text',{class:'two-link-object-label'},'tool'),elbowLabel=make('text',{class:'two-link-object-label'},'elbow');svg.append(title,shoulder,reach,angle,toolLabel,elbowLabel);
 const aside=document.createElement('div');aside.className='two-link-metrics';
 const heading=document.createElement('p');heading.className='two-link-demand';heading.textContent=`Requested tool speed: ${D.speed.toFixed(1)} m/s`;
 const direction=document.createElement('p');direction.textContent='Pose comparison, not an executed trajectory. Orange: the local weakest direction; it changes with posture.';
 const metric=document.createElement('div');metric.className='two-link-metric';
 const jointsText=document.createElement('p'),cost=document.createElement('p'),legend=document.createElement('p');
 const calculation=document.createElement('details');calculation.className='two-link-calculation';const calculationLabel=document.createElement('summary');calculationLabel.textContent='Inspect the calculation';const formula=document.createElement('p'),sigmaText=document.createElement('p'),jointDetail=document.createElement('p');calculation.append(calculationLabel,formula,sigmaText,jointDetail);legend.className='two-link-legend';legend.textContent='Teal ellipse: tool velocities from joint-rate vectors of norm 1 rad/s. Ellipse drawing scale stays fixed; the orange arrow marks direction, not speed.';
 aside.append(heading,direction,metric,jointsText,cost,calculation);wrap.append(svg,aside);
 const compact=document.createElement('p');compact.className='two-link-compact';
 const summary=document.createElement('p');summary.className='two-link-summary';
 const definitions=document.createElement('p');definitions.className='two-link-definitions';calculation.append(definitions);definitions.textContent='σmin has units m/rad. ‖q̇‖ is the two-joint vector norm, not a per-joint rate. Kinematic snapshots, not a timed trajectory or a safety certificate.';
 const angles=[90,90,60,30,10,10],labels=['Establish the bent arm','Map joint rates to tool velocity','Open the elbow to 60°','Open the elbow to 30°','Approach the straight arm at 10°','Check the individual drives'];
 const descriptions=[
  'The links retain the original lengths and orientation. The tool lies on the same 45° bearing as in the static figure.',
  'A unit joint-rate circle maps to this teal ellipse. Its short axis identifies the direction that joint motion produces least effectively.',
  'As the elbow opens, the short axis shrinks. The same tool-speed magnitude now requires more joint rate along the new weakest direction.',
  'The arm continues toward its reach boundary. The live rate demand rises while the requested tool-speed magnitude stays unchanged.',
  'Watch the ellipse narrow toward the original 10° pose. The live readout reports the current pose; the final endpoint matches the original right panel.',
  'Check each joint against its own drive limit. Reducing tool speed can satisfy these velocity bounds at this posture; it does not establish acceleration, torque, or collision feasibility.'
 ];
 const steps=labels.map((label,i)=>({label,description:descriptions[i]}));
 function draw(deg,phase){const k=Math.max(0,Math.min(D.samples.length-1,Math.round((90-deg)*10))),s=D.samples[k],p0=px([0,0]),p1=px(s.elbow),p2=px(s.tool);
  arm.setAttribute('d',path([p0,p1,p2]));[p0,p1,p2].forEach((p,i)=>{joints[i].setAttribute('cx',p[0]);joints[i].setAttribute('cy',p[1]);});
  toolLabel.setAttribute('x',p2[0]+12);toolLabel.setAttribute('y',p2[1]+24);elbowLabel.setAttribute('x',p1[0]+12);elbowLabel.setAttribute('y',p1[1]+24);
  const reference=D.samples[0];ghost.setAttribute('d',path([p0,px(reference.elbow),px(reference.tool)]));ghost.style.opacity=deg<89?'0.25':'0';
  const points=[];for(let j=0;j<=80;j++){const t=j*2*Math.PI/80,c=Math.cos(t),v=Math.sin(t);points.push(px([s.tool[0]+D.ellipse_drawing_scale*(s.jac[0][0]*c+s.jac[0][1]*v),s.tool[1]+D.ellipse_drawing_scale*(s.jac[1][0]*c+s.jac[1][1]*v)]));}
  ellipse.setAttribute('d',path(points)+'Z');ellipse.style.opacity=phase===0?'0':'1';
  const a=px([s.tool[0]-s.weak[0]*.13,s.tool[1]-s.weak[1]*.13]),b=px([s.tool[0]+s.weak[0]*.13,s.tool[1]+s.weak[1]*.13]);
  const head=(tip,other)=>{const dx=tip[0]-other[0],dy=tip[1]-other[1],len=Math.hypot(dx,dy),ux=dx/len,uy=dy/len;return `M${tip[0]-ux*11-uy*5},${tip[1]-uy*11+ux*5}L${tip}L${tip[0]-ux*11+uy*5},${tip[1]-uy*11-ux*5}`;};
  arrow.setAttribute('d',path([a,b])+head(a,b)+head(b,a));arrow.style.opacity=phase===0?'0':'1';
  title.textContent=`Current elbow ${s.angle.toFixed(1)}°`;angle.textContent=`Links ${D.l1.toFixed(3)} m + ${D.l2.toFixed(3)} m`;
  compact.textContent=`Requested tool speed ${D.speed.toFixed(1)} m/s → joint-rate norm ${s.norm.toFixed(1)} rad/s`;
  metric.replaceChildren();const label=document.createElement('span');label.textContent='Joint-rate norm required';const value=document.createElement('strong');value.textContent=`${s.norm.toFixed(1)} rad/s`;const calc=document.createElement('span');calc.textContent=`‖q̇‖ = v / σmin ≈ ${s.norm.toFixed(1)} rad/s`;metric.append(label,value);formula.textContent=calc.textContent;sigmaText.textContent=`σmin = ${s.sigma[1].toFixed(4)} m/rad. The denominator comes from the current Jacobian’s smallest singular value.`;
  jointsText.textContent=`Joint 1: ${s.qdot[0].toFixed(2)} rad/s · Joint 2: ${s.qdot[1].toFixed(2)} rad/s`;
  jointDetail.textContent=jointsText.textContent;jointDetail.hidden=phase===5;jointsText.hidden=phase!==5;cost.hidden=phase!==5;
  cost.textContent=phase===5?`Each drive limit: ±${D.joint_limit.toFixed(3)} rad/s. Maximum tool speed along this direction ≈ ${s.feasible.toFixed(3)} m/s (instantaneous velocity bounds only).`:`Smallest singular value: ${s.sigma[1].toFixed(4)} m/rad. Same tool-speed magnitude; changing direction and rate demand.`;
  summary.textContent=phase===0?'First identify shoulder, elbow, and tool. Next adds the velocity ellipse.':phase===5?'A tool-speed limit alone cannot enforce individual joint-speed limits.':`As the elbow opens, the ellipse narrows and joint-rate demand grows. Grey links retain the initial pose for comparison.`;
  root.dataset.values=JSON.stringify({...s,joint_limit:D.joint_limit,requested_speed:D.speed,phase});root.dataset.sceneStep=String(phase);
 }
 function render(i,{stage}){if(!wrap.isConnected)stage.append(compact,wrap,legend,summary);draw(angles[i],i);}
 function transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>draw(angles[from]+(angles[to]-angles[from])*p,p===0?from:to));}
 return window.MLSysOriginalPlayer.create({root,outer:context.outer,title:'Why does the same tool speed demand faster joints?',scope:'The original two-link arm comes to life. Compare a fixed speed magnitude along each posture’s current weakest direction; the direction is not a fixed Cartesian vector.',steps,render,transition});
};})();
