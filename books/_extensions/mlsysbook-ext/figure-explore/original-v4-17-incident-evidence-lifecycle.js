/* Record identities and independent copy lifetimes on the original incident diagram. */
(() => {
  'use strict';
  const NAME = 'original-v4-17-incident-evidence-lifecycle';
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons[NAME] = async (root, context) => {
    if (!context.verifiedArtwork || !window.MLSysOriginalPlayer) throw Error('Verified incident artwork and shared player required');
    const svg = context.verifiedArtwork.cloneNode(true), ns = 'http://www.w3.org/2000/svg';
    if (svg.getAttribute('viewBox') !== '0 0 680 525') throw Error('Incident artwork geometry changed');
    root.classList.add(NAME);
    svg.querySelectorAll('text').forEach(n=>{const color=n.getAttribute('fill');if(color)n.style.setProperty('fill',color,'important');});
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Original incident timeline and purpose branches with record identities and separate retention lifetimes');
    const canvas = document.createElement('div'); canvas.className = 'incident-canvas'; canvas.append(svg);
    const overlay = document.createElementNS(ns, 'g'); svg.append(overlay);
    const make = (tag, attrs, text) => { const n = document.createElementNS(ns, tag); for (const [k,v] of Object.entries(attrs)) n.setAttribute(k, String(v)); if(text !== undefined) n.textContent = text; if(tag==='text'&&attrs.fill)n.style.setProperty('fill',attrs.fill,'important'); return n; };
    const token = (id,x,y,color='#007A87',deleted=false) => {
      const g = make('g', {'data-record':id, 'data-deleted':String(deleted), transform:`translate(${x} ${y})`});
      g.append(make('circle',{r:12,fill:'#fff',stroke:color,'stroke-width':1.8}),make('text',{x:0,y:4,'text-anchor':'middle','font-size':12,fill:color,'font-weight':700},id));
      if(deleted)g.append(make('path',{d:'M-10 -10 L10 10',stroke:'#A51C30','stroke-width':2.2}));
      overlay.append(g); return g;
    };
    const viewLabel = document.createElement('label'); viewLabel.className='incident-view';viewLabel.textContent='Diagram view ';
    const view = document.createElement('select');view.setAttribute('aria-label','Diagram view');view.append(new Option('Follow','follow'),new Option('Whole','whole'));view.value=matchMedia('(max-width:640px)').matches?'follow':'whole';viewLabel.append(view);
    const readout = document.createElement('p');readout.className='incident-readout';
    const legend = document.createElement('p');legend.className='incident-legend';legend.textContent='A, B, and C are illustrative frame identities. E is their selected event window; E in each retained-copy box denotes a separate copy. A question mark is a pending review; a slash means that copy has expired.';
    const steps = [
      ['Capture a bounded history','Stage 1 of 9 in panel (a): the routine ring buffer holds frame A and pre-trigger frame B while the shaded event window marks what a future incident trigger would capture.'],
      ['Expire routine A; select pre-trigger B','Stage 2 of 9: older frame A ages out of the bounded ring buffer (marked with a red deletion slash) before the incident trigger fires, so the trigger selects only in-window pre-trigger frame B and cannot recover expired frame A.'],
      ['Collect response C into event E','Stage 3 of 9: post-trigger response frame C arrives in the routine buffer and joins pre-trigger frame B inside the bounded event window E, separating the triggered slice from surrounding routine capture.'],
      ['Retain an incident copy of E','Stage 4 of 9: the left branch of panel (b) creates an immutable incident copy of E (green token) under its declared investigation purpose, retention clock, authorized readers, and redaction policy while B and C still sit in the routine buffer.'],
      ['Let routine B and C roll off','Stage 5 of 9: routine frames B and C age out of the panel (a) ring buffer on their normal overwrite schedule, yet the retained incident copy of E in panel (b) persists intact on its own retention clock.'],
      ['Review a proposed training use','Stage 6 of 9: a secondary request to use E for model training enters the right branch of panel (b) with a pending review token (?); incident retention never automatically authorizes training reuse without an explicit eligibility, consent, and redaction decision.'],
      ['Follow an explicitly approved example','Stage 7 of 9: assuming the governance review approves the request, a separate training copy of E (bronze token) is created with its own purpose limitation, access boundary, and expiration schedule independent of the incident record.'],
      ['Expire the incident copy','Stage 8 of 9: when the declared incident retention window ends without a legal-hold extension, the left-branch incident copy of E is deleted (red slash) while the approved right-branch training copy of E continues under its own lifetime.'],
      ['Expire the training copy separately','Stage 9 of 9: the right-branch training copy of E reaches its own declared expiration and is deleted (red slash), showing that approval never grants indefinite retention or unspecified downstream uses.']
    ].map(([label,description])=>({label,description}));
    const crops = ['0 35 360 145','15 35 350 145','155 40 365 240','15 305 290 205','15 305 290 205','375 305 290 205','375 305 290 205','15 305 290 205','375 305 290 205'];
    const focus = [[45,62,270,58],[158,62,157,58],[190,225,300,44],[30,320,260,80],[30,320,260,80],[390,320,260,80],[390,435,260,65],[30,435,260,65],[390,435,260,65]];
    let current=0;
    function camera(){svg.setAttribute('viewBox',view.value==='whole'?'0 0 680 525':crops[current]);svg.querySelectorAll('text').forEach(n=>{if(n.textContent==='b  Purpose limits each retained copy')n.style.visibility=view.value==='follow'&&current===2?'hidden':'';});const b=svg.getAttribute('viewBox').split(' ');svg.setAttribute('width',b[2]);svg.setAttribute('height',b[3]);}
    view.addEventListener('change',camera);
    function render(i,{stage,details}) {
      current=i;if(!canvas.isConnected){stage.append(canvas);details.append(readout,legend);}
      overlay.replaceChildren();camera();
      if(i===0)token('A',70,100);if(i<=3)token('B',230,100);if(i>=2&&i<=3)token('C',415,100);
      if(i===1)token('A',70,100,'#A51C30',true);
      if(i>=2&&i<=3)token('E',210,247);
      if(i>=3&&i<7)token('E',48,345,'#1A4D3E');
      if(i===5)token('?',408,345,'#B87333');
      if(i>=6&&i<8)token('E',408,462,'#B87333');
      if(i>=7)token('E',48,462,'#A51C30',true);
      if(i===8)token('E',408,462,'#A51C30',true);
      const [x,y,w,h]=focus[i];overlay.append(make('rect',{x:x-4,y:y-4,width:w+8,height:h+8,rx:3,fill:'none',stroke:'#007A87','stroke-width':2,'stroke-dasharray':'5 3','data-mlsys-focus':''}));
      const state={routine:i===0?['A','B']:i===1?['B']:i<=3?['B','C']:[],eventSelected:i>=2,incidentRetained:i>=3&&i<7,trainingReview:i<5?'not requested':i===5?'pending':'approved example',trainingRetained:i>=6&&i<8,incidentDeleted:i>=7,trainingDeleted:i===8};
      root.dataset.incidentState=JSON.stringify(state);
      readout.textContent=`Routine buffer: ${state.routine.join(', ')||'event frames rolled off'} · Incident copy: ${state.incidentRetained?'E retained':state.incidentDeleted?'deleted':'not retained yet'} · Training copy: ${state.trainingRetained?'E retained':state.trainingDeleted?'deleted':i===5?'not authorized; review pending':'none'}`;
    }
    const routes={1:[[70,100],[40,100]],2:[[415,100],[415,192],[210,247]],3:[[340,269],[340,295],[160,295],[160,310]],6:[[520,400],[520,425]],7:[[160,400],[160,425]]};
    function transition({from,to,onFrame}) {
      if(to!==from+1||!routes[to])return;
      if(view.value==='follow'){svg.setAttribute('viewBox','0 0 680 525');svg.setAttribute('width','680');svg.setAttribute('height','525');}
      const pts=routes[to],g=to===1?overlay.querySelector('[data-record="A"]'):token(to===2?'C':'E',...pts[0],to>=5?'#B87333':'#007A87');g.dataset.transient='true';
      const lengths=pts.slice(1).map((p,i)=>Math.hypot(p[0]-pts[i][0],p[1]-pts[i][1])),total=lengths.reduce((a,b)=>a+b,0);
      onFrame(p=>{let d=p*total,j=0;while(j<lengths.length-1&&d>lengths[j]){d-=lengths[j];j++;}const q=lengths[j]?d/lengths[j]:1;g.setAttribute('transform',`translate(${pts[j][0]+q*(pts[j+1][0]-pts[j][0])} ${pts[j][1]+q*(pts[j+1][1]-pts[j][1])})`);});
    }
    const player=window.MLSysOriginalPlayer.create({root,outer:context.outer,layout:'compact',title:'Which copy survives when routine frames expire?',intro:'Follow frames A, B, and C across the 9 lifecycle stages to see how event window E is retained and expired separately as incident evidence and training data.',scope:'Schematic 9-stage incident evidence and retention lifecycle (ordered events with no prescribed wall-clock duration): routine capture ring buffer holding frames A, B, and C in panel (a), triggered extraction of pre-trigger frame B and response frame C into bounded event window E, and 2 independent purpose-limited retention branches in panel (b) (incident review copy E and explicitly reviewed training copy E, each with separate access controls, redaction rules, and expiration events).',viewControls:viewLabel,steps,render,transition});
    const destroy=player.destroy;player.destroy=()=>{view.removeEventListener('change',camera);destroy();};return player;
  };
})();
