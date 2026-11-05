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
      g.append(make('circle',{r:9,fill:'#fff',stroke:color,'stroke-width':1.5}),make('text',{x:0,y:3,'text-anchor':'middle','font-size':9,fill:color,'font-weight':700},id));
      if(deleted)g.append(make('path',{d:'M-8 -8 L8 8',stroke:'#A51C30','stroke-width':2}));
      overlay.append(g); return g;
    };
    const viewLabel = document.createElement('label'); viewLabel.className='incident-view';viewLabel.textContent='Diagram view ';
    const view = document.createElement('select');view.setAttribute('aria-label','Diagram view');view.append(new Option('Follow','follow'),new Option('Whole','whole'));view.value=matchMedia('(max-width:640px)').matches?'follow':'whole';viewLabel.append(view);
    const readout = document.createElement('p');readout.className='incident-readout';
    const legend = document.createElement('p');legend.className='incident-legend';legend.textContent='A, B, and C are illustrative frame identities. E is their selected event window; E in each retained-copy box denotes a separate copy. A question mark is a pending review; a slash means that copy has expired.';
    const steps = [
      ['Capture a bounded history','Frames A and B are routine capture history. The future response has not arrived. The original shaded region marks the window that a trigger would select.'],
      ['Expire routine A; select pre-trigger B','A has aged out of the routine buffer. The trigger selects B for this schematic event; selection does not recover A or authorize a training use.'],
      ['Collect response C into event E','Response frame C joins pre-trigger B. E denotes this bounded event window, not the surrounding routine footage. The figure uses event order, not a universal window duration.'],
      ['Retain an incident copy of E','A separate incident copy retains E under its declared purpose, retention, readers, and justified redaction. B and C can still be present in the routine buffer.'],
      ['Let routine B and C roll off','Routine B and C expire. The incident copy of E remains: removing routine frames does not delete a retained incident record.'],
      ['Review a proposed training use','The retained incident E is considered for a proposed training use in the separate review. E remains incident evidence; no training copy is authorized yet. Consent, eligibility, and purpose require an explicit decision.'],
      ['Follow an explicitly approved example','In this illustrative approved branch, a separate training copy of E is created with its own purpose, retention, and access limits. Approval is an assumed review outcome, not a conclusion supplied by the animation. Refusal would create no training copy.'],
      ['Expire the incident copy','The declared incident retention expires without an extension in this example. That incident copy is deleted. The reviewed training copy has a separate lifetime and remains until its own expiry.'],
      ['Expire the training copy separately','The training copy now reaches its own declared expiry and is deleted. Approval did not make retention indefinite, and neither branch authorizes unspecified later uses.']
    ].map(([label,description])=>({label,description}));
    const crops = ['0 35 360 145','15 35 350 145','155 40 365 240','15 305 290 205','15 305 290 205','375 305 290 205','375 305 290 205','15 305 290 205','375 305 290 205'];
    const focus = [[45,92,200,20],[315,52,30,80],[190,225,300,44],[30,320,260,80],[30,320,260,80],[390,320,260,80],[390,435,260,65],[30,435,260,65],[390,435,260,65]];
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
    const player=window.MLSysOriginalPlayer.create({root,outer:context.outer,title:'Which copy survives when routine frames expire?',scope:'Schematic frame identities A/B/C and event window E on the original diagram. Ordered events have no prescribed duration. One explicitly approved training branch is assumed; no legal-compliance or eligibility verdict is computed.',viewControls:viewLabel,steps,render,transition});
    const destroy=player.destroy;player.destroy=()=>{view.removeEventListener('change',camera);destroy();};return player;
  };
})();
