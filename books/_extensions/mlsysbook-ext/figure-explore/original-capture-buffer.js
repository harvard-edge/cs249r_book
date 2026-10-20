/* Frame identity and ownership on the verified three-panel capture-buffer SVG. */
(() => {
  'use strict';
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons['original-capture-buffer'] = async (root, context) => {
    if (!context.verifiedArtwork || !window.MLSysOriginalPlayer) throw Error('Verified capture-buffer artwork and shared player required');
    const svg=context.verifiedArtwork.cloneNode(true), ns='http://www.w3.org/2000/svg';
    const rects=[...svg.querySelectorAll('rect')], texts=[...svg.querySelectorAll('text')];
    const box=(x,y,w)=>rects.find(n=>+n.getAttribute('x')===x&&+n.getAttribute('y')===y&&+n.getAttribute('width')===w);
    const copy=[30,195,360,525].map(x=>box(x,50,125)), dma=box(30,190,130), pages=box(220,190,180), mappings=[30,230,430].map(x=>box(x,290,160)), converted=box(230,360,160), owners=[box(30,485,130),box(200,485,130),box(370,485,130),box(540,485,110)];
    if([...copy,dma,pages,...mappings,converted,...owners].some(n=>!n)||!svg.getAttribute('viewBox')?.endsWith('680 630')) throw Error('Capture-buffer source geometry changed; retain Diagram until reviewed');
    const text=(x,y)=>texts.find(n=>+n.getAttribute('x')===x&&+n.getAttribute('y')===y);
    const pageStatus=text(310,228), fence=text(95,228), convertedStatus=text(310,398), guard=text(430,580);
    const reuseBracket=[...svg.querySelectorAll('path')].find(n=>n.getAttribute('d')==='M265 527 V560 H595 V527');
    if(!pageStatus||!fence||!convertedStatus||!guard||!reuseBracket)throw Error('Capture-buffer labels changed');
    const overlay=document.createElementNS(ns,'g');overlay.classList.add('ocb-overlay');svg.append(overlay);
    const add=(attrs,value)=>{const n=document.createElementNS(ns,'text');Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));n.textContent=value;overlay.append(n);return n;};
    function caption(rect,titleY,valueY){const x=+rect.getAttribute('x') + +rect.getAttribute('width')/2;const title=text(x,titleY);if(!title)throw Error('Capture-buffer box title changed');title.setAttribute('y',String(valueY-17));return add({x,y:valueY,'text-anchor':'middle','font-size':'10.5',fill:'#1A202C'},'');}
    const copyValues=copy.map(r=>caption(r,77,85)), mapValues=mappings.map(r=>caption(r,315,322)), ownerValues=owners.map(r=>caption(r,511,519));
    const canvas=document.createElement('div');canvas.className='ocb-canvas';canvas.tabIndex=0;canvas.setAttribute('aria-label','Original three-panel comparison. Scroll to inspect; the active region stays visible. Expand for more room.');canvas.append(svg);
    svg.setAttribute('role','img');svg.setAttribute('aria-label','Payload-copy alternative, shared capture pages, and capture-buffer ownership with synchronized frame identity and reference counts');
    root.classList.add('mlsys-original-capture-buffer');
    const summary=document.createElement('p');summary.className='ocb-summary';
    const inspect=document.createElement('button');inspect.type='button';inspect.className='ocb-inspect';inspect.textContent='Inspect retained B';inspect.hidden=true;
    let inspectingB=false;
    function follow(x,y){const scale=svg.getBoundingClientRect().width/680;canvas.scrollLeft=Math.max(0,x*scale-canvas.clientWidth/2);canvas.scrollTop=Math.max(0,y*scale-canvas.clientHeight/2);}
    inspect.onclick=()=>{inspectingB=!inspectingB;inspect.textContent=inspectingB?'Return to requeue':'Inspect retained B';follow(...(inspectingB?[310,365]:[595,460]));};
    const invariant=document.createElement('p');invariant.className='ocb-invariant';invariant.textContent='Mappings refer to A; conversion writes separate pages B for the same frame F. Completing B does not release other consumers of A. The driver requeues A only after its final consumer releases it.';
    const steps=[
      ['Compare two alternatives','Panel a copies one frame through distinct payload buffers. Panel b is an alternative shared-page design; it does not run after panel a. Panel c expands the lifetime of capture pages A.'],
      ['Capture F in the copy alternative','Frame F first occupies the kernel DMA buffer. The other payload buffers have not received it.'],
      ['Copy F into the user buffer','The payload is duplicated into a distinct user buffer. The kernel buffer still contains frame F; this is a copy, not a move.'],
      ['Copy F into pinned host storage','Another payload copy creates a pinned-host representation of F. Shared frame identity does not mean shared physical storage.'],
      ['Copy F into device storage','The final copy writes F into the device buffer. All four boxes represent distinct payload storage in this alternative.'],
      ['Switch to the shared-page alternative','Start a separate illustration of the same frame F. Capture DMA writes physical pages A directly. No consumer may read A before write completion and the required synchronization.'],
      ['Complete the DMA write','The completion fence is satisfied. Device ordering and the required cache maintenance still precede the admitted reads; completion alone does not authorize reuse.'],
      ['Admit three mapped readers','After the required handoffs, CPU, ISP, and GPU each hold a consumer reference to physical pages A. These mappings alias A; they do not allocate three new payload copies.'],
      ['Convert into separate pages B','In this declared conversion case, the ISP reads A and writes a new representation into B. Both are frame F, but B is separate physical storage. This transformation is not a zero-copy operation.'],
      ['Release the CPU reference','The CPU is finished with A and releases its reference. ISP and GPU still hold A, so capture reuse remains blocked.'],
      ['Release the ISP reference','The ISP releases A after conversion. GPU still holds the source pages; the completed output B does not remove that obligation.'],
      ['Release the last reference','The GPU releases the last consumer reference to A. DMA completion and handoff requirements are already satisfied, so A is now eligible for requeue. Eligibility is not a new DMA write.'],
      ['Requeue A for a later capture','The driver returns A to capture. No consumer from F still holds A. Converted pages B keep F independently; requeuing A does not overwrite B. A later frame has not been captured in this illustration.']
    ].map(([label,description])=>({label,description}));
    steps.forEach((step,i)=>{const description=step.description;Object.defineProperty(step,'description',{get(){return root.dataset.lessonCommittedIndex===String(i)?description:'On completion: '+description;}});});
    const initialFills=new Map([...copy,dma,pages,...mappings,converted,...owners].map(n=>[n,n.getAttribute('fill')]));
    const focusPoints=[[340,75],[92.5,72],[257.5,72],[422.5,72],[587.5,72],[200,215],[200,215],[310,310],[310,365],[110,310],[310,310],[435,460],[595,460]];
    function render(index,{stage,details}){root.dataset.lessonCommittedIndex=String(index);
      if(!canvas.isConnected)stage.append(canvas,summary,inspect);inspect.hidden=index!==12;inspectingB=false;inspect.textContent='Inspect retained B';if(!invariant.isConnected)details.append(invariant);
      const shared=index>=5, admitted=index>=7, holds=mappings.map((_,i)=>admitted&&index<9+i), refs=holds.filter(Boolean).length, bExists=index>=8, eligible=index>=11;
      [...copy,dma,pages,...mappings,converted,...owners].forEach(n=>{n.classList.remove('ocb-active');n.setAttribute('fill',initialFills.get(n));});
      copy.forEach((n,i)=>{const present=index>=i+1;n.dataset.frame=present?'F':'';copyValues[i].textContent=present?'F · distinct buffer':'empty';n.style.opacity=shared?'0.55':'1';copyValues[i].style.opacity=shared?'0.55':'1';});
      pages.dataset.frame=shared?'F':'';pages.dataset.references=String(refs);pages.dataset.requeueEligible=String(eligible);pages.dataset.requeued=String(index===12);
      pageStatus.textContent=shared?`F · A references: ${refs}`:'frame identity F';fence.textContent=shared?(index>=6?'write complete':'write pending'):'completion fence';
      mappings.forEach((n,i)=>{n.dataset.references=holds[i]?'1':'0';n.dataset.storage='A';mapValues[i].textContent=!admitted?'not admitted':holds[i]?'A · F · holds 1':'A · released';});
      converted.dataset.frame=bExists?'F':'';converted.dataset.storage='B';converted.dataset.allocated=String(bExists);converted.style.opacity=bExists?'1':'0.6';convertedStatus.textContent=bExists?'F · separate storage':'same frame F (not yet written)';convertedStatus.setAttribute('font-size',bExists?'11':'9.5');
      ownerValues[0].textContent=!shared?'pending':index===5?'writing F → A':'completed';ownerValues[1].textContent=!admitted?'not admitted':`${refs} references to A`;ownerValues[2].textContent=eligible?'0 refs · released':'not complete';ownerValues[3].textContent=index===12?'A returned':eligible?'eligible':'blocked';
      const active=index===0?null:index<=4?copy[index-1]:index===5?dma:index===6?pages:index===7?pages:index===8?converted:index<=11?mappings[index-9]:owners[3];
      if(active)active.classList.add('ocb-active');if(index>=7&&index<=10)owners[1].classList.add('ocb-active');if(index===11)owners[2].classList.add('ocb-active');
      guard.textContent=eligible?(index===12?'A requeued; B retains frame F':'Last consumer released: A may now be requeued'):'Reuse blocked while any consumer holds A';guard.setAttribute('fill',eligible?'#1A4D3E':'#A51C30');reuseBracket.setAttribute('stroke',eligible?'#1A4D3E':'#A51C30');
      root.dataset.captureStep=String(index);root.dataset.references=String(refs);root.dataset.requeueEligible=String(eligible);root.dataset.requeued=String(index===12);root.dataset.dmaComplete=String(index>=6);root.dataset.converted=String(bExists);
      summary.textContent=!shared?'Alternative a: payload copies into distinct storage.':`Alternative b: F in capture A · ${refs} consumer references · ${eligible?(index===12?'A requeued':'A eligible for requeue'):'reuse blocked'}${bExists?' · F also in separate B':''}`;
      follow(...focusPoints[index]);
    }
    function transition({from,to,onFrame}){
      if(to!==from+1)return;
      let route;
      if(to>=2&&to<=4)route=[[155+165*(to-2),72],[195+165*(to-2),72]];
      else if(to===5) return; // Comparison switch: deliberately no invented connecting transfer.
      else if(to===8)route=[[310,330],[310,360]];
      else if(to===12)route=[[500,506],[540,506]];
      if(!route)return;
      const token=document.createElementNS(ns,'circle');token.setAttribute('r','4');token.setAttribute('fill',to===8?'#B87333':to===12?'#1A4D3E':'#007A87');overlay.append(token);
      onFrame(p=>{token.setAttribute('cx',String(route[0][0]+(route[1][0]-route[0][0])*p));token.setAttribute('cy',String(route[0][1]+(route[1][1]-route[0][1])*p));});
      // Parent owns cancellation and endpoint commit; transient token is cleared on render.
      token.classList.add('ocb-transient');
    }
    const commit=(i,context)=>{overlay.querySelectorAll('.ocb-transient').forEach(n=>n.remove());render(i,context);};
    return window.MLSysOriginalPlayer.create({root,outer:context.outer,title:'When can capture pages be reused?',scope:'Two alternative designs for one schematic frame F. Logical ownership order, not latency or simultaneous device execution. Conversion into B is an explicit separate-storage case.',steps,render:commit,transition});
  };
})();
