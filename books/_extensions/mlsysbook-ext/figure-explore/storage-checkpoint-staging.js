/* Specialized, deterministic lesson. Animation time is reading time, not a hardware model. */
(() => {
'use strict';
let instances = 0;
window.MLSysFigureLessons = window.MLSysFigureLessons || {};
window.MLSysFigureLessons['storage-checkpoint-staging'] = (root, context) => {
 const prefix = 'mlsys-' + 'storage-checkpoint-staging' + '-' + (++instances);
 const abort = new AbortController();
 const on = (el, type, fn) => el.addEventListener(type, fn, {signal:abort.signal});
 const el = (tag, cls, text) => { const n=document.createElement(tag); if(cls)n.className=cls; if(text!==undefined)n.textContent=text; return n; };
 const box = (cls, text) => `<div class="cell ${cls}">${text}</div>`;
 root.classList.add('mlsys-v2-lesson', 'mlsys-' + 'storage-checkpoint-staging');
 let step=0, running=false, timer=null, destroyed=false;
 const intro=el('p','lesson-question',"When can training resume, and when can this checkpoint be restored?"), scope=el('p','lesson-scope',"The original four-lane t\u2080\u2013t\u2083 staging schedule. One immutable generation is followed across storage owners; no hardware rates, time estimates, or instantaneous durability are assumed.");
 const controls=el('div','lesson-controls'); controls.setAttribute('role','group'); controls.setAttribute('aria-label',intro.textContent+' playback');
 const buttons={};
 for(const [key,label] of [['play','Play'],['back','Back'],['next','Next'],['restart','Restart']]) {const b=el('button','',label);b.type='button'; b.id=prefix+'-'+key;buttons[key]=b;controls.append(b);}
 const label=el('label','','Sequence '), menu=el('select');menu.id=prefix+'-sequence';label.htmlFor=menu.id;label.append(menu);controls.append(label);
 const current=el('div','lesson-current');current.setAttribute('aria-live','polite');current.setAttribute('aria-atomic','true');
 const title=el('p','lesson-step'), narration=el('p','lesson-narration');current.append(title,narration);
 const art=el('div','lesson-art');art.setAttribute('role','group');art.setAttribute('aria-label',intro.textContent);
 const detail=el('div','lesson-detail');
 root.append(intro,scope,controls,current,art,detail);


 const stages=[
 ['Before t₀: live training','Training uses live state at step i. The previous committed checkpoint remains the only restore point. The new generation has not been written.'],
 ['t₀: write local shards','Training pauses while each node writes its shard of the same snapshot generation to local NVMe. A partial write is not a completed snapshot.'],
 ['t₁: snapshot staged; resume','All required local writes have completed and the snapshot is immutable. Training resumes on live state; the background writer owns the staged shard until it is safe to release.'],
 ['Copy overlaps training','The background process reads the unchanged local shard and writes the parallel file system while training changes live state. The staged source remains retained; an incomplete durable set is not restorable.'],
 ['t₂: every shard on PFS','Every required shard has persisted on the parallel file system. The coordinator still must verify the complete generation and publish its manifest. Restore continues to use the previous generation.'],
 ['Checksums verified','The coordinator verifies the full shard set for the immutable generation. Verification alone does not publish a restore point; the manifest remains unpublished.'],
 ['t₃: manifest committed last','The atomic manifest publication names the complete verified shard set. The new generation is now a restore point. Training has already resumed and local staging may now be cleared.'],
 ['Release local staging','Local staging is cleared after its copy consumer and publication obligations finish. Durable shard copies and the committed manifest remain; clearing the staging buffer does not delete the checkpoint.']
 ];
 function draw(s){
  const state=(start,end,role)=>s<start?'future':s<=end?role+' active':role+' done';
  art.innerHTML=`<div class="legend"><span>Outline: current state</span><span>✓: completed</span><span>Dashed: scheduled region or wait</span></div><p class="scroll-hint">Scroll the timeline horizontally →. The current state is also listed below.</p><div class="scroll-diagram" tabindex="0" aria-label="Four-lane staging timeline; scroll horizontally on a narrow screen"><div class="checkpoint-grid timeline">
   <div class="lane-name" style="grid-row:1">Training<small>accelerators</small></div>
   <div class="cell compute ${s===0?'active':'done'}" style="grid-column:2;grid-row:1">step i</div>
   <div class="cell ${state(1,1,'fault')}" style="grid-column:3;grid-row:1">Twrite<br>paused</div>
   <div class="cell ${state(2,7,'compute')}" style="grid-column:4/12;grid-row:1">Training continues on live state</div>
   <div class="lane-name" style="grid-row:2">Local NVMe<small>per node</small></div>
   <div class="cell ${state(1,1,'storage')}" style="grid-column:3;grid-row:2">write shard</div>
   <div class="cell ${state(2,5,'storage')}" style="grid-column:4/10;grid-row:2">Shard staged locally<br><small>immutable; retained for copy</small></div>
   <div class="cell ${s>=7?'empty':s>=6?'storage active':'future'}" style="grid-column:10/12;grid-row:2">${s>=7?'cleared':'can be cleared'}</div>
   <div class="lane-name" style="grid-row:3">Parallel FS<small>shared</small></div>
   <div class="cell ${state(3,3,'storage')}" style="grid-column:4/8;grid-row:3">Async copy to PFS<br><small>overlaps training</small></div>
   <div class="cell ${s>=4?'storage':'future'}" style="grid-column:8/12;grid-row:3">Durable shard copies</div>
   <div class="lane-name" style="grid-row:4">Coordinator<small>manifest</small></div>
   <div class="cell ${s<4?'future':'done'}" style="grid-column:4/8;grid-row:4">wait for every rank’s shard</div>
   <div class="cell ${state(4,5,'coord')}" style="grid-column:8/10;grid-row:4">verify checksums</div>
   <div class="cell ${s>=6?'coord done':'future'}" style="grid-column:10/12;grid-row:4">publish<br>manifest last</div>
   <div class="milestone" style="grid-column:3;grid-row:5">t₀<br>trigger</div><div class="milestone" style="grid-column:4;grid-row:5">t₁<br>resume</div><div class="milestone" style="grid-column:8/10;grid-row:5">t₂<br>all shards on PFS</div><div class="milestone" style="grid-column:10/12;grid-row:5">t₃<br>committed</div>
   <div class="restore-bracket" style="grid-column:4/10;grid-row:6">Staged, not restorable</div><div class="restore-bracket storage" style="grid-column:10/12;grid-row:6">Restorable</div>
  </div></div>`;
  const local=s===0?'absent':s===1?'write in progress':s>=7?'released':'immutable shard retained';
  const durable=s<3?'absent':s===3?'copy in progress; incomplete':'all required shard copies persisted';
  detail.innerHTML=`<table class="state-table"><caption>Generation ownership at this boundary</caption><thead><tr><th>Owner / location</th><th>Current state</th></tr></thead><tbody><tr><th>Training / live state</th><td>${s===1?'paused at snapshot boundary':s===0?'step i':'advancing independently of immutable snapshot'}</td></tr><tr><th>Background writer / local NVMe</th><td>${local}</td></tr><tr><th>Shared PFS / new generation</th><td>${durable}</td></tr><tr><th>Coordinator / manifest</th><td>${s>=6?'committed':s===5?'checks verified; unpublished':s===4?'checking all shards':'waiting; unpublished'}</td></tr></tbody></table><p><strong>Restore point:</strong> ${s>=6?'new committed generation':'previous committed generation'}.</p><p>Local staging alone can be lost with its node. The copy still consumes shared resources; overlap does not make its bandwidth cost disappear.</p>`;
 }
 stages.forEach((s,i)=>{const o=el('option','',`${i+1}. ${s[0]}`);o.value=i;menu.append(o);});
 function render(){
   root.dataset.lessonStep=String(step);root.dataset.lessonRunning=String(running);
   title.textContent=`${step+1} / ${stages.length} · ${stages[step][0]}`;narration.textContent=stages[step][1];menu.value=String(step);
   buttons.play.textContent=running?'Pause':step===stages.length-1?'Replay':'Play';buttons.play.setAttribute('aria-pressed',String(running));
   buttons.back.disabled=step===0;buttons.next.disabled=step===stages.length-1;
   draw(step);
 }
 function pause(){running=false;clearTimeout(timer);timer=null;if(!destroyed)render();}
 function schedule(){timer=setTimeout(()=>{if(!running||destroyed)return;if(document.hidden||root.closest('[hidden]')){pause();return;}if(step<stages.length-1)step++;if(step===stages.length-1)running=false;render();if(running)schedule();},Math.max(5500,stages[step][1].split(/\s+/).length*260));}
 on(buttons.play,'click',()=>{if(running){pause();return;}if(step===stages.length-1)step=0;running=true;render();schedule();});
 on(buttons.back,'click',()=>{pause();step=Math.max(0,step-1);render();});
 on(buttons.next,'click',()=>{pause();step=Math.min(stages.length-1,step+1);render();});
 on(buttons.restart,'click',()=>{pause();step=0;render();});
 on(menu,'change',()=>{const selected=Number(menu.value);pause();step=selected;render();});
 on(document,'visibilitychange',()=>{if(document.hidden)pause();});on(window,'beforeprint',pause);
 on(root,'mlsys-view-change',e=>{if(e.detail?.view!=='explore')pause();});
 const intersection=typeof IntersectionObserver==='function'?new IntersectionObserver(items=>{if(items.some(x=>!x.isIntersecting)&&running)pause();}):null;intersection?.observe(root);
 const hidden=new MutationObserver(()=>{if(running&&root.closest('[hidden]'))pause();});hidden.observe(document.documentElement,{attributes:true,attributeFilter:['hidden'],subtree:true});
 const motion=window.matchMedia('(prefers-reduced-motion: reduce)');on(motion,'change',()=>{if(motion.matches)pause();});
 render();
 return {pause,destroy(){pause();destroyed=true;abort.abort();intersection?.disconnect();hidden.disconnect();root.replaceChildren();}};
};
})();
