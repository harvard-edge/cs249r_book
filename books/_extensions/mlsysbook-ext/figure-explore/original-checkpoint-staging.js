/* The original four-lane staging timeline, with immutable generation identity. */
(() => {
  'use strict';
  const NS='http://www.w3.org/2000/svg';
  const svg=(tag,attrs={},text)=>{const element=document.createElementNS(NS,tag);for(const [key,value] of Object.entries(attrs))element.setAttribute(key,value);if(text!==undefined)element.textContent=text;return element;};
  const steps=[
    {label:'Pause at a consistent step',description:'Both ranks stop at the same completed step i. Generation g0 is already committed; the new generation g1 has not completed its local writes.'},
    {label:'Complete every local NVMe write',description:'Once every required local shard is written and acknowledged, the fixed step-i snapshot is ready and the completion barrier can release training.'},
    {label:'Resume on separate live state',description:'Resume training on separate live state from step i+1. The local g1 shards stay at step i; background copying reads those immutable shards.'},
    {label:'Copy the immutable snapshot',description:'Start copying rank A and rank B shards from local NVMe to the parallel file system. An in-flight copy is not a completed durable write.'},
    {label:'Finish rank A’s durable write',description:'When A’s write finishes, its receipt becomes durable while B remains pending. One completed shard cannot make the generation restorable; g0 remains the restore point.'},
    {label:'Finish rank B’s durable write',description:'Once B’s write finishes, every required g1 shard is persisted. The generation still lacks its verified, published manifest, so restore must still use g0.'},
    {label:'Verify every required shard',description:'Verify the complete shard set and checksums. Successful verification permits publication; it does not itself change the committed restore point.'},
    {label:'Publish the manifest last',description:'Publish the complete, verified g1 manifest atomically. Restore stays on g0 until publication finishes; the committed g1 manifest then permits restoration of step i.'},
    {label:'Release local staging',description:'After publication, release local g1 staging. The committed snapshot remains on the parallel file system; releasing local staging does not delete the previous generation.'},
  ];
  const phaseX=[230,350,350,326,326,527,525,615,615];
  const state=index=>({local:index>=1&&index<8,live:index>=2?'i+1':'i',running:index>=2,copiedA:index>=4,copiedB:index>=5,verified:index>=6,committed:index>=7});
  window.MLSysFigureLessons=window.MLSysFigureLessons||{};
  window.MLSysFigureLessons['original-checkpoint-staging']=async(root,context)=>{
    if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw new Error('Verified staging artwork and shared player required');
    const art=context.verifiedArtwork.cloneNode(true);
    if(art.getAttribute('viewBox')!=='0 0 680 386')throw new Error('Checkpoint staging geometry requires review');
    art.classList.add('original-checkpoint-art');art.setAttribute('role','img');art.setAttribute('aria-label','Original four concurrent lanes: training, local NVMe, parallel file system, and coordinator. Captured snapshot i stays fixed while live training advances.');
    const rectangles=[...art.querySelectorAll('rect')],texts=[...art.querySelectorAll('text')];
    const specification={
      previous:[130,20,60,44,[160]],pause:[190,20,80,44,[230]],live:[270,20,390,44,[465]],
      write:[190,82,80,44,[230]],local:[270,82,300,44,[420]],clear:[570,82,90,44,[615]],
      copy:[270,144,210,44,[375]],durable:[480,144,180,44,[570]],
      wait:[270,206,210,44,[375]],verify:[480,206,90,44,[525]],publish:[570,206,90,44,[615]],
    };
    const groups={};
    for(const [key,[x,y,width,height,centers]]of Object.entries(specification)){
      const rect=rectangles.find(node=>+node.getAttribute('x')===x&&+node.getAttribute('y')===y&&+node.getAttribute('width')===width&&+node.getAttribute('height')===height);
      if(!rect)throw new Error('Source staging block missing: '+key);
      const labels=texts.filter(node=>centers.includes(+node.getAttribute('x'))&&+node.getAttribute('y')>y&&+node.getAttribute('y')<y+height);
      const group=svg('g',{'data-staging-group':key});group.append(rect,...labels);art.append(group);groups[key]={group,rect,labels};
    }
    groups.live.labels[0].classList.add('original-checkpoint-live-title');
    const liveBadge=svg('g',{'data-live-badge':'true'});liveBadge.append(svg('rect',{x:282,y:31,width:100,height:22,rx:2,fill:'#EAF1F8',stroke:'#1F407A'}),svg('text',{x:332,y:46,'text-anchor':'middle','font-size':11,'font-weight':700,fill:'#14294E'},'Live: i+1'));groups.live.group.append(liveBadge);
    const localTitle=groups.local.labels[0],localWarning=groups.local.labels[1];
    localTitle.setAttribute('y','94');localTitle.setAttribute('font-size','11');
    localWarning.setAttribute('y','122');localWarning.setAttribute('font-size','9');
    const copyTitle=groups.copy.labels[0],copySub=groups.copy.labels[1];
    const durableTitle=groups.durable.labels[0];durableTitle.setAttribute('y','156');durableTitle.setAttribute('font-size','10');durableTitle.textContent='PFS copies · snapshot i';
    const localShards=[],receipts=[],durableShards=[];
    function shard(parent,x,y,width,height,label,rank,kind){
      const group=svg('g',{'data-staging-shard':rank,'data-staging-kind':kind,'data-generation':'g1','data-snapshot':'i'});
      const rect=svg('rect',{x,y,width,height,rx:2,fill:'#dcfce7',stroke:'#16a34a','stroke-width':1});
      const text=svg('text',{x:x+width/2,y:y+height/2+3.5,'text-anchor':'middle','font-size':10,fill:'#15803d','font-weight':700},label);
      group.append(rect,text);parent.append(group);return{group,rect,text};
    }
    for(const [index,rank]of ['A','B'].entries()){
      localShards.push(shard(groups.local.group,306+index*140,99,88,15,`Rank ${rank} · i`,rank,'local'));
      receipts.push(shard(groups.copy.group,280+index*98,162,92,19,`${rank}: pending`,rank,'receipt'));
      durableShards.push(shard(groups.durable.group,486+index*86,163,82,19,`Rank ${rank} · i`,rank,'durable'));
    }
    const stagedInterval=svg('g'),restorableInterval=svg('g');
    const paths=[...art.querySelectorAll('path')];
    const stagedPath=paths.find(node=>node.getAttribute('d')?.startsWith('M270,322'));
    const restorablePath=paths.find(node=>node.getAttribute('d')?.startsWith('M573,322'));
    if(!stagedPath||!restorablePath)throw new Error('Checkpoint interval brackets require review');
    stagedInterval.append(stagedPath,...texts.filter(node=>+node.getAttribute('x')===420&&+node.getAttribute('y')>=346));
    restorableInterval.append(restorablePath,...texts.filter(node=>+node.getAttribute('x')===615&&+node.getAttribute('y')===346));
    art.append(stagedInterval,restorableInterval);
    for(const item of receipts){const x=+item.rect.getAttribute('x'),y=+item.rect.getAttribute('y');item.progress=svg('rect',{x,y,width:0,height:19,rx:2,fill:'#fef3c7','pointer-events':'none'});item.group.insertBefore(item.progress,item.text);}
    const markers=svg('g',{'aria-hidden':'true','data-copy-packets':'true'});art.append(markers);
    // Source lane names remain source objects, pinned only to the local viewport.
    const labelLayer=svg('g'),backing=svg('rect',{x:0,y:8,width:125,height:246,fill:'#fff'});labelLayer.append(backing);
    for(const label of texts.filter(node=>+node.getAttribute('x')===116))labelLayer.append(label);
    art.append(labelLayer);
    const restore=document.createElement('p');restore.className='original-checkpoint-restore';restore.setAttribute('aria-live','polite');
    const current=document.createElement('p');current.className='original-checkpoint-current';
    const statusBoard=document.createElement('div');statusBoard.className='original-checkpoint-ranks';statusBoard.setAttribute('aria-label','Both required snapshot shards');
    const rankStatus=['A','B'].map(rank=>{
      const card=document.createElement('section');card.className='original-checkpoint-rank';card.dataset.rankStatus=rank;
      const title=document.createElement('strong');title.textContent=`Rank ${rank} · step i`;
      const local=document.createElement('div');local.className='original-checkpoint-rank-local';
      const receipt=document.createElement('div');receipt.className='original-checkpoint-rank-receipt';
      const label=document.createElement('span');label.className='original-checkpoint-receipt-label';receipt.append(label);
      const fill=document.createElement('span');fill.className='original-checkpoint-receipt-progress';fill.setAttribute('aria-hidden','true');receipt.append(fill);
      const checks=document.createElement('div');checks.className='original-checkpoint-rank-checks';
      card.append(title,local,receipt,checks);statusBoard.append(card);return{card,local,receipt,label,fill,checks};
    });
    const note=document.createElement('p');note.className='original-checkpoint-note';note.textContent='The lanes share a schematic time axis. Pale regions are later phases. A and B represent the required shard set; receipt completion is distinct from manifest publication.';
    let stageNode=null,mounted=false,currentIndex=0,observer=null;
    function keepLabels(){if(!stageNode)return;const scale=art.getBoundingClientRect().width/680;const offset=scale?stageNode.scrollLeft/scale:0;labelLayer.setAttribute('transform',`translate(${offset},0)`);liveBadge.setAttribute('transform',`translate(${Math.max(282,Math.min(550,offset+138))-282},0)`);}
    function center(index){if(!stageNode)return;const scale=art.getBoundingClientRect().width/680;stageNode.scrollLeft=Math.max(0,phaseX[index]*scale-(stageNode.clientWidth+125*scale)/2);keepLabels();}
    function opacity(index,key){
      const visible={previous:true,pause:index<2,live:index>=2,write:index>=1,local:index>=1,clear:index>=8,copy:index>=3,durable:index>=5,wait:index>=3,verify:index>=6,publish:index>=7};
      if(!visible[key])return .1;
      if(key==='pause'||key==='write')return index>=2?.4:1;
      if(key==='copy'||key==='wait')return index>=5?.4:1;
      if(key==='local')return index>=8?.3:1;
      return 1;
    }
    function apply(index){
      const s=state(index);
      stagedInterval.style.opacity=String(index<1?.1:index>=7?.4:1);restorableInterval.style.opacity=String(s.committed?1:.1);
      for(const [key,{group}]of Object.entries(groups))group.style.opacity=String(opacity(index,key));
      rankStatus.forEach((item,i)=>{const durable=i===0?s.copiedA:s.copiedB;item.card.dataset.snapshot='i';item.local.textContent=`Local: ${s.local?'snapshot i retained':index>=8?'released':'not yet written'}`;item.receipt.dataset.durable=String(durable);item.receipt.dataset.inflight='false';item.label.textContent=`PFS: ${durable?'durable':index>=3?'pending':'not started'}`;item.fill.style.width='0%';item.checks.textContent=s.verified?'Checks: verified':'Checks: not verified';item.checks.dataset.verified=String(s.verified);item.checks.style.opacity='1';});
      groups.verify.labels[0].textContent=s.verified?'verified':'verify';groups.verify.labels[1].textContent=s.verified?'A ✓ · B ✓':'checksums';
      groups.publish.labels[0].textContent=s.committed?'g1 manifest':'publish';groups.publish.labels[0].setAttribute('font-size',s.committed?'11':'12.5');groups.publish.labels[1].textContent=s.committed?'A@i · B@i':'manifest last';groups.publish.labels[1].setAttribute('font-size',s.committed?'10':'11.5');groups.publish.group.dataset.manifestShards=s.committed?'A,B':'';
      // The source still shows its full reference timeline; actual resident
      // shards have their own explicit state and never become durable in flight.
      groups.live.labels[0].textContent=s.running?'Live state · step i+1':'Training continues on live state';
      liveBadge.style.visibility=s.running?'visible':'hidden';
      groups.live.group.dataset.liveStep=s.live;groups.live.group.dataset.running=String(s.running);
      localTitle.textContent=index>=8?'Local staging released':'Snapshot i · immutable local shards';
      localWarning.textContent=index>=8?'committed g1 remains on PFS':'local copies can be lost with their nodes';
      localShards.forEach(item=>{item.group.hidden=false;item.group.style.visibility=s.local?'visible':'hidden';item.group.dataset.resident=String(s.local);item.group.style.opacity='1';});
      const progress=index>=3&&index<5;
      copyTitle.style.visibility='visible';copyTitle.setAttribute('y',progress?'155':'163');copyTitle.setAttribute('font-size',progress?'10':'12.5');copyTitle.textContent=progress?'PFS write receipts':'Async copy to PFS';
      copySub.style.visibility=progress?'hidden':'visible';
      receipts.forEach((item,i)=>{const durable=i===0?s.copiedA:s.copiedB;item.group.style.visibility=progress?'visible':'hidden';item.group.dataset.durable=String(durable);item.progress.setAttribute('width','0');item.text.textContent=`${i===0?'A':'B'}: ${durable?'durable':'pending'}`;item.rect.setAttribute('fill',durable?'#dcfce7':'#fff');item.rect.setAttribute('stroke-dasharray',durable?'none':'3,2');});
      durableShards.forEach(item=>{item.group.style.visibility=s.copiedB?'visible':'hidden';item.group.dataset.resident=String(s.copiedB);item.group.dataset.durable=String(s.copiedB);item.group.style.opacity='1';});
      groups.verify.group.dataset.verified=String(s.verified);groups.publish.group.dataset.committed=String(s.committed);
      restore.textContent=s.committed?'Restore point: g1 · snapshot at step i':'Restore point: g0 · previous committed generation';
      restore.dataset.restoreGeneration=s.committed?'g1':'g0';
      current.textContent=`Live training: ${s.running?'running from step i+1':'paused at step i'} · Captured g1: step i${s.local?' · local snapshot retained':index>=8?' · local staging released':''}`;
      current.dataset.liveStep=s.live;current.dataset.snapshotStep='i';
    }
    const player=window.MLSysOriginalPlayer.create({root,outer:context.outer,title:'Why can training resume before a checkpoint is restorable?',scope:'Generation g1 captures step i on representative ranks A and B; g0 is already committed. This local-NVMe-first schedule waits for every local write before resuming separate live state. Motion does not encode measured duration or bandwidth.',steps:steps.map((step,index)=>({label:step.label,get description(){return currentIndex===index?step.description:'On completion: '+step.description;}})),
      render(index,{stage,details}){
        currentIndex=index;
        if(!mounted){stageNode=stage;stage.classList.add('original-checkpoint-stage');stage.tabIndex=0;stage.setAttribute('aria-label','Original checkpoint timeline; scroll horizontally to inspect stages');stage.before(restore,current,statusBoard);stage.append(art);details.append(note);stage.addEventListener('scroll',keepLabels,{passive:true});observer=new ResizeObserver(()=>center(currentIndex));observer.observe(stage);mounted=true;}
        markers.replaceChildren();apply(index);center(index);
      },
      transition({from,to,onFrame}){
        if(to!==from+1)return;
        center(to);
        const entering=Object.keys(groups).filter(key=>opacity(to,key)>opacity(from,key));
        const packets=[];
        if(to===4||to===5)rankStatus[to===4?0:1].receipt.dataset.inflight='true';
        if(to===3){copyTitle.style.visibility='hidden';copySub.style.visibility='hidden';const packet=svg('g');packet.append(svg('rect',{x:-30,y:-9,width:60,height:18,rx:2,fill:'#dcfce7',stroke:'#16a34a'}),svg('text',{'text-anchor':'middle',y:4,'font-size':11,'font-weight':700,fill:'#15803d'},'g1 @ i'));markers.append(packet);packets.push({packet});}
        if(to===6){groups.verify.labels[0].textContent='verify A/B';rankStatus.forEach(item=>item.checks.textContent='Checks: checking…');}
        if(to===7){groups.publish.labels[0].textContent='g1 pending';groups.publish.labels[0].setAttribute('font-size','11');groups.publish.labels[1].textContent='A@i · B@i';groups.publish.labels[1].setAttribute('font-size','10');}
        if(to===1)for(const item of localShards){item.group.style.visibility='visible';item.group.style.opacity='0';}
        onFrame(progress=>{
          for(const key of entering)groups[key].group.style.opacity=String(opacity(from,key)+(opacity(to,key)-opacity(from,key))*progress);
          if(to===6){groups.verify.labels[0].textContent='verify A/B';rankStatus.forEach(item=>item.checks.textContent='Checks: checking…');}
        if(to===7){groups.publish.labels[0].textContent='g1 pending';groups.publish.labels[0].setAttribute('font-size','11');groups.publish.labels[1].textContent='A@i · B@i';groups.publish.labels[1].setAttribute('font-size','10');}
        if(to===1)for(const item of localShards)item.group.style.opacity=String(progress);
          if(to===8)for(const item of localShards)item.group.style.opacity=String(1-progress);
          if(to===4||to===5){const rank=to===4?0:1;receipts[rank].progress.setAttribute('width',String(92*progress));rankStatus[rank].fill.style.width=`${100*progress}%`;}
          if(to===6)rankStatus.forEach(item=>item.checks.style.opacity=String(.4+.6*progress));
          for(const {packet}of packets){const q=Math.max(0,(progress-.4)/.6),x=progress<.4?294:294+81*q,y=progress<.4?126+17*progress/.4:143+28*q;packet.setAttribute('transform',`translate(${x},${y})`);}
        });
      },
    });
    return{pause:()=>player.pause(),destroy:()=>{stageNode?.removeEventListener('scroll',keepLabels);observer?.disconnect();player.destroy();}};
  };
})();
