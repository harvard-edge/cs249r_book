(() => {
 'use strict';
 const NAME='original-v2-hierarchical-allreduce',NS='http://www.w3.org/2000/svg';
 const el=(tag,attrs={},text)=>{const n=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;return n;};
 const sub=set=>set.map(i=>'₀₁₂₃'[i]).join('');
 const steps=[
  {label:'Begin with local contributions',description:'Start in panel 1 with 2 nodes (Node 0: ranks 0, 1; Node 1: ranks 2, 3) and 2 GPUs per node (4 GPUs total). Every GPU begins with its own unreduced local gradient split into 2 shards of size M/2: shard A (A₀, A₁, A₂, A₃) and shard B (B₀, B₁, B₂, B₃).'},
  {label:'ReduceScatter inside each node',description:'In Phase 1 (1. ReduceScatter), the thick amber intra-node NVLink bars exchange and sum shards inside each node while inter-node InfiniBand stays idle. On Node 0, GPU 0 accumulates node partial sum A₀₁ and GPU 1 accumulates B₀₁; on Node 1, GPU 0 accumulates A₂₃ and GPU 1 accumulates B₂₃ (2 of 4 rank contributions per shard, 1 shard of size M/2 per GPU).'},
  {label:'Reduce matching shards at node 1',description:'In Phase 2 (2. Shard AllReduce), only the dashed amber inter-node InfiniBand links are active, carrying 1 shard of size M/2 per GPU instead of the full M payload. Node 0 sends A₀₁ from GPU 0 and B₀₁ from GPU 1 across InfiniBand to matching GPUs on Node 1, forming full 4-rank global sums ΣA (A₀₁₂₃) and ΣB (B₀₁₂₃) on Node 1.'},
  {label:'Return the completed global shards',description:'Completing Phase 2 over the dashed amber InfiniBand links returns the finished 4-rank global sums ΣA to Node 0 GPU 0 and ΣB to Node 0 GPU 1. Both nodes now hold the complete global sums ΣA and ΣB, still partitioned 1 shard (M/2) per local GPU.'},
  {label:'AllGather inside each node',description:'In Phase 3 (3. AllGather), the thick amber intra-node NVLink bars copy finished shard ΣA from GPU 0 to GPU 1 and finished shard ΣB from GPU 1 to GPU 0 inside each of the 2 nodes. All 4 GPUs finish with both green full sums (ΣA and ΣB) without sending any additional bytes over InfiniBand.'},
 ];
 window.MLSysFigureLessons=window.MLSysFigureLessons||{};
 window.MLSysFigureLessons[NAME]=async(root,context)=>{let visualCameraIndex=0;
  if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified artwork/player required');
  const art=context.verifiedArtwork.cloneNode(true);if(art.getAttribute('viewBox')!=='0 0 880 462')throw Error('Hierarchical geometry changed');
  art.classList.add(NAME+'-art');art.setAttribute('role','img');art.setAttribute('aria-label','Original three-phase two-node, two-GPU hierarchical AllReduce');
  const base=art.querySelector('g[transform]'),rects=[...base.querySelectorAll('rect')],texts=[...base.querySelectorAll('text')],tiles=[];
  for(let phase=0;phase<3;phase++)for(let node=0;node<2;node++)for(let gpu=0;gpu<2;gpu++)for(let chunk=0;chunk<2;chunk++){
   const x=59+295*phase+120*gpu+38*chunk,y=128+180*node;
   const shapes=rects.filter(n=>+n.getAttribute('x')===x&&+n.getAttribute('y')>=y&&+n.getAttribute('y')<y+26&&+n.getAttribute('width')===34);
   if(!shapes.length)throw Error('Hierarchical tile mapping changed');
   const group=el('g',{'data-phase':phase,'data-node':node,'data-gpu':gpu,'data-chunk':'AB'[chunk]});
   shapes.forEach(n=>group.append(n));
   const old=texts.find(n=>+n.getAttribute('x')===x+17&&Math.abs(+n.getAttribute('y')-(y+17.9))<.01);
   const text=old||el('text',{x:x+17,y:y+17.9,'text-anchor':'middle','font-size':14.5});const fraction=el('rect',{x:x+1,y:y+13,width:32,height:12,fill:'#d5dde8'});group.append(fraction,text);base.append(group);
   tiles.push({phase,node,gpu,chunk,x,y,group,text,shapes,fraction});
  }
  const carriers=[...rects.filter(n=>[398,518].includes(+n.getAttribute('x'))&&+n.getAttribute('y')>=208&&+n.getAttribute('y')<234),...texts.filter(n=>[415,535].includes(+n.getAttribute('x'))&&Math.abs(+n.getAttribute('y')-225.9)<.01)];
  const phaseOf=i=>i<2?0:i<4?1:2;
  const moving=el('g',{'aria-hidden':'true'});base.append(moving);
  const proof=document.createElement('p');proof.className=NAME+'-proof';
  let stageNode=null,indexNow=0,observer;
  function camera(index){visualCameraIndex=index;root.dataset.visualCameraIndex=String(index);if(!stageNode)return;const scale=art.getBoundingClientRect().width/880,phase=phaseOf(index);stageNode.scrollLeft=Math.max(0,(155+phase*295)*scale-stageNode.clientWidth/2);}
  function contributors(index,t){
   if(t.phase===0&&index===0)return[t.node*2+t.gpu];
   if(t.gpu!==t.chunk&&!(t.phase===2&&index>=4))return[];
   if(t.phase===0||index<2||(index===2&&t.node===0))return[t.node*2,t.node*2+1];return[0,1,2,3];
  }
  function render(index,{stage,details}){narrationIndex=index;
   indexNow=index;if(!stageNode){stageNode=stage;stage.classList.add(NAME+'-stage');stage.tabIndex=0;stage.append(art);details.append(proof);observer=new ResizeObserver(()=>camera(visualCameraIndex));observer.observe(stage);}
   moving.replaceChildren();carriers.forEach(n=>n.style.visibility=index>=2?'visible':'hidden');root.dataset.animationProgress='0';
   for(const t of tiles){const set=contributors(index,t),shown=t.phase<=phaseOf(index),full=set.length===4;
    t.group.style.opacity=shown?'1':'.1';t.group.dataset.contributors=set.join(',');t.text.textContent=!set.length?'':full?'Σ'+'AB'[t.chunk]:'AB'[t.chunk]+sub(set);
    const height=26*set.length/4;t.fraction.style.visibility=set.length&& !full?'visible':'hidden';t.fraction.setAttribute('height',String(Math.max(0,height-1)));t.fraction.setAttribute('y',String(t.y+26-height));
    t.text.setAttribute('fill',full?'#15803d':'#14294E');t.text.setAttribute('font-size','14');
    t.shapes.forEach((n,i)=>{if(+n.getAttribute('height')!==26){n.style.visibility='hidden';return;}n.setAttribute('fill',i===0?(full?'#dcfce7':set.length?'#fff':'#fff'):'none');n.setAttribute('stroke',full?'#16a34a':'#94a3b8');n.setAttribute('stroke-dasharray',!set.length?'3,2':'none');});
   }
   proof.textContent=index===0?'Phase 1 setup · Node 0 (ranks 0, 1) holds A₀, B₀ and A₁, B₁; Node 1 (ranks 2, 3) holds A₂, B₂ and A₃, B₃. Inter-node InfiniBand traffic so far: 0 shards.':index===1?'Phase 1 complete (NVLink ReduceScatter) · Node 0 holds A₀₁ on GPU 0 and B₀₁ on GPU 1; Node 1 holds A₂₃ on GPU 0 and B₂₃ on GPU 1. Each GPU now owns 1 shard of size M / 2.':index===2?'Phase 2 step 1 (InfiniBand Shard AllReduce) · Node 1 combines A₀₁ + A₂₃ → ΣA on GPU 0 and B₀₁ + B₂₃ → ΣB on GPU 1 (all 4 ranks {0,1,2,3}); Node 0 still holds A₀₁ and B₀₁.':index===3?'Phase 2 complete (InfiniBand Shard AllReduce) · Both Node 0 and Node 1 hold ΣA on GPU 0 and ΣB on GPU 1. Each InfiniBand port carried only 1 shard (M / 2) instead of the full M gradient.':'Phase 3 complete (NVLink AllGather) · Every GPU on both nodes holds full sums ΣA and ΣB ({0,1,2,3}). Intra-node NVLink copied the finished shards without adding them a second time.';
   root.dataset.originalState=JSON.stringify({phase:index,nodeContributors:[0,1].map(node=>contributors(index,{phase:phaseOf(index),node,gpu:0,chunk:0}).length),allShards:index===4});camera(index);
  }
  let narrationIndex=0;const narratedSteps=steps.map((step,i)=>({label:step.label,get description(){return narrationIndex===i?step.description:'This action is still in progress. Expected after completion: '+step.description;}}));
const player=window.MLSysOriginalPlayer.create({root,outer:context.outer,layout:'compact',title:'How do local partial sums become a full result on every GPU?',intro:'Follow 2 nodes of 2 GPUs each across 3 hierarchical phases to see how intra-node NVLink ReduceScatter halves the inter-node InfiniBand payload per GPU.',scope:'A 3-phase Hierarchical AllReduce across 2 nodes (Node 0: ranks 0 and 1; Node 1: ranks 2 and 3) with G_node = 2 GPUs per node (4 GPUs total) and a payload of size M split into 2 shards (A and B, each M/2). Phase 1 (ReduceScatter) and Phase 3 (AllGather) use intra-node NVLink; only Phase 2 (Shard AllReduce) crosses inter-node InfiniBand, carrying 1 shard (M/2) per GPU.',steps:narratedSteps,render,
   transition({from,to,onFrame}){
    if(to!==from+1)return;camera(to);const packets=[];
    function packet(label,a,b){const g=el('g',{'data-motion-object':'true'});g.append(el('rect',{x:-15,y:-8,width:30,height:16,rx:2,fill:to>=3?'#dcfce7':'#fef3c7',stroke:'#d97706'}),el('text',{'text-anchor':'middle',y:4,'font-size':12,fill:'#14294E'},label));moving.append(g);packets.push({g,a,b});}
    if(to===1)for(let n=0;n<2;n++){packet('A'+sub([2*n+1]),[165,131+180*n],[145,131+180*n]);packet('B'+sub([2*n]),[145,153+180*n],[165,153+180*n]);}
    if(to===2||to===3){carriers.forEach(n=>n.style.visibility='hidden');for(let g=0;g<2;g++)packet(to===2?'AB'[g]+sub([0,1]):'Σ'+'AB'[g],[390+120*g,to===2?164:278],[390+120*g,to===2?278:164]);}
    if(to===4)for(let n=0;n<2;n++){packet('ΣA',[735,131+180*n],[755,131+180*n]);packet('ΣB',[755,153+180*n],[735,153+180*n]);}
    const arriving=tiles.filter(t=>t.phase===phaseOf(to));arriving.forEach(t=>{t.group.dataset.motionObject='true';t.group.style.opacity='1';});
    onFrame(p=>{root.dataset.animationProgress=String(p);for(const{g,a,b}of packets)g.setAttribute('transform',`translate(${a[0]+(b[0]-a[0])*p},${a[1]+(b[1]-a[1])*p})`);});
   }
  });return{pause:()=>player.pause(),destroy:()=>{observer?.disconnect();player.destroy();}};
 };
})();
