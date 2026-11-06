(() => {
 'use strict';
 const NAME='original-v2-hierarchical-allreduce',NS='http://www.w3.org/2000/svg';
 const el=(tag,attrs={},text)=>{const n=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;return n;};
 const sub=set=>set.map(i=>'₀₁₂₃'[i]).join('');
 const steps=[
  {label:'Begin with local contributions',description:'The same four GPUs are repeated across the three phase panels. Initially every GPU contributes to both A and B.'},
  {label:'ReduceScatter inside each node',description:'Exchange and add the node’s local contributions. After completion, GPU 0 owns that node’s A partial sum and GPU 1 owns its B partial sum.'},
  {label:'Reduce matching shards at node 1',description:'Send node 0’s partial shards to their matching GPUs at node 1. After arrival, node 1 combines disjoint contributions into global sums; node 0 retains its partials.'},
  {label:'Return the completed global shards',description:'Copy the completed A and B sums back to node 0. This illustrative reduction-and-return completes the matching-shard AllReduce; returning sums are not added again.'},
  {label:'AllGather inside each node',description:'Copy the finished shards across the local links. AllGather adds nothing: every GPU finishes with the same global A and B sums.'},
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
   proof.textContent=index===0?'Subscripts identify the contributing GPUs: node 0 contains ranks 0/1; node 1 contains ranks 2/3.':index===1?'Node 0 partials include {0,1}; node 1 partials include {2,3}. A and B are split across the two local GPUs.':index===2?'Node 1 has full {0,1,2,3} sums; node 0 retains its {0,1} partials until the return.':index===3?'Both nodes now hold global sums, still split between their local GPUs.':'Every GPU owns ΣA and ΣB. The last phase copied complete sums without adding them again.';
   root.dataset.originalState=JSON.stringify({phase:index,nodeContributors:[0,1].map(node=>contributors(index,{phase:phaseOf(index),node,gpu:0,chunk:0}).length),allShards:index===4});camera(index);
  }
  let narrationIndex=0;const narratedSteps=steps.map((step,i)=>({label:step.label,get description(){return narrationIndex===i?step.description:'This action is still in progress. Expected after completion: '+step.description;}}));
const player=window.MLSysOriginalPlayer.create({root,outer:context.outer,title:'How do local partial sums become a full result on every GPU?',scope:'The same two nodes and four GPUs repeat across three phase panels. Subscripts name rank contributions. The inter-node AllReduce uses an illustrative reduction-and-return; no particular library schedule or hardware timing is claimed. Pale tiles preview later phases.',steps:narratedSteps,render,
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
