(() => {
 'use strict';
 const NAME="original-v4-03-request-local-prefill-decode", CFG={"question": "When does a sampled token acquire KV state?", "scope": "Original request-local schematic. Three input tiles and two decode passes illustrate ownership; tile counts are schematic, not tensor sizes or timing.", "purpose": "Follow consumed inputs into one growing request-local KV cache, then release that cache before the next independent evaluation.", "steps": [["Observe inputs", "Camera and instruction are inputs to this request. No KV has been written yet."], ["Prefill writes initial KV", "The three schematic input tiles acquire KV state. Prefill samples token 1; that new output has not yet acquired KV."], ["Consume token 1", "Decode pass 1 reads prior KV and consumes token 1, appending its KV. It samples token 2, which has no KV yet."], ["Consume token 2", "Decode pass 2 reads the growing cache and appends KV for consumed token 2. Token 3 is sampled next; it is not another KV entry yet."], ["Independent evaluation", "The old request ends and its KV is released. The next observation requires a new prefill and new KV state."]], "guard": "Snapshots repeat one growing request cache; sampled output acquires KV only on consumption. No KV survives into an independent control evaluation.", "key": "vol4:fig-03-request-local-prefill-decode"}, NS='http://www.w3.org/2000/svg';
 const el=(tag,attrs={},text)=>{const n=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;return n;};
 window.MLSysFigureLessons=window.MLSysFigureLessons||{};
 window.MLSysFigureLessons[NAME]=async(root,context)=>{
  if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
  const art=context.verifiedArtwork.cloneNode(true),vb=art.viewBox.baseVal;
  if(!vb.width||!vb.height)throw Error('Original geometry missing');
  art.classList.add(NAME+'-art');art.setAttribute('role','img');art.setAttribute('aria-label',CFG.question+' Original artwork with synchronized mechanism overlay.');
  const layer=el('svg',{viewBox:'0 0 1000 1000',width:vb.width,height:vb.height,preserveAspectRatio:'none','aria-hidden':'true'});art.append(layer);
  const proof=document.createElement('p');proof.className=NAME+'-proof';
  let mounted=false,current=0,stageNode;
  const tiles=[...art.querySelectorAll('rect')].filter(n=>+n.getAttribute('width')===25);const panels=[...art.querySelectorAll('rect')].filter(n=>+n.getAttribute('width')===180);
  function draw(from,to,p){
   layer.replaceChildren();root.dataset.animationProgress=String(p);
   const phase=p<1?from:to;const live=phase<4;
   tiles.forEach(n=>{const x=+n.getAttribute('x'),col=x<240?0:x<460?1:2;const count=col+3,pos=Math.round((x-(45+220*col))/30);n.style.opacity=live&&phase>=col+1?'1':'.12';n.setAttribute('fill',pos>=3?'#F7EDE3':'#E3F3F5');});
   panels.forEach(n=>n.style.opacity=phase===4&&+n.getAttribute('y')<380?'.22':'1');
   if(from!==to&&p<1&&to<4){const cx=120+220*(to-1),cy=to===1?150:50;const g=el('g',{'data-motion-object':'true',transform:`translate(${cx/680*1000},${(cy+(272-cy)*p)/500*1000})`});g.append(el('rect',{x:-17,y:-25,width:34,height:50,fill:'#F7EDE3',stroke:'#B87333','stroke-width':3}));layer.append(g);}
   proof.textContent=['No request KV exists yet.','KV contains input-prefix state. Sampled token 1 is still an output.','KV = prefix + consumed token 1; sampled token 2 is not cached yet.','KV = prefix + consumed tokens 1 and 2; sampled token 3 is not cached yet.','Old request KV is released. New observation → new prefill → new KV.'][phase];root.dataset.kvConsumed=String(phase===4?0:Math.max(0,phase-1));root.dataset.kvReleased=String(phase===4);
   if(stageNode)stageNode.scrollLeft=Math.max(0,art.getBoundingClientRect().width*([120,120,340,560,560][to]/680)-stageNode.clientWidth/2);
  }
  const steps=CFG.steps.map(([label,description],i)=>({label,get description(){return i!==Number(root.dataset.step)?'Next endpoint will show the following (not yet completed): '+description:description;}}));
  const player=window.MLSysOriginalPlayer.create({root,outer:context.outer,title:CFG.question,scope:CFG.scope,steps,
   render(index,{stage,details}){current=index;root.dataset.step=String(index);if(!mounted){stageNode=stage;stage.tabIndex=0;stage.setAttribute('aria-label','Original diagram; scroll horizontally or use Expand for dense artwork');stage.classList.add(NAME+'-stage');stage.append(art);details.append(proof);mounted=true;}draw(index,index,1);root.dataset.originalState=JSON.stringify({index,kind:CFG.key});},
   transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>draw(from,to,p));}
  });return{pause:()=>player.pause(),destroy:()=>player.destroy()};
 };
})();
