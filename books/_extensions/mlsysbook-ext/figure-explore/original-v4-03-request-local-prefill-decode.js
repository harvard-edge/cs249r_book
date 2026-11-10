(() => {
 'use strict';
 const NAME="original-v4-03-request-local-prefill-decode", CFG={"question": "When does a sampled token acquire KV state?", "intro": "Follow 3 input prefix tiles and 2 decode passes as one intent evaluation grows its request-local KV cache before releasing it.", "scope": "Request-local KV cache lifecycle within one single-observation intent-model evaluation. Three schematic prefix tiles represent camera patches plus instruction tokens during compute-bound prefill, and two sequential decode passes illustrate bandwidth-bound autoregressive generation; tile counts are schematic rather than full tensor dimensions.", "purpose": "Follow consumed inputs into one growing request-local KV cache, then release that cache before the next independent evaluation.", "steps": [["Observe inputs", "Camera patches and instruction tokens arrive at the prefill block for the current control evaluation; all 3 snapshot panels remain dimmed because 0 KV tiles have been written."], ["Prefill writes initial KV", "Prefill processes the camera and instruction prefix in parallel, lighting the 3 teal input KV tiles in snapshot 0 and sampling token 1 at the top; token 1 is still an unconsumed output and holds no KV entry."], ["Consume token 1", "Decode pass 1 feeds sampled token 1 back as an input, reads the 3 prefix KV tiles, and appends 1 copper KV tile in snapshot 1 (4 tiles total) before sampling token 2."], ["Consume token 2", "Decode pass 2 reads all 4 prior KV tiles and appends a second copper KV tile for consumed token 2 in snapshot 2 (5 tiles total), then samples token 3 as the next output."], ["Independent evaluation", "When the current intent evaluation finishes, the upper cache panels dim and release all 5 KV tiles; in the lower strip, the next camera observation starts a fresh prefill and builds a new KV cache."]], "guard": "Snapshots repeat one growing request cache; sampled output acquires KV only on consumption. No KV survives into an independent control evaluation.", "key": "vol4:fig-03-request-local-prefill-decode"}, NS='http://www.w3.org/2000/svg';
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
  // Repaint the exact paused frame after a width/context change, never on scroll.
  let lastFrame=null, resizeFrame=0, resizeGeometry='';
  const resizeObserver=new ResizeObserver(()=>{
   cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(()=>{
    if(!stageNode||!lastFrame||!root.isConnected)return;
    const geometry=[root.clientWidth,stageNode.clientWidth,art.getBoundingClientRect().width,!!root.closest('dialog')].join(':');
    if(geometry===resizeGeometry)return;resizeGeometry=geometry;draw(...lastFrame);
   });
  });resizeObserver.observe(root);
  const tiles=[...art.querySelectorAll('rect')].filter(n=>+n.getAttribute('width')===25);const panels=[...art.querySelectorAll('rect')].filter(n=>+n.getAttribute('width')===180);
  function draw(from,to,p){
   lastFrame=[from,to,p];
   layer.replaceChildren();root.dataset.animationProgress=String(p);
   const phase=p<1?from:to;const live=phase<4;
   tiles.forEach(n=>{const x=+n.getAttribute('x'),col=x<240?0:x<460?1:2;const count=col+3,pos=Math.round((x-(45+220*col))/30);n.style.opacity=live&&phase>=col+1?'1':'.12';n.setAttribute('fill',pos>=3?'#F7EDE3':'#E3F3F5');});
   panels.forEach(n=>n.style.opacity=phase===4&&+n.getAttribute('y')<380?'.22':'1');
   if(from!==to&&p<1&&to<4){const cx=120+220*(to-1),cy=to===1?150:50;const g=el('g',{'data-motion-object':'true',transform:`translate(${cx/680*1000},${(cy+(272-cy)*p)/500*1000})`});g.append(el('rect',{x:-17,y:-25,width:34,height:50,fill:'#F7EDE3',stroke:'#B87333','stroke-width':3}));layer.append(g);}
   proof.textContent=['Request state: 0 cached KV tiles · 0 decode passes completed · prefill pending for current observation.','Request state: 3 prefix KV tiles cached (snapshot 0) · sampled token 1 emitted · 0 action tokens cached in KV.','Request state: 4 KV tiles cached in snapshot 1 (3 prefix + 1 consumed token) · sampled token 2 emitted without KV.','Request state: 5 KV tiles cached in snapshot 2 (3 prefix + 2 consumed tokens) · sampled token 3 emitted without KV.','Request state: 0 KV tiles retained across evaluations · prior 5-tile cache released · next observation starts fresh prefill.'][phase];root.dataset.kvConsumed=String(phase===4?0:Math.max(0,phase-1));root.dataset.kvReleased=String(phase===4);
   if(stageNode)stageNode.scrollLeft=Math.max(0,art.getBoundingClientRect().width*([120,120,340,560,560][to]/680)-stageNode.clientWidth/2);
  }
  const steps=CFG.steps.map(([label,description],i)=>({label,get description(){return i!==Number(root.dataset.step)?'Next endpoint will show the following (not yet completed): '+description:description;}}));
  const player=window.MLSysOriginalPlayer.create({root,outer:context.outer,layout:'compact',intro:CFG.intro,title:CFG.question,scope:CFG.scope,steps,
   render(index,{stage,details}){current=index;root.dataset.step=String(index);if(!mounted){stageNode=stage;stage.tabIndex=0;stage.setAttribute('aria-label','Original diagram; scroll horizontally or use Expand for dense artwork');stage.classList.add(NAME+'-stage');stage.append(art);details.append(proof);mounted=true;}draw(index,index,1);root.dataset.originalState=JSON.stringify({index,kind:CFG.key});},
   transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>draw(from,to,p));}
  });return{pause:()=>player.pause(),destroy:()=>{resizeObserver.disconnect();cancelAnimationFrame(resizeFrame);player.destroy();}};
 };
})();
