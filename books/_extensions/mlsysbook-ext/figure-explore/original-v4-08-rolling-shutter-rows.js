(() => {
 'use strict';
 const NAME="original-v4-08-rolling-shutter-rows", CFG={"question": "How does row timing differ from exposure blur?", "scope": "Canonical navigation-camera plot and its RollingShutterSkew LEGO inputs. A cursor samples rows; playback duration does not represent acquisition time.", "purpose": "Follow each row\u2019s changing sample position while keeping its separate exposure-width blur visible.", "steps": [["Top row", "The first row is the time origin for scanline skew. Its own exposure still spans the full blur band."], ["Middle rows", "Later rows observe the upright from farther along the aisle. The band width is within-row exposure blur; the centerline displacement is between-row skew."], ["Bottom row", "The last row reaches the canonical skew endpoint. This offset and the per-row blur arise from distinct time intervals; neither is a new localization allowance."]], "guard": "Preserve original axes, band and allowance. Row readout skew is distinct from exposure blur; do not add the two as if they were independent measured errors.", "key": "vol4:fig-08-rolling-shutter-rows"}, NS='http://www.w3.org/2000/svg';
 const el=(tag,attrs={},text)=>{const n=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;return n;};
 window.MLSysFigureLessons=window.MLSysFigureLessons||{};
 window.MLSysFigureLessons[NAME]=async(root,context)=>{
  if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
  const art=context.verifiedArtwork.cloneNode(true),vb=art.viewBox.baseVal;
  if(!vb.width||!vb.height)throw Error('Original geometry missing');
  art.classList.add(NAME+'-art');art.setAttribute('role','img');art.setAttribute('aria-label',CFG.question+' Original artwork with synchronized mechanism overlay.');
  const layer=el('svg',{viewBox:'0 0 1000 1000',width:vb.width,height:vb.height,preserveAspectRatio:'none','aria-hidden':'true'});art.append(layer);
  const scene=el('svg',{viewBox:'0 0 680 185',role:'img','aria-label':'Synchronized mechanism assembly',class:NAME+'-scene'});const proof=document.createElement('p');proof.className=NAME+'-proof';
  let mounted=false,current=0,stageNode;
  const D={
  "rows": 3040,
  "row_ms": 0.005460526315789474,
  "v_mm_ms": 1.5,
  "blur_mm": 24.0,
  "exposure_ms": 16.0
}
;
  function draw(from,to,p){
   layer.replaceChildren();root.dataset.animationProgress=String(p);
   const f=(from+(to-from)*p)/2,row=Math.round((D.rows-1)*f),t=row*D.row_ms,x=t*D.v_mm_ms,px=286+x*12.45,py=153.2+(837.6-153.2)*(row+D.rows*.03)/(D.rows*1.06);layer.append(el('line',{x1:112,x2:980,y1:py,y2:py,stroke:'#B87333','stroke-width':2,'stroke-dasharray':'5 4','data-motion-object':'true'}),el('line',{x1:px-D.blur_mm*12.45/2,x2:px+D.blur_mm*12.45/2,y1:py,y2:py,stroke:'#007A87','stroke-width':10,'stroke-opacity':.5,'data-motion-object':'true'}),el('circle',{cx:px,cy:py,r:7,fill:'#B87333',stroke:'#fff','stroke-width':2,'data-motion-object':'true'}));proof.textContent=`Row ${row}: read offset = row × ${D.row_ms.toFixed(6)} ms = ${t.toFixed(3)} ms; camera displacement = ${x.toFixed(2)} mm. Within-row blur stays ${D.blur_mm.toFixed(1)} mm over ${D.exposure_ms.toFixed(1)} ms exposure.`;root.dataset.row=String(row);root.dataset.readMs=String(t);root.dataset.displacementMm=String(x);
   scene.replaceChildren();const cameraX=35+f*180;scene.append(el('text',{x:12,y:18,'font-size':13,fill:'#1A202C'},'Camera travels while rows accumulate (schematic geometry)'));scene.append(el('line',{x1:25,x2:310,y1:110,y2:110,stroke:'#4A5568'}),el('rect',{x:cameraX,y:80,width:36,height:24,fill:'#E3F3F5',stroke:'#007A87','data-motion-object':'true'}),el('circle',{cx:cameraX+30,cy:92,r:6,fill:'#007A87'}),el('line',{x1:380,x2:380,y1:32,y2:125,stroke:'#1A4D3E','stroke-width':8}),el('line',{x1:cameraX+36,x2:380,y1:92,y2:35+85*f,stroke:'#B87333','stroke-width':2,'data-motion-object':'true'}),el('text',{x:340,y:148,'font-size':12,fill:'#1A202C'},'rack upright'),el('text',{x:20,y:148,'font-size':12,fill:'#1A202C'},`travel ${x.toFixed(2)} mm`),el('text',{x:490,y:35,'font-size':12,fill:'#1A202C'},'representative rows'));for(let j=0;j<16;j++){const ff=j/15;if(ff<=f)scene.append(el('line',{x1:510+ff*D.v_mm_ms*D.row_ms*(D.rows-1),x2:510+ff*D.v_mm_ms*D.row_ms*(D.rows-1),y1:45+j*6,y2:50+j*6,stroke:'#007A87','stroke-width':5,'data-assembled-row':j}));}scene.append(el('text',{x:470,y:161,'font-size':11,fill:'#4A5568'},'retained samples shear the post'));
   if(root.clientWidth<500){scene.setAttribute('viewBox','0 0 320 195');scene.replaceChildren();scene.append(el('text',{x:8,y:20,'font-size':13,fill:'#1A202C'},'Camera motion → retained row samples'));const cx=25+f*70;scene.append(el('rect',{x:cx,y:75,width:27,height:20,fill:'#E3F3F5',stroke:'#007A87','data-motion-object':'true'}),el('line',{x1:165,x2:165,y1:40,y2:130,stroke:'#1A4D3E','stroke-width':6}),el('line',{x1:cx+27,x2:165,y1:85,y2:40+90*f,stroke:'#B87333','data-motion-object':'true'}),el('text',{x:15,y:153,'font-size':12,fill:'#1A202C'},'camera'),el('text',{x:139,y:153,'font-size':12,fill:'#1A202C'},'upright'),el('text',{x:225,y:153,'font-size':12,fill:'#1A202C'},'rows'));for(let j=0;j<16;j++)if(j/15<=f)scene.append(el('line',{x1:240+j/15*25,x2:240+j/15*25,y1:40+j*6,y2:45+j*6,stroke:'#007A87','stroke-width':4}));scene.append(el('text',{x:8,y:179,'font-size':12,fill:'#1A202C'},`travel ${x.toFixed(2)} mm; representative rows`));}else scene.setAttribute('viewBox','0 0 680 185');
   if(stageNode)stageNode.scrollLeft=Math.max(0,art.getBoundingClientRect().width*(px/1000)-stageNode.clientWidth/2);
  }
  const steps=CFG.steps.map(([label,description],i)=>({label,get description(){return i!==Number(root.dataset.step)?'Next endpoint will show the following (not yet completed): '+description:description;}}));
  const player=window.MLSysOriginalPlayer.create({root,outer:context.outer,title:CFG.question,scope:CFG.scope,steps,
   render(index,{stage,details}){current=index;root.dataset.step=String(index);if(!mounted){const artScroll=document.createElement('div');artScroll.className=NAME+'-art-scroll';stageNode=artScroll;stage.tabIndex=0;stage.setAttribute('aria-label','Original diagram; scroll horizontally or use Expand for dense artwork');stage.classList.add(NAME+'-stage');artScroll.append(art);stage.append(artScroll,scene);details.append(proof);mounted=true;}draw(index,index,1);root.dataset.originalState=JSON.stringify({index,kind:CFG.key});},
   transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>draw(from,to,p));}
  });return{pause:()=>player.pause(),destroy:()=>player.destroy()};
 };
})();
