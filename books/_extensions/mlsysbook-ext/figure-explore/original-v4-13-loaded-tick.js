(() => {
 'use strict';
 const NAME="original-v4-13-loaded-tick", CFG={"question": "Does fitting within a period establish the timing premise?", "scope": "Canonical loaded-tick FIFO model, with all original rows and thresholds. Aggregate interleaving and one-channel cases retain their original assumptions. Playback duration is illustrative; each operation can be inspected regardless of its modeled duration.", "purpose": "Accumulate queued wait before the read and unchanged later computation, then compare the correct threshold.", "steps": [["Unloaded execution", "The isolated sub-costs establish the unloaded tick. The period, deadline, and declared WCET are different limits."], ["Loaded snapshot is queued", "The representative sense-allowance row begins with a snapshot waiting behind queued traffic. The permission core cannot use that snapshot yet."], ["Drain the queue", "The FIFO backlog completes before the snapshot read can begin. The snapshot remains a distinct waiting object."], ["Read and retain the snapshot", "The snapshot has reached the permission core and remains available to constraint evaluation. No completed drive frame exists yet."], ["Evaluate constraints and finish the frame", "The unchanged later work completes. Only now is the drive frame ready; the snapshot has served its consumer."], ["Declared WCET", "The next row reaches the largest backlog count admitted by the declared WCET. One more backlog would refute that execution bound even inside the period."], ["Deadline allocation", "The largest deadline backlog row already exceeds declared WCET. Blocking and interference still reserve the shaded allocation before the deadline."], ["One-channel mapping", "Confining traffic to one channel changes the service premise. Far fewer backlogs approach WCET. All rows fit the period; that does not validate the placement."]], "guard": "Period pass is not timing-premise pass. Keep FIFO queueing assumptions, sense read, WCET, full response deadline and channel mapping distinct.", "key": "vol4:fig-13-loaded-tick"}, NS='http://www.w3.org/2000/svg';
 const el=(tag,attrs={},text)=>{const n=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;return n;};
 window.MLSysFigureLessons=window.MLSysFigureLessons||{};
 window.MLSysFigureLessons[NAME]=async(root,context)=>{
  if(!context.verifiedArtwork||!window.MLSysOriginalPlayer)throw Error('Verified original artwork and shared player required');
  const art=context.verifiedArtwork.cloneNode(true),vb=art.viewBox.baseVal;
  if(!vb.width||!vb.height)throw Error('Original geometry missing');
  art.classList.add(NAME+'-art');art.setAttribute('role','img');art.setAttribute('aria-label',CFG.question+' Original artwork with synchronized mechanism overlay.');
  const layer=el('svg',{viewBox:'0 0 1000 1000',width:vb.width,height:vb.height,preserveAspectRatio:'none','aria-hidden':'true'});art.append(layer);
  const scene=el('svg',{viewBox:'0 0 680 235',role:'img','aria-label':'Synchronized mechanism assembly',class:NAME+'-scene'});const proof=document.createElement('p');proof.className=NAME+'-proof';
  let mounted=false,current=0,stageNode;
  const D={
  "rows": [
    {
      "label": "unloaded",
      "wait": 0,
      "read": 40.0,
      "compute": 95.0
    },
    {
      "label": "sense allowance",
      "wait": 39.46744117647059,
      "read": 0.02007843137254902,
      "compute": 95.0
    },
    {
      "label": "declared WCET",
      "wait": 154.1109607843137,
      "read": 0.02007843137254902,
      "compute": 95.0
    },
    {
      "label": "deadline allocation",
      "wait": 214.25182352941175,
      "read": 0.02007843137254902,
      "compute": 95.0
    },
    {
      "label": "one channel",
      "wait": 141.9146568627451,
      "read": 0.3212549019607843,
      "compute": 95.0
    }
  ],
  "wcet_us": 250.0,
  "deadline_us": 400.0,
  "reserve_us": 90.0,
  "period_us": 1000.0
}
;
  function draw(from,to,p){
   layer.replaceChildren();root.dataset.animationProgress=String(p);
   const row=to===0?0:to<=4?1:to-3,r=D.rows[row],x0=150,scale=.81,y=[306,429,552,675,798][row],end=r.wait+r.read+r.compute;const stops=[0,r.wait,r.wait+r.read,end],target=to>=1&&to<=4?stops[to-1]:end,start=from===to?target:from>=1&&from<=4&&to<=4?stops[from-1]:0,t=start+(target-start)*p;layer.append(el('line',{x1:x0+scale*t,x2:x0+scale*t,y1:80,y2:920,stroke:'#B87333','stroke-width':2,'stroke-dasharray':'6 5','data-motion-object':'true'}),el('line',{x1:x0,x2:x0+scale*Math.min(t,r.wait),y1:y,y2:y,stroke:'#007A87','stroke-width':17,'stroke-opacity':.45,'data-motion-object':'true'}));if(t>r.wait)layer.append(el('line',{x1:x0+scale*r.wait,x2:x0+scale*Math.min(t,r.wait+r.read),y1:y,y2:y,stroke:'#007A87','stroke-width':17,'data-motion-object':'true'}));if(t>r.wait+r.read)layer.append(el('line',{x1:x0+scale*(r.wait+r.read),x2:x0+scale*t,y1:y,y2:y,stroke:'#B87333','stroke-width':17,'stroke-opacity':.65,'data-motion-object':'true'}));proof.textContent=`${r.label}: cursor ${t.toFixed(3)} µs into this row (${t<r.wait?'queued wait':t<r.wait+r.read?'snapshot read':t<end?'later computation':'complete'}). Full-row modeled summary: queue ${r.wait.toFixed(3)} µs + snapshot ${r.read.toFixed(3)} µs + later work ${r.compute.toFixed(0)} µs = ${end.toFixed(3)} µs. Declared WCET ${end<=D.wcet_us?'holds':'is exceeded'}; deadline including B + I ${end+D.reserve_us<=D.deadline_us?'holds':'is exceeded'}; period ${end<D.period_us?'holds':'is exceeded'}. Period pass alone does not establish the timing premise.`;root.dataset.row=String(row);root.dataset.tickEndUs=String(end);root.dataset.wcetPass=String(end<=D.wcet_us);
   scene.replaceChildren();scene.append(el('text',{x:12,y:18,'font-size':13,fill:'#1A202C'},'One permission tick: memory service precedes the unchanged work'));const labels=['queued tile / camera / telemetry','snapshot → permission core','constraints → drive-frame write'];const boundaries=[0,r.wait,r.wait+r.read,end];for(let j=0;j<3;j++){const xx=12+j*222,done=t>=boundaries[j+1],active=t>=boundaries[j]&&t<boundaries[j+1];scene.append(el('rect',{x:xx,y:40,width:208,height:100,fill:done?'#E8F0EC':active?'#E3F3F5':'#F8FAFC',stroke:active?'#007A87':'#4A5568'}),el('text',{x:xx+8,y:59,'font-size':10,fill:'#1A202C'},labels[j]),el('text',{x:xx+8,y:86,'font-size':11,fill:'#1A202C'},j===0?`FIFO wait ${r.wait.toFixed(3)} µs`:j===1?`read ${r.read.toFixed(3)} µs`:`later work ${r.compute.toFixed(0)} µs`),el('text',{x:xx+8,y:112,'font-size':12,fill:done?'#1A4D3E':'#4A5568'},done?'complete':active?'owns service now':'blocked by predecessor'));if(active)scene.append(el('rect',{x:xx+8,y:121,width:190*Math.min(1,(t-boundaries[j])/Math.max(.0001,boundaries[j+1]-boundaries[j])),height:6,fill:'#007A87','data-motion-object':'true'}));}scene.append(el('text',{x:12,y:163,'font-size':12,fill:'#1A202C'},`cursor ${t.toFixed(3)} µs / modeled row ${end.toFixed(3)} µs; all thresholds above retain their original meaning`));
   scene.append(el('line',{x1:20,x2:650,y1:205,y2:205,stroke:'#94A3B8','stroke-width':2}));for(let j=0;j<3;j++){if(r.wait>0&&t<r.wait*(j+1)/3)scene.append(el('rect',{x:20+j*45,y:190,width:38,height:28,fill:'#E3F3F5',stroke:'#007A87','data-queued-object':j}),el('text',{x:25+j*45,y:208,'font-size':10,fill:'#1A202C'},['tile','cam','log'][j]));}let sx=t<r.wait?185:t<r.wait+r.read?185+130*(t-r.wait)/r.read:315;scene.append(el('rect',{x:sx,y:190,width:70,height:28,fill:'#E3F3F5',stroke:'#007A87','data-motion-object':'true'}),el('text',{x:sx+5,y:208,'font-size':10,fill:'#1A202C'},t<r.wait?'snapshot Q':'snapshot'));if(t>=r.wait+r.read)scene.append(el('rect',{x:535+60*Math.min(1,(t-r.wait-r.read)/r.compute),y:190,width:75,height:28,fill:'#F7EDE3',stroke:'#B87333','data-motion-object':'true'}),el('text',{x:540+60*Math.min(1,(t-r.wait-r.read)/r.compute),y:208,'font-size':10,fill:'#1A202C'},t<end?'frame pending':'drive frame'));
   if(root.clientWidth<500){scene.setAttribute('viewBox','0 0 320 335');scene.replaceChildren();scene.append(el('text',{x:8,y:18,'font-size':13,fill:'#1A202C'},'FIFO → snapshot → constraints / frame'));for(let j=0;j<3;j++){const yy=38+j*57,done=t>=boundaries[j+1],active=t>=boundaries[j]&&t<boundaries[j+1];scene.append(el('rect',{x:8,y:yy,width:304,height:49,fill:done?'#E8F0EC':active?'#E3F3F5':'#F8FAFC',stroke:'#4A5568'}),el('text',{x:16,y:yy+18,'font-size':12,fill:'#1A202C'},['queued tile / camera / telemetry','snapshot → permission core','constraints → drive-frame write'][j]),el('text',{x:16,y:yy+36,'font-size':12,fill:'#1A202C'},(done?'complete':active?'owns service':'blocked')+' · '+[r.wait.toFixed(3),r.read.toFixed(3),r.compute.toFixed(0)][j]+' µs'));}scene.append(el('text',{x:8,y:228,'font-size':12,fill:'#1A202C'},`cursor ${t.toFixed(3)} / ${end.toFixed(3)} µs`));scene.append(el('line',{x1:8,x2:208,y1:261,y2:261,stroke:'#94A3B8'}),el('line',{x1:175,x2:175,y1:261,y2:299,stroke:'#94A3B8'}),el('line',{x1:175,x2:307,y1:299,y2:299,stroke:'#94A3B8'}));for(let j=0;j<3;j++)if(r.wait>0&&t<r.wait*(j+1)/3)scene.append(el('rect',{x:8+j*41,y:247,width:37,height:28,fill:'#E3F3F5',stroke:'#007A87'}),el('text',{x:11+j*41,y:265,'font-size':10,fill:'#1A202C'},['tile','cam','log'][j]));const sy=t<r.wait?247:t<r.wait+r.read?247+38*(t-r.wait)/r.read:285;scene.append(el('rect',{x:140,y:sy,width:70,height:28,fill:'#E3F3F5',stroke:'#007A87','data-motion-object':'true'}),el('text',{x:144,y:sy+18,'font-size':11,fill:'#1A202C'},'snapshot'),el('text',{x:130,y:329,'font-size':10,fill:'#1A202C'},'permission core'));if(t>=r.wait+r.read)scene.append(el('rect',{x:237,y:285,width:75,height:28,fill:'#F7EDE3',stroke:'#B87333','data-motion-object':'true'}),el('text',{x:241,y:303,'font-size':10,fill:'#1A202C'},t<end?'frame pending':'drive frame'));}else scene.setAttribute('viewBox','0 0 680 235');
   if(stageNode)stageNode.scrollLeft=Math.max(0,art.getBoundingClientRect().width*((x0+scale*t)/1000)-stageNode.clientWidth/2);
  }
  const steps=CFG.steps.map(([label,description],i)=>({label,get description(){return i!==Number(root.dataset.step)?'Next endpoint will show the following (not yet completed): '+description:description;}}));
  const player=window.MLSysOriginalPlayer.create({root,outer:context.outer,title:CFG.question,scope:CFG.scope,steps,
   render(index,{stage,details}){current=index;root.dataset.step=String(index);if(!mounted){const artScroll=document.createElement('div');artScroll.className=NAME+'-art-scroll';stageNode=artScroll;stage.tabIndex=0;stage.setAttribute('aria-label','Original diagram; scroll horizontally or use Expand for dense artwork');stage.classList.add(NAME+'-stage');artScroll.append(art);stage.append(artScroll,scene);details.append(proof);mounted=true;}draw(index,index,1);root.dataset.originalState=JSON.stringify({index,kind:CFG.key});},
   transition({from,to,onFrame}){if(to!==from+1)return;onFrame(p=>draw(from,to,p));}
  });return{pause:()=>player.pause(),destroy:()=>player.destroy()};
 };
})();
