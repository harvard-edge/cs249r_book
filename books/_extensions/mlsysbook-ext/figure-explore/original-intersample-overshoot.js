/* Signed acceleration area and continuous velocity on the original planning SVG. */
(() => {
  'use strict';
  let serial=0;
  window.MLSysFigureLessons=window.MLSysFigureLessons||{};
  window.MLSysFigureLessons['original-intersample-overshoot']=async(root,context)=>{
    const d=window.MLSysOriginalIntersampleData;
    if(!context.verifiedArtwork||!window.MLSysOriginalPlayer||!d?.normalized)throw Error('Verified intersample artwork, computed data, and shared player required');
    const svg=context.verifiedArtwork.cloneNode(true), ns='http://www.w3.org/2000/svg', g=d.geometry;
    const paths=[...svg.querySelectorAll('path')], texts=[...svg.querySelectorAll('text')];
    // This original drawing remains a white paper surface in both chapter themes.
    texts.forEach(n=>{const fill=n.getAttribute('fill');if(fill)n.style.setProperty('fill',fill,'important');});
    const bronze=paths.find(n=>n.getAttribute('d')==='M140 330 L370 270 L600 330');
    const pine=paths.find(n=>n.getAttribute('d')==='M140 360 L370 300 L600 360');
    const positive=[...svg.querySelectorAll('rect')].find(n=>n.getAttribute('x')==='140'&&n.getAttribute('y')==='70'&&n.getAttribute('width')==='230');
    const gainLabel=texts.find(n=>n.textContent==='velocity gained'), peakLabel=texts.find(n=>n.textContent==='hidden peak');
    const panelTitles=texts.filter(n=>n.textContent==='a  Acceleration'||n.textContent==='b  Velocity');
    const peak=[...svg.querySelectorAll('circle')].find(n=>n.getAttribute('cx')==='370'&&n.getAttribute('cy')==='270');
    if(!bronze||!pine||!positive||!gainLabel||!peakLabel||!peak||svg.getAttribute('viewBox')!=='0 0 680 500')throw Error('Intersample artwork changed; retain Diagram until reviewed');
    function node(tag,attrs={},value){const n=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(value!==undefined)n.textContent=value;if(tag==='text'&&attrs.fill)n.style.setProperty('fill',attrs.fill,'important');return n;}
    const defs=svg.querySelector('defs'), prefix='intersample-'+(++serial)+'-';
    const clips=[bronze,pine].map((path,i)=>{const ghost=path.cloneNode(true);ghost.removeAttribute('id');ghost.setAttribute('opacity','.16');path.before(ghost);const clip=node('clipPath',{id:prefix+i,clipPathUnits:'userSpaceOnUse'}),r=node('rect',{x:g.x0-5,y:260,width:5,height:165});clip.append(r);defs.append(clip);path.setAttribute('clip-path',`url(#${prefix+i})`);return r;});
    const negative=node('rect',{x:g.xmid,y:g.acceleration_zero,width:0,height:55,fill:'#F7EDE3'});positive.after(negative);
    const overlay=node('g',{'data-intersample-overlay':''});svg.append(overlay);
    const cursor=node('line',{y1:50,y2:420,stroke:'#007A87','stroke-width':1.2,'stroke-dasharray':'2,3'});overlay.append(cursor);
    const markerBronze=node('circle',{r:5,fill:'#B87333',stroke:'#fff','stroke-width':1.5,'data-intersample-marker':'bronze'}),markerPine=node('circle',{r:5,fill:'#1A4D3E',stroke:'#fff','stroke-width':1.5,'data-intersample-marker':'pine'});
    const bronzeValue=node('text',{'font-size':12,'font-weight':700,fill:'#B87333'}),pineValue=node('text',{'font-size':12,'font-weight':700,fill:'#1A4D3E'});
    const accelerationValue=node('text',{'font-size':12,'font-weight':700,fill:'#B87333'});
    const lossLabel=node('text',{y:159,'text-anchor':'middle','font-size':12,fill:'#B87333'},'velocity lost');
    overlay.append(lossLabel,accelerationValue,markerBronze,markerPine,bronzeValue,pineValue);
    // The four source endpoint checks remain fixed; the filled markers inspect the interpolant.
    const samples=[...svg.querySelectorAll('circle')].filter(n=>n.getAttribute('cx')==='140'||n.getAttribute('cx')==='600');samples.forEach(n=>n.dataset.sampleCheck='passed');
    svg.setAttribute('role','img');svg.setAttribute('aria-label','Original aligned acceleration and velocity diagrams: both sampled bronze endpoints pass, while the continuous peak exceeds the limit; an equal acceleration excursion from the contracted pine baseline reaches the ceiling');
    const canvas=document.createElement('div');canvas.className='oio-canvas';canvas.tabIndex=0;canvas.setAttribute('aria-label','Original acceleration and velocity panels. Scroll horizontally to inspect; the time cursor stays visible. Expand for more space.');canvas.append(svg);
    const key=document.createElement('p');key.className='oio-key';key.textContent=`Dashed lines: red = physical ceiling ${d.limit}; pine = contracted sample ceiling ${d.contracted_endpoint}.`;
    const hint=document.createElement('p');hint.className='oio-inspection-hint';hint.textContent='Swipe the diagram to inspect the endpoints, or choose Expand for more space.';
    const status=document.createElement('p');status.className='oio-status';const equation=document.createElement('p');equation.className='oio-equation';const caveat=document.createElement('p');caveat.className='oio-caveat';caveat.textContent='This establishes only the velocity reserve for the stated acceleration-bounded interpolant. It is not a finite-jerk, tracking, torque, collision, or overall motion-safety certification.';
    root.classList.add('mlsys-original-intersample-overshoot');
    const states=[{t:0,pine:false},{t:.25,pine:false},{t:.5,pine:false},{t:.75,pine:false},{t:1,pine:false},{t:0,pine:true},{t:.5,pine:true},{t:1,pine:true}];
    const steps=[
      ['Check the sampled endpoints','At normalized time t/T = 0, both hollow bronze sample points (at t/T = 0 and t/T = 1) sit at v = 2.5 below the physical velocity ceiling of 3, showing that endpoint checks alone say nothing about continuous velocity across interval 1.'],
      ['Reach the ceiling between samples','Across the first quarter of interval 1 (t/T = 0.25), positive acceleration a = +2 accumulates area Δv = 2(0.25) = 0.5, raising the bronze velocity curve from 2.5 to the dashed red ceiling of 3 before the midpoint.'],
      ['Expose the hidden peak','At the midpoint t/T = 0.5, the full positive acceleration area Δv = a_max T / 2 = 2(1)/2 = 1.0 drives the bronze velocity curve from 2.5 to a hidden peak of 3.5, exceeding the physical ceiling of 3 by 0.5 despite passing both endpoint checks.'],
      ['Reduce velocity with negative area','After the instantaneous sign reversal to a = -2 at t/T = 0.5, negative acceleration removes 0.5 of velocity by t/T = 0.75 (net Δv = 0.5), bringing the bronze curve back down to the ceiling of 3 after its earlier overspeed excursion.'],
      ['Return to the next valid sample','At t/T = 1, the full negative acceleration area (-1.0) cancels the earlier +1.0 velocity gain and returns the bronze curve to v = 2.5; the second sampled endpoint passes, but the interior violation up to 3.5 already occurred.'],
      ['Reserve the full possible rise','Resetting to t/T = 0 with the same acceleration bound of 2 and interval 1, contract the pine sample ceiling by the worst-case rise Δv_max = a_max T / 2 = 1.0 so the sampled endpoints start at 3 - 1 = 2.0 instead of 2.5.'],
      ['Use the same rise from the contracted baseline','At the midpoint t/T = 0.5, the identical positive acceleration area Δv = 1.0 raises the pine velocity curve from its contracted 2.0 baseline to 3.0, touching the physical velocity ceiling of 3 without exceeding it.'],
      ['Complete the bounded comparison','At t/T = 1, the negative acceleration area returns the pine curve to its contracted endpoint of 2.0 while the bronze counterexample (peak 3.5) remains visible, proving that reserving a margin of 1 on the endpoints certifies the continuous interpolant.']
    ].map(([label,description])=>({label,description}));
    steps.forEach((step,i)=>{const description=step.description;Object.defineProperty(step,'description',{get(){return root.dataset.lessonCommittedIndex===String(i)?description:'On completion: '+description;}});});
    const f=v=>String(Number(v.toFixed(4)));
    function relation(value){
      const difference=value-d.limit, tolerance=Number.EPSILON*Math.max(1,Math.abs(value),Math.abs(d.limit))*8;
      return difference>tolerance?'above':difference<-tolerance?'below':'at';
    }
    function display(value,guardLimit=false){
      const rounded=Number(value.toFixed(4)), roundedAway=rounded!==value;
      if(guardLimit&&relation(value)!=='at'&&rounded===d.limit){
        const difference=value-d.limit;
        return {operator:'≈',text:`(${f(d.limit)} ${difference>0?'+':'−'} ${Math.abs(difference).toExponential(2)})`};
      }
      return {operator:roundedAway?'≈':'=',text:f(value)};
    }
    function values(t){const rise=d.acceleration_bound*Math.min(t,d.duration-t);return {rise,bronze:d.candidate_endpoint+rise,pine:d.contracted_endpoint+rise,x:g.x0+(g.x1-g.x0)*t/d.duration};}
    function draw(t,compare){
      const v=values(t), fraction=t/d.duration, first=Math.min(fraction,.5),second=Math.max(0,fraction-.5);
      positive.setAttribute('width',String((g.x1-g.x0)*first));negative.setAttribute('width',String((g.x1-g.x0)*second));gainLabel.setAttribute('opacity',first>=.25?'1':'.2');lossLabel.setAttribute('opacity',second>=.25?'1':'0');lossLabel.setAttribute('x',String(g.xmid+(g.x1-g.x0)*second/2));
      clips[0].setAttribute('width',String(compare?g.x1-g.x0+10:v.x-g.x0+5));clips[1].setAttribute('width',String(compare?v.x-g.x0+5:0));bronze.setAttribute('opacity',compare?'.5':'1');
      cursor.setAttribute('x1',v.x);cursor.setAttribute('x2',v.x);
      const labelSide=v.x>500?-1:1;
      for(const [m,label,value,shown]of[[markerBronze,bronzeValue,v.bronze,!compare],[markerPine,pineValue,v.pine,compare]]){m.setAttribute('cx',v.x);m.setAttribute('cy',g.velocity_offset-g.velocity_scale*value);m.setAttribute('visibility',shown?'visible':'hidden');label.setAttribute('x',v.x+labelSide*11);const y=g.velocity_offset-g.velocity_scale*value;label.setAttribute('y',!compare&&v.x>350&&v.x<450&&y<300?248:y+(compare?19:-12));label.setAttribute('text-anchor',labelSide<0?'end':'start');label.textContent=relation(value)!=='at'&&Number(value.toFixed(4))===d.limit?`${relation(value)==='above'?'>':'<'} ${f(d.limit)}`:f(value);label.setAttribute('visibility',shown?'visible':'hidden');}
      const seenPeak=compare||t>=d.duration/2;peak.setAttribute('opacity',seenPeak?'1':'.15');peakLabel.setAttribute('opacity',seenPeak?'1':'0');
      samples.filter(n=>n.getAttribute('cy')==='360').forEach(n=>n.setAttribute('opacity',compare?'1':'.2'));
      root.dataset.time=String(t);root.dataset.currentVelocity=String(compare?v.pine:v.bronze);root.dataset.rise=String(v.rise);root.dataset.compare=String(compare);root.dataset.endpointChecks='passed';root.dataset.observedOverspeed=String(compare||relation(v.bronze)==='above'||t>=d.duration/2);
      markerBronze.setAttribute('fill',relation(v.bronze)==='above'?'#A51C30':'#B87333');bronzeValue.setAttribute('fill',relation(v.bronze)==='above'?'#A51C30':'#B87333');bronzeValue.style.setProperty('fill',relation(v.bronze)==='above'?'#A51C30':'#B87333','important');
      if(canvas.isConnected){
        const scale=svg.getBoundingClientRect().width/680;canvas.scrollLeft=Math.max(0,v.x*scale-canvas.clientWidth/2);
        const left=canvas.scrollLeft/scale,right=left+canvas.clientWidth/scale;
        panelTitles.forEach(n=>n.setAttribute('x',canvas.scrollWidth>canvas.clientWidth?left+12:30));
        const gainLeft=Math.max(g.x0,left+5),gainRight=Math.min(g.x0+(g.x1-g.x0)*first,right-5);
        gainLabel.setAttribute('x',(gainLeft+gainRight)/2);gainLabel.setAttribute('opacity',gainRight-gainLeft>=95?'1':'0');
        const lossLeft=Math.max(g.xmid,left+5),lossRight=Math.min(g.xmid+(g.x1-g.x0)*second,right-5);
        lossLabel.setAttribute('x',(lossLeft+lossRight)/2);lossLabel.setAttribute('opacity',lossRight-lossLeft>=95?'1':'0');
      }
      const acceleration=t<d.duration/2?d.acceleration_bound:-d.acceleration_bound;
      accelerationValue.textContent=`a = ${acceleration>0?'+':''}${f(acceleration)}`;accelerationValue.setAttribute('x',v.x+labelSide*11);accelerationValue.setAttribute('text-anchor',labelSide<0?'end':'start');accelerationValue.setAttribute('y',g.acceleration_zero-g.acceleration_scale*acceleration-12);
      const current=compare?v.pine:v.bronze, currentRelation=relation(current), shownTime=display(t/d.duration),shownVelocity=display(current,true),shownRise=display(v.rise);
      const classification=currentRelation==='above'?'ABOVE ceiling':currentRelation==='at'?'at ceiling':'below ceiling';
      status.textContent=`Normalized t/T ${shownTime.operator} ${shownTime.text} · ${compare?'pine':'bronze'} v ${shownVelocity.operator} ${shownVelocity.text} (${classification}) · sampled endpoints: pass`;
      equation.textContent=`Signed acceleration area: Δv ${shownRise.operator} ${shownRise.text}; v = ${f(compare?d.contracted_endpoint:d.candidate_endpoint)} + Δv ${shownVelocity.operator} ${shownVelocity.text}. ${compare?'Same acceleration bound and interval.':''}`;
      root.dataset.currentRelation=currentRelation;
      status.dataset.velocity=String(current);status.dataset.time=String(t);equation.dataset.velocity=String(current);equation.dataset.rise=String(v.rise);
      return v;
    }
    function render(index,{stage,details}){root.dataset.lessonCommittedIndex=String(index);
      if(!canvas.isConnected)stage.append(canvas,key,hint);if(details&&!status.isConnected)details.append(status,equation,caveat);
      const s=states[index],v=draw(s.t,s.pine);
      if(index===0)equation.textContent=`Bronze samples: ${f(d.candidate_endpoint)} ≤ ${f(d.limit)}. Interior extrema have not been checked by these two tests.`;
      else if(index===5)equation.textContent=`Reserve = aT/2 = ${f(d.reserve)}; contracted sample ceiling = ${f(d.limit)} − ${f(d.reserve)} = ${f(d.contracted_endpoint)}.`;
      else if(index===4)equation.append(' The earlier peak still exceeded the ceiling.');
      root.dataset.intersampleStep=String(index);
      const scale=svg.getBoundingClientRect().width/680;canvas.scrollLeft=Math.max(0,v.x*scale-canvas.clientWidth/2);
    }
    function transition({from,to,onFrame}){if(to!==from+1||states[from].pine!==states[to].pine)return;const a=states[from],b=states[to];onFrame(p=>draw(a.t+(b.t-a.t)*p,b.pine));}
    return window.MLSysOriginalPlayer.create({root,outer:context.outer,layout:'compact',title:'Can both samples pass while velocity exceeds its limit?',intro:'Trace how a bang-bang acceleration pulse of ±2 over interval 1 raises velocity by 1.0 past the limit of 3 unless endpoints reserve a margin of 1.',scope:'Normalized illustrative values (not machine measurements): sample interval T = 1, acceleration bound a_max = 2 (bang-bang +2 on [0, 0.5] and -2 on [0.5, 1] with instantaneous sign reversal and no finite jerk bound), physical velocity ceiling v_max = 3, uncontracted bronze sample endpoints v = 2.5 (peak 3.5), velocity reserve Δv_max = a_max T / 2 = 1.0, and contracted pine sample ceiling v_max - Δv_max = 2.0 (peak 3.0).',steps,render,transition});
  };
})();
