import"./chunk-TCCFYFTB-ClOYb5MI.js";import"./main-B0OX4z33.js";import"./purify.es-DBmVz-Wl.js";import{n as W}from"./ordinal-K0JYqmCT.js";import"./src-nKy6a3tt.js";import{r as h}from"./path-LDcaQ64B.js";import{p as k}from"./math-Bf0RgaKx.js";import{t as F}from"./arc-ONRhTAm2.js";import{t as _}from"./array-BCwZq3ts.js";import{i as B,p as P}from"./chunk-S3R3BYOJ-DGKPoF4U.js";import{n as p,r as M}from"./src-CmKhyEBC.js";import{B as L,C as V,U as j,_ as U,a as q,b as G,c as H,d as I,v as J,z as K}from"./chunk-ABZYJK2D-BInZW7OW.js";import{t as Q}from"./chunk-EXTU4WIE-Cwo9-IiR.js";import"./dist-DxxvVPQH.js";import"./chunk-UMXZTB3W-DsE6dCbX.js";import"./chunk-4F5CHEZ2-BBsStyPF.js";import"./chunk-FRFDVMJY-D6StbPqE.js";import"./chunk-SJTYNZTY-BNaCydHS.js";import"./chunk-PL6DKKU2-x3Qoia3Y.js";import"./chunk-TQ3KTPDO-DfYTN8Ct.js";import"./chunk-B2363JML-CPCjJ6tH.js";import{t as X}from"./chunk-4BX2VUAB-mPGRpph5.js";import{t as Y}from"./mermaid-parser.core-BQULBKwL.js";function Z(t,a){return a<t?-1:a>t?1:a>=t?0:NaN}function tt(t){return t}function et(){var t=tt,a=Z,d=null,o=h(0),s=h(k),x=h(0);function n(e){var l,i=(e=_(e)).length,m,b,$=0,y=Array(i),u=Array(i),v=+o.apply(this,arguments),A=Math.min(k,Math.max(-k,s.apply(this,arguments)-v)),f,T=Math.min(Math.abs(A)/i,x.apply(this,arguments)),w=T*(A<0?-1:1),c;for(l=0;l<i;++l)(c=u[y[l]=l]=+t(e[l],l,e))>0&&($+=c);for(a==null?d!=null&&y.sort(function(g,S){return d(e[g],e[S])}):y.sort(function(g,S){return a(u[g],u[S])}),l=0,b=$?(A-i*w)/$:0;l<i;++l,v=f)m=y[l],c=u[m],f=v+(c>0?c*b:0)+w,u[m]={data:e[m],index:l,value:c,startAngle:v,endAngle:f,padAngle:T};return u}return n.value=function(e){return arguments.length?(t=typeof e=="function"?e:h(+e),n):t},n.sortValues=function(e){return arguments.length?(a=e,d=null,n):a},n.sort=function(e){return arguments.length?(d=e,a=null,n):d},n.startAngle=function(e){return arguments.length?(o=typeof e=="function"?e:h(+e),n):o},n.endAngle=function(e){return arguments.length?(s=typeof e=="function"?e:h(+e),n):s},n.padAngle=function(e){return arguments.length?(x=typeof e=="function"?e:h(+e),n):x},n}var R=I.pie,O={sections:new Map,showData:!1,config:R},C=O.sections,z=O.showData,at=structuredClone(R),E={getConfig:p(()=>structuredClone(at),"getConfig"),clear:p(()=>{C=new Map,z=O.showData,q()},"clear"),setDiagramTitle:j,getDiagramTitle:V,setAccTitle:L,getAccTitle:J,setAccDescription:K,getAccDescription:U,addSection:p(({label:t,value:a})=>{if(a<0)throw Error(`"${t}" has invalid value: ${a}. Negative values are not allowed in pie charts. All slice values must be >= 0.`);C.has(t)||(C.set(t,a),M.debug(`added new section: ${t}, with value: ${a}`))},"addSection"),getSections:p(()=>C,"getSections"),setShowData:p(t=>{z=t},"setShowData"),getShowData:p(()=>z,"getShowData")},rt=p((t,a)=>{X(t,a),a.setShowData(t.showData),t.sections.map(a.addSection)},"populateDb"),it={parse:p(async t=>{let a=await Y("pie",t);M.debug(a),rt(a,E)},"parse")},lt=p(t=>`
  .pieCircle{
    stroke: ${t.pieStrokeColor};
    stroke-width : ${t.pieStrokeWidth};
    opacity : ${t.pieOpacity};
  }
  .pieOuterCircle{
    stroke: ${t.pieOuterStrokeColor};
    stroke-width: ${t.pieOuterStrokeWidth};
    fill: none;
  }
  .pieTitleText {
    text-anchor: middle;
    font-size: ${t.pieTitleTextSize};
    fill: ${t.pieTitleTextColor};
    font-family: ${t.fontFamily};
  }
  .slice {
    font-family: ${t.fontFamily};
    fill: ${t.pieSectionTextColor};
    font-size:${t.pieSectionTextSize};
    // fill: white;
  }
  .legend text {
    fill: ${t.pieLegendTextColor};
    font-family: ${t.fontFamily};
    font-size: ${t.pieLegendTextSize};
  }
`,"getStyles"),nt=p(t=>{let a=[...t.values()].reduce((o,s)=>o+s,0),d=[...t.entries()].map(([o,s])=>({label:o,value:s})).filter(o=>o.value/a*100>=1).sort((o,s)=>s.value-o.value);return et().value(o=>o.value)(d)},"createPieArcs"),ot={parser:it,db:E,renderer:{draw:p((t,a,d,o)=>{M.debug(`rendering pie chart
`+t);let s=o.db,x=G(),n=B(s.getConfig(),x.pie),e=Q(a),l=e.append("g");l.attr("transform","translate(225,225)");let{themeVariables:i}=x,[m]=P(i.pieOuterStrokeWidth);m??(m=2);let b=n.textPosition,$=F().innerRadius(0).outerRadius(185),y=F().innerRadius(185*b).outerRadius(185*b);l.append("circle").attr("cx",0).attr("cy",0).attr("r",185+m/2).attr("class","pieOuterCircle");let u=s.getSections(),v=nt(u),A=[i.pie1,i.pie2,i.pie3,i.pie4,i.pie5,i.pie6,i.pie7,i.pie8,i.pie9,i.pie10,i.pie11,i.pie12],f=0;u.forEach(r=>{f+=r});let T=v.filter(r=>(r.data.value/f*100).toFixed(0)!=="0"),w=W(A);l.selectAll("mySlices").data(T).enter().append("path").attr("d",$).attr("fill",r=>w(r.data.label)).attr("class","pieCircle"),l.selectAll("mySlices").data(T).enter().append("text").text(r=>(r.data.value/f*100).toFixed(0)+"%").attr("transform",r=>"translate("+y.centroid(r)+")").style("text-anchor","middle").attr("class","slice"),l.append("text").text(s.getDiagramTitle()).attr("x",0).attr("y",-400/2).attr("class","pieTitleText");let c=[...u.entries()].map(([r,D])=>({label:r,value:D})),g=l.selectAll(".legend").data(c).enter().append("g").attr("class","legend").attr("transform",(r,D)=>{let N=22*c.length/2;return"translate(216,"+(D*22-N)+")"});g.append("rect").attr("width",18).attr("height",18).style("fill",r=>w(r.label)).style("stroke",r=>w(r.label)),g.append("text").attr("x",22).attr("y",14).text(r=>s.getShowData()?`${r.label} [${r.value}]`:r.label);let S=512+Math.max(...g.selectAll("text").nodes().map(r=>(r==null?void 0:r.getBoundingClientRect().width)??0));e.attr("viewBox",`0 0 ${S} 450`),H(e,450,S,n.useMaxWidth)},"draw")},styles:lt};export{ot as diagram};
