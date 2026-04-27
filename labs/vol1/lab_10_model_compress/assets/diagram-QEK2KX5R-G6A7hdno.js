import"./chunk-TCCFYFTB-ClOYb5MI.js";import"./main-B0OX4z33.js";import"./purify.es-DC4RGS9t.js";import"./src-nKy6a3tt.js";import{i as y}from"./chunk-S3R3BYOJ-GS_DuFun.js";import{n as o,r as k}from"./src-CmKhyEBC.js";import{B as O,C as S,T as I,U as z,_ as E,a as F,d as P,v as R,y as w,z as D}from"./chunk-ABZYJK2D-X_7JfTbo.js";import{t as B}from"./chunk-EXTU4WIE-B7W_a4w2.js";import"./dist-DxxvVPQH.js";import"./chunk-UMXZTB3W-DsE6dCbX.js";import"./chunk-4F5CHEZ2-BBsStyPF.js";import"./chunk-FRFDVMJY-D6StbPqE.js";import"./chunk-SJTYNZTY-BNaCydHS.js";import"./chunk-PL6DKKU2-x3Qoia3Y.js";import"./chunk-TQ3KTPDO-DfYTN8Ct.js";import"./chunk-B2363JML-CPCjJ6tH.js";import{t as G}from"./chunk-4BX2VUAB-mPGRpph5.js";import{t as V}from"./mermaid-parser.core-DP6UBthl.js";var h={showLegend:!0,ticks:5,max:null,min:0,graticule:"circle"},b={axes:[],curves:[],options:h},g=structuredClone(b),_=P.radar,j=o(()=>y({..._,...w().radar}),"getConfig"),C=o(()=>g.axes,"getAxes"),W=o(()=>g.curves,"getCurves"),H=o(()=>g.options,"getOptions"),N=o(a=>{g.axes=a.map(t=>({name:t.name,label:t.label??t.name}))},"setAxes"),U=o(a=>{g.curves=a.map(t=>({name:t.name,label:t.label??t.name,entries:Z(t.entries)}))},"setCurves"),Z=o(a=>{if(a[0].axis==null)return a.map(e=>e.value);let t=C();if(t.length===0)throw Error("Axes must be populated before curves for reference entries");return t.map(e=>{let r=a.find(i=>{var n;return((n=i.axis)==null?void 0:n.$refText)===e.name});if(r===void 0)throw Error("Missing entry for axis "+e.label);return r.value})},"computeCurveEntries"),x={getAxes:C,getCurves:W,getOptions:H,setAxes:N,setCurves:U,setOptions:o(a=>{var e,r,i,n,l;let t=a.reduce((s,c)=>(s[c.name]=c,s),{});g.options={showLegend:((e=t.showLegend)==null?void 0:e.value)??h.showLegend,ticks:((r=t.ticks)==null?void 0:r.value)??h.ticks,max:((i=t.max)==null?void 0:i.value)??h.max,min:((n=t.min)==null?void 0:n.value)??h.min,graticule:((l=t.graticule)==null?void 0:l.value)??h.graticule}},"setOptions"),getConfig:j,clear:o(()=>{F(),g=structuredClone(b)},"clear"),setAccTitle:O,getAccTitle:R,setDiagramTitle:z,getDiagramTitle:S,getAccDescription:E,setAccDescription:D},q=o(a=>{G(a,x);let{axes:t,curves:e,options:r}=a;x.setAxes(t),x.setCurves(e),x.setOptions(r)},"populate"),J={parse:o(async a=>{let t=await V("radar",a);k.debug(t),q(t)},"parse")},K=o((a,t,e,r)=>{let i=r.db,n=i.getAxes(),l=i.getCurves(),s=i.getOptions(),c=i.getConfig(),p=i.getDiagramTitle(),d=Q(B(t),c),m=s.max??Math.max(...l.map($=>Math.max(...$.entries))),u=s.min,f=Math.min(c.width,c.height)/2;X(d,n,f,s.ticks,s.graticule),Y(d,n,f,c),M(d,n,l,u,m,s.graticule,c),A(d,l,s.showLegend,c),d.append("text").attr("class","radarTitle").text(p).attr("x",0).attr("y",-c.height/2-c.marginTop)},"draw"),Q=o((a,t)=>{let e=t.width+t.marginLeft+t.marginRight,r=t.height+t.marginTop+t.marginBottom,i={x:t.marginLeft+t.width/2,y:t.marginTop+t.height/2};return a.attr("viewbox",`0 0 ${e} ${r}`).attr("width",e).attr("height",r),a.append("g").attr("transform",`translate(${i.x}, ${i.y})`)},"drawFrame"),X=o((a,t,e,r,i)=>{if(i==="circle")for(let n=0;n<r;n++){let l=e*(n+1)/r;a.append("circle").attr("r",l).attr("class","radarGraticule")}else if(i==="polygon"){let n=t.length;for(let l=0;l<r;l++){let s=e*(l+1)/r,c=t.map((p,d)=>{let m=2*d*Math.PI/n-Math.PI/2;return`${s*Math.cos(m)},${s*Math.sin(m)}`}).join(" ");a.append("polygon").attr("points",c).attr("class","radarGraticule")}}},"drawGraticule"),Y=o((a,t,e,r)=>{let i=t.length;for(let n=0;n<i;n++){let l=t[n].label,s=2*n*Math.PI/i-Math.PI/2;a.append("line").attr("x1",0).attr("y1",0).attr("x2",e*r.axisScaleFactor*Math.cos(s)).attr("y2",e*r.axisScaleFactor*Math.sin(s)).attr("class","radarAxisLine"),a.append("text").text(l).attr("x",e*r.axisLabelFactor*Math.cos(s)).attr("y",e*r.axisLabelFactor*Math.sin(s)).attr("class","radarAxisLabel")}},"drawAxes");function M(a,t,e,r,i,n,l){let s=t.length,c=Math.min(l.width,l.height)/2;e.forEach((p,d)=>{if(p.entries.length!==s)return;let m=p.entries.map((u,f)=>{let $=2*Math.PI*f/s-Math.PI/2,v=L(u,r,i,c);return{x:v*Math.cos($),y:v*Math.sin($)}});n==="circle"?a.append("path").attr("d",T(m,l.curveTension)).attr("class",`radarCurve-${d}`):n==="polygon"&&a.append("polygon").attr("points",m.map(u=>`${u.x},${u.y}`).join(" ")).attr("class",`radarCurve-${d}`)})}o(M,"drawCurves");function L(a,t,e,r){return r*(Math.min(Math.max(a,t),e)-t)/(e-t)}o(L,"relativeRadius");function T(a,t){let e=a.length,r=`M${a[0].x},${a[0].y}`;for(let i=0;i<e;i++){let n=a[(i-1+e)%e],l=a[i],s=a[(i+1)%e],c=a[(i+2)%e],p={x:l.x+(s.x-n.x)*t,y:l.y+(s.y-n.y)*t},d={x:s.x-(c.x-l.x)*t,y:s.y-(c.y-l.y)*t};r+=` C${p.x},${p.y} ${d.x},${d.y} ${s.x},${s.y}`}return`${r} Z`}o(T,"closedRoundCurve");function A(a,t,e,r){if(!e)return;let i=(r.width/2+r.marginRight)*3/4,n=-(r.height/2+r.marginTop)*3/4;t.forEach((l,s)=>{let c=a.append("g").attr("transform",`translate(${i}, ${n+s*20})`);c.append("rect").attr("width",12).attr("height",12).attr("class",`radarLegendBox-${s}`),c.append("text").attr("x",16).attr("y",0).attr("class","radarLegendText").text(l.label)})}o(A,"drawLegend");var tt={draw:K},et=o((a,t)=>{let e="";for(let r=0;r<a.THEME_COLOR_LIMIT;r++){let i=a[`cScale${r}`];e+=`
		.radarCurve-${r} {
			color: ${i};
			fill: ${i};
			fill-opacity: ${t.curveOpacity};
			stroke: ${i};
			stroke-width: ${t.curveStrokeWidth};
		}
		.radarLegendBox-${r} {
			fill: ${i};
			fill-opacity: ${t.curveOpacity};
			stroke: ${i};
		}
		`}return e},"genIndexStyles"),at=o(a=>{let t=y(I(),w().themeVariables);return{themeVariables:t,radarOptions:y(t.radar,a)}},"buildRadarStyleOptions"),rt={parser:J,db:x,renderer:tt,styles:o(({radar:a}={})=>{let{themeVariables:t,radarOptions:e}=at(a);return`
	.radarTitle {
		font-size: ${t.fontSize};
		color: ${t.titleColor};
		dominant-baseline: hanging;
		text-anchor: middle;
	}
	.radarAxisLine {
		stroke: ${e.axisColor};
		stroke-width: ${e.axisStrokeWidth};
	}
	.radarAxisLabel {
		dominant-baseline: middle;
		text-anchor: middle;
		font-size: ${e.axisLabelFontSize}px;
		color: ${e.axisColor};
	}
	.radarGraticule {
		fill: ${e.graticuleColor};
		fill-opacity: ${e.graticuleOpacity};
		stroke: ${e.graticuleColor};
		stroke-width: ${e.graticuleStrokeWidth};
	}
	.radarLegendText {
		text-anchor: start;
		font-size: ${e.legendFontSize}px;
		dominant-baseline: hanging;
	}
	${et(t,e)}
	`},"styles")};export{rt as diagram};
