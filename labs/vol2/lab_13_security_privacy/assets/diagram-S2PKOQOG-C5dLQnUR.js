var g;import"./chunk-TCCFYFTB-ByQR3Ftq.js";import"./main-BNL5rxQw.js";import"./purify.es-BFDc-_i_.js";import"./src-lJewPvdk.js";import{i as u}from"./chunk-S3R3BYOJ-QzFziHq0.js";import{n as f,r as y}from"./src-DSbF9mby.js";import{B as C,C as v,U as P,_ as z,a as S,c as E,d as F,v as T,y as W,z as D}from"./chunk-ABZYJK2D-C6__zbIg.js";import{t as A}from"./chunk-EXTU4WIE-B_Um-dIN.js";import"./dist-D-j2KFQ4.js";import"./chunk-UMXZTB3W-C4miLMVf.js";import"./chunk-4F5CHEZ2-BXUyRfDO.js";import"./chunk-FRFDVMJY-CjptiqNO.js";import"./chunk-SJTYNZTY-rJzTNWIe.js";import"./chunk-PL6DKKU2-BdoRD2nE.js";import"./chunk-TQ3KTPDO-CURFAmMV.js";import"./chunk-B2363JML-CY6QiDyK.js";import{t as R}from"./chunk-4BX2VUAB-hp7-9s5b.js";import{t as Y}from"./mermaid-parser.core-CvPV98yl.js";var _=F.packet,w=(g=class{constructor(){this.packet=[],this.setAccTitle=C,this.getAccTitle=T,this.setDiagramTitle=P,this.getDiagramTitle=v,this.getAccDescription=z,this.setAccDescription=D}getConfig(){let t=u({..._,...W().packet});return t.showBits&&(t.paddingY+=10),t}getPacket(){return this.packet}pushWord(t){t.length>0&&this.packet.push(t)}clear(){S(),this.packet=[]}},f(g,"PacketDB"),g),H=1e4,L=f((e,t)=>{R(e,t);let i=-1,a=[],l=1,{bitsPerRow:n}=t.getConfig();for(let{start:r,end:s,bits:c,label:d}of e.blocks){if(r!==void 0&&s!==void 0&&s<r)throw Error(`Packet block ${r} - ${s} is invalid. End must be greater than start.`);if(r??(r=i+1),r!==i+1)throw Error(`Packet block ${r} - ${s??r} is not contiguous. It should start from ${i+1}.`);if(c===0)throw Error(`Packet block ${r} is invalid. Cannot have a zero bit field.`);for(s??(s=r+(c??1)-1),c??(c=s-r+1),i=s,y.debug(`Packet block ${r} - ${i} with label ${d}`);a.length<=n+1&&t.getPacket().length<H;){let[p,o]=M({start:r,end:s,bits:c,label:d},l,n);if(a.push(p),p.end+1===l*n&&(t.pushWord(a),a=[],l++),!o)break;({start:r,end:s,bits:c,label:d}=o)}}t.pushWord(a)},"populate"),M=f((e,t,i)=>{if(e.start===void 0)throw Error("start should have been set during first phase");if(e.end===void 0)throw Error("end should have been set during first phase");if(e.start>e.end)throw Error(`Block start ${e.start} is greater than block end ${e.end}.`);if(e.end+1<=t*i)return[e,void 0];let a=t*i-1,l=t*i;return[{start:e.start,end:a,label:e.label,bits:a-e.start},{start:l,end:e.end,label:e.label,bits:e.end-l}]},"getNextFittingBlock"),x={parser:{yy:void 0},parse:f(async e=>{var a;let t=await Y("packet",e),i=(a=x.parser)==null?void 0:a.yy;if(!(i instanceof w))throw Error("parser.parser?.yy was not a PacketDB. This is due to a bug within Mermaid, please report this issue at https://github.com/mermaid-js/mermaid/issues.");y.debug(t),L(t,i)},"parse")},j=f((e,t,i,a)=>{let l=a.db,n=l.getConfig(),{rowHeight:r,paddingY:s,bitWidth:c,bitsPerRow:d}=n,p=l.getPacket(),o=l.getDiagramTitle(),b=r+s,h=b*(p.length+1)-(o?0:r),k=c*d+2,m=A(t);m.attr("viewbox",`0 0 ${k} ${h}`),E(m,h,k,n.useMaxWidth);for(let[$,B]of p.entries())I(m,B,$,n);m.append("text").text(o).attr("x",k/2).attr("y",h-b/2).attr("dominant-baseline","middle").attr("text-anchor","middle").attr("class","packetTitle")},"draw"),I=f((e,t,i,{rowHeight:a,paddingX:l,paddingY:n,bitWidth:r,bitsPerRow:s,showBits:c})=>{let d=e.append("g"),p=i*(a+n)+n;for(let o of t){let b=o.start%s*r+1,h=(o.end-o.start+1)*r-l;if(d.append("rect").attr("x",b).attr("y",p).attr("width",h).attr("height",a).attr("class","packetBlock"),d.append("text").attr("x",b+h/2).attr("y",p+a/2).attr("class","packetLabel").attr("dominant-baseline","middle").attr("text-anchor","middle").text(o.label),!c)continue;let k=o.end===o.start,m=p-2;d.append("text").attr("x",b+(k?h/2:0)).attr("y",m).attr("class","packetByte start").attr("dominant-baseline","auto").attr("text-anchor",k?"middle":"start").text(o.start),k||d.append("text").attr("x",b+h).attr("y",m).attr("class","packetByte end").attr("dominant-baseline","auto").attr("text-anchor","end").text(o.end)}},"drawWord"),N={draw:j},U={byteFontSize:"10px",startByteColor:"black",endByteColor:"black",labelColor:"black",labelFontSize:"12px",titleColor:"black",titleFontSize:"14px",blockStrokeColor:"black",blockStrokeWidth:"1",blockFillColor:"#efefef"},X={parser:x,get db(){return new w},renderer:N,styles:f(({packet:e}={})=>{let t=u(U,e);return`
	.packetByte {
		font-size: ${t.byteFontSize};
	}
	.packetByte.start {
		fill: ${t.startByteColor};
	}
	.packetByte.end {
		fill: ${t.endByteColor};
	}
	.packetLabel {
		fill: ${t.labelColor};
		font-size: ${t.labelFontSize};
	}
	.packetTitle {
		fill: ${t.titleColor};
		font-size: ${t.titleFontSize};
	}
	.packetBlock {
		stroke: ${t.blockStrokeColor};
		stroke-width: ${t.blockStrokeWidth};
		fill: ${t.blockFillColor};
	}
	`},"styles")};export{X as diagram};
