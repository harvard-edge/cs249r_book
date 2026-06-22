import{s as ct,t as vt}from"./chunk-LvLJmgfZ.js";import"./purify.es-BFDc-_i_.js";import{t as me}from"./linear-BICTUQLW.js";import{n as ye,o as ke,s as pe}from"./time-DLcLlk0u.js";import"./defaultLocale-JieDVWC_.js";import{C as Zt,N as Ut,T as qt,c as Xt,d as ge,f as be,g as ve,h as Te,m as xe,p as we,t as Qt,u as $e,v as Jt,x as Kt}from"./defaultLocale-BLne0bXb.js";import{o as _e}from"./timer-DBQ_H0G1.js";import{u as Dt}from"./src-lJewPvdk.js";import{g as De}from"./chunk-S3R3BYOJ-QzFziHq0.js";import{a as te,n as u,r as st}from"./src-DSbF9mby.js";import{B as Se,C as Me,U as Ce,_ as Ee,a as Ye,b as lt,c as Ae,s as Ie,v as Le,z as Fe}from"./chunk-ABZYJK2D-C6__zbIg.js";import{t as Oe}from"./dist-D-j2KFQ4.js";function We(t){return t}var Tt=1,St=2,Mt=3,xt=4,ee=1e-6;function Pe(t){return"translate("+t+",0)"}function ze(t){return"translate(0,"+t+")"}function He(t){return i=>+t(i)}function Ne(t,i){return i=Math.max(0,t.bandwidth()-i*2)/2,t.round()&&(i=Math.round(i)),r=>+t(r)+i}function Be(){return!this.__axis}function ie(t,i){var r=[],n=null,o=null,d=6,k=6,M=3,I=typeof window<"u"&&window.devicePixelRatio>1?0:.5,D=t===Tt||t===xt?-1:1,x=t===xt||t===St?"x":"y",F=t===Tt||t===Mt?Pe:ze;function w($){var z=n??(i.ticks?i.ticks.apply(i,r):i.domain()),Y=o??(i.tickFormat?i.tickFormat.apply(i,r):We),v=Math.max(d,0)+M,C=i.range(),L=+C[0]+I,O=+C[C.length-1]+I,H=(i.bandwidth?Ne:He)(i.copy(),I),N=$.selection?$.selection():$,A=N.selectAll(".domain").data([null]),p=N.selectAll(".tick").data(z,i).order(),m=p.exit(),h=p.enter().append("g").attr("class","tick"),g=p.select("line"),y=p.select("text");A=A.merge(A.enter().insert("path",".tick").attr("class","domain").attr("stroke","currentColor")),p=p.merge(h),g=g.merge(h.append("line").attr("stroke","currentColor").attr(x+"2",D*d)),y=y.merge(h.append("text").attr("fill","currentColor").attr(x,D*v).attr("dy",t===Tt?"0em":t===Mt?"0.71em":"0.32em")),$!==N&&(A=A.transition($),p=p.transition($),g=g.transition($),y=y.transition($),m=m.transition($).attr("opacity",ee).attr("transform",function(s){return isFinite(s=H(s))?F(s+I):this.getAttribute("transform")}),h.attr("opacity",ee).attr("transform",function(s){var c=this.parentNode.__axis;return F((c&&isFinite(c=c(s))?c:H(s))+I)})),m.remove(),A.attr("d",t===xt||t===St?k?"M"+D*k+","+L+"H"+I+"V"+O+"H"+D*k:"M"+I+","+L+"V"+O:k?"M"+L+","+D*k+"V"+I+"H"+O+"V"+D*k:"M"+L+","+I+"H"+O),p.attr("opacity",1).attr("transform",function(s){return F(H(s)+I)}),g.attr(x+"2",D*d),y.attr(x,D*v).text(Y),N.filter(Be).attr("fill","none").attr("font-size",10).attr("font-family","sans-serif").attr("text-anchor",t===St?"start":t===xt?"end":"middle"),N.each(function(){this.__axis=H})}return w.scale=function($){return arguments.length?(i=$,w):i},w.ticks=function(){return r=Array.from(arguments),w},w.tickArguments=function($){return arguments.length?(r=$==null?[]:Array.from($),w):r.slice()},w.tickValues=function($){return arguments.length?(n=$==null?null:Array.from($),w):n&&n.slice()},w.tickFormat=function($){return arguments.length?(o=$,w):o},w.tickSize=function($){return arguments.length?(d=k=+$,w):d},w.tickSizeInner=function($){return arguments.length?(d=+$,w):d},w.tickSizeOuter=function($){return arguments.length?(k=+$,w):k},w.tickPadding=function($){return arguments.length?(M=+$,w):M},w.offset=function($){return arguments.length?(I=+$,w):I},w}function je(t){return ie(Tt,t)}function Ge(t){return ie(Mt,t)}var Ve=vt(((t,i)=>{(function(r,n){typeof t=="object"&&i!==void 0?i.exports=n():typeof define=="function"&&define.amd?define(n):(r=typeof globalThis<"u"?globalThis:r||self).dayjs_plugin_isoWeek=n()})(t,(function(){var r="day";return function(n,o,d){var k=function(D){return D.add(4-D.isoWeekday(),r)},M=o.prototype;M.isoWeekYear=function(){return k(this).year()},M.isoWeek=function(D){if(!this.$utils().u(D))return this.add(7*(D-this.isoWeek()),r);var x,F,w,$,z=k(this),Y=(x=this.isoWeekYear(),F=this.$u,w=(F?d.utc:d)().year(x).startOf("year"),$=4-w.isoWeekday(),w.isoWeekday()>4&&($+=7),w.add($,r));return z.diff(Y,"week")+1},M.isoWeekday=function(D){return this.$utils().u(D)?this.day()||7:this.day(this.day()%7?D:D-7)};var I=M.startOf;M.startOf=function(D,x){var F=this.$utils(),w=!!F.u(x)||x;return F.p(D)==="isoweek"?w?this.date(this.date()-(this.isoWeekday()-1)).startOf("day"):this.date(this.date()-1-(this.isoWeekday()-1)+7).endOf("day"):I.bind(this)(D,x)}}}))})),Re=vt(((t,i)=>{(function(r,n){typeof t=="object"&&i!==void 0?i.exports=n():typeof define=="function"&&define.amd?define(n):(r=typeof globalThis<"u"?globalThis:r||self).dayjs_plugin_customParseFormat=n()})(t,(function(){var r={LTS:"h:mm:ss A",LT:"h:mm A",L:"MM/DD/YYYY",LL:"MMMM D, YYYY",LLL:"MMMM D, YYYY h:mm A",LLLL:"dddd, MMMM D, YYYY h:mm A"},n=/(\[[^[]*\])|([-_:/.,()\s]+)|(A|a|Q|YYYY|YY?|ww?|MM?M?M?|Do|DD?|hh?|HH?|mm?|ss?|S{1,3}|z|ZZ?)/g,o=/\d/,d=/\d\d/,k=/\d\d?/,M=/\d*[^-_:/,()\s\d]+/,I={},D=function(v){return(v=+v)+(v>68?1900:2e3)},x=function(v){return function(C){this[v]=+C}},F=[/[+-]\d\d:?(\d\d)?|Z/,function(v){(this.zone||(this.zone={})).offset=(function(C){if(!C||C==="Z")return 0;var L=C.match(/([+-]|\d\d)/g),O=60*L[1]+(+L[2]||0);return O===0?0:L[0]==="+"?-O:O})(v)}],w=function(v){var C=I[v];return C&&(C.indexOf?C:C.s.concat(C.f))},$=function(v,C){var L,O=I.meridiem;if(O){for(var H=1;H<=24;H+=1)if(v.indexOf(O(H,0,C))>-1){L=H>12;break}}else L=v===(C?"pm":"PM");return L},z={A:[M,function(v){this.afternoon=$(v,!1)}],a:[M,function(v){this.afternoon=$(v,!0)}],Q:[o,function(v){this.month=3*(v-1)+1}],S:[o,function(v){this.milliseconds=100*v}],SS:[d,function(v){this.milliseconds=10*v}],SSS:[/\d{3}/,function(v){this.milliseconds=+v}],s:[k,x("seconds")],ss:[k,x("seconds")],m:[k,x("minutes")],mm:[k,x("minutes")],H:[k,x("hours")],h:[k,x("hours")],HH:[k,x("hours")],hh:[k,x("hours")],D:[k,x("day")],DD:[d,x("day")],Do:[M,function(v){var C=I.ordinal;if(this.day=v.match(/\d+/)[0],C)for(var L=1;L<=31;L+=1)C(L).replace(/\[|\]/g,"")===v&&(this.day=L)}],w:[k,x("week")],ww:[d,x("week")],M:[k,x("month")],MM:[d,x("month")],MMM:[M,function(v){var C=w("months"),L=(w("monthsShort")||C.map((function(O){return O.slice(0,3)}))).indexOf(v)+1;if(L<1)throw Error();this.month=L%12||L}],MMMM:[M,function(v){var C=w("months").indexOf(v)+1;if(C<1)throw Error();this.month=C%12||C}],Y:[/[+-]?\d+/,x("year")],YY:[d,function(v){this.year=D(v)}],YYYY:[/\d{4}/,x("year")],Z:F,ZZ:F};function Y(v){for(var C=v,L=I&&I.formats,O=(v=C.replace(/(\[[^\]]+])|(LTS?|l{1,4}|L{1,4})/g,(function(g,y,s){var c=s&&s.toUpperCase();return y||L[s]||r[s]||L[c].replace(/(\[[^\]]+])|(MMMM|MM|DD|dddd)/g,(function(f,l,b){return l||b.slice(1)}))}))).match(n),H=O.length,N=0;N<H;N+=1){var A=O[N],p=z[A],m=p&&p[0],h=p&&p[1];O[N]=h?{regex:m,parser:h}:A.replace(/^\[|\]$/g,"")}return function(g){for(var y={},s=0,c=0;s<H;s+=1){var f=O[s];if(typeof f=="string")c+=f.length;else{var l=f.regex,b=f.parser,a=g.slice(c),_=l.exec(a)[0];b.call(y,_),g=g.replace(_,"")}}return(function(e){var T=e.afternoon;if(T!==void 0){var S=e.hours;T?S<12&&(e.hours+=12):S===12&&(e.hours=0),delete e.afternoon}})(y),y}}return function(v,C,L){L.p.customParseFormat=!0,v&&v.parseTwoDigitYear&&(D=v.parseTwoDigitYear);var O=C.prototype,H=O.parse;O.parse=function(N){var A=N.date,p=N.utc,m=N.args;this.$u=p;var h=m[1];if(typeof h=="string"){var g=m[2]===!0,y=m[3]===!0,s=g||y,c=m[2];y&&(c=m[2]),I=this.$locale(),!g&&c&&(I=L.Ls[c]),this.$d=(function(a,_,e,T){try{if(["x","X"].indexOf(_)>-1)return new Date((_==="X"?1e3:1)*a);var S=Y(_)(a),E=S.year,W=S.month,P=S.day,j=S.hours,B=S.minutes,X=S.seconds,ht=S.milliseconds,at=S.zone,gt=S.week,ft=new Date,ot=P||(E||W?1:ft.getDate()),V=E||ft.getFullYear(),et=0;E&&!W||(et=W>0?W-1:ft.getMonth());var Z,R=j||0,nt=B||0,Q=X||0,it=ht||0;return at?new Date(Date.UTC(V,et,ot,R,nt,Q,it+60*at.offset*1e3)):e?new Date(Date.UTC(V,et,ot,R,nt,Q,it)):(Z=new Date(V,et,ot,R,nt,Q,it),gt&&(Z=T(Z).week(gt).toDate()),Z)}catch{return new Date("")}})(A,h,p,L),this.init(),c&&c!==!0&&(this.$L=this.locale(c).$L),s&&A!=this.format(h)&&(this.$d=new Date("")),I={}}else if(h instanceof Array)for(var f=h.length,l=1;l<=f;l+=1){m[1]=h[l-1];var b=L.apply(this,m);if(b.isValid()){this.$d=b.$d,this.$L=b.$L,this.init();break}l===f&&(this.$d=new Date(""))}else H.call(this,N)}}}))})),Ze=vt(((t,i)=>{(function(r,n){typeof t=="object"&&i!==void 0?i.exports=n():typeof define=="function"&&define.amd?define(n):(r=typeof globalThis<"u"?globalThis:r||self).dayjs_plugin_advancedFormat=n()})(t,(function(){return function(r,n){var o=n.prototype,d=o.format;o.format=function(k){var M=this,I=this.$locale();if(!this.isValid())return d.bind(this)(k);var D=this.$utils(),x=(k||"YYYY-MM-DDTHH:mm:ssZ").replace(/\[([^\]]+)]|Q|wo|ww|w|WW|W|zzz|z|gggg|GGGG|Do|X|x|k{1,2}|S/g,(function(F){switch(F){case"Q":return Math.ceil((M.$M+1)/3);case"Do":return I.ordinal(M.$D);case"gggg":return M.weekYear();case"GGGG":return M.isoWeekYear();case"wo":return I.ordinal(M.week(),"W");case"w":case"ww":return D.s(M.week(),F==="w"?1:2,"0");case"W":case"WW":return D.s(M.isoWeek(),F==="W"?1:2,"0");case"k":case"kk":return D.s(String(M.$H===0?24:M.$H),F==="k"?1:2,"0");case"X":return Math.floor(M.$d.getTime()/1e3);case"x":return M.$d.getTime();case"z":return"["+M.offsetName()+"]";case"zzz":return"["+M.offsetName("long")+"]";default:return F}}));return d.bind(this)(x)}}}))})),Ue=vt(((t,i)=>{(function(r,n){typeof t=="object"&&i!==void 0?i.exports=n():typeof define=="function"&&define.amd?define(n):(r=typeof globalThis<"u"?globalThis:r||self).dayjs_plugin_duration=n()})(t,(function(){var r,n,o=1e3,d=6e4,k=36e5,M=864e5,I=/\[([^\]]+)]|Y{1,4}|M{1,4}|D{1,2}|d{1,4}|H{1,2}|h{1,2}|a|A|m{1,2}|s{1,2}|Z{1,2}|SSS/g,D=31536e6,x=2628e6,F=/^(-|\+)?P(?:([-+]?[0-9,.]*)Y)?(?:([-+]?[0-9,.]*)M)?(?:([-+]?[0-9,.]*)W)?(?:([-+]?[0-9,.]*)D)?(?:T(?:([-+]?[0-9,.]*)H)?(?:([-+]?[0-9,.]*)M)?(?:([-+]?[0-9,.]*)S)?)?$/,w={years:D,months:x,days:M,hours:k,minutes:d,seconds:o,milliseconds:1,weeks:6048e5},$=function(A){return A instanceof H},z=function(A,p,m){return new H(A,m,p.$l)},Y=function(A){return n.p(A)+"s"},v=function(A){return A<0},C=function(A){return v(A)?Math.ceil(A):Math.floor(A)},L=function(A){return Math.abs(A)},O=function(A,p){return A?v(A)?{negative:!0,format:""+L(A)+p}:{negative:!1,format:""+A+p}:{negative:!1,format:""}},H=(function(){function A(m,h,g){var y=this;if(this.$d={},this.$l=g,m===void 0&&(this.$ms=0,this.parseFromMilliseconds()),h)return z(m*w[Y(h)],this);if(typeof m=="number")return this.$ms=m,this.parseFromMilliseconds(),this;if(typeof m=="object")return Object.keys(m).forEach((function(f){y.$d[Y(f)]=m[f]})),this.calMilliseconds(),this;if(typeof m=="string"){var s=m.match(F);if(s){var c=s.slice(2).map((function(f){return f==null?0:Number(f)}));return this.$d.years=c[0],this.$d.months=c[1],this.$d.weeks=c[2],this.$d.days=c[3],this.$d.hours=c[4],this.$d.minutes=c[5],this.$d.seconds=c[6],this.calMilliseconds(),this}}return this}var p=A.prototype;return p.calMilliseconds=function(){var m=this;this.$ms=Object.keys(this.$d).reduce((function(h,g){return h+(m.$d[g]||0)*w[g]}),0)},p.parseFromMilliseconds=function(){var m=this.$ms;this.$d.years=C(m/D),m%=D,this.$d.months=C(m/x),m%=x,this.$d.days=C(m/M),m%=M,this.$d.hours=C(m/k),m%=k,this.$d.minutes=C(m/d),m%=d,this.$d.seconds=C(m/o),m%=o,this.$d.milliseconds=m},p.toISOString=function(){var m=O(this.$d.years,"Y"),h=O(this.$d.months,"M"),g=+this.$d.days||0;this.$d.weeks&&(g+=7*this.$d.weeks);var y=O(g,"D"),s=O(this.$d.hours,"H"),c=O(this.$d.minutes,"M"),f=this.$d.seconds||0;this.$d.milliseconds&&(f+=this.$d.milliseconds/1e3,f=Math.round(1e3*f)/1e3);var l=O(f,"S"),b=m.negative||h.negative||y.negative||s.negative||c.negative||l.negative,a=s.format||c.format||l.format?"T":"",_=(b?"-":"")+"P"+m.format+h.format+y.format+a+s.format+c.format+l.format;return _==="P"||_==="-P"?"P0D":_},p.toJSON=function(){return this.toISOString()},p.format=function(m){var h=m||"YYYY-MM-DDTHH:mm:ss",g={Y:this.$d.years,YY:n.s(this.$d.years,2,"0"),YYYY:n.s(this.$d.years,4,"0"),M:this.$d.months,MM:n.s(this.$d.months,2,"0"),D:this.$d.days,DD:n.s(this.$d.days,2,"0"),H:this.$d.hours,HH:n.s(this.$d.hours,2,"0"),m:this.$d.minutes,mm:n.s(this.$d.minutes,2,"0"),s:this.$d.seconds,ss:n.s(this.$d.seconds,2,"0"),SSS:n.s(this.$d.milliseconds,3,"0")};return h.replace(I,(function(y,s){return s||String(g[y])}))},p.as=function(m){return this.$ms/w[Y(m)]},p.get=function(m){var h=this.$ms,g=Y(m);return g==="milliseconds"?h%=1e3:h=g==="weeks"?C(h/w[g]):this.$d[g],h||0},p.add=function(m,h,g){var y;return y=h?m*w[Y(h)]:$(m)?m.$ms:z(m,this).$ms,z(this.$ms+y*(g?-1:1),this)},p.subtract=function(m,h){return this.add(m,h,!0)},p.locale=function(m){var h=this.clone();return h.$l=m,h},p.clone=function(){return z(this.$ms,this)},p.humanize=function(m){return r().add(this.$ms,"ms").locale(this.$l).fromNow(!m)},p.valueOf=function(){return this.asMilliseconds()},p.milliseconds=function(){return this.get("milliseconds")},p.asMilliseconds=function(){return this.as("milliseconds")},p.seconds=function(){return this.get("seconds")},p.asSeconds=function(){return this.as("seconds")},p.minutes=function(){return this.get("minutes")},p.asMinutes=function(){return this.as("minutes")},p.hours=function(){return this.get("hours")},p.asHours=function(){return this.as("hours")},p.days=function(){return this.get("days")},p.asDays=function(){return this.as("days")},p.weeks=function(){return this.get("weeks")},p.asWeeks=function(){return this.as("weeks")},p.months=function(){return this.get("months")},p.asMonths=function(){return this.as("months")},p.years=function(){return this.get("years")},p.asYears=function(){return this.as("years")},A})(),N=function(A,p,m){return A.add(p.years()*m,"y").add(p.months()*m,"M").add(p.days()*m,"d").add(p.hours()*m,"h").add(p.minutes()*m,"m").add(p.seconds()*m,"s").add(p.milliseconds()*m,"ms")};return function(A,p,m){r=m,n=m().$utils(),m.duration=function(y,s){return z(y,{$l:m.locale()},s)},m.isDuration=$;var h=p.prototype.add,g=p.prototype.subtract;p.prototype.add=function(y,s){return $(y)?N(this,y,1):h.bind(this)(y,s)},p.prototype.subtract=function(y,s){return $(y)?N(this,y,-1):g.bind(this)(y,s)}}}))})),qe=Oe(),q=ct(te(),1),Xe=ct(Ve(),1),Qe=ct(Re(),1),Je=ct(Ze(),1),mt=ct(te(),1),Ke=ct(Ue(),1),Ct=(function(){var t=u(function(s,c,f,l){for(f||(f={}),l=s.length;l--;f[s[l]]=c);return f},"o"),i=[6,8,10,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,33,35,36,38,40],r=[1,26],n=[1,27],o=[1,28],d=[1,29],k=[1,30],M=[1,31],I=[1,32],D=[1,33],x=[1,34],F=[1,9],w=[1,10],$=[1,11],z=[1,12],Y=[1,13],v=[1,14],C=[1,15],L=[1,16],O=[1,19],H=[1,20],N=[1,21],A=[1,22],p=[1,23],m=[1,25],h=[1,35],g={trace:u(function(){},"trace"),yy:{},symbols_:{error:2,start:3,gantt:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NL:10,weekday:11,weekday_monday:12,weekday_tuesday:13,weekday_wednesday:14,weekday_thursday:15,weekday_friday:16,weekday_saturday:17,weekday_sunday:18,weekend:19,weekend_friday:20,weekend_saturday:21,dateFormat:22,inclusiveEndDates:23,topAxis:24,axisFormat:25,tickInterval:26,excludes:27,includes:28,todayMarker:29,title:30,acc_title:31,acc_title_value:32,acc_descr:33,acc_descr_value:34,acc_descr_multiline_value:35,section:36,clickStatement:37,taskTxt:38,taskData:39,click:40,callbackname:41,callbackargs:42,href:43,clickStatementDebug:44,$accept:0,$end:1},terminals_:{2:"error",4:"gantt",6:"EOF",8:"SPACE",10:"NL",12:"weekday_monday",13:"weekday_tuesday",14:"weekday_wednesday",15:"weekday_thursday",16:"weekday_friday",17:"weekday_saturday",18:"weekday_sunday",20:"weekend_friday",21:"weekend_saturday",22:"dateFormat",23:"inclusiveEndDates",24:"topAxis",25:"axisFormat",26:"tickInterval",27:"excludes",28:"includes",29:"todayMarker",30:"title",31:"acc_title",32:"acc_title_value",33:"acc_descr",34:"acc_descr_value",35:"acc_descr_multiline_value",36:"section",38:"taskTxt",39:"taskData",40:"click",41:"callbackname",42:"callbackargs",43:"href"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[19,1],[19,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,1],[9,2],[37,2],[37,3],[37,3],[37,4],[37,3],[37,4],[37,2],[44,2],[44,3],[44,3],[44,4],[44,3],[44,4],[44,2]],performAction:u(function(s,c,f,l,b,a,_){var e=a.length-1;switch(b){case 1:return a[e-1];case 2:this.$=[];break;case 3:a[e-1].push(a[e]),this.$=a[e-1];break;case 4:case 5:this.$=a[e];break;case 6:case 7:this.$=[];break;case 8:l.setWeekday("monday");break;case 9:l.setWeekday("tuesday");break;case 10:l.setWeekday("wednesday");break;case 11:l.setWeekday("thursday");break;case 12:l.setWeekday("friday");break;case 13:l.setWeekday("saturday");break;case 14:l.setWeekday("sunday");break;case 15:l.setWeekend("friday");break;case 16:l.setWeekend("saturday");break;case 17:l.setDateFormat(a[e].substr(11)),this.$=a[e].substr(11);break;case 18:l.enableInclusiveEndDates(),this.$=a[e].substr(18);break;case 19:l.TopAxis(),this.$=a[e].substr(8);break;case 20:l.setAxisFormat(a[e].substr(11)),this.$=a[e].substr(11);break;case 21:l.setTickInterval(a[e].substr(13)),this.$=a[e].substr(13);break;case 22:l.setExcludes(a[e].substr(9)),this.$=a[e].substr(9);break;case 23:l.setIncludes(a[e].substr(9)),this.$=a[e].substr(9);break;case 24:l.setTodayMarker(a[e].substr(12)),this.$=a[e].substr(12);break;case 27:l.setDiagramTitle(a[e].substr(6)),this.$=a[e].substr(6);break;case 28:this.$=a[e].trim(),l.setAccTitle(this.$);break;case 29:case 30:this.$=a[e].trim(),l.setAccDescription(this.$);break;case 31:l.addSection(a[e].substr(8)),this.$=a[e].substr(8);break;case 33:l.addTask(a[e-1],a[e]),this.$="task";break;case 34:this.$=a[e-1],l.setClickEvent(a[e-1],a[e],null);break;case 35:this.$=a[e-2],l.setClickEvent(a[e-2],a[e-1],a[e]);break;case 36:this.$=a[e-2],l.setClickEvent(a[e-2],a[e-1],null),l.setLink(a[e-2],a[e]);break;case 37:this.$=a[e-3],l.setClickEvent(a[e-3],a[e-2],a[e-1]),l.setLink(a[e-3],a[e]);break;case 38:this.$=a[e-2],l.setClickEvent(a[e-2],a[e],null),l.setLink(a[e-2],a[e-1]);break;case 39:this.$=a[e-3],l.setClickEvent(a[e-3],a[e-1],a[e]),l.setLink(a[e-3],a[e-2]);break;case 40:this.$=a[e-1],l.setLink(a[e-1],a[e]);break;case 41:case 47:this.$=a[e-1]+" "+a[e];break;case 42:case 43:case 45:this.$=a[e-2]+" "+a[e-1]+" "+a[e];break;case 44:case 46:this.$=a[e-3]+" "+a[e-2]+" "+a[e-1]+" "+a[e];break}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},t(i,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:17,12:r,13:n,14:o,15:d,16:k,17:M,18:I,19:18,20:D,21:x,22:F,23:w,24:$,25:z,26:Y,27:v,28:C,29:L,30:O,31:H,33:N,35:A,36:p,37:24,38:m,40:h},t(i,[2,7],{1:[2,1]}),t(i,[2,3]),{9:36,11:17,12:r,13:n,14:o,15:d,16:k,17:M,18:I,19:18,20:D,21:x,22:F,23:w,24:$,25:z,26:Y,27:v,28:C,29:L,30:O,31:H,33:N,35:A,36:p,37:24,38:m,40:h},t(i,[2,5]),t(i,[2,6]),t(i,[2,17]),t(i,[2,18]),t(i,[2,19]),t(i,[2,20]),t(i,[2,21]),t(i,[2,22]),t(i,[2,23]),t(i,[2,24]),t(i,[2,25]),t(i,[2,26]),t(i,[2,27]),{32:[1,37]},{34:[1,38]},t(i,[2,30]),t(i,[2,31]),t(i,[2,32]),{39:[1,39]},t(i,[2,8]),t(i,[2,9]),t(i,[2,10]),t(i,[2,11]),t(i,[2,12]),t(i,[2,13]),t(i,[2,14]),t(i,[2,15]),t(i,[2,16]),{41:[1,40],43:[1,41]},t(i,[2,4]),t(i,[2,28]),t(i,[2,29]),t(i,[2,33]),t(i,[2,34],{42:[1,42],43:[1,43]}),t(i,[2,40],{41:[1,44]}),t(i,[2,35],{43:[1,45]}),t(i,[2,36]),t(i,[2,38],{42:[1,46]}),t(i,[2,37]),t(i,[2,39])],defaultActions:{},parseError:u(function(s,c){if(c.recoverable)this.trace(s);else{var f=Error(s);throw f.hash=c,f}},"parseError"),parse:u(function(s){var c=this,f=[0],l=[],b=[null],a=[],_=this.table,e="",T=0,S=0,E=0,W=2,P=1,j=a.slice.call(arguments,1),B=Object.create(this.lexer),X={yy:{}};for(var ht in this.yy)Object.prototype.hasOwnProperty.call(this.yy,ht)&&(X.yy[ht]=this.yy[ht]);B.setInput(s,X.yy),X.yy.lexer=B,X.yy.parser=this,B.yylloc===void 0&&(B.yylloc={});var at=B.yylloc;a.push(at);var gt=B.options&&B.options.ranges;typeof X.yy.parseError=="function"?this.parseError=X.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function ft(U){f.length-=2*U,b.length-=U,a.length-=U}u(ft,"popStack");function ot(){var U=l.pop()||B.lex()||P;return typeof U!="number"&&(U instanceof Array&&(l=U,U=l.pop()),U=c.symbols_[U]||U),U}u(ot,"lex");for(var V,et,Z,R,nt,Q={},it,K,Vt,bt;;){if(Z=f[f.length-1],this.defaultActions[Z]?R=this.defaultActions[Z]:(V??(V=ot()),R=_[Z]&&_[Z][V]),R===void 0||!R.length||!R[0]){var Rt="";for(it in bt=[],_[Z])this.terminals_[it]&&it>W&&bt.push("'"+this.terminals_[it]+"'");Rt=B.showPosition?"Parse error on line "+(T+1)+`:
`+B.showPosition()+`
Expecting `+bt.join(", ")+", got '"+(this.terminals_[V]||V)+"'":"Parse error on line "+(T+1)+": Unexpected "+(V==P?"end of input":"'"+(this.terminals_[V]||V)+"'"),this.parseError(Rt,{text:B.match,token:this.terminals_[V]||V,line:B.yylineno,loc:at,expected:bt})}if(R[0]instanceof Array&&R.length>1)throw Error("Parse Error: multiple actions possible at state: "+Z+", token: "+V);switch(R[0]){case 1:f.push(V),b.push(B.yytext),a.push(B.yylloc),f.push(R[1]),V=null,et?(V=et,et=null):(S=B.yyleng,e=B.yytext,T=B.yylineno,at=B.yylloc,E>0&&E--);break;case 2:if(K=this.productions_[R[1]][1],Q.$=b[b.length-K],Q._$={first_line:a[a.length-(K||1)].first_line,last_line:a[a.length-1].last_line,first_column:a[a.length-(K||1)].first_column,last_column:a[a.length-1].last_column},gt&&(Q._$.range=[a[a.length-(K||1)].range[0],a[a.length-1].range[1]]),nt=this.performAction.apply(Q,[e,S,T,X.yy,R[1],b,a].concat(j)),nt!==void 0)return nt;K&&(f=f.slice(0,-1*K*2),b=b.slice(0,-1*K),a=a.slice(0,-1*K)),f.push(this.productions_[R[1]][0]),b.push(Q.$),a.push(Q._$),Vt=_[f[f.length-2]][f[f.length-1]],f.push(Vt);break;case 3:return!0}}return!0},"parse")};g.lexer=(function(){return{EOF:1,parseError:u(function(s,c){if(this.yy.parser)this.yy.parser.parseError(s,c);else throw Error(s)},"parseError"),setInput:u(function(s,c){return this.yy=c||this.yy||{},this._input=s,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:u(function(){var s=this._input[0];return this.yytext+=s,this.yyleng++,this.offset++,this.match+=s,this.matched+=s,s.match(/(?:\r\n?|\n).*/g)?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),s},"input"),unput:u(function(s){var c=s.length,f=s.split(/(?:\r\n?|\n)/g);this._input=s+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-c),this.offset-=c;var l=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),f.length-1&&(this.yylineno-=f.length-1);var b=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:f?(f.length===l.length?this.yylloc.first_column:0)+l[l.length-f.length].length-f[0].length:this.yylloc.first_column-c},this.options.ranges&&(this.yylloc.range=[b[0],b[0]+this.yyleng-c]),this.yyleng=this.yytext.length,this},"unput"),more:u(function(){return this._more=!0,this},"more"),reject:u(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:u(function(s){this.unput(this.match.slice(s))},"less"),pastInput:u(function(){var s=this.matched.substr(0,this.matched.length-this.match.length);return(s.length>20?"...":"")+s.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:u(function(){var s=this.match;return s.length<20&&(s+=this._input.substr(0,20-s.length)),(s.substr(0,20)+(s.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:u(function(){var s=this.pastInput(),c=Array(s.length+1).join("-");return s+this.upcomingInput()+`
`+c+"^"},"showPosition"),test_match:u(function(s,c){var f,l,b;if(this.options.backtrack_lexer&&(b={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(b.yylloc.range=this.yylloc.range.slice(0))),l=s[0].match(/(?:\r\n?|\n).*/g),l&&(this.yylineno+=l.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:l?l[l.length-1].length-l[l.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+s[0].length},this.yytext+=s[0],this.match+=s[0],this.matches=s,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(s[0].length),this.matched+=s[0],f=this.performAction.call(this,this.yy,this,c,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),f)return f;if(this._backtrack){for(var a in b)this[a]=b[a];return!1}return!1},"test_match"),next:u(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var s,c,f,l;this._more||(this.yytext="",this.match="");for(var b=this._currentRules(),a=0;a<b.length;a++)if(f=this._input.match(this.rules[b[a]]),f&&(!c||f[0].length>c[0].length)){if(c=f,l=a,this.options.backtrack_lexer){if(s=this.test_match(f,b[a]),s!==!1)return s;if(this._backtrack){c=!1;continue}else return!1}else if(!this.options.flex)break}return c?(s=this.test_match(c,b[l]),s===!1?!1:s):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:u(function(){return this.next()||this.lex()},"lex"),begin:u(function(s){this.conditionStack.push(s)},"begin"),popState:u(function(){return this.conditionStack.length-1>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:u(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:u(function(s){return s=this.conditionStack.length-1-Math.abs(s||0),s>=0?this.conditionStack[s]:"INITIAL"},"topState"),pushState:u(function(s){this.begin(s)},"pushState"),stateStackSize:u(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:u(function(s,c,f,l){switch(f){case 0:return this.begin("open_directive"),"open_directive";case 1:return this.begin("acc_title"),31;case 2:return this.popState(),"acc_title_value";case 3:return this.begin("acc_descr"),33;case 4:return this.popState(),"acc_descr_value";case 5:this.begin("acc_descr_multiline");break;case 6:this.popState();break;case 7:return"acc_descr_multiline_value";case 8:break;case 9:break;case 10:break;case 11:return 10;case 12:break;case 13:break;case 14:this.begin("href");break;case 15:this.popState();break;case 16:return 43;case 17:this.begin("callbackname");break;case 18:this.popState();break;case 19:this.popState(),this.begin("callbackargs");break;case 20:return 41;case 21:this.popState();break;case 22:return 42;case 23:this.begin("click");break;case 24:this.popState();break;case 25:return 40;case 26:return 4;case 27:return 22;case 28:return 23;case 29:return 24;case 30:return 25;case 31:return 26;case 32:return 28;case 33:return 27;case 34:return 29;case 35:return 12;case 36:return 13;case 37:return 14;case 38:return 15;case 39:return 16;case 40:return 17;case 41:return 18;case 42:return 20;case 43:return 21;case 44:return"date";case 45:return 30;case 46:return"accDescription";case 47:return 36;case 48:return 38;case 49:return 39;case 50:return":";case 51:return 6;case 52:return"INVALID"}},"anonymous"),rules:[/^(?:%%\{)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:%%(?!\{)*[^\n]*)/i,/^(?:[^\}]%%*[^\n]*)/i,/^(?:%%*[^\n]*[\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:%[^\n]*)/i,/^(?:href[\s]+["])/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:call[\s]+)/i,/^(?:\([\s]*\))/i,/^(?:\()/i,/^(?:[^(]*)/i,/^(?:\))/i,/^(?:[^)]*)/i,/^(?:click[\s]+)/i,/^(?:[\s\n])/i,/^(?:[^\s\n]*)/i,/^(?:gantt\b)/i,/^(?:dateFormat\s[^#\n;]+)/i,/^(?:inclusiveEndDates\b)/i,/^(?:topAxis\b)/i,/^(?:axisFormat\s[^#\n;]+)/i,/^(?:tickInterval\s[^#\n;]+)/i,/^(?:includes\s[^#\n;]+)/i,/^(?:excludes\s[^#\n;]+)/i,/^(?:todayMarker\s[^\n;]+)/i,/^(?:weekday\s+monday\b)/i,/^(?:weekday\s+tuesday\b)/i,/^(?:weekday\s+wednesday\b)/i,/^(?:weekday\s+thursday\b)/i,/^(?:weekday\s+friday\b)/i,/^(?:weekday\s+saturday\b)/i,/^(?:weekday\s+sunday\b)/i,/^(?:weekend\s+friday\b)/i,/^(?:weekend\s+saturday\b)/i,/^(?:\d\d\d\d-\d\d-\d\d\b)/i,/^(?:title\s[^\n]+)/i,/^(?:accDescription\s[^#\n;]+)/i,/^(?:section\s[^\n]+)/i,/^(?:[^:\n]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[6,7],inclusive:!1},acc_descr:{rules:[4],inclusive:!1},acc_title:{rules:[2],inclusive:!1},callbackargs:{rules:[21,22],inclusive:!1},callbackname:{rules:[18,19,20],inclusive:!1},href:{rules:[15,16],inclusive:!1},click:{rules:[24,25],inclusive:!1},INITIAL:{rules:[0,1,3,5,8,9,10,11,12,13,14,17,23,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52],inclusive:!0}}}})();function y(){this.yy={}}return u(y,"Parser"),y.prototype=g,g.Parser=y,new y})();Ct.parser=Ct;var ti=Ct;q.default.extend(Xe.default),q.default.extend(Qe.default),q.default.extend(Je.default);var ne={friday:5,saturday:6},J="",Et="",Yt=void 0,At="",yt=[],kt=[],It=new Map,Lt=[],wt=[],ut="",Ft="",se=["active","done","crit","milestone","vert"],Ot=[],pt=!1,Wt=!1,Pt="sunday",$t="saturday",zt=0,ei=u(function(){Lt=[],wt=[],ut="",Ot=[],Nt=0,Bt=void 0,_t=void 0,G=[],J="",Et="",Ft="",Yt=void 0,At="",yt=[],kt=[],pt=!1,Wt=!1,zt=0,It=new Map,Ye(),Pt="sunday",$t="saturday"},"clear"),ii=u(function(t){Et=t},"setAxisFormat"),ni=u(function(){return Et},"getAxisFormat"),si=u(function(t){Yt=t},"setTickInterval"),ri=u(function(){return Yt},"getTickInterval"),ai=u(function(t){At=t},"setTodayMarker"),oi=u(function(){return At},"getTodayMarker"),ci=u(function(t){J=t},"setDateFormat"),li=u(function(){pt=!0},"enableInclusiveEndDates"),ui=u(function(){return pt},"endDatesAreInclusive"),di=u(function(){Wt=!0},"enableTopAxis"),hi=u(function(){return Wt},"topAxisEnabled"),fi=u(function(t){Ft=t},"setDisplayMode"),mi=u(function(){return Ft},"getDisplayMode"),yi=u(function(){return J},"getDateFormat"),ki=u(function(t){yt=t.toLowerCase().split(/[\s,]+/)},"setIncludes"),pi=u(function(){return yt},"getIncludes"),gi=u(function(t){kt=t.toLowerCase().split(/[\s,]+/)},"setExcludes"),bi=u(function(){return kt},"getExcludes"),vi=u(function(){return It},"getLinks"),Ti=u(function(t){ut=t,Lt.push(t)},"addSection"),xi=u(function(){return Lt},"getSections"),wi=u(function(){let t=ue(),i=0;for(;!t&&i<10;)t=ue(),i++;return wt=G,wt},"getTasks"),re=u(function(t,i,r,n){let o=t.format(i.trim()),d=t.format("YYYY-MM-DD");return n.includes(o)||n.includes(d)?!1:r.includes("weekends")&&(t.isoWeekday()===ne[$t]||t.isoWeekday()===ne[$t]+1)||r.includes(t.format("dddd").toLowerCase())?!0:r.includes(o)||r.includes(d)},"isInvalidDate"),$i=u(function(t){Pt=t},"setWeekday"),_i=u(function(){return Pt},"getWeekday"),Di=u(function(t){$t=t},"setWeekend"),ae=u(function(t,i,r,n){if(!r.length||t.manualEndTime)return;let o;o=t.startTime instanceof Date?(0,q.default)(t.startTime):(0,q.default)(t.startTime,i,!0),o=o.add(1,"d");let d;d=t.endTime instanceof Date?(0,q.default)(t.endTime):(0,q.default)(t.endTime,i,!0);let[k,M]=Si(o,d,i,r,n);t.endTime=k.toDate(),t.renderEndTime=M},"checkTaskDates"),Si=u(function(t,i,r,n,o){let d=!1,k=null;for(;t<=i;)d||(k=i.toDate()),d=re(t,r,n,o),d&&(i=i.add(1,"d")),t=t.add(1,"d");return[i,k]},"fixTaskDates"),Ht=u(function(t,i,r){if(r=r.trim(),u(d=>{let k=d.trim();return k==="x"||k==="X"},"isTimestampFormat")(i)&&/^\d+$/.test(r))return new Date(Number(r));let n=/^after\s+(?<ids>[\d\w- ]+)/.exec(r);if(n!==null){let d=null;for(let M of n.groups.ids.split(" ")){let I=rt(M);I!==void 0&&(!d||I.endTime>d.endTime)&&(d=I)}if(d)return d.endTime;let k=new Date;return k.setHours(0,0,0,0),k}let o=(0,q.default)(r,i.trim(),!0);if(o.isValid())return o.toDate();{st.debug("Invalid date:"+r),st.debug("With date format:"+i.trim());let d=new Date(r);if(d===void 0||isNaN(d.getTime())||d.getFullYear()<-1e4||d.getFullYear()>1e4)throw Error("Invalid date:"+r);return d}},"getStartDate"),oe=u(function(t){let i=/^(\d+(?:\.\d+)?)([Mdhmswy]|ms)$/.exec(t.trim());return i===null?[NaN,"ms"]:[Number.parseFloat(i[1]),i[2]]},"parseDuration"),ce=u(function(t,i,r,n=!1){r=r.trim();let o=/^until\s+(?<ids>[\d\w- ]+)/.exec(r);if(o!==null){let D=null;for(let F of o.groups.ids.split(" ")){let w=rt(F);w!==void 0&&(!D||w.startTime<D.startTime)&&(D=w)}if(D)return D.startTime;let x=new Date;return x.setHours(0,0,0,0),x}let d=(0,q.default)(r,i.trim(),!0);if(d.isValid())return n&&(d=d.add(1,"d")),d.toDate();let k=(0,q.default)(t),[M,I]=oe(r);if(!Number.isNaN(M)){let D=k.add(M,I);D.isValid()&&(k=D)}return k.toDate()},"getEndDate"),Nt=0,dt=u(function(t){return t===void 0?(Nt+=1,"task"+Nt):t},"parseId"),Mi=u(function(t,i){let r;r=i.substr(0,1)===":"?i.substr(1,i.length):i;let n=r.split(","),o={};jt(n,o,se);for(let k=0;k<n.length;k++)n[k]=n[k].trim();let d="";switch(n.length){case 1:o.id=dt(),o.startTime=t.endTime,d=n[0];break;case 2:o.id=dt(),o.startTime=Ht(void 0,J,n[0]),d=n[1];break;case 3:o.id=dt(n[0]),o.startTime=Ht(void 0,J,n[1]),d=n[2];break;default:}return d&&(o.endTime=ce(o.startTime,J,d,pt),o.manualEndTime=(0,q.default)(d,"YYYY-MM-DD",!0).isValid(),ae(o,J,kt,yt)),o},"compileData"),Ci=u(function(t,i){let r;r=i.substr(0,1)===":"?i.substr(1,i.length):i;let n=r.split(","),o={};jt(n,o,se);for(let d=0;d<n.length;d++)n[d]=n[d].trim();switch(n.length){case 1:o.id=dt(),o.startTime={type:"prevTaskEnd",id:t},o.endTime={data:n[0]};break;case 2:o.id=dt(),o.startTime={type:"getStartDate",startData:n[0]},o.endTime={data:n[1]};break;case 3:o.id=dt(n[0]),o.startTime={type:"getStartDate",startData:n[1]},o.endTime={data:n[2]};break;default:}return o},"parseData"),Bt,_t,G=[],le={},Ei=u(function(t,i){let r={section:ut,type:ut,processed:!1,manualEndTime:!1,renderEndTime:null,raw:{data:i},task:t,classes:[]},n=Ci(_t,i);r.raw.startTime=n.startTime,r.raw.endTime=n.endTime,r.id=n.id,r.prevTaskId=_t,r.active=n.active,r.done=n.done,r.crit=n.crit,r.milestone=n.milestone,r.vert=n.vert,r.order=zt,zt++;let o=G.push(r);_t=r.id,le[r.id]=o-1},"addTask"),rt=u(function(t){let i=le[t];return G[i]},"findTaskById"),Yi=u(function(t,i){let r={section:ut,type:ut,description:t,task:t,classes:[]},n=Mi(Bt,i);r.startTime=n.startTime,r.endTime=n.endTime,r.id=n.id,r.active=n.active,r.done=n.done,r.crit=n.crit,r.milestone=n.milestone,r.vert=n.vert,Bt=r,wt.push(r)},"addTaskOrg"),ue=u(function(){let t=u(function(r){let n=G[r],o="";switch(G[r].raw.startTime.type){case"prevTaskEnd":n.startTime=rt(n.prevTaskId).endTime;break;case"getStartDate":o=Ht(void 0,J,G[r].raw.startTime.startData),o&&(G[r].startTime=o);break}return G[r].startTime&&(G[r].endTime=ce(G[r].startTime,J,G[r].raw.endTime.data,pt),G[r].endTime&&(G[r].processed=!0,G[r].manualEndTime=(0,q.default)(G[r].raw.endTime.data,"YYYY-MM-DD",!0).isValid(),ae(G[r],J,kt,yt))),G[r].processed},"compileTask"),i=!0;for(let[r,n]of G.entries())t(r),i&&(i=n.processed);return i},"compileTasks"),Ai=u(function(t,i){let r=i;lt().securityLevel!=="loose"&&(r=(0,qe.sanitizeUrl)(i)),t.split(",").forEach(function(n){rt(n)!==void 0&&(he(n,()=>{window.open(r,"_self")}),It.set(n,r))}),de(t,"clickable")},"setLink"),de=u(function(t,i){t.split(",").forEach(function(r){let n=rt(r);n!==void 0&&n.classes.push(i)})},"setClass"),Ii=u(function(t,i,r){if(lt().securityLevel!=="loose"||i===void 0)return;let n=[];if(typeof r=="string"){n=r.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/);for(let o=0;o<n.length;o++){let d=n[o].trim();d.startsWith('"')&&d.endsWith('"')&&(d=d.substr(1,d.length-2)),n[o]=d}}n.length===0&&n.push(t),rt(t)!==void 0&&he(t,()=>{De.runFunc(i,...n)})},"setClickFun"),he=u(function(t,i){Ot.push(function(){let r=document.querySelector(`[id="${t}"]`);r!==null&&r.addEventListener("click",function(){i()})},function(){let r=document.querySelector(`[id="${t}-text"]`);r!==null&&r.addEventListener("click",function(){i()})})},"pushFun"),Li={getConfig:u(()=>lt().gantt,"getConfig"),clear:ei,setDateFormat:ci,getDateFormat:yi,enableInclusiveEndDates:li,endDatesAreInclusive:ui,enableTopAxis:di,topAxisEnabled:hi,setAxisFormat:ii,getAxisFormat:ni,setTickInterval:si,getTickInterval:ri,setTodayMarker:ai,getTodayMarker:oi,setAccTitle:Se,getAccTitle:Le,setDiagramTitle:Ce,getDiagramTitle:Me,setDisplayMode:fi,getDisplayMode:mi,setAccDescription:Fe,getAccDescription:Ee,addSection:Ti,getSections:xi,getTasks:wi,addTask:Ei,findTaskById:rt,addTaskOrg:Yi,setIncludes:ki,getIncludes:pi,setExcludes:gi,getExcludes:bi,setClickEvent:u(function(t,i,r){t.split(",").forEach(function(n){Ii(n,i,r)}),de(t,"clickable")},"setClickEvent"),setLink:Ai,getLinks:vi,bindFunctions:u(function(t){Ot.forEach(function(i){i(t)})},"bindFunctions"),parseDuration:oe,isInvalidDate:re,setWeekday:$i,getWeekday:_i,setWeekend:Di};function jt(t,i,r){let n=!0;for(;n;)n=!1,r.forEach(function(o){let d="^\\s*"+o+"\\s*$",k=new RegExp(d);t[0].match(k)&&(i[o]=!0,t.shift(1),n=!0)})}u(jt,"getTaskTags"),mt.default.extend(Ke.default);var Fi=u(function(){st.debug("Something is calling, setConf, remove the call")},"setConf"),fe={monday:ge,tuesday:Te,wednesday:ve,thursday:xe,friday:$e,saturday:be,sunday:we},Oi=u((t,i)=>{let r=[...t].map(()=>-1/0),n=[...t].sort((d,k)=>d.startTime-k.startTime||d.order-k.order),o=0;for(let d of n)for(let k=0;k<r.length;k++)if(d.startTime>=r[k]){r[k]=d.endTime,d.order=k+i,k>o&&(o=k);break}return o},"getMaxIntersections"),tt,Gt=1e4,Wi={parser:ti,db:Li,renderer:{setConf:Fi,draw:u(function(t,i,r,n){let o=lt().gantt,d=lt().securityLevel,k;d==="sandbox"&&(k=Dt("#i"+i));let M=Dt(d==="sandbox"?k.nodes()[0].contentDocument.body:"body"),I=d==="sandbox"?k.nodes()[0].contentDocument:document,D=I.getElementById(i);tt=D.parentElement.offsetWidth,tt===void 0&&(tt=1200),o.useWidth!==void 0&&(tt=o.useWidth);let x=n.db.getTasks(),F=[];for(let h of x)F.push(h.type);F=m(F);let w={},$=2*o.topPadding;if(n.db.getDisplayMode()==="compact"||o.displayMode==="compact"){let h={};for(let y of x)h[y.section]===void 0?h[y.section]=[y]:h[y.section].push(y);let g=0;for(let y of Object.keys(h)){let s=Oi(h[y],g)+1;g+=s,$+=s*(o.barHeight+o.barGap),w[y]=s}}else{$+=x.length*(o.barHeight+o.barGap);for(let h of F)w[h]=x.filter(g=>g.type===h).length}D.setAttribute("viewBox","0 0 "+tt+" "+$);let z=M.select(`[id="${i}"]`),Y=ye().domain([ke(x,function(h){return h.startTime}),pe(x,function(h){return h.endTime})]).rangeRound([0,tt-o.leftPadding-o.rightPadding]);function v(h,g){let y=h.startTime,s=g.startTime,c=0;return y>s?c=1:y<s&&(c=-1),c}u(v,"taskCompare"),x.sort(v),C(x,tt,$),Ae(z,$,tt,o.useMaxWidth),z.append("text").text(n.db.getDiagramTitle()).attr("x",tt/2).attr("y",o.titleTopMargin).attr("class","titleText");function C(h,g,y){let s=o.barHeight,c=s+o.barGap,f=o.topPadding,l=o.leftPadding,b=me().domain([0,F.length]).range(["#00B9FA","#F95002"]).interpolate(_e);O(c,f,l,g,y,h,n.db.getExcludes(),n.db.getIncludes()),N(l,f,g,y),L(h,c,f,l,s,b,g,y),A(c,f,l,s,b),p(l,f,g,y)}u(C,"makeGantt");function L(h,g,y,s,c,f,l){h.sort((e,T)=>e.vert===T.vert?0:e.vert?1:-1);let b=[...new Set(h.map(e=>e.order))].map(e=>h.find(T=>T.order===e));z.append("g").selectAll("rect").data(b).enter().append("rect").attr("x",0).attr("y",function(e,T){return T=e.order,T*g+y-2}).attr("width",function(){return l-o.rightPadding/2}).attr("height",g).attr("class",function(e){for(let[T,S]of F.entries())if(e.type===S)return"section section"+T%o.numberSectionStyles;return"section section0"}).enter();let a=z.append("g").selectAll("rect").data(h).enter(),_=n.db.getLinks();if(a.append("rect").attr("id",function(e){return e.id}).attr("rx",3).attr("ry",3).attr("x",function(e){return e.milestone?Y(e.startTime)+s+.5*(Y(e.endTime)-Y(e.startTime))-.5*c:Y(e.startTime)+s}).attr("y",function(e,T){return T=e.order,e.vert?o.gridLineStartPadding:T*g+y}).attr("width",function(e){return e.milestone?c:e.vert?.08*c:Y(e.renderEndTime||e.endTime)-Y(e.startTime)}).attr("height",function(e){return e.vert?x.length*(o.barHeight+o.barGap)+o.barHeight*2:c}).attr("transform-origin",function(e,T){return T=e.order,(Y(e.startTime)+s+.5*(Y(e.endTime)-Y(e.startTime))).toString()+"px "+(T*g+y+.5*c).toString()+"px"}).attr("class",function(e){let T="";e.classes.length>0&&(T=e.classes.join(" "));let S=0;for(let[W,P]of F.entries())e.type===P&&(S=W%o.numberSectionStyles);let E="";return e.active?e.crit?E+=" activeCrit":E=" active":e.done?E=e.crit?" doneCrit":" done":e.crit&&(E+=" crit"),E.length===0&&(E=" task"),e.milestone&&(E=" milestone "+E),e.vert&&(E=" vert "+E),E+=S,E+=" "+T,"task"+E}),a.append("text").attr("id",function(e){return e.id+"-text"}).text(function(e){return e.task}).attr("font-size",o.fontSize).attr("x",function(e){let T=Y(e.startTime),S=Y(e.renderEndTime||e.endTime);if(e.milestone&&(T+=.5*(Y(e.endTime)-Y(e.startTime))-.5*c,S=T+c),e.vert)return Y(e.startTime)+s;let E=this.getBBox().width;return E>S-T?S+E+1.5*o.leftPadding>l?T+s-5:S+s+5:(S-T)/2+T+s}).attr("y",function(e,T){return e.vert?o.gridLineStartPadding+x.length*(o.barHeight+o.barGap)+60:(T=e.order,T*g+o.barHeight/2+(o.fontSize/2-2)+y)}).attr("text-height",c).attr("class",function(e){let T=Y(e.startTime),S=Y(e.endTime);e.milestone&&(S=T+c);let E=this.getBBox().width,W="";e.classes.length>0&&(W=e.classes.join(" "));let P=0;for(let[B,X]of F.entries())e.type===X&&(P=B%o.numberSectionStyles);let j="";return e.active&&(j=e.crit?"activeCritText"+P:"activeText"+P),e.done?j=e.crit?j+" doneCritText"+P:j+" doneText"+P:e.crit&&(j=j+" critText"+P),e.milestone&&(j+=" milestoneText"),e.vert&&(j+=" vertText"),E>S-T?S+E+1.5*o.leftPadding>l?W+" taskTextOutsideLeft taskTextOutside"+P+" "+j:W+" taskTextOutsideRight taskTextOutside"+P+" "+j+" width-"+E:W+" taskText taskText"+P+" "+j+" width-"+E}),lt().securityLevel==="sandbox"){let e;e=Dt("#i"+i);let T=e.nodes()[0].contentDocument;a.filter(function(S){return _.has(S.id)}).each(function(S){var E=T.querySelector("#"+S.id),W=T.querySelector("#"+S.id+"-text");let P=E.parentNode;var j=T.createElement("a");j.setAttribute("xlink:href",_.get(S.id)),j.setAttribute("target","_top"),P.appendChild(j),j.appendChild(E),j.appendChild(W)})}}u(L,"drawRects");function O(h,g,y,s,c,f,l,b){if(l.length===0&&b.length===0)return;let a,_;for(let{startTime:W,endTime:P}of f)(a===void 0||W<a)&&(a=W),(_===void 0||P>_)&&(_=P);if(!a||!_)return;if((0,mt.default)(_).diff((0,mt.default)(a),"year")>5){st.warn("The difference between the min and max time is more than 5 years. This will cause performance issues. Skipping drawing exclude days.");return}let e=n.db.getDateFormat(),T=[],S=null,E=(0,mt.default)(a);for(;E.valueOf()<=_;)n.db.isInvalidDate(E,e,l,b)?S?S.end=E:S={start:E,end:E}:S&&(S=(T.push(S),null)),E=E.add(1,"d");z.append("g").selectAll("rect").data(T).enter().append("rect").attr("id",W=>"exclude-"+W.start.format("YYYY-MM-DD")).attr("x",W=>Y(W.start.startOf("day"))+y).attr("y",o.gridLineStartPadding).attr("width",W=>Y(W.end.endOf("day"))-Y(W.start.startOf("day"))).attr("height",c-g-o.gridLineStartPadding).attr("transform-origin",function(W,P){return(Y(W.start)+y+.5*(Y(W.end)-Y(W.start))).toString()+"px "+(P*h+.5*c).toString()+"px"}).attr("class","exclude-range")}u(O,"drawExcludeDays");function H(h,g,y,s){if(y<=0||h>g)return 1/0;let c=g-h,f=mt.default.duration({[s??"day"]:y}).asMilliseconds();return f<=0?1/0:Math.ceil(c/f)}u(H,"getEstimatedTickCount");function N(h,g,y,s){let c=n.db.getDateFormat(),f=n.db.getAxisFormat(),l;l=f||(c==="D"?"%d":o.axisFormat??"%Y-%m-%d");let b=Ge(Y).tickSize(-s+g+o.gridLineStartPadding).tickFormat(Qt(l)),a=/^([1-9]\d*)(millisecond|second|minute|hour|day|week|month)$/.exec(n.db.getTickInterval()||o.tickInterval);if(a!==null){let _=parseInt(a[1],10);if(isNaN(_)||_<=0)st.warn(`Invalid tick interval value: "${a[1]}". Skipping custom tick interval.`);else{let e=a[2],T=n.db.getWeekday()||o.weekday,S=Y.domain(),E=S[0],W=S[1],P=H(E,W,_,e);if(P>Gt)st.warn(`The tick interval "${_}${e}" would generate ${P} ticks, which exceeds the maximum allowed (${Gt}). This may indicate an invalid date or time range. Skipping custom tick interval.`);else switch(e){case"millisecond":b.ticks(Ut.every(_));break;case"second":b.ticks(qt.every(_));break;case"minute":b.ticks(Zt.every(_));break;case"hour":b.ticks(Kt.every(_));break;case"day":b.ticks(Jt.every(_));break;case"week":b.ticks(fe[T].every(_));break;case"month":b.ticks(Xt.every(_));break}}}if(z.append("g").attr("class","grid").attr("transform","translate("+h+", "+(s-50)+")").call(b).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10).attr("dy","1em"),n.db.topAxisEnabled()||o.topAxis){let _=je(Y).tickSize(-s+g+o.gridLineStartPadding).tickFormat(Qt(l));if(a!==null){let e=parseInt(a[1],10);if(isNaN(e)||e<=0)st.warn(`Invalid tick interval value: "${a[1]}". Skipping custom tick interval.`);else{let T=a[2],S=n.db.getWeekday()||o.weekday,E=Y.domain(),W=E[0],P=E[1];if(H(W,P,e,T)<=Gt)switch(T){case"millisecond":_.ticks(Ut.every(e));break;case"second":_.ticks(qt.every(e));break;case"minute":_.ticks(Zt.every(e));break;case"hour":_.ticks(Kt.every(e));break;case"day":_.ticks(Jt.every(e));break;case"week":_.ticks(fe[S].every(e));break;case"month":_.ticks(Xt.every(e));break}}}z.append("g").attr("class","grid").attr("transform","translate("+h+", "+g+")").call(_).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10)}}u(N,"makeGrid");function A(h,g){let y=0,s=Object.keys(w).map(c=>[c,w[c]]);z.append("g").selectAll("text").data(s).enter().append(function(c){let f=c[0].split(Ie.lineBreakRegex),l=-(f.length-1)/2,b=I.createElementNS("http://www.w3.org/2000/svg","text");b.setAttribute("dy",l+"em");for(let[a,_]of f.entries()){let e=I.createElementNS("http://www.w3.org/2000/svg","tspan");e.setAttribute("alignment-baseline","central"),e.setAttribute("x","10"),a>0&&e.setAttribute("dy","1em"),e.textContent=_,b.appendChild(e)}return b}).attr("x",10).attr("y",function(c,f){if(f>0)for(let l=0;l<f;l++)return y+=s[f-1][1],c[1]*h/2+y*h+g;else return c[1]*h/2+g}).attr("font-size",o.sectionFontSize).attr("class",function(c){for(let[f,l]of F.entries())if(c[0]===l)return"sectionTitle sectionTitle"+f%o.numberSectionStyles;return"sectionTitle"})}u(A,"vertLabels");function p(h,g,y,s){let c=n.db.getTodayMarker();if(c==="off")return;let f=z.append("g").attr("class","today"),l=new Date,b=f.append("line");b.attr("x1",Y(l)+h).attr("x2",Y(l)+h).attr("y1",o.titleTopMargin).attr("y2",s-o.titleTopMargin).attr("class","today"),c!==""&&b.attr("style",c.replace(/,/g,";"))}u(p,"drawToday");function m(h){let g={},y=[];for(let s=0,c=h.length;s<c;++s)Object.prototype.hasOwnProperty.call(g,h[s])||(g[h[s]]=!0,y.push(h[s]));return y}u(m,"checkUnique")},"draw")},styles:u(t=>`
  .mermaid-main-font {
        font-family: ${t.fontFamily};
  }

  .exclude-range {
    fill: ${t.excludeBkgColor};
  }

  .section {
    stroke: none;
    opacity: 0.2;
  }

  .section0 {
    fill: ${t.sectionBkgColor};
  }

  .section2 {
    fill: ${t.sectionBkgColor2};
  }

  .section1,
  .section3 {
    fill: ${t.altSectionBkgColor};
    opacity: 0.2;
  }

  .sectionTitle0 {
    fill: ${t.titleColor};
  }

  .sectionTitle1 {
    fill: ${t.titleColor};
  }

  .sectionTitle2 {
    fill: ${t.titleColor};
  }

  .sectionTitle3 {
    fill: ${t.titleColor};
  }

  .sectionTitle {
    text-anchor: start;
    font-family: ${t.fontFamily};
  }


  /* Grid and axis */

  .grid .tick {
    stroke: ${t.gridColor};
    opacity: 0.8;
    shape-rendering: crispEdges;
  }

  .grid .tick text {
    font-family: ${t.fontFamily};
    fill: ${t.textColor};
  }

  .grid path {
    stroke-width: 0;
  }


  /* Today line */

  .today {
    fill: none;
    stroke: ${t.todayLineColor};
    stroke-width: 2px;
  }


  /* Task styling */

  /* Default task */

  .task {
    stroke-width: 2;
  }

  .taskText {
    text-anchor: middle;
    font-family: ${t.fontFamily};
  }

  .taskTextOutsideRight {
    fill: ${t.taskTextDarkColor};
    text-anchor: start;
    font-family: ${t.fontFamily};
  }

  .taskTextOutsideLeft {
    fill: ${t.taskTextDarkColor};
    text-anchor: end;
  }


  /* Special case clickable */

  .task.clickable {
    cursor: pointer;
  }

  .taskText.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideLeft.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideRight.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }


  /* Specific task settings for the sections*/

  .taskText0,
  .taskText1,
  .taskText2,
  .taskText3 {
    fill: ${t.taskTextColor};
  }

  .task0,
  .task1,
  .task2,
  .task3 {
    fill: ${t.taskBkgColor};
    stroke: ${t.taskBorderColor};
  }

  .taskTextOutside0,
  .taskTextOutside2
  {
    fill: ${t.taskTextOutsideColor};
  }

  .taskTextOutside1,
  .taskTextOutside3 {
    fill: ${t.taskTextOutsideColor};
  }


  /* Active task */

  .active0,
  .active1,
  .active2,
  .active3 {
    fill: ${t.activeTaskBkgColor};
    stroke: ${t.activeTaskBorderColor};
  }

  .activeText0,
  .activeText1,
  .activeText2,
  .activeText3 {
    fill: ${t.taskTextDarkColor} !important;
  }


  /* Completed task */

  .done0,
  .done1,
  .done2,
  .done3 {
    stroke: ${t.doneTaskBorderColor};
    fill: ${t.doneTaskBkgColor};
    stroke-width: 2;
  }

  .doneText0,
  .doneText1,
  .doneText2,
  .doneText3 {
    fill: ${t.taskTextDarkColor} !important;
  }


  /* Tasks on the critical line */

  .crit0,
  .crit1,
  .crit2,
  .crit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.critBkgColor};
    stroke-width: 2;
  }

  .activeCrit0,
  .activeCrit1,
  .activeCrit2,
  .activeCrit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.activeTaskBkgColor};
    stroke-width: 2;
  }

  .doneCrit0,
  .doneCrit1,
  .doneCrit2,
  .doneCrit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.doneTaskBkgColor};
    stroke-width: 2;
    cursor: pointer;
    shape-rendering: crispEdges;
  }

  .milestone {
    transform: rotate(45deg) scale(0.8,0.8);
  }

  .milestoneText {
    font-style: italic;
  }
  .doneCritText0,
  .doneCritText1,
  .doneCritText2,
  .doneCritText3 {
    fill: ${t.taskTextDarkColor} !important;
  }

  .vert {
    stroke: ${t.vertLineColor};
  }

  .vertText {
    font-size: 15px;
    text-anchor: middle;
    fill: ${t.vertLineColor} !important;
  }

  .activeCritText0,
  .activeCritText1,
  .activeCritText2,
  .activeCritText3 {
    fill: ${t.taskTextDarkColor} !important;
  }

  .titleText {
    text-anchor: middle;
    font-size: 18px;
    fill: ${t.titleColor||t.textColor};
    font-family: ${t.fontFamily};
  }
`,"getStyles")};export{Wi as diagram};
