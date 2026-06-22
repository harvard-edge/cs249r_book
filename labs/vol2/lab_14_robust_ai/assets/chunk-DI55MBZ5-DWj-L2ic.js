var K;import{g as te,s as ee}from"./chunk-S3R3BYOJ-QzFziHq0.js";import{n as u,r as T}from"./src-DSbF9mby.js";import{B as se,C as ie,U as re,_ as ne,a as ae,b as w,s as U,v as oe,z as le}from"./chunk-ABZYJK2D-C6__zbIg.js";import{r as ce}from"./chunk-N4CR4FBY-DEy2vl_t.js";import{t as he}from"./chunk-55IACEB6-CRnvfQlo.js";import{t as de}from"./chunk-QN33PNHL-CC2wokMI.js";var kt=(function(){var e=u(function(a,h,g,S){for(g||(g={}),S=a.length;S--;g[a[S]]=h);return g},"o"),t=[1,2],s=[1,3],n=[1,4],i=[2,4],o=[1,9],c=[1,11],y=[1,16],p=[1,17],_=[1,18],m=[1,19],D=[1,33],A=[1,20],x=[1,21],R=[1,22],$=[1,23],O=[1,24],d=[1,26],E=[1,27],v=[1,28],G=[1,29],j=[1,30],Y=[1,31],F=[1,32],rt=[1,35],nt=[1,36],at=[1,37],ot=[1,38],H=[1,34],f=[1,4,5,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,41,45,48,51,52,53,54,57],lt=[1,4,5,14,15,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,39,40,41,45,48,51,52,53,54,57],$t=[4,5,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,41,45,48,51,52,53,54,57],mt={trace:u(function(){},"trace"),yy:{},symbols_:{error:2,start:3,SPACE:4,NL:5,SD:6,document:7,line:8,statement:9,classDefStatement:10,styleStatement:11,cssClassStatement:12,idStatement:13,DESCR:14,"-->":15,HIDE_EMPTY:16,scale:17,WIDTH:18,COMPOSIT_STATE:19,STRUCT_START:20,STRUCT_STOP:21,STATE_DESCR:22,AS:23,ID:24,FORK:25,JOIN:26,CHOICE:27,CONCURRENT:28,note:29,notePosition:30,NOTE_TEXT:31,direction:32,acc_title:33,acc_title_value:34,acc_descr:35,acc_descr_value:36,acc_descr_multiline_value:37,CLICK:38,STRING:39,HREF:40,classDef:41,CLASSDEF_ID:42,CLASSDEF_STYLEOPTS:43,DEFAULT:44,style:45,STYLE_IDS:46,STYLEDEF_STYLEOPTS:47,class:48,CLASSENTITY_IDS:49,STYLECLASS:50,direction_tb:51,direction_bt:52,direction_rl:53,direction_lr:54,eol:55,";":56,EDGE_STATE:57,STYLE_SEPARATOR:58,left_of:59,right_of:60,$accept:0,$end:1},terminals_:{2:"error",4:"SPACE",5:"NL",6:"SD",14:"DESCR",15:"-->",16:"HIDE_EMPTY",17:"scale",18:"WIDTH",19:"COMPOSIT_STATE",20:"STRUCT_START",21:"STRUCT_STOP",22:"STATE_DESCR",23:"AS",24:"ID",25:"FORK",26:"JOIN",27:"CHOICE",28:"CONCURRENT",29:"note",31:"NOTE_TEXT",33:"acc_title",34:"acc_title_value",35:"acc_descr",36:"acc_descr_value",37:"acc_descr_multiline_value",38:"CLICK",39:"STRING",40:"HREF",41:"classDef",42:"CLASSDEF_ID",43:"CLASSDEF_STYLEOPTS",44:"DEFAULT",45:"style",46:"STYLE_IDS",47:"STYLEDEF_STYLEOPTS",48:"class",49:"CLASSENTITY_IDS",50:"STYLECLASS",51:"direction_tb",52:"direction_bt",53:"direction_rl",54:"direction_lr",56:";",57:"EDGE_STATE",58:"STYLE_SEPARATOR",59:"left_of",60:"right_of"},productions_:[0,[3,2],[3,2],[3,2],[7,0],[7,2],[8,2],[8,1],[8,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,3],[9,4],[9,1],[9,2],[9,1],[9,4],[9,3],[9,6],[9,1],[9,1],[9,1],[9,1],[9,4],[9,4],[9,1],[9,2],[9,2],[9,1],[9,5],[9,5],[10,3],[10,3],[11,3],[12,3],[32,1],[32,1],[32,1],[32,1],[55,1],[55,1],[13,1],[13,1],[13,3],[13,3],[30,1],[30,1]],performAction:u(function(a,h,g,S,b,r,Q){var l=r.length-1;switch(b){case 3:return S.setRootDoc(r[l]),r[l];case 4:this.$=[];break;case 5:r[l]!="nl"&&(r[l-1].push(r[l]),this.$=r[l-1]);break;case 6:case 7:this.$=r[l];break;case 8:this.$="nl";break;case 12:this.$=r[l];break;case 13:let ht=r[l-1];ht.description=S.trimColon(r[l]),this.$=ht;break;case 14:this.$={stmt:"relation",state1:r[l-2],state2:r[l]};break;case 15:let dt=S.trimColon(r[l]);this.$={stmt:"relation",state1:r[l-3],state2:r[l-1],description:dt};break;case 19:this.$={stmt:"state",id:r[l-3],type:"default",description:"",doc:r[l-1]};break;case 20:var z=r[l],X=r[l-2].trim();if(r[l].match(":")){var Z=r[l].split(":");z=Z[0],X=[X,Z[1]]}this.$={stmt:"state",id:z,type:"default",description:X};break;case 21:this.$={stmt:"state",id:r[l-3],type:"default",description:r[l-5],doc:r[l-1]};break;case 22:this.$={stmt:"state",id:r[l],type:"fork"};break;case 23:this.$={stmt:"state",id:r[l],type:"join"};break;case 24:this.$={stmt:"state",id:r[l],type:"choice"};break;case 25:this.$={stmt:"state",id:S.getDividerId(),type:"divider"};break;case 26:this.$={stmt:"state",id:r[l-1].trim(),note:{position:r[l-2].trim(),text:r[l].trim()}};break;case 29:this.$=r[l].trim(),S.setAccTitle(this.$);break;case 30:case 31:this.$=r[l].trim(),S.setAccDescription(this.$);break;case 32:this.$={stmt:"click",id:r[l-3],url:r[l-2],tooltip:r[l-1]};break;case 33:this.$={stmt:"click",id:r[l-3],url:r[l-1],tooltip:""};break;case 34:case 35:this.$={stmt:"classDef",id:r[l-1].trim(),classes:r[l].trim()};break;case 36:this.$={stmt:"style",id:r[l-1].trim(),styleClass:r[l].trim()};break;case 37:this.$={stmt:"applyClass",id:r[l-1].trim(),styleClass:r[l].trim()};break;case 38:S.setDirection("TB"),this.$={stmt:"dir",value:"TB"};break;case 39:S.setDirection("BT"),this.$={stmt:"dir",value:"BT"};break;case 40:S.setDirection("RL"),this.$={stmt:"dir",value:"RL"};break;case 41:S.setDirection("LR"),this.$={stmt:"dir",value:"LR"};break;case 44:case 45:this.$={stmt:"state",id:r[l].trim(),type:"default",description:""};break;case 46:this.$={stmt:"state",id:r[l-2].trim(),classes:[r[l].trim()],type:"default",description:""};break;case 47:this.$={stmt:"state",id:r[l-2].trim(),classes:[r[l].trim()],type:"default",description:""};break}},"anonymous"),table:[{3:1,4:t,5:s,6:n},{1:[3]},{3:5,4:t,5:s,6:n},{3:6,4:t,5:s,6:n},e([1,4,5,16,17,19,22,24,25,26,27,28,29,33,35,37,38,41,45,48,51,52,53,54,57],i,{7:7}),{1:[2,1]},{1:[2,2]},{1:[2,3],4:o,5:c,8:8,9:10,10:12,11:13,12:14,13:15,16:y,17:p,19:_,22:m,24:D,25:A,26:x,27:R,28:$,29:O,32:25,33:d,35:E,37:v,38:G,41:j,45:Y,48:F,51:rt,52:nt,53:at,54:ot,57:H},e(f,[2,5]),{9:39,10:12,11:13,12:14,13:15,16:y,17:p,19:_,22:m,24:D,25:A,26:x,27:R,28:$,29:O,32:25,33:d,35:E,37:v,38:G,41:j,45:Y,48:F,51:rt,52:nt,53:at,54:ot,57:H},e(f,[2,7]),e(f,[2,8]),e(f,[2,9]),e(f,[2,10]),e(f,[2,11]),e(f,[2,12],{14:[1,40],15:[1,41]}),e(f,[2,16]),{18:[1,42]},e(f,[2,18],{20:[1,43]}),{23:[1,44]},e(f,[2,22]),e(f,[2,23]),e(f,[2,24]),e(f,[2,25]),{30:45,31:[1,46],59:[1,47],60:[1,48]},e(f,[2,28]),{34:[1,49]},{36:[1,50]},e(f,[2,31]),{13:51,24:D,57:H},{42:[1,52],44:[1,53]},{46:[1,54]},{49:[1,55]},e(lt,[2,44],{58:[1,56]}),e(lt,[2,45],{58:[1,57]}),e(f,[2,38]),e(f,[2,39]),e(f,[2,40]),e(f,[2,41]),e(f,[2,6]),e(f,[2,13]),{13:58,24:D,57:H},e(f,[2,17]),e($t,i,{7:59}),{24:[1,60]},{24:[1,61]},{23:[1,62]},{24:[2,48]},{24:[2,49]},e(f,[2,29]),e(f,[2,30]),{39:[1,63],40:[1,64]},{43:[1,65]},{43:[1,66]},{47:[1,67]},{50:[1,68]},{24:[1,69]},{24:[1,70]},e(f,[2,14],{14:[1,71]}),{4:o,5:c,8:8,9:10,10:12,11:13,12:14,13:15,16:y,17:p,19:_,21:[1,72],22:m,24:D,25:A,26:x,27:R,28:$,29:O,32:25,33:d,35:E,37:v,38:G,41:j,45:Y,48:F,51:rt,52:nt,53:at,54:ot,57:H},e(f,[2,20],{20:[1,73]}),{31:[1,74]},{24:[1,75]},{39:[1,76]},{39:[1,77]},e(f,[2,34]),e(f,[2,35]),e(f,[2,36]),e(f,[2,37]),e(lt,[2,46]),e(lt,[2,47]),e(f,[2,15]),e(f,[2,19]),e($t,i,{7:78}),e(f,[2,26]),e(f,[2,27]),{5:[1,79]},{5:[1,80]},{4:o,5:c,8:8,9:10,10:12,11:13,12:14,13:15,16:y,17:p,19:_,21:[1,81],22:m,24:D,25:A,26:x,27:R,28:$,29:O,32:25,33:d,35:E,37:v,38:G,41:j,45:Y,48:F,51:rt,52:nt,53:at,54:ot,57:H},e(f,[2,32]),e(f,[2,33]),e(f,[2,21])],defaultActions:{5:[2,1],6:[2,2],47:[2,48],48:[2,49]},parseError:u(function(a,h){if(h.recoverable)this.trace(a);else{var g=Error(a);throw g.hash=h,g}},"parseError"),parse:u(function(a){var h=this,g=[0],S=[],b=[null],r=[],Q=this.table,l="",z=0,X=0,Z=0,ht=2,dt=1,qt=r.slice.call(arguments,1),k=Object.create(this.lexer),W={yy:{}};for(var St in this.yy)Object.prototype.hasOwnProperty.call(this.yy,St)&&(W.yy[St]=this.yy[St]);k.setInput(a,W.yy),W.yy.lexer=k,W.yy.parser=this,k.yylloc===void 0&&(k.yylloc={});var _t=k.yylloc;r.push(_t);var Qt=k.options&&k.options.ranges;typeof W.yy.parseError=="function"?this.parseError=W.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function Zt(N){g.length-=2*N,b.length-=N,r.length-=N}u(Zt,"popStack");function Lt(){var N=S.pop()||k.lex()||dt;return typeof N!="number"&&(N instanceof Array&&(S=N,N=S.pop()),N=h.symbols_[N]||N),N}u(Lt,"lex");for(var L,bt,M,I,Tt,V={},ut,B,At,pt;;){if(M=g[g.length-1],this.defaultActions[M]?I=this.defaultActions[M]:(L??(L=Lt()),I=Q[M]&&Q[M][L]),I===void 0||!I.length||!I[0]){var vt="";for(ut in pt=[],Q[M])this.terminals_[ut]&&ut>ht&&pt.push("'"+this.terminals_[ut]+"'");vt=k.showPosition?"Parse error on line "+(z+1)+`:
`+k.showPosition()+`
Expecting `+pt.join(", ")+", got '"+(this.terminals_[L]||L)+"'":"Parse error on line "+(z+1)+": Unexpected "+(L==dt?"end of input":"'"+(this.terminals_[L]||L)+"'"),this.parseError(vt,{text:k.match,token:this.terminals_[L]||L,line:k.yylineno,loc:_t,expected:pt})}if(I[0]instanceof Array&&I.length>1)throw Error("Parse Error: multiple actions possible at state: "+M+", token: "+L);switch(I[0]){case 1:g.push(L),b.push(k.yytext),r.push(k.yylloc),g.push(I[1]),L=null,bt?(L=bt,bt=null):(X=k.yyleng,l=k.yytext,z=k.yylineno,_t=k.yylloc,Z>0&&Z--);break;case 2:if(B=this.productions_[I[1]][1],V.$=b[b.length-B],V._$={first_line:r[r.length-(B||1)].first_line,last_line:r[r.length-1].last_line,first_column:r[r.length-(B||1)].first_column,last_column:r[r.length-1].last_column},Qt&&(V._$.range=[r[r.length-(B||1)].range[0],r[r.length-1].range[1]]),Tt=this.performAction.apply(V,[l,X,z,W.yy,I[1],b,r].concat(qt)),Tt!==void 0)return Tt;B&&(g=g.slice(0,-1*B*2),b=b.slice(0,-1*B),r=r.slice(0,-1*B)),g.push(this.productions_[I[1]][0]),b.push(V.$),r.push(V._$),At=Q[g[g.length-2]][g[g.length-1]],g.push(At);break;case 3:return!0}}return!0},"parse")};mt.lexer=(function(){return{EOF:1,parseError:u(function(a,h){if(this.yy.parser)this.yy.parser.parseError(a,h);else throw Error(a)},"parseError"),setInput:u(function(a,h){return this.yy=h||this.yy||{},this._input=a,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:u(function(){var a=this._input[0];return this.yytext+=a,this.yyleng++,this.offset++,this.match+=a,this.matched+=a,a.match(/(?:\r\n?|\n).*/g)?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),a},"input"),unput:u(function(a){var h=a.length,g=a.split(/(?:\r\n?|\n)/g);this._input=a+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-h),this.offset-=h;var S=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),g.length-1&&(this.yylineno-=g.length-1);var b=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:g?(g.length===S.length?this.yylloc.first_column:0)+S[S.length-g.length].length-g[0].length:this.yylloc.first_column-h},this.options.ranges&&(this.yylloc.range=[b[0],b[0]+this.yyleng-h]),this.yyleng=this.yytext.length,this},"unput"),more:u(function(){return this._more=!0,this},"more"),reject:u(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:u(function(a){this.unput(this.match.slice(a))},"less"),pastInput:u(function(){var a=this.matched.substr(0,this.matched.length-this.match.length);return(a.length>20?"...":"")+a.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:u(function(){var a=this.match;return a.length<20&&(a+=this._input.substr(0,20-a.length)),(a.substr(0,20)+(a.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:u(function(){var a=this.pastInput(),h=Array(a.length+1).join("-");return a+this.upcomingInput()+`
`+h+"^"},"showPosition"),test_match:u(function(a,h){var g,S,b;if(this.options.backtrack_lexer&&(b={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(b.yylloc.range=this.yylloc.range.slice(0))),S=a[0].match(/(?:\r\n?|\n).*/g),S&&(this.yylineno+=S.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:S?S[S.length-1].length-S[S.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+a[0].length},this.yytext+=a[0],this.match+=a[0],this.matches=a,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(a[0].length),this.matched+=a[0],g=this.performAction.call(this,this.yy,this,h,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),g)return g;if(this._backtrack){for(var r in b)this[r]=b[r];return!1}return!1},"test_match"),next:u(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var a,h,g,S;this._more||(this.yytext="",this.match="");for(var b=this._currentRules(),r=0;r<b.length;r++)if(g=this._input.match(this.rules[b[r]]),g&&(!h||g[0].length>h[0].length)){if(h=g,S=r,this.options.backtrack_lexer){if(a=this.test_match(g,b[r]),a!==!1)return a;if(this._backtrack){h=!1;continue}else return!1}else if(!this.options.flex)break}return h?(a=this.test_match(h,b[S]),a===!1?!1:a):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:u(function(){return this.next()||this.lex()},"lex"),begin:u(function(a){this.conditionStack.push(a)},"begin"),popState:u(function(){return this.conditionStack.length-1>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:u(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:u(function(a){return a=this.conditionStack.length-1-Math.abs(a||0),a>=0?this.conditionStack[a]:"INITIAL"},"topState"),pushState:u(function(a){this.begin(a)},"pushState"),stateStackSize:u(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:u(function(a,h,g,S){switch(g){case 0:return 38;case 1:return 40;case 2:return 39;case 3:return 44;case 4:return 51;case 5:return 52;case 6:return 53;case 7:return 54;case 8:break;case 9:break;case 10:return 5;case 11:break;case 12:break;case 13:break;case 14:break;case 15:return this.pushState("SCALE"),17;case 16:return 18;case 17:this.popState();break;case 18:return this.begin("acc_title"),33;case 19:return this.popState(),"acc_title_value";case 20:return this.begin("acc_descr"),35;case 21:return this.popState(),"acc_descr_value";case 22:this.begin("acc_descr_multiline");break;case 23:this.popState();break;case 24:return"acc_descr_multiline_value";case 25:return this.pushState("CLASSDEF"),41;case 26:return this.popState(),this.pushState("CLASSDEFID"),"DEFAULT_CLASSDEF_ID";case 27:return this.popState(),this.pushState("CLASSDEFID"),42;case 28:return this.popState(),43;case 29:return this.pushState("CLASS"),48;case 30:return this.popState(),this.pushState("CLASS_STYLE"),49;case 31:return this.popState(),50;case 32:return this.pushState("STYLE"),45;case 33:return this.popState(),this.pushState("STYLEDEF_STYLES"),46;case 34:return this.popState(),47;case 35:return this.pushState("SCALE"),17;case 36:return 18;case 37:this.popState();break;case 38:this.pushState("STATE");break;case 39:return this.popState(),h.yytext=h.yytext.slice(0,-8).trim(),25;case 40:return this.popState(),h.yytext=h.yytext.slice(0,-8).trim(),26;case 41:return this.popState(),h.yytext=h.yytext.slice(0,-10).trim(),27;case 42:return this.popState(),h.yytext=h.yytext.slice(0,-8).trim(),25;case 43:return this.popState(),h.yytext=h.yytext.slice(0,-8).trim(),26;case 44:return this.popState(),h.yytext=h.yytext.slice(0,-10).trim(),27;case 45:return 51;case 46:return 52;case 47:return 53;case 48:return 54;case 49:this.pushState("STATE_STRING");break;case 50:return this.pushState("STATE_ID"),"AS";case 51:return this.popState(),"ID";case 52:this.popState();break;case 53:return"STATE_DESCR";case 54:return 19;case 55:this.popState();break;case 56:return this.popState(),this.pushState("struct"),20;case 57:break;case 58:return this.popState(),21;case 59:break;case 60:return this.begin("NOTE"),29;case 61:return this.popState(),this.pushState("NOTE_ID"),59;case 62:return this.popState(),this.pushState("NOTE_ID"),60;case 63:this.popState(),this.pushState("FLOATING_NOTE");break;case 64:return this.popState(),this.pushState("FLOATING_NOTE_ID"),"AS";case 65:break;case 66:return"NOTE_TEXT";case 67:return this.popState(),"ID";case 68:return this.popState(),this.pushState("NOTE_TEXT"),24;case 69:return this.popState(),h.yytext=h.yytext.substr(2).trim(),31;case 70:return this.popState(),h.yytext=h.yytext.slice(0,-8).trim(),31;case 71:return 6;case 72:return 6;case 73:return 16;case 74:return 57;case 75:return 24;case 76:return h.yytext=h.yytext.trim(),14;case 77:return 15;case 78:return 28;case 79:return 58;case 80:return 5;case 81:return"INVALID"}},"anonymous"),rules:[/^(?:click\b)/i,/^(?:href\b)/i,/^(?:"[^"]*")/i,/^(?:default\b)/i,/^(?:.*direction\s+TB[^\n]*)/i,/^(?:.*direction\s+BT[^\n]*)/i,/^(?:.*direction\s+RL[^\n]*)/i,/^(?:.*direction\s+LR[^\n]*)/i,/^(?:%%(?!\{)[^\n]*)/i,/^(?:[^\}]%%[^\n]*)/i,/^(?:[\n]+)/i,/^(?:[\s]+)/i,/^(?:((?!\n)\s)+)/i,/^(?:#[^\n]*)/i,/^(?:%[^\n]*)/i,/^(?:scale\s+)/i,/^(?:\d+)/i,/^(?:\s+width\b)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:classDef\s+)/i,/^(?:DEFAULT\s+)/i,/^(?:\w+\s+)/i,/^(?:[^\n]*)/i,/^(?:class\s+)/i,/^(?:(\w+)+((,\s*\w+)*))/i,/^(?:[^\n]*)/i,/^(?:style\s+)/i,/^(?:[\w,]+\s+)/i,/^(?:[^\n]*)/i,/^(?:scale\s+)/i,/^(?:\d+)/i,/^(?:\s+width\b)/i,/^(?:state\s+)/i,/^(?:.*<<fork>>)/i,/^(?:.*<<join>>)/i,/^(?:.*<<choice>>)/i,/^(?:.*\[\[fork\]\])/i,/^(?:.*\[\[join\]\])/i,/^(?:.*\[\[choice\]\])/i,/^(?:.*direction\s+TB[^\n]*)/i,/^(?:.*direction\s+BT[^\n]*)/i,/^(?:.*direction\s+RL[^\n]*)/i,/^(?:.*direction\s+LR[^\n]*)/i,/^(?:["])/i,/^(?:\s*as\s+)/i,/^(?:[^\n\{]*)/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:[^\n\s\{]+)/i,/^(?:\n)/i,/^(?:\{)/i,/^(?:%%(?!\{)[^\n]*)/i,/^(?:\})/i,/^(?:[\n])/i,/^(?:note\s+)/i,/^(?:left of\b)/i,/^(?:right of\b)/i,/^(?:")/i,/^(?:\s*as\s*)/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:[^\n]*)/i,/^(?:\s*[^:\n\s\-]+)/i,/^(?:\s*:[^:\n;]+)/i,/^(?:[\s\S]*?end note\b)/i,/^(?:stateDiagram\s+)/i,/^(?:stateDiagram-v2\s+)/i,/^(?:hide empty description\b)/i,/^(?:\[\*\])/i,/^(?:[^:\n\s\-\{]+)/i,/^(?:\s*:[^:\n;]+)/i,/^(?:-->)/i,/^(?:--)/i,/^(?::::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{LINE:{rules:[12,13],inclusive:!1},struct:{rules:[12,13,25,29,32,38,45,46,47,48,57,58,59,60,74,75,76,77,78],inclusive:!1},FLOATING_NOTE_ID:{rules:[67],inclusive:!1},FLOATING_NOTE:{rules:[64,65,66],inclusive:!1},NOTE_TEXT:{rules:[69,70],inclusive:!1},NOTE_ID:{rules:[68],inclusive:!1},NOTE:{rules:[61,62,63],inclusive:!1},STYLEDEF_STYLEOPTS:{rules:[],inclusive:!1},STYLEDEF_STYLES:{rules:[34],inclusive:!1},STYLE_IDS:{rules:[],inclusive:!1},STYLE:{rules:[33],inclusive:!1},CLASS_STYLE:{rules:[31],inclusive:!1},CLASS:{rules:[30],inclusive:!1},CLASSDEFID:{rules:[28],inclusive:!1},CLASSDEF:{rules:[26,27],inclusive:!1},acc_descr_multiline:{rules:[23,24],inclusive:!1},acc_descr:{rules:[21],inclusive:!1},acc_title:{rules:[19],inclusive:!1},SCALE:{rules:[16,17,36,37],inclusive:!1},ALIAS:{rules:[],inclusive:!1},STATE_ID:{rules:[51],inclusive:!1},STATE_STRING:{rules:[52,53],inclusive:!1},FORK_STATE:{rules:[],inclusive:!1},STATE:{rules:[12,13,39,40,41,42,43,44,49,50,54,55,56],inclusive:!1},ID:{rules:[12,13],inclusive:!1},INITIAL:{rules:[0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,18,20,22,25,29,32,35,38,56,60,71,72,73,74,75,76,77,79,80,81],inclusive:!0}}}})();function ct(){this.yy={}}return u(ct,"Parser"),ct.prototype=mt,mt.Parser=ct,new ct})();kt.parser=kt;var ue=kt,pe="TB",It="TB",Nt="dir",J="state",q="root",Et="relation",ye="classDef",ge="style",fe="applyClass",tt="default",Ot="divider",Rt="fill:none",wt="fill: #333",Bt="c",Yt="text",Ft="normal",Dt="rect",Ct="rectWithTitle",me="stateStart",Se="stateEnd",Pt="divider",Gt="roundedWithTitle",_e="note",be="noteGroup",et="statediagram",Te=`${et}-state`,jt="transition",ke="note",Ee=`${jt} note-edge`,De=`${et}-${ke}`,Ce=`${et}-cluster`,xe=`${et}-cluster-alt`,zt="parent",Wt="note",$e="state",xt="----",Le=`${xt}${Wt}`,Mt=`${xt}${zt}`,Ut=u((e,t=It)=>{if(!e.doc)return t;let s=t;for(let n of e.doc)n.stmt==="dir"&&(s=n.value);return s},"getDir"),Ae={getClasses:u(function(e,t){return t.db.getClasses()},"getClasses"),draw:u(async function(e,t,s,n){T.info("REF0:"),T.info("Drawing state diagram (v2)",t);let{securityLevel:i,state:o,layout:c}=w();n.db.extract(n.db.getRootDocV2());let y=n.db.getData(),p=he(t,i);y.type=n.type,y.layoutAlgorithm=c,y.nodeSpacing=(o==null?void 0:o.nodeSpacing)||50,y.rankSpacing=(o==null?void 0:o.rankSpacing)||50,y.markers=["barb"],y.diagramId=t,await ce(y,p);try{(typeof n.db.getLinks=="function"?n.db.getLinks():new Map).forEach((_,m)=>{var d;let D=typeof m=="string"?m:typeof(m==null?void 0:m.id)=="string"?m.id:"";if(!D){T.warn("\u26A0\uFE0F Invalid or missing stateId from key:",JSON.stringify(m));return}let A=(d=p.node())==null?void 0:d.querySelectorAll("g"),x;if(A==null||A.forEach(E=>{var v;((v=E.textContent)==null?void 0:v.trim())===D&&(x=E)}),!x){T.warn("\u26A0\uFE0F Could not find node matching text:",D);return}let R=x.parentNode;if(!R){T.warn("\u26A0\uFE0F Node has no parent, cannot wrap:",D);return}let $=document.createElementNS("http://www.w3.org/2000/svg","a"),O=_.url.replace(/^"+|"+$/g,"");if($.setAttributeNS("http://www.w3.org/1999/xlink","xlink:href",O),$.setAttribute("target","_blank"),_.tooltip){let E=_.tooltip.replace(/^"+|"+$/g,"");$.setAttribute("title",E)}R.replaceChild($,x),$.appendChild(x),T.info("\u{1F517} Wrapped node in <a> tag for:",D,_.url)})}catch(_){T.error("\u274C Error injecting clickable links:",_)}te.insertTitle(p,"statediagramTitleText",(o==null?void 0:o.titleTopMargin)??25,n.db.getDiagramTitle()),de(p,8,et,(o==null?void 0:o.useMaxWidth)??!0)},"draw"),getDir:Ut},yt=new Map,P=0;function gt(e="",t=0,s="",n=xt){return`${$e}-${e}${s!==null&&s.length>0?`${n}${s}`:""}-${t}`}u(gt,"stateDomId");var ve=u((e,t,s,n,i,o,c,y)=>{T.trace("items",t),t.forEach(p=>{switch(p.stmt){case J:it(e,p,s,n,i,o,c,y);break;case tt:it(e,p,s,n,i,o,c,y);break;case Et:{it(e,p.state1,s,n,i,o,c,y),it(e,p.state2,s,n,i,o,c,y);let _={id:"edge"+P,start:p.state1.id,end:p.state2.id,arrowhead:"normal",arrowTypeEnd:"arrow_barb",style:Rt,labelStyle:"",label:U.sanitizeText(p.description??"",w()),arrowheadStyle:wt,labelpos:Bt,labelType:Yt,thickness:Ft,classes:jt,look:c};i.push(_),P++}break}})},"setupDoc"),Kt=u((e,t=It)=>{let s=t;if(e.doc)for(let n of e.doc)n.stmt==="dir"&&(s=n.value);return s},"getDir");function st(e,t,s){if(!t.id||t.id==="</join></fork>"||t.id==="</choice>")return;t.cssClasses&&(Array.isArray(t.cssCompiledStyles)||(t.cssCompiledStyles=[]),t.cssClasses.split(" ").forEach(i=>{let o=s.get(i);o&&(t.cssCompiledStyles=[...t.cssCompiledStyles??[],...o.styles])}));let n=e.find(i=>i.id===t.id);n?Object.assign(n,t):e.push(t)}u(st,"insertOrUpdateNode");function Ht(e){var t;return((t=e==null?void 0:e.classes)==null?void 0:t.join(" "))??""}u(Ht,"getClassesFromDbInfo");function Xt(e){return(e==null?void 0:e.styles)??[]}u(Xt,"getStylesFromDbInfo");var it=u((e,t,s,n,i,o,c,y)=>{var x,R,$;let p=t.id,_=s.get(p),m=Ht(_),D=Xt(_),A=w();if(T.info("dataFetcher parsedItem",t,_,D),p!=="root"){let O=Dt;t.start===!0?O=me:t.start===!1&&(O=Se),t.type!==tt&&(O=t.type),yt.get(p)||yt.set(p,{id:p,shape:O,description:U.sanitizeText(p,A),cssClasses:`${m} ${Te}`,cssStyles:D});let d=yt.get(p);t.description&&(Array.isArray(d.description)?(d.shape=Ct,d.description.push(t.description)):(x=d.description)!=null&&x.length&&d.description.length>0?(d.shape=Ct,d.description===p?d.description=[t.description]:d.description=[d.description,t.description]):(d.shape=Dt,d.description=t.description),d.description=U.sanitizeTextOrArray(d.description,A)),((R=d.description)==null?void 0:R.length)===1&&d.shape===Ct&&(d.type==="group"?d.shape=Gt:d.shape=Dt),!d.type&&t.doc&&(T.info("Setting cluster for XCX",p,Kt(t)),d.type="group",d.isGroup=!0,d.dir=Kt(t),d.shape=t.type===Ot?Pt:Gt,d.cssClasses=`${d.cssClasses} ${Ce} ${o?xe:""}`);let E={labelStyle:"",shape:d.shape,label:d.description,cssClasses:d.cssClasses,cssCompiledStyles:[],cssStyles:d.cssStyles,id:p,dir:d.dir,domId:gt(p,P),type:d.type,isGroup:d.type==="group",padding:8,rx:10,ry:10,look:c};if(E.shape===Pt&&(E.label=""),e&&e.id!=="root"&&(T.trace("Setting node ",p," to be child of its parent ",e.id),E.parentId=e.id),E.centerLabel=!0,t.note){let v={labelStyle:"",shape:_e,label:t.note.text,cssClasses:De,cssStyles:[],cssCompiledStyles:[],id:p+Le+"-"+P,domId:gt(p,P,Wt),type:d.type,isGroup:d.type==="group",padding:($=A.flowchart)==null?void 0:$.padding,look:c,position:t.note.position},G=p+Mt,j={labelStyle:"",shape:be,label:t.note.text,cssClasses:d.cssClasses,cssStyles:[],id:p+Mt,domId:gt(p,P,zt),type:"group",isGroup:!0,padding:16,look:c,position:t.note.position};P++,j.id=G,v.parentId=G,st(n,j,y),st(n,v,y),st(n,E,y);let Y=p,F=v.id;t.note.position==="left of"&&(Y=v.id,F=p),i.push({id:Y+"-"+F,start:Y,end:F,arrowhead:"none",arrowTypeEnd:"",style:Rt,labelStyle:"",classes:Ee,arrowheadStyle:wt,labelpos:Bt,labelType:Yt,thickness:Ft,look:c})}else st(n,E,y)}t.doc&&(T.trace("Adding nodes children "),ve(t,t.doc,s,n,i,!o,c,y))},"dataFetcher"),Ie=u(()=>{yt.clear(),P=0},"reset"),C={START_NODE:"[*]",START_TYPE:"start",END_NODE:"[*]",END_TYPE:"end",COLOR_KEYWORD:"color",FILL_KEYWORD:"fill",BG_FILL:"bgFill",STYLECLASS_SEP:","},Vt=u(()=>new Map,"newClassesList"),Jt=u(()=>({relations:[],states:new Map,documents:{}}),"newDoc"),ft=u(e=>JSON.parse(JSON.stringify(e)),"clone"),Ne=(K=class{constructor(t){this.version=t,this.nodes=[],this.edges=[],this.rootDoc=[],this.classes=Vt(),this.documents={root:Jt()},this.currentDocument=this.documents.root,this.startEndCount=0,this.dividerCnt=0,this.links=new Map,this.getAccTitle=oe,this.setAccTitle=se,this.getAccDescription=ne,this.setAccDescription=le,this.setDiagramTitle=re,this.getDiagramTitle=ie,this.clear(),this.setRootDoc=this.setRootDoc.bind(this),this.getDividerId=this.getDividerId.bind(this),this.setDirection=this.setDirection.bind(this),this.trimColon=this.trimColon.bind(this)}extract(t){this.clear(!0);for(let i of Array.isArray(t)?t:t.doc)switch(i.stmt){case J:this.addState(i.id.trim(),i.type,i.doc,i.description,i.note);break;case Et:this.addRelation(i.state1,i.state2,i.description);break;case ye:this.addStyleClass(i.id.trim(),i.classes);break;case ge:this.handleStyleDef(i);break;case fe:this.setCssClass(i.id.trim(),i.styleClass);break;case"click":this.addLink(i.id,i.url,i.tooltip);break}let s=this.getStates(),n=w();Ie(),it(void 0,this.getRootDocV2(),s,this.nodes,this.edges,!0,n.look,this.classes);for(let i of this.nodes)if(Array.isArray(i.label)){if(i.description=i.label.slice(1),i.isGroup&&i.description.length>0)throw Error(`Group nodes can only have label. Remove the additional description for node [${i.id}]`);i.label=i.label[0]}}handleStyleDef(t){let s=t.id.trim().split(","),n=t.styleClass.split(",");for(let i of s){let o=this.getState(i);if(!o){let c=i.trim();this.addState(c),o=this.getState(c)}o&&(o.styles=n.map(c=>{var y;return(y=c.replace(/;/g,""))==null?void 0:y.trim()}))}}setRootDoc(t){T.info("Setting root doc",t),this.rootDoc=t,this.version===1?this.extract(t):this.extract(this.getRootDocV2())}docTranslator(t,s,n){if(s.stmt===Et){this.docTranslator(t,s.state1,!0),this.docTranslator(t,s.state2,!1);return}if(s.stmt===J&&(s.id===C.START_NODE?(s.id=t.id+(n?"_start":"_end"),s.start=n):s.id=s.id.trim()),s.stmt!==q&&s.stmt!==J||!s.doc)return;let i=[],o=[];for(let c of s.doc)if(c.type===Ot){let y=ft(c);y.doc=ft(o),i.push(y),o=[]}else o.push(c);if(i.length>0&&o.length>0){let c={stmt:J,id:ee(),type:"divider",doc:ft(o)};i.push(ft(c)),s.doc=i}s.doc.forEach(c=>this.docTranslator(s,c,!0))}getRootDocV2(){return this.docTranslator({id:q,stmt:q},{id:q,stmt:q,doc:this.rootDoc},!0),{id:q,doc:this.rootDoc}}addState(t,s=tt,n=void 0,i=void 0,o=void 0,c=void 0,y=void 0,p=void 0){let _=t==null?void 0:t.trim();if(!this.currentDocument.states.has(_))T.info("Adding state ",_,i),this.currentDocument.states.set(_,{stmt:J,id:_,descriptions:[],type:s,doc:n,note:o,classes:[],styles:[],textStyles:[]});else{let m=this.currentDocument.states.get(_);if(!m)throw Error(`State not found: ${_}`);m.doc||(m.doc=n),m.type||(m.type=s)}if(i&&(T.info("Setting state description",_,i),(Array.isArray(i)?i:[i]).forEach(m=>this.addDescription(_,m.trim()))),o){let m=this.currentDocument.states.get(_);if(!m)throw Error(`State not found: ${_}`);m.note=o,m.note.text=U.sanitizeText(m.note.text,w())}c&&(T.info("Setting state classes",_,c),(Array.isArray(c)?c:[c]).forEach(m=>this.setCssClass(_,m.trim()))),y&&(T.info("Setting state styles",_,y),(Array.isArray(y)?y:[y]).forEach(m=>this.setStyle(_,m.trim()))),p&&(T.info("Setting state styles",_,y),(Array.isArray(p)?p:[p]).forEach(m=>this.setTextStyle(_,m.trim())))}clear(t){this.nodes=[],this.edges=[],this.documents={root:Jt()},this.currentDocument=this.documents.root,this.startEndCount=0,this.classes=Vt(),t||(this.links=new Map,ae())}getState(t){return this.currentDocument.states.get(t)}getStates(){return this.currentDocument.states}logDocuments(){T.info("Documents = ",this.documents)}getRelations(){return this.currentDocument.relations}addLink(t,s,n){this.links.set(t,{url:s,tooltip:n}),T.warn("Adding link",t,s,n)}getLinks(){return this.links}startIdIfNeeded(t=""){return t===C.START_NODE?(this.startEndCount++,`${C.START_TYPE}${this.startEndCount}`):t}startTypeIfNeeded(t="",s=tt){return t===C.START_NODE?C.START_TYPE:s}endIdIfNeeded(t=""){return t===C.END_NODE?(this.startEndCount++,`${C.END_TYPE}${this.startEndCount}`):t}endTypeIfNeeded(t="",s=tt){return t===C.END_NODE?C.END_TYPE:s}addRelationObjs(t,s,n=""){let i=this.startIdIfNeeded(t.id.trim()),o=this.startTypeIfNeeded(t.id.trim(),t.type),c=this.startIdIfNeeded(s.id.trim()),y=this.startTypeIfNeeded(s.id.trim(),s.type);this.addState(i,o,t.doc,t.description,t.note,t.classes,t.styles,t.textStyles),this.addState(c,y,s.doc,s.description,s.note,s.classes,s.styles,s.textStyles),this.currentDocument.relations.push({id1:i,id2:c,relationTitle:U.sanitizeText(n,w())})}addRelation(t,s,n){if(typeof t=="object"&&typeof s=="object")this.addRelationObjs(t,s,n);else if(typeof t=="string"&&typeof s=="string"){let i=this.startIdIfNeeded(t.trim()),o=this.startTypeIfNeeded(t),c=this.endIdIfNeeded(s.trim()),y=this.endTypeIfNeeded(s);this.addState(i,o),this.addState(c,y),this.currentDocument.relations.push({id1:i,id2:c,relationTitle:n?U.sanitizeText(n,w()):void 0})}}addDescription(t,s){var o;let n=this.currentDocument.states.get(t),i=s.startsWith(":")?s.replace(":","").trim():s;(o=n==null?void 0:n.descriptions)==null||o.push(U.sanitizeText(i,w()))}cleanupLabel(t){return t.startsWith(":")?t.slice(2).trim():t.trim()}getDividerId(){return this.dividerCnt++,`divider-id-${this.dividerCnt}`}addStyleClass(t,s=""){this.classes.has(t)||this.classes.set(t,{id:t,styles:[],textStyles:[]});let n=this.classes.get(t);s&&n&&s.split(C.STYLECLASS_SEP).forEach(i=>{let o=i.replace(/([^;]*);/,"$1").trim();if(RegExp(C.COLOR_KEYWORD).exec(i)){let c=o.replace(C.FILL_KEYWORD,C.BG_FILL).replace(C.COLOR_KEYWORD,C.FILL_KEYWORD);n.textStyles.push(c)}n.styles.push(o)})}getClasses(){return this.classes}setCssClass(t,s){t.split(",").forEach(n=>{var o;let i=this.getState(n);if(!i){let c=n.trim();this.addState(c),i=this.getState(c)}(o=i==null?void 0:i.classes)==null||o.push(s)})}setStyle(t,s){var n,i;(i=(n=this.getState(t))==null?void 0:n.styles)==null||i.push(s)}setTextStyle(t,s){var n,i;(i=(n=this.getState(t))==null?void 0:n.textStyles)==null||i.push(s)}getDirectionStatement(){return this.rootDoc.find(t=>t.stmt===Nt)}getDirection(){var t;return((t=this.getDirectionStatement())==null?void 0:t.value)??pe}setDirection(t){let s=this.getDirectionStatement();s?s.value=t:this.rootDoc.unshift({stmt:Nt,value:t})}trimColon(t){return t.startsWith(":")?t.slice(1).trim():t.trim()}getData(){let t=w();return{nodes:this.nodes,edges:this.edges,other:{},config:t,direction:Ut(this.getRootDocV2())}}getConfig(){return w().state}},u(K,"StateDB"),K.relationType={AGGREGATION:0,EXTENSION:1,COMPOSITION:2,DEPENDENCY:3},K),Oe=u(e=>`
defs #statediagram-barbEnd {
    fill: ${e.transitionColor};
    stroke: ${e.transitionColor};
  }
g.stateGroup text {
  fill: ${e.nodeBorder};
  stroke: none;
  font-size: 10px;
}
g.stateGroup text {
  fill: ${e.textColor};
  stroke: none;
  font-size: 10px;

}
g.stateGroup .state-title {
  font-weight: bolder;
  fill: ${e.stateLabelColor};
}

g.stateGroup rect {
  fill: ${e.mainBkg};
  stroke: ${e.nodeBorder};
}

g.stateGroup line {
  stroke: ${e.lineColor};
  stroke-width: 1;
}

.transition {
  stroke: ${e.transitionColor};
  stroke-width: 1;
  fill: none;
}

.stateGroup .composit {
  fill: ${e.background};
  border-bottom: 1px
}

.stateGroup .alt-composit {
  fill: #e0e0e0;
  border-bottom: 1px
}

.state-note {
  stroke: ${e.noteBorderColor};
  fill: ${e.noteBkgColor};

  text {
    fill: ${e.noteTextColor};
    stroke: none;
    font-size: 10px;
  }
}

.stateLabel .box {
  stroke: none;
  stroke-width: 0;
  fill: ${e.mainBkg};
  opacity: 0.5;
}

.edgeLabel .label rect {
  fill: ${e.labelBackgroundColor};
  opacity: 0.5;
}
.edgeLabel {
  background-color: ${e.edgeLabelBackground};
  p {
    background-color: ${e.edgeLabelBackground};
  }
  rect {
    opacity: 0.5;
    background-color: ${e.edgeLabelBackground};
    fill: ${e.edgeLabelBackground};
  }
  text-align: center;
}
.edgeLabel .label text {
  fill: ${e.transitionLabelColor||e.tertiaryTextColor};
}
.label div .edgeLabel {
  color: ${e.transitionLabelColor||e.tertiaryTextColor};
}

.stateLabel text {
  fill: ${e.stateLabelColor};
  font-size: 10px;
  font-weight: bold;
}

.node circle.state-start {
  fill: ${e.specialStateColor};
  stroke: ${e.specialStateColor};
}

.node .fork-join {
  fill: ${e.specialStateColor};
  stroke: ${e.specialStateColor};
}

.node circle.state-end {
  fill: ${e.innerEndBackground};
  stroke: ${e.background};
  stroke-width: 1.5
}
.end-state-inner {
  fill: ${e.compositeBackground||e.background};
  // stroke: ${e.background};
  stroke-width: 1.5
}

.node rect {
  fill: ${e.stateBkg||e.mainBkg};
  stroke: ${e.stateBorder||e.nodeBorder};
  stroke-width: 1px;
}
.node polygon {
  fill: ${e.mainBkg};
  stroke: ${e.stateBorder||e.nodeBorder};;
  stroke-width: 1px;
}
#statediagram-barbEnd {
  fill: ${e.lineColor};
}

.statediagram-cluster rect {
  fill: ${e.compositeTitleBackground};
  stroke: ${e.stateBorder||e.nodeBorder};
  stroke-width: 1px;
}

.cluster-label, .nodeLabel {
  color: ${e.stateLabelColor};
  // line-height: 1;
}

.statediagram-cluster rect.outer {
  rx: 5px;
  ry: 5px;
}
.statediagram-state .divider {
  stroke: ${e.stateBorder||e.nodeBorder};
}

.statediagram-state .title-state {
  rx: 5px;
  ry: 5px;
}
.statediagram-cluster.statediagram-cluster .inner {
  fill: ${e.compositeBackground||e.background};
}
.statediagram-cluster.statediagram-cluster-alt .inner {
  fill: ${e.altBackground?e.altBackground:"#efefef"};
}

.statediagram-cluster .inner {
  rx:0;
  ry:0;
}

.statediagram-state rect.basic {
  rx: 5px;
  ry: 5px;
}
.statediagram-state rect.divider {
  stroke-dasharray: 10,10;
  fill: ${e.altBackground?e.altBackground:"#efefef"};
}

.note-edge {
  stroke-dasharray: 5;
}

.statediagram-note rect {
  fill: ${e.noteBkgColor};
  stroke: ${e.noteBorderColor};
  stroke-width: 1px;
  rx: 0;
  ry: 0;
}
.statediagram-note rect {
  fill: ${e.noteBkgColor};
  stroke: ${e.noteBorderColor};
  stroke-width: 1px;
  rx: 0;
  ry: 0;
}

.statediagram-note text {
  fill: ${e.noteTextColor};
}

.statediagram-note .nodeLabel {
  color: ${e.noteTextColor};
}
.statediagram .edgeLabel {
  color: red; // ${e.noteTextColor};
}

#dependencyStart, #dependencyEnd {
  fill: ${e.lineColor};
  stroke: ${e.lineColor};
  stroke-width: 1;
}

.statediagramTitleText {
  text-anchor: middle;
  font-size: 18px;
  fill: ${e.textColor};
}
`,"getStyles");export{Oe as i,ue as n,Ae as r,Ne as t};
