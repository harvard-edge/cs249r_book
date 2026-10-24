#!/usr/bin/env node
/** Browser QA for real Quarto figure enhancements. Artifacts are intentionally external.
 * node check_figure_explore_browser.cjs --build-root books/_build/parallel/RUN
 *   --base-url http://127.0.0.1:8768 --playwright-module /path/to/playwright
 * PLAYWRIGHT_MODULE is also supported; otherwise normal require('playwright') resolves.
 * --screenshots all|none (default all), --volumes 1,2,3,4, --workers 3,
 * --max-pages N, --keys vol1:fig-id,vol2:fig-id, --output /tmp/mlsys-figure-qa,
 * --mount-timeout 45000.
 * --integrity-only checks enrollment/artwork mounts and IDs without stepping/captures.
 * --capture-only selects a representative paused stage without rerunning controls/timers.
 * --profiles desktop-light,desktop-dark selects capture/geometry profiles.
 * --baseline-root books/_build/parallel/RUN classifies identical baseline pageerrors.
 * Exit 1 means actual failures; exit 2 means incomplete build coverage only.
 * Screenshots require human inspection; passing assertions are not artwork approval.
 */
'use strict';
const fs=require('node:fs'), path=require('node:path');
const argv=process.argv.slice(2), opts={};
for(let i=0;i<argv.length;i++){if(!argv[i].startsWith('--'))throw Error('Unexpected argument '+argv[i]);opts[argv[i].slice(2)]=argv[i+1]&&!argv[i+1].startsWith('--')?argv[++i]:true;}
if(opts.help){console.log(fs.readFileSync(__filename,'utf8').split("'use strict';")[0]);process.exit(0);}
const repo=path.resolve(opts.repo||path.join(__dirname,'../../..'));
if(!opts['build-root'])throw Error('--build-root is required; use the actual parallel build directory.');
const build=path.resolve(repo,opts['build-root']), books=path.join(repo,'books');
const output=path.resolve(opts.output||'/tmp/mlsys-figure-qa');fs.mkdirSync(output,{recursive:true});
const moduleName=opts['playwright-module']||process.env.PLAYWRIGHT_MODULE||'playwright';
let playwright;try{playwright=require(moduleName);}catch(e){throw Error('Cannot load Playwright; install playwright or set PLAYWRIGHT_MODULE / --playwright-module. '+e.message);}
const registryPath=path.join(books,'_extensions/mlsysbook-ext/figure-explore/figure-registry.json');
const registry=JSON.parse(fs.readFileSync(registryPath,'utf8'));
if(opts.keys)for(const key of String(opts.keys).split(','))if(!Object.hasOwn(registry,key))throw Error('Unknown registered key: '+key);
const volumes=String(opts.volumes||'1,2,3,4').split(',').map(Number), base=String(opts['base-url']||'http://127.0.0.1:8768').replace(/\/$/,'');
const mountTimeout=Number(opts['mount-timeout']||45000), workers=Math.max(1,Number(opts.workers||3));
const allProfiles=[{name:'desktop-light',width:1440,height:1100,dark:false},{name:'desktop-dark',width:1440,height:1100,dark:true},{name:'narrow-light',width:390,height:1000,dark:false},{name:'narrow-dark',width:390,height:1000,dark:true}];
const profiles=opts.profiles?String(opts.profiles).split(',').map(name=>{const profile=allProfiles.find(p=>p.name===name);if(!profile)throw Error('Unknown profile: '+name);return profile;}):allProfiles;
const report={started:new Date().toISOString(),running:true,build,base,registry:registryPath,mode:opts['integrity-only']?'integrity-only':opts['capture-only']?'capture-only':'full-runtime',reviewBoundary:'Automated runtime/geometry checks and screenshot capture, not human artwork or pedagogical approval.',expected:0,figures:[],pages:[],missing:[],failures:[],warnings:[],baselineErrors:[]};
function walk(dir){if(!fs.existsSync(dir))return[];return fs.readdirSync(dir,{withFileTypes:true}).flatMap(e=>e.isDirectory()?walk(path.join(dir,e.name)):e.name.endsWith('.html')?[path.join(dir,e.name)]:[]);}
const plans=[];
for(const volume of volumes){
 const entries=Object.entries(registry).filter(([key,v])=>Number(v.volume)===volume&&(!opts.keys||String(opts.keys).split(',').includes(key)));report.expected+=entries.length;
 const dir=path.join(build,`vol${volume}-html/output`), files=walk(dir), byKey=new Map();
 for(const file of files){const html=fs.readFileSync(file,'utf8');for(const m of html.matchAll(/data-mlsys-explore-key=["']([^"']+)["']/g)){if(!byKey.has(m[1]))byKey.set(m[1],file);}}
 const groups=new Map();
 for(const [key,lesson] of entries){
  const file=byKey.get(key);
  if(!file){report.missing.push({key,source:lesson.source,reason:'Registered figure not found in available built HTML; build may be incomplete or filter enrollment failed.'});continue;}
  if(!groups.has(file))groups.set(file,[]);groups.get(file).push({key,...lesson});
 }
 let volumePage=0;for(const [file,entries] of groups){const relative=path.relative(books,file).split(path.sep).map(encodeURIComponent).join('/');plans.push({file,url:base+'/'+relative,entries,volume,volumePage:volumePage++});}
}
// Interleave volumes so independent visual reviewers receive every volume early.
plans.sort((a,b)=>a.volumePage-b.volumePage||a.volume-b.volume);
const limited=opts['max-pages']?plans.slice(0,Number(opts['max-pages'])):plans;
if(limited.length<plans.length)report.warnings.push(`Scan limited to ${limited.length} of ${plans.length} pages by --max-pages.`);
function save(){report.updatedAt=new Date().toISOString();report.complete=!report.running&&!report.interrupted&&report.missing.length===0&&report.failures.length===0&&report.figures.length===report.expected&&report.figures.every(r=>['passed','integrity-passed','captured'].includes(r.status));report.counts={expected:report.expected,availablePages:plans.length,scannedPages:report.pages.length,scannedFigures:report.figures.length,missing:report.missing.length,failures:report.failures.length,screenshots:report.figures.reduce((n,r)=>n+r.screenshots.length,0)};fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');}
for(const signal of ['SIGINT','SIGTERM'])process.on(signal,()=>{report.interrupted=signal;report.running=false;save();process.exit(signal==='SIGINT'?130:143);});
const fail=(key,check,error,extra={})=>{const f={key,check,error:String(error?.message||error),...extra};report.failures.push(f);process.stderr.write(JSON.stringify(f)+'\n');};
async function theme(page,dark){await page.emulateMedia({colorScheme:dark?'dark':'light',reducedMotion:'reduce'});await page.evaluate(dark=>{document.body.classList.toggle('quarto-dark',dark);document.body.classList.toggle('quarto-light',!dark);document.documentElement.dataset.bsTheme=dark?'dark':'light';const d=document.querySelector('link.quarto-color-scheme.quarto-color-alternate');const l=document.querySelector('link.quarto-color-scheme:not(.quarto-color-alternate)');if(d&&l){d.disabled=!dark;l.disabled=dark;}},dark);}
async function captureFigure(page,figure,destination){
 // Element screenshots silently scroll the figure back beneath fixed navigation.
 // Position it ourselves, then use document-coordinate page clipping. fullPage
 // selects Playwright's document-rectangle path, which enables Chromium's
 // captureBeyondViewport without resizing the reading layout or scrolling again.
 await page.evaluate(async()=>{if(window.MathJax?.startup?.promise)await window.MathJax.startup.promise.catch(()=>{});await document.fonts.ready;await Promise.all([...document.images].map(image=>image.decode().catch(()=>{})));});
 await page.mouse.move(0,0);
 await figure.evaluate(element=>element.scrollIntoView({block:'start',inline:'nearest',behavior:'instant'}));
 let frame;
 for(let attempt=0;attempt<4;attempt++){
  await page.clock.runFor(200);
  frame=await figure.evaluate(element=>{
   const candidates=[...document.querySelectorAll('header,nav,.fixed-top,#quarto-announcement')].map(node=>({rect:node.getBoundingClientRect(),style:getComputedStyle(node)})).filter(({rect,style})=>['fixed','sticky'].includes(style.position)&&rect.width>=innerWidth*.5&&rect.bottom>0&&style.visibility!=='hidden');
   let clearance=0;
   for(let i=0;i<4;i++)for(const {rect} of candidates)if(rect.top<=clearance+4)clearance=Math.max(clearance,rect.bottom);
   const rect=element.getBoundingClientRect();
   return{clearance,viewportTop:rect.top,clip:{x:Math.max(0,Math.floor(rect.x+scrollX)),y:Math.max(0,Math.floor(rect.y+scrollY)),width:Math.ceil(rect.width),height:Math.ceil(rect.height)}};
  });
  if(frame.viewportTop>=frame.clearance+16)break;
  await page.evaluate(delta=>window.scrollBy({top:delta,left:0,behavior:'instant'}),frame.viewportTop-frame.clearance-24);
 }
 if(frame.viewportTop<frame.clearance+8)throw Error('Cannot frame figure below fixed navigation: '+JSON.stringify(frame));
 await page.mouse.move(0,0);await page.clock.runFor(200);
 await page.waitForFunction(()=>![...document.querySelectorAll('.tippy-box')].some(e=>e.dataset.state==='visible'),null,{timeout:3000});
 // Re-read after delayed math/fonts/margin work: screenshot preparation must
 // never use a document position measured before that work settled.
 await page.evaluate(()=>document.fonts.ready);
 const finalRect=await figure.evaluate(element=>{const r=element.getBoundingClientRect();return{x:r.x+scrollX,y:r.y+scrollY,width:r.width,height:r.height,viewportTop:r.top};});
 frame.viewportTop=finalRect.viewportTop;frame.clip={x:Math.max(0,Math.floor(finalRect.x)),y:Math.max(0,Math.floor(finalRect.y)),width:Math.ceil(finalRect.width),height:Math.ceil(finalRect.height)};
 if(frame.viewportTop<frame.clearance+8)throw Error('Figure shifted under navigation before capture: '+JSON.stringify(frame));
 await page.screenshot({path:destination,fullPage:true,clip:frame.clip,animations:'allow',timeout:30000});
 return frame;
}
async function inspectFigure(page,entry){
 const result={key:entry.key,page:page.url(),lesson:entry.name,status:'checking',screenshots:[],states:0};report.figures.push(result);
 // IDs contain no CSS quoting assumptions: select the exact data key in the DOM.
 const selector=`[data-mlsys-explore-key=${JSON.stringify(entry.key)}]`, figure=page.locator(selector);
 try{
  if(await figure.count()!==1)throw Error('Expected exactly one enrolled outer figure, got '+await figure.count());
  const state=await figure.getAttribute('data-mlsys-explore-state');if(state!=='ready')throw Error('Mount is '+(state||'unset')+'; inspect console for canonical artwork integrity or lesson exception.');
  const initial=await figure.evaluate(root=>(root.matches('figure')?root:root.querySelector(':scope > figure'))?.dataset.figureView);if(initial!=='diagram')throw Error('Diagram must be initial/default view, got '+initial);
  const duplicate=await figure.evaluate(root=>{const ids=[...root.querySelectorAll('[id]')].map(e=>e.id);return ids.filter((id,i)=>ids.indexOf(id)!==i);});if(duplicate.length)throw Error('Duplicate component IDs: '+duplicate.join(','));
  if(opts['integrity-only']){result.status='integrity-passed';return result;}
  const exploreTab=figure.getByRole('tab',{name:'Explore',exact:true}), diagramTab=figure.getByRole('tab',{name:'Diagram',exact:true});
  await exploreTab.click();const panel=figure.locator('.mlsys-explore-panel');
  const menus=panel.locator('select');const menu=menus.first();if(!await menu.count())throw Error('No step sequence menu in Explore.');
  const values=await menu.locator('option').evaluateAll(nodes=>nodes.map(n=>n.value));if(!values.length)throw Error('Empty step menu.');
  if(!opts['capture-only']){
  for(const value of values){
   await menu.selectOption(value);result.states++;
   const focus=await panel.evaluate((root,{key,value})=>{
    if(!root.classList.contains('mlsys-guided'))return null;
    const lesson=window.MLSysGuidedLessons?.[key];if(!lesson)throw Error('Guided lesson missing from window registry');
    const ids=Object.keys(lesson.regions), rects=[...root.querySelectorAll('.mlsys-guided-overlay rect')];
    const actual=rects.flatMap((r,i)=>r.classList.contains('is-focused')?[ids[i]]:[]).sort();
    const expected=[...lesson.steps[Number(value)].focus].sort();
    const heading=root.querySelector('.mlsys-guided-step')?.textContent||'';
    return {actual,expected,heading,title:lesson.steps[Number(value)].title};
   },{key:entry.key,value});
   if(focus&&(JSON.stringify(focus.actual)!==JSON.stringify(focus.expected)||!focus.heading.includes(focus.title)))throw Error('Focus/narration differs from authored stage: '+JSON.stringify(focus));
   const stageGeometry=await panel.evaluate(root=>({scroll:root.scrollWidth,client:root.clientWidth,empty:!root.textContent.trim()}));
   if(stageGeometry.empty||stageGeometry.scroll>stageGeometry.client+2)throw Error('Invalid/overflowing stage '+value+': '+JSON.stringify(stageGeometry));
  }
  const terminalNext=panel.getByRole('button',{name:/^(Next|Next →)$/});
  if(await terminalNext.count()&&!await terminalNext.isDisabled())throw Error('Next should be disabled at last stage');
  const restart=panel.getByRole('button',{name:'Restart',exact:true});await restart.click();if(await menu.inputValue()!==values[0])throw Error('Restart did not restore first stage');
  await menu.selectOption(values[0]);const back=panel.getByRole('button',{name:/^(Back|← Back)$/}),next=panel.getByRole('button',{name:/^(Next|Next →)$/});
  if(await back.count()&&!await back.isDisabled())throw Error('Back should be disabled at first stage');
  if(values.length>1&&await next.count()){await next.click();if(await menu.inputValue()!==values[1])throw Error('Next did not advance one stage');await back.click();if(await menu.inputValue()!==values[0])throw Error('Back did not restore first stage');}
  const play=panel.getByRole('button',{name:/^(Play|Play sequence|▶ Play|▶ Play sequence|Replay|Replay sequence|↻ Replay)$/}).first();
  if(!await play.count())throw Error('No Play/Replay control found');await play.click();
  const pause=panel.getByRole('button',{name:/^(Pause|Pause sequence|Ⅱ Pause)$/}).first();if(!await pause.count())throw Error('Play did not expose Pause');
  // Advance due timers without waiting minutes per figure. fastForward fires each
  // existing timer at most once; a newly scheduled next stage remains future work.
  if(values.length>1){await page.clock.fastForward(60000);if(await menu.inputValue()===values[0])throw Error('Play timer did not advance the sequence');}
  if(await pause.count())await pause.click();
  const pausedStage=await menu.inputValue();await page.clock.fastForward(60000);if(await menu.inputValue()!==pausedStage)throw Error('Paused sequence advanced');
  if(values.length>1){
   await menu.selectOption(values[values.length-2]);await play.click();await page.clock.fastForward(60000);await page.clock.fastForward(60000);
   if(await menu.inputValue()!==values[values.length-1]||await pause.count())throw Error('Playback did not finish once at the final state');
   await play.click();if(await menu.inputValue()!==values[0])throw Error('Replay did not reset the sequence');if(await pause.count())await pause.click();
  }
  const preserved=await menu.inputValue();await diagramTab.click();await exploreTab.click();if(await menu.inputValue()!==preserved)throw Error('Tab switch reset current stage');
  await exploreTab.focus();await page.keyboard.press('Home');if(await diagramTab.getAttribute('aria-selected')!=='true')throw Error('Home did not activate Diagram tab');
  await page.keyboard.press('End');if(await exploreTab.getAttribute('aria-selected')!=='true')throw Error('End did not activate Explore tab');
  await page.keyboard.press('ArrowLeft');if(await diagramTab.getAttribute('aria-selected')!=='true')throw Error('Left arrow did not activate Diagram tab');
  await page.keyboard.press('ArrowRight');if(await exploreTab.getAttribute('aria-selected')!=='true')throw Error('Right arrow did not activate Explore tab');
  // Confirm an active player is paused on tab switch, not simply a manually paused one.
  await play.click();await diagramTab.click();await exploreTab.click();if(await panel.getByRole('button',{name:/^(Pause|Pause sequence|Ⅱ Pause)$/}).count())throw Error('Playback still running after hiding Explore');
  }
  await menu.selectOption(values[Math.floor(values.length/2)]);
  for(const profile of profiles){
   await page.setViewportSize({width:profile.width,height:profile.height});await theme(page,profile.dark);await page.mouse.move(0,0);if(await page.locator('.tippy-box[data-state=visible]').count())await page.clock.runFor(600);await page.evaluate(()=>Promise.all([...document.querySelectorAll('link[rel=stylesheet]:not([disabled])')].map(link=>link.sheet?Promise.resolve():new Promise(resolve=>{link.addEventListener('load',resolve,{once:true});link.addEventListener('error',resolve,{once:true});setTimeout(resolve,2000);}))));await figure.scrollIntoViewIfNeeded();
   if(profile.width<992){await page.clock.runFor(250);const stale=await page.evaluate(()=>[...document.querySelectorAll('.column-margin')].filter(e=>e.style.transform).map(e=>({transform:e.style.transform,text:e.textContent.trim().slice(0,80)})));if(stale.length)throw Error('Desktop margin transforms survived narrow resize: '+JSON.stringify(stale));}
   await panel.locator('img').evaluateAll(images=>Promise.all(images.map(img=>img.decode().catch(()=>{}))));
   const geometry=await panel.evaluate(async root=>{const box=root.getBoundingClientRect();const art=root.querySelector('.mlsys-guided-art');const img=art?.querySelector('img');const overlay=art?.querySelector('svg');const r=img?.getBoundingClientRect(),o=overlay?.getBoundingClientRect(),style=img&&getComputedStyle(img);let aspectWidth=img?.naturalWidth,aspectHeight=img?.naturalHeight,aspectSource='natural';if(img&&/\.svg(?:[?#]|$)/i.test(img.currentSrc||img.src)){const svg=new DOMParser().parseFromString(await (await fetch(img.currentSrc||img.src)).text(),'image/svg+xml').documentElement;const vb=svg.getAttribute('viewBox')?.trim().split(/[\s,]+/).map(Number);if(vb?.length===4&&vb[2]>0&&vb[3]>0){aspectWidth=vb[2];aspectHeight=vb[3];aspectSource='svg-viewBox';}}const padding=style?['paddingTop','paddingRight','paddingBottom','paddingLeft'].map(k=>parseFloat(style[k])):[],border=style?['borderTopWidth','borderRightWidth','borderBottomWidth','borderLeftWidth'].map(k=>parseFloat(style[k])):[];return {guidedImage:img?{padding,border,naturalWidth:img.naturalWidth,naturalHeight:img.naturalHeight,aspectSource,aspectWidth,aspectHeight,width:r.width,height:r.height,aspectErrorPx:Math.abs(r.height-r.width*aspectHeight/aspectWidth)}:null,width:box.width,scroll:root.scrollWidth,client:root.clientWidth,documentOverflow:document.documentElement.scrollWidth>innerWidth+2,brokenImages:[...root.querySelectorAll('img')].filter(i=>!i.complete||!i.naturalWidth).map(i=>i.src),overlayMismatch:r&&o?Math.max(Math.abs(r.width-o.width),Math.abs(r.height-o.height),Math.abs(r.x-o.x),Math.abs(r.y-o.y)):0};});
   if(geometry.scroll>geometry.client+2)fail(entry.key,'panel-overflow',JSON.stringify(geometry),{profile:profile.name});
   if(geometry.guidedImage&&(geometry.guidedImage.padding.some(n=>n!==0)||geometry.guidedImage.border.some(n=>n!==0)||geometry.guidedImage.aspectErrorPx>1))fail(entry.key,'art-content-projection',JSON.stringify(geometry.guidedImage),{profile:profile.name});
   if(geometry.overlayMismatch>2)fail(entry.key,'overlay-registration',JSON.stringify(geometry),{profile:profile.name});
   if(geometry.brokenImages.length)fail(entry.key,'broken-image',JSON.stringify(geometry),{profile:profile.name});
   if(geometry.documentOverflow)report.warnings.push({key:entry.key,profile:profile.name,message:'Whole document overflows viewport; may originate outside component.'});
   const enlarge=panel.getByRole('button',{name:'Enlarge figure',exact:true});
   if(!opts['capture-only']&&await enlarge.count()){
    const retained=await menu.inputValue();await page.mouse.move(0,0);await page.clock.runFor(400);await enlarge.click();
    const dialog=panel.locator('dialog.mlsys-guided-dialog');await dialog.waitFor({state:'visible'});
    const close=dialog.getByRole('button',{name:'Close enlarged figure',exact:true});
    if(!await close.evaluate(e=>document.activeElement===e))throw Error('Enlarge did not focus Close');
    await page.keyboard.press('Shift+Tab');if(!await dialog.evaluate(e=>e.contains(document.activeElement)))throw Error('Modal Shift+Tab escaped focus');
    await page.keyboard.press('Tab');if(!await close.evaluate(e=>document.activeElement===e))throw Error('Modal focus did not wrap back to Close');
    if(await menu.inputValue()!==retained)throw Error('Opening Enlarge changed stage');
    const bounds=await dialog.evaluate(e=>{const r=e.getBoundingClientRect();return{left:r.left,right:r.right,top:r.top,bottom:r.bottom,w:innerWidth,h:innerHeight};});
    if(bounds.left< -1||bounds.right>bounds.w+1||bounds.top< -1||bounds.bottom>bounds.h+1)throw Error('Modal outside viewport: '+JSON.stringify(bounds));
    await page.keyboard.press('Escape');await dialog.waitFor({state:'hidden'});
    if(await menu.inputValue()!==retained||!await enlarge.evaluate(e=>document.activeElement===e))throw Error('Escape lost stage or opener focus');
    if(await panel.getByRole('button',{name:/^(Pause|Pause sequence|Ⅱ Pause)$/}).count())throw Error('Closing Enlarge left playback running');
    await page.mouse.move(0,0);await page.clock.runFor(400);await enlarge.click();await close.click();await dialog.waitFor({state:'hidden'});
    if(await menu.inputValue()!==retained||!await enlarge.evaluate(e=>document.activeElement===e))throw Error('Close button lost stage or opener focus');
    // A closing modal can reveal a reference under the mouse. Move away before
    // screenshots so a legitimate Quarto hover preview does not cover the figure.
    await page.mouse.move(0,0);await page.clock.runFor(600);
    await page.waitForFunction(()=>![...document.querySelectorAll('.tippy-box')].some(e=>e.dataset.state==='visible'),null,{timeout:3000});
    result.modalProfiles=(result.modalProfiles||[]).concat(profile.name);
   }
   if(opts.screenshots!=='none'){
    const filename=entry.key.replace(/[^a-zA-Z0-9_-]/g,'_')+'-'+profile.name+'.png', destination=path.join(output,filename);
    const framing=await captureFigure(page,figure,destination);result.screenshots.push({profile:profile.name,file:destination,stage:await menu.inputValue(),framing});
   }
  }
  if(!opts['capture-only']){
  // Print must restore Diagram while retaining the user's chosen Explore state afterwards.
  await page.emulateMedia({media:'print'});
  if(!await figure.locator('.mlsys-diagram-panel').isVisible()||await panel.isVisible())throw Error('Print does not show Diagram exclusively');
  await page.emulateMedia({media:'screen'});await page.mouse.move(0,0);await page.clock.runFor(250);
  if(page.viewportSize().width<992){const stale=await page.evaluate(()=>[...document.querySelectorAll('.column-margin')].filter(e=>e.style.transform).map(e=>e.style.transform));if(stale.length)throw Error('Desktop margin transforms survived narrow print transition: '+JSON.stringify(stale));}
  }
  await diagramTab.click();result.status=opts['capture-only']?'captured':'passed';
 }catch(error){result.status='failed';result.failureContext=await figure.evaluateAll(es=>es.map(root=>{const e=root.querySelector('.mlsys-guided-enlarge');const r=e?.getBoundingClientRect();const hit=r&&document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);let ancestry=[];for(let n=hit;n&&ancestry.length<8;n=n.parentElement)ancestry.push({tag:n.tagName,cls:typeof n.className==='string'?n.className:'',id:n.id});return{enlargeRect:r?.toJSON(),hitAncestry:ancestry,active:document.activeElement?.outerHTML.slice(0,160),popups:[...document.querySelectorAll('.tippy-box')].map(n=>({state:n.dataset.state,rect:n.getBoundingClientRect().toJSON(),text:n.textContent.trim().slice(0,120)}))}})).catch(()=>[]);const failurePath=path.join(output,entry.key.replace(/[^a-zA-Z0-9_-]/g,'_')+'-FAILED.png');await page.screenshot({path:failurePath,animations:'allow',timeout:10000}).then(()=>result.failureScreenshot=failurePath).catch(()=>{});const evidence=await figure.evaluateAll(es=>es.map(e=>({id:e.id,state:e.dataset.mlsysExploreState||'unset',lesson:e.dataset.mlsysExplore}))).catch(()=>[]);fail(entry.key,'figure-runtime',error,{evidence});}
 return result;
}
save();
(async()=>{
 const browser=await playwright.chromium.launch({headless:true});let cursor=0;
 async function worker(){const context=await browser.newContext({viewport:{width:1440,height:1100},reducedMotion:'reduce'});context.setDefaultTimeout(Number(opts['action-timeout']||6000));const page=await context.newPage();let clockInstalled=false;
  while(cursor<limited.length){const plan=limited[cursor++];const errors=[],warnings=[];const onError=e=>errors.push(e.stack||e.message);const onConsole=m=>{if(m.text().includes('[figure-explore]'))(m.type()==='error'?errors:warnings).push(m.text());};page.on('pageerror',onError);page.on('console',onConsole);
   const pageResult={url:plan.url,volume:plan.volume,figures:plan.entries.length};report.pages.push(pageResult);
   try{const response=await page.goto(plan.url,{waitUntil:'domcontentloaded',timeout:90000});if(!response?.ok())throw Error('HTTP '+response?.status());
    const runtime=await page.evaluate(()=>typeof window.MLSysFigureExplore?.initialize==='function');
    if(!runtime){for(const entry of plan.entries){report.figures.push({key:entry.key,page:plan.url,lesson:entry.name,status:'failed',screenshots:[],states:0});}throw Error('Enrolled figures have no loaded MLSysFigureExplore runtime; inspect emitted dependencies.');}
    await page.waitForFunction(()=>[...document.querySelectorAll('[data-mlsys-explore-key]')].every(e=>['ready','failed'].includes(e.dataset.mlsysExploreState)),null,{timeout:mountTimeout}).catch(()=>{});
    pageResult.mountSummary=await page.evaluate(()=>[...document.querySelectorAll('[data-mlsys-explore-key]')].reduce((counts,e)=>{const state=e.dataset.mlsysExploreState||'unset';counts[state]=(counts[state]||0)+1;return counts;},{}));
    console.log(JSON.stringify({page:plan.url,mounts:pageResult.mountSummary}));
    if(!opts['integrity-only']&&!clockInstalled){await page.clock.install();clockInstalled=true;}
    for(const entry of plan.entries)await inspectFigure(page,entry);
    // Exercise Quarto's real hover preview: cloning enhanced controls must not duplicate IDs.
    const previewKey=await page.evaluate(keys=>[...document.querySelectorAll('a.quarto-xref')].map(a=>{try{const u=new URL(a.href);return u.pathname===location.pathname?u.hash.slice(1):null;}catch{return null;}}).find(id=>keys.includes(id)),plan.entries.map(e=>e.figure_id));
    if(!opts['capture-only']&&previewKey){
     const link=page.locator('a.quarto-xref[href$='+JSON.stringify('#'+previewKey)+']').first();
     if(await link.isVisible()){
      await link.hover();await page.waitForTimeout(600);
      const previews=await page.locator('.tippy-content').evaluateAll(nodes=>nodes.filter(e=>e.getBoundingClientRect().width&&e.getBoundingClientRect().height).map(e=>({interactive:e.querySelectorAll('.mlsys-figure-tabs,.mlsys-explore-panel').length,ids:[...e.querySelectorAll('[id]')].map(n=>n.id)})));
      if(previews.some(p=>p.interactive))fail(plan.url,'xref-preview-controls','Cross-reference preview cloned enhanced controls rather than a static-only figure',{previews});
      await page.mouse.move(0,0);
     }
    }
    const collisions=await page.evaluate(()=>{const ids=[...document.querySelectorAll('[role=tab],.mlsys-explore-panel,[data-mlsys-explore]')].map(e=>e.id).filter(Boolean);return ids.filter((id,i)=>ids.indexOf(id)!==i);});if(collisions.length)fail(plan.url,'page-component-ids',collisions.join(','));
   }catch(e){fail(plan.url,'page-navigation',e);}
   page.off('pageerror',onError);page.off('console',onConsole);
   const baselineErrors=new Set();
   if(errors.length&&opts['baseline-root']){
    const baselineFile=path.join(path.resolve(repo,opts['baseline-root']),path.relative(build,plan.file));
    if(fs.existsSync(baselineFile)){
     const baseline=await context.newPage();baseline.on('pageerror',e=>baselineErrors.add(e.stack||e.message));
     try{await baseline.goto(base+'/'+path.relative(books,baselineFile).split(path.sep).map(encodeURIComponent).join('/'),{waitUntil:'load',timeout:60000});await baseline.waitForTimeout(1500);}catch(e){report.warnings.push({page:plan.url,message:'Baseline error comparison unavailable: '+e.message});}finally{await baseline.close();}
    }
   }
   for(const error of [...new Set(errors)]){if(baselineErrors.has(error))report.baselineErrors.push({page:plan.url,error});else fail(plan.url,'browser-error',error);}
   report.warnings.push(...warnings.map(message=>({page:plan.url,message})));save();console.log(JSON.stringify({page:plan.url,scanned:report.figures.length,failures:report.failures.length}));
  }await context.close();}
 try{await Promise.all(Array.from({length:Math.min(workers,limited.length)},worker));}finally{await browser.close();report.running=false;report.finished=new Date().toISOString();save();}
 console.log(JSON.stringify(report.counts));process.exitCode=report.failures.length?1:report.missing.length?2:0;
})().catch(error=>{fail('scan','fatal',error);save();process.exitCode=1;});
