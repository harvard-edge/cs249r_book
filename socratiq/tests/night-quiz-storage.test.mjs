import test from 'node:test';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import {readFile, writeFile} from 'node:fs/promises';
import {testBrowser} from './legacy-test-browser.mjs';

// Constructed questions and scores exercise browser IndexedDB; no provider calls.
test('legacy quizzes isolate same-title volumes, retain old records, and preserve selected source scope', {timeout:90000}, async () => {
  const server=await createServer({
    configFile:fileURLToPath(new URL('../vite.config.dev.mjs',import.meta.url)),
    root:fileURLToPath(new URL('../src_shadow',import.meta.url)),
    server:{port:0,host:'127.0.0.1',open:false}
  });
  let browser;
  try {
    await server.listen();
    const baseUrl=`http://127.0.0.1:${server.httpServer.address().port}`;
    browser=await testBrowser.launch({headless:true});
    const page=await browser.newPage();
    await page.route('**/night-quiz-harness.html',route=>route.fulfill({contentType:'text/html',body:'<!doctype html><title>Constructed quiz storage harness</title><body></body>'}));
    // Keep ini_quiz's chat callback isolated from the widget bootstrap.
    await page.route('**/js/index.js',route=>route.fulfill({contentType:'text/javascript',body:'export async function saveChatHistory() {}'}));
    if(process.env.QUIZ_STORAGE_BASELINE) {
      const source=await readFile(process.env.QUIZ_STORAGE_BASELINE,'utf8');
      await page.route('**/js/components/quiz/quiz-storage.js',route=>route.fulfill({contentType:'text/javascript',body:source}));
    }
    await page.goto(`${baseUrl}/night-quiz-harness.html`);
    const receipt=await page.evaluate(async()=>{
      const api=await import('/js/components/quiz/quiz-storage.js');
      const {getDBInstance}=await import('/js/libs/utils/indexDb.js');
      const db=await getDBInstance();
      const urls=[1,2,3,4].map(volume=>`${location.origin}/books/vol${volume}/introduction.html`);
      const title='Cumulative Quiz - Introduction';
      const legacyHigh={quizTitle:title,score:{correct:1,total:3},percentageScore:'33.33',legacyMarker:'unassigned'};
      const legacyHistory={id:title,quizTitle:title,attempts:[{date:'2020-01-01',score:{correct:1,total:3},answers:[],quizData:[]}],legacyMarker:'unassigned'};
      await db.add('quizHighScores',legacyHigh);
      await db.add('quizHistory',legacyHistory);
      for(const url of urls) await db.add('chapterMap',{url,title:'Introduction',toc:{headers:[]}});
      let successCallbacks=0;
      let lastCommitted=false;
      let precommitSuccess=false;
      const submit=async(url, quizTitle, correct, sourceUrls=url?[url]:[])=>{
        const form=document.createElement('form');
        if (url) form.setAttribute('data-context-url',`${url}?mode=quiz#section`);
        form.setAttribute('data-quiz-title',quizTitle);
        form.setAttribute('data-source-page-urls',typeof sourceUrls==='string'?sourceUrls:JSON.stringify(sourceUrls));
        form.innerHTML=[0,1].map(index=>`<div class="mb-8"><h4>Constructed question ${index}</h4><ul><li><input type="radio" name="q${index}" value="0" data-correct="true" ${index<correct?'checked':''}><label>Constructed correct answer</label><p class="explanation">Constructed explanation</p></li><li><input type="radio" name="q${index}" value="1" data-correct="false" ${index>=correct?'checked':''}><label>Constructed wrong answer</label><p class="explanation">Constructed explanation</p></li></ul></div>`).join('');
        return await api.enhanceQuizSubmit(()=>{successCallbacks++;if(!lastCommitted) precommitSuccess=true;},form,quizTitle)(new Event('submit',{cancelable:true}));
      };
      for(const url of urls) await submit(url,title,2);
      const firstPass={
        highs:await api.getAllHighScores(),histories:await db.getAll('quizHistory'),completions:await db.getAll('pageCompletions'),
        legacyHigh:await db.getByKey('quizHighScores',title),legacyHistory:await db.getByKey('quizHistory',title),
        scopedReads:await Promise.all(urls.map(url=>api.getQuizHistory(title,url)))
      };
      await submit(urls[2],title,0);
      await submit(urls[2],'Section Quiz',1);
      await submit(urls[3],'Cumulative Quiz - selected material',2,[urls[0],urls[1]]);
      await submit(urls[0],'Same question across volumes',0);
      await submit(urls[1],'Same question across volumes',0);
      await submit(null,'Unassigned restored quiz',1);
      const unassigned=(await api.getAllHighScores()).find(row=>row.quizTitle==='Unassigned restored quiz');
      await submit(urls[0],'Damaged restored scope',1,'{broken JSON');
      const damaged=(await api.getAllHighScores()).find(row=>row.quizTitle==='Damaged restored scope');
      const concurrentScores=[2,0,1,0,2,1];
      const attemptsBefore=(await db.getAll('quizAttempts'))[0].attempts;
      const enhancedBefore=(await api.getCumulativeQuizRecord(urls[0]))?.quizHistory.length || 0;
      await Promise.all(concurrentScores.map(correct=>submit(urls[0],'Concurrent attempts',correct)));
      const concurrent={
        scores:concurrentScores,
        enhancedDelta:((await api.getCumulativeQuizRecord(urls[0]))?.quizHistory.length || 0)-enhancedBefore,
        high:await api.getHighScore('Concurrent attempts',urls[0]),
        history:await api.getQuizHistory('Concurrent attempts',urls[0]),
        attemptsDelta:(await db.getAll('quizAttempts'))[0].attempts-attemptsBefore
      };
      const raceUrl=`${location.origin}/books/vol4/concurrency.html`;
      await db.add('chapterMap',{url:raceUrl,title:'Concurrent page',toc:[]});
      await Promise.all([submit(raceUrl,'Cumulative Quiz - concurrent pass',2),submit(raceUrl,'Concurrent section',0)]);
      const enhancedRace=await api.getCumulativeQuizRecord(raceUrl);
      const post={
        unassigned,unassignedHistory:unassigned?.quizId ? await api.getQuizHistory(unassigned) : [],
        high:await api.getHighScore(title,urls[2]),completion:await api.isPageCompletedByCumulativeQuiz(urls[2]),
        mixedRecord:await api.getCumulativeQuizRecord(urls[3]),
        mixedStats:(await db.getAll('quizStats')).find(row=>row.title==='Cumulative Quiz - selected material'),
        questions:await api.getIncorrectQuestions(),
        historyCounts:await Promise.all(urls.map(async url=>(await api.getQuizHistory(title,url)).length))
      };
      const stores=['quizHighScores','quizHistory','quizStats','quizAttempts','ongoingIncorrectQuestions','cumulativeQuizScores','pageCompletions','chapterMap'];
      const snapshot=async()=>Object.fromEntries(await Promise.all(stores.map(async store=>[store,await db.getAll(store)])));
      const beforeAbort=await snapshot();
      const countersBefore=[localStorage.getItem('quizStreak'),localStorage.getItem('totalQuizzes')];
      const callbacksBefore=successCallbacks;
      const nativePut=IDBObjectStore.prototype.put;
      const nativeTransaction=IDBDatabase.prototype.transaction;
      lastCommitted=false;precommitSuccess=false;
      IDBDatabase.prototype.transaction=function(...args){
        const transaction=nativeTransaction.apply(this,args);
        if(Array.isArray(args[0])&&args[0].includes('quizStats')) transaction.addEventListener('complete',()=>{lastCommitted=true;});
        return transaction;
      };
      let aborted;
      try {
        IDBObjectStore.prototype.put=function(data,...args){
          if(this.name==='pageCompletions'&&data.cumulativeQuizTitle==='Cumulative Quiz - injected abort') throw new DOMException('Constructed storage failure','QuotaExceededError');
          return nativePut.call(this,data,...args);
        };
        aborted=await submit(urls[0],'Cumulative Quiz - injected abort',2);
      } finally {IDBObjectStore.prototype.put=nativePut;}
      const abortResult={saved:aborted,before:beforeAbort,after:await snapshot(),callbacksDelta:successCallbacks-callbacksBefore,countersBefore,countersAfter:[localStorage.getItem('quizStreak'),localStorage.getItem('totalQuizzes')]};
      await submit(urls[0],'Post-commit callback probe',2);
      const commitResult={lastCommitted,precommitSuccess};
      IDBDatabase.prototype.transaction=nativeTransaction;
      // Render a generated quiz, serialize it like a saved chat, then rehydrate it.
      const {ini_quiz,reinitializeQuizButtons}=await import('/js/components/quiz/load_quiz.js');
      const host=document.createElement('div');document.body.append(host);
      const root=host.attachShadow({mode:'open'});
      const markup=()=>'<div class="ai-message-chat"><div id="quiz"><form id="quiz-form"></form><button id="submit-quiz-btn">Submit</button><p id="result"></p></div></div>';
      root.innerHTML=markup();
      const question=[{question:'Constructed restored question',answers:[{text:'Correct',correct:true,explanation:'Constructed'},{text:'Wrong',correct:false,explanation:'Constructed'}]}];
      await ini_quiz(root,question,'Captured section','', '', {url:urls[1],title:'Introduction',content:{totalLength:30,sourceCount:1},position:{scrollPercentage:{y:0}},timestamp:'constructed'});
      const sectionUrl=root.querySelector('form').getAttribute('data-context-url');
      root.innerHTML=markup();
      await ini_quiz(root,question,'Selected source quiz','', '', [{pageUrl:urls[0],label:'Vol1',position:0},{pageUrl:urls[1],label:'Vol2',position:0}]);
      const generatedScope=JSON.parse(root.querySelector('form').getAttribute('data-source-page-urls'));
      root.innerHTML=root.innerHTML;
      reinitializeQuizButtons(root);
      root.querySelector('input').checked=true;
      const submitted=new Promise(resolve=>window.addEventListener('quizSubmitted',resolve,{once:true}));
      root.querySelector('button.socratiq-quiz-submit').click();
      await submitted;
      const restoredStats=(await db.getAll('quizStats')).find(row=>row.title==='Selected source quiz');
      return {urls,title,legacyHigh,legacyHistory,firstPass,post,sectionUrl,generatedScope,restoredStats,concurrent,damaged,enhancedRace,abortResult,commitResult};
    });
    if(process.env.QUIZ_STORAGE_RECEIPT) await writeFile(process.env.QUIZ_STORAGE_RECEIPT,JSON.stringify(receipt,null,2));
    const {urls,title,firstPass,post}=receipt;
    assert.deepEqual(firstPass.legacyHigh,receipt.legacyHigh,'unassigned legacy high score retained');
    assert.deepEqual(firstPass.legacyHistory,receipt.legacyHistory,'unassigned legacy history retained');
    assert.equal(firstPass.highs.filter(row=>row.pageUrl).length,urls.length);
    assert.deepEqual(firstPass.scopedReads.map(attempts=>attempts.length),urls.map(()=>1));
    assert.deepEqual(firstPass.completions.map(row=>row.pageUrl).sort(),urls.slice().sort());
    assert.ok(firstPass.highs.filter(row=>row.pageUrl).every(row=>row.quizTitle===title));
    assert.deepEqual(post.historyCounts,[1,1,2,1]);
    assert.equal(receipt.abortResult.saved,false);
    assert.deepEqual(receipt.abortResult.after,receipt.abortResult.before,'aborted submission changes no store');
    assert.equal(receipt.abortResult.callbacksDelta,0,'failure invokes no success UI callback');
    assert.deepEqual(receipt.abortResult.countersAfter,receipt.abortResult.countersBefore);
    assert.deepEqual(receipt.commitResult,{lastCommitted:true,precommitSuccess:false});
    assert.equal(receipt.damaged.pageUrl,null);
    assert.equal(receipt.concurrent.high.score.correct,Math.max(...receipt.concurrent.scores));
    assert.equal(receipt.concurrent.history.length,receipt.concurrent.scores.length);
    assert.equal(receipt.concurrent.attemptsDelta,receipt.concurrent.scores.length);
    assert.equal(receipt.concurrent.enhancedDelta,receipt.concurrent.scores.length);
    assert.equal(receipt.enhancedRace.quizHistory.length,2);
    assert.equal(receipt.enhancedRace.completedByCumulativeQuiz,true);
    assert.equal(post.unassigned.pageUrl,null);
    assert.equal(post.unassignedHistory.length,1);
    assert.equal(post.high.score.correct,2);
    assert.equal(post.high.isCumulative,true);
    assert.equal(post.completion,true);
    assert.deepEqual(post.mixedStats.sourcePageUrls,urls.slice(0,2));
    assert.ok(!post.mixedRecord.quizHistory.some(row=>row.quizTitle==='Cumulative Quiz - selected material'));
    assert.equal(new Set(post.questions.filter(row=>row.quizTitle==='Same question across volumes').map(row=>row.pageUrl)).size,2);
    assert.equal(receipt.sectionUrl,urls[1]);
    assert.deepEqual(receipt.generatedScope,urls.slice(0,2));
    assert.deepEqual(receipt.restoredStats.sourcePageUrls,urls.slice(0,2));
  } finally {await browser?.close();await server.close();}
});
