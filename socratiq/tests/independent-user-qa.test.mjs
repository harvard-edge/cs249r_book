// Author-constructed chapter, questions, and provider responses. These checks measure
// browser behavior, not learning outcomes or live provider quality.
import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {hostname, platform, arch} from 'node:os';
import {resolve, join} from 'node:path';
import {chromium, webkit, firefox, expect} from '@playwright/test';

const repo = new URL('../../', import.meta.url);
const assets = new URL('../../books/shared/assets/scripts/socratiq/', import.meta.url);
const output = resolve(process.env.SOCRATIQ_INDEPENDENT_QA_OUTPUT || '/tmp/socratiq-independent-user-qa');
const browsers = {chromium, webkit, firefox};
const devices = {desktop: {width: 1366, height: 900}, phone: {width: 390, height: 844}, tablet: {width: 768, height: 1024}};
const filter = (name, values) => process.env[name]?.split(',') || values;
const passage = 'Memory bandwidth measures how much data can move per unit time. Latency measures how long an individual request takes.';
const explanation = '**Bandwidth** measures transfer rate; latency measures the delay of an individual request.\n\n$$t = D/B$$\n\n```python\ntransfer_time = data_bytes / bandwidth\n```\n\nA wider road carries more cars without shortening the journey.';
const quiz = [{question: 'Constructed QA: what does memory bandwidth measure?', answers: [{text: 'Data transferred per unit time', correct: true, explanation: 'Bandwidth is a rate.'}, {text: 'Delay of a single request', correct: false, explanation: 'That describes latency.'}]}];
const sse = content => `data: ${JSON.stringify({choices: [{delta: {content}, finish_reason: null}]})}\n\ndata: ${JSON.stringify({choices: [{delta: {}, finish_reason: 'stop'}]})}\n\ndata: [DONE]\n\n`;
const digest = data => createHash('sha256').update(data).digest('hex');
function runGit(...args) {return execFileSync('git', args, {cwd: repo, encoding: 'utf8'}).trim();}

async function selected(page) {
  await page.locator('#passage').evaluate(node => {
    const range = document.createRange(); range.selectNodeContents(node);
    const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
    document.dispatchEvent(new Event('selectionchange'));
  });
}
async function screenshot(page, directory, name, receipt) {
  const file = join(directory, `${name}.png`);
  await page.screenshot({path: file, animations: 'disabled'});
  receipt.screenshots.push({name, file, sha256: digest(await readFile(file))});
}
async function visibleInside(locator, page) {
  await expect(locator).toBeVisible();
  const box = await locator.boundingBox();
  const viewport = page.viewportSize();
  assert.ok(box.x >= -1 && box.x + box.width <= viewport.width + 1, `Horizontal clipping: ${JSON.stringify(box)}`);
  assert.ok(box.y >= -1 && box.y + box.height <= viewport.height + 1, `Vertical clipping: ${JSON.stringify(box)}`);
}

test('independent production-bundle browser and viewport matrix', {timeout: 2400000}, async t => {
  await mkdir(output, {recursive: true});
  const bundleBytes = Object.fromEntries(await Promise.all(['reader.js', 'study.js'].map(async name => [name, await readFile(new URL(name, assets))])));
  const manifest = {
    classification: 'Measured browser interactions on authored HTML and constructed provider responses',
    started: new Date().toISOString(), codeSHA: runGit('rev-parse', 'HEAD'), dirty: Boolean(runGit('status', '--porcelain')),
    command: 'node --test tests/independent-user-qa.test.mjs',
    testHash: digest(await readFile(new URL(import.meta.url))),
    filters: {browsers: process.env.QA_BROWSERS || null, volumes: process.env.QA_VOLUMES || null, devices: process.env.QA_DEVICES || null},
    workingDiffHash: digest(execFileSync('git', ['diff', 'HEAD'], {cwd: repo, maxBuffer: 64 * 1024 * 1024})),
    seeds: 'Not applicable: deterministic authored fixtures; generated browser origin/IDs do not determine pass labels',
    environment: {hostname: hostname(), platform: platform(), arch: arch(), node: process.version},
    inputs: Object.fromEntries(Object.entries(bundleBytes).map(([name, bytes]) => [name, digest(bytes)])),
    limits: 'Emulated viewport and DOM selection only. No physical devices, OS selection handles, native virtual keyboards, or live inference. All observed page errors fail the test.',
    thresholds: 'Every named action completes; selected controls stay in viewport; answers render math/code; answer switching grades the final choice; saved flashcard survives reload; no page errors.',
    cases: [],
  };
  await writeFile(join(output, 'manifest.json'), JSON.stringify(manifest, null, 2));
  const server = createServer((req, res) => {
    const path = new URL(req.url, 'http://localhost').pathname;
    if (/\/(reader|study)\.js$/.test(path)) {
      res.setHeader('Content-Type', 'text/javascript; charset=utf-8');
      res.end(bundleBytes[path.endsWith('study.js') ? 'study.js' : 'reader.js']); return;
    }
    const volume = path.match(/\bvol[1-4]\b/)?.[0] || 'vol1';
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    res.end(`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Constructed ${volume} QA chapter</title><style>body{font:18px/1.65 Georgia,serif;margin:0;background:#fff;color:#252525}nav{padding:16px;background:#eee}main{max-width:720px;margin:48px auto;padding:20px}h1,h2{line-height:1.2}@media(prefers-color-scheme:dark){body{background:#1b1b1b;color:#eee}nav{background:#333}}</style><script src="/${volume}/shared/assets/scripts/socratiq/reader.js" data-volume="${volume}" data-endpoint="https://mlsysbook.ai/api/socratiq/ai/stream" defer></script></head><body><nav>Constructed navigation excluded from context</nav><main id="quarto-document-content"><h1>Memory bandwidth and latency</h1><section id="bandwidth"><h2>Data movement</h2><p id="passage">${passage}</p><p>A system can have high bandwidth and still have high latency. A wider road carries more cars without shortening the journey.</p><p>Transfer time is data volume divided by bandwidth. The engineering decision depends on whether throughput or single-request latency limits the workload.</p></section></main></body></html>`);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  try {
    for (const engine of filter('QA_BROWSERS', Object.keys(browsers))) {
      const browser = await browsers[engine].launch({headless: true});
      try {
        for (const volume of filter('QA_VOLUMES', ['vol1', 'vol2', 'vol3', 'vol4'])) {
          for (const device of filter('QA_DEVICES', Object.keys(devices))) {
            await t.test(`${engine} ${volume} ${device}`, {timeout: 180000}, async () => {
              const directory = join(output, `${engine}-${volume}-${device}`);
              await mkdir(directory, {recursive: true});
              const context = await browser.newContext({viewport: devices[device], hasTouch: device !== 'desktop', colorScheme: volume === 'vol2' || volume === 'vol4' ? 'dark' : 'light', reducedMotion: 'reduce', acceptDownloads: true});
              const page = await context.newPage(); page.setDefaultTimeout(15000);
              const receipt = {engine, browserVersion: browser.version(), volume, device, viewport: devices[device], started: new Date().toISOString(), errors: [], requests: [], screenshots: [], status: 'running'};
              let responseKind = 'explanation';
              page.on('pageerror', error => receipt.errors.push(error.message));
              await page.addInitScript(() => {
                window.constructedClipboard = [];
                Object.defineProperty(navigator, 'clipboard', {configurable: true, value: {writeText: async text => window.constructedClipboard.push(text)}});
              });
              await page.route('**/*', route => {
                const request = route.request(), url = new URL(request.url());
                if (url.origin === origin) return route.continue();
                receipt.requests.push({url: request.url(), method: request.method(), body: request.postData()});
                if (/\/api\/socratiq\/ai(?:\/stream)?$/.test(url.pathname) || url.hostname.endsWith('workers.dev')) {
                  if (responseKind === 'error') return route.fulfill({status: 503, body: 'Constructed provider diagnostic must stay private'});
                  const content = responseKind === 'quiz' ? JSON.stringify(quiz) : explanation;
                  if (url.pathname.endsWith('/ai') || request.postDataJSON()?.stream === false) return route.fulfill({contentType: 'application/json', body: JSON.stringify({choices: [{message: {role: 'assistant', content}, finish_reason: 'stop'}]})});
                  return route.fulfill({contentType: 'text/event-stream', body: sse(content)});
                }
                return route.abort();
              });
              try {
                await page.goto(`${origin}/${volume}/chapter/`);
                await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).click();
                const tour = page.getByRole('dialog', {name: 'SocratiQ guided tour'});
                await visibleInside(tour, page);
                await screenshot(page, directory, 'fresh-tour', receipt);
                await tour.getByRole('button', {name: 'Next', exact: true}).click();
                await tour.getByRole('button', {name: 'Back', exact: true}).click();
                await tour.getByRole('button', {name: 'Next', exact: true}).click();
                await tour.getByRole('button', {name: 'Next', exact: true}).click();
                await tour.getByRole('button', {name: 'Done', exact: true}).click();
                await expect(tour).toBeHidden();
                await page.locator('.reader-preferences summary').click();
                await page.getByRole('combobox', {name: 'Explanation level'}).selectOption('beginner');
                await page.locator('.reader-background').fill('Constructed QA Python background');
                await page.locator('.reader-teaching').fill('Constructed QA request: define symbols');
                receipt.preferencesBeforeSave = await page.locator('.reader-preferences').evaluate(node => ({level: node.querySelector('select').value, fields: [...node.querySelectorAll('textarea')].map(node => node.value)}));
                await page.getByRole('button', {name: 'Save preferences', exact: true}).click();
                receipt.storedPreferences = await page.evaluate(() => localStorage.getItem('socratiq_learner_preferences_v1'));
                await expect(page.locator('.reader-preferences-status')).toContainText('saved');
                await screenshot(page, directory, 'learner-preferences', receipt);
                await page.locator('.reader-preferences summary').click();
                assert.equal(receipt.requests.length, 0, 'Opening a launcher does not call a provider');
                await page.getByRole('button', {name: 'Close SocratiQ reader'}).click();
                await selected(page);
                await page.getByRole('button', {name: 'Explain this', exact: true}).click();
                await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
                await expect(page.locator('.reader-answer .katex')).toBeVisible();
                await expect(page.locator('.reader-answer pre')).toContainText('transfer_time');
                await visibleInside(page.locator('.reader-panel'), page);
                await visibleInside(page.getByRole('button', {name: 'Send question', exact: true}), page);
                await screenshot(page, directory, 'reader-answer', receipt);
                const outgoing = receipt.requests.filter(request => request.body)[0].body;
                assert.ok(outgoing.includes(volume)); assert.ok(outgoing.includes(passage));
                assert.ok(outgoing.includes('Constructed QA Python background'));
                assert.ok(outgoing.includes('Constructed QA request: define symbols'));
                assert.ok(!outgoing.includes('Constructed navigation'));
                for (const action of ['Simplify', 'Show an example', 'Check my understanding']) {
                  await page.getByRole('button', {name: action, exact: true}).click();
                  await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
                }
                await page.getByRole('button', {name: 'Copy answer as Markdown'}).last().click();
                assert.ok((await page.evaluate(() => window.constructedClipboard.at(-1))).includes('Source:'));
                responseKind = 'error';
                await page.getByRole('textbox', {name: 'Ask a follow-up question'}).fill('Constructed retry question');
                await page.getByRole('button', {name: 'Send question', exact: true}).click();
                await expect(page.getByRole('button', {name: 'Retry', exact: true})).toBeVisible();
                await expect(page.locator('#socratiq-reader')).not.toContainText('Constructed provider diagnostic');
                responseKind = 'explanation';
                await page.getByRole('button', {name: 'Retry', exact: true}).click();
                await expect(page.getByText('Explanation ready.', {exact: true})).toBeVisible();
                await page.getByRole('button', {name: 'Study tools', exact: true}).click();
                const study = page.locator('#widget-chat-container');
                await expect(study.getByRole('button', {name: 'Close study tools', exact: true})).toBeVisible({timeout: 60000});
                await screenshot(page, directory, 'study-chat', receipt);
                await study.locator('#settings-btn').click();
                await expect(study.locator('#modal1')).toBeVisible();
                await study.locator('#study-learner-preferences summary').click();
                await expect(study.locator('#study-learner-background')).toHaveValue('Constructed QA Python background');
                await expect(study.locator('#study-learner-instructions')).toHaveValue('Constructed QA request: define symbols');
                await screenshot(page, directory, 'settings', receipt);
                await study.locator('#close-btn').click();
                await study.locator('#flashcards-button').click();
                await study.locator('#newDeckBtn').click();
                await study.locator('#deckNameInput').fill(`Constructed ${engine} ${volume} ${device} deck`);
                await study.locator('#createDeck').click();
                await study.locator('#showAddCard').click();
                await study.locator('#questionEditor [contenteditable=true]').pressSequentially('Constructed QA flashcard question');
                await study.locator('#answerEditor [contenteditable=true]').pressSequentially('Constructed QA flashcard answer');
                await study.locator('#saveCard').click();
                await expect(study.locator('#question-0 [contenteditable=true]')).toHaveText('Constructed QA flashcard question');
                await screenshot(page, directory, 'flashcards', receipt);
                await study.locator('#closeModal_sr').click();
                await study.locator('#quiz-button').click();
                responseKind = 'quiz';
                await study.locator('#section-quiz-btn').click();
                const form = study.locator('form[data-quiz]').last();
                await expect(form).toContainText(quiz[0].question, {timeout: 30000});
                await expect(form.locator('h4').first()).toBeInViewport();
                receipt.quizInitialGeometry = await form.locator('h4').first().evaluate(node => {const a = node.getBoundingClientRect(), b = node.getRootNode().querySelector('#message-container').getBoundingClientRect(); return {question: {top:a.top,bottom:a.bottom}, scroller:{top:b.top,bottom:b.bottom}};});
                await screenshot(page, directory, 'quiz-initial-anchor', receipt);
                await form.locator('input[type=radio][data-correct=false]').check();
                await form.locator('input[type=radio][data-correct=true]').check();
                await study.locator('button.socratiq-quiz-submit').last().click();
                await expect(study.locator('#result').last()).toContainText(`You got ${quiz.length} out of ${quiz.length} questions correct.`);
                await screenshot(page, directory, 'quiz-graded', receipt);
                await study.locator('#chat-quiz-btn').click();
                await expect(study.locator('.stats-modal')).toBeVisible();
                await expect(study.locator('.stats-modal .chart-loader')).toHaveCount(0, {timeout: 30000});
                await screenshot(page, directory, 'progress', receipt);
                await study.locator('.stats-modal .close-modal').click();
                await study.locator('#knowledge-graph-btn').click();
                const graph = study.locator('#socratiq-knowledge-graph-modal');
                await expect(graph).toBeVisible();
                if (device === 'phone') await expect(graph.locator('#list-view .list-item').first()).toBeVisible({timeout: 30000});
                else await expect(graph.locator('.knowledge-graph-viewport > svg')).toBeVisible({timeout: 30000});
                await screenshot(page, directory, 'knowledge-graph', receipt);
                await graph.locator(':scope > div > button').click();
                await study.locator('#help-btn').click();
                await expect(study.locator('#helpModalContainer')).toBeVisible();
                await expect(study.locator('#youtubeVideo')).toHaveAttribute('src', 'https://www.youtube.com/embed/mIT9nIsxCe0?enablejsapi=1');
                receipt.helpMedia = 'Embed URL and controls tested; external YouTube media blocked by fixture, playback untested';
                await screenshot(page, directory, 'help', receipt);
                await study.locator('#closeButtonHelpModal').click();
                await study.locator('#more-options-button').click();
                await page.locator('#socratiq-more-dropdown #meditation-btn').click();
                await expect(study.locator('#med-card')).toBeVisible();
                await study.locator('#med-svg').click({position: {x: 195, y: 105}});
                await expect(study.locator('#med-time-display')).not.toHaveText('0:00');
                const configuredDuration = await study.locator('#med-time-display').textContent();
                await study.locator('#med-play').click();
                await expect(study.locator('#med-status')).toContainText('Breathe');
                await study.locator('#med-play').click();
                await study.locator('#med-reset').click();
                await expect(study.locator('#med-time-display')).toHaveText(configuredDuration);
                await study.locator('#med-pulse-toggle').click();
                await expect(study.locator('#med-pulse-label')).toHaveText('Pulse: off');
                await screenshot(page, directory, 'meditation-timer', receipt);
                await study.locator('#med-close').click();
                await study.locator('#more-options-button').click();
                await page.locator('#socratiq-more-dropdown #draw-select-btn').click();
                await expect(page.locator('#socratiq-lasso-canvas')).toBeVisible();
                await expect(study).toBeHidden();
                await page.keyboard.press('Escape');
                await expect(page.locator('#socratiq-lasso-canvas')).toHaveCount(0);
                await expect(study).toHaveJSProperty('hidden', false);
                await expect(study.getByRole('button', {name: 'Close study tools', exact: true})).toBeVisible();
                await study.getByRole('button', {name: 'Close study tools', exact: true}).click();
                await expect(page.getByRole('button', {name: 'Study tools', exact: true})).toBeFocused();
                await page.reload();
                await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).click();
                await page.getByRole('button', {name: 'Study tools', exact: true}).click();
                await study.locator('#flashcards-button').click();
                await expect(study.getByText(`Constructed ${engine} ${volume} ${device} deck`, {exact: true}).first()).toBeVisible();
                await expect(study.locator('#question-0 [contenteditable=true]')).toHaveText('Constructed QA flashcard question');
                await study.locator('#closeModal_sr').click();
                if (volume === 'vol1' && device === 'desktop') {
                  receipt.settingsBeforeReset = await page.evaluate(() => ({preferences: JSON.parse(localStorage.getItem('socratiq_learner_preferences_v1')), customAPI: JSON.parse(localStorage.getItem('userSettings'))?.customAPI}));
                  await study.locator('#new-chat-btn').click();
                  await expect(study.locator('#reset-btn')).toBeVisible();
                  await screenshot(page, directory, 'global-reset-control', receipt);
                  page.once('dialog', async dialog => {receipt.resetConfirmation = dialog.message(); await dialog.accept();});
                  const navigation = page.waitForEvent('load');
                  await study.locator('#reset-btn').click();
                  await navigation;
                  assert.match(receipt.resetConfirmation, /Learning preferences and custom API settings will remain/);
                  assert.doesNotMatch(receipt.resetConfirmation, /clear all data/i);
                  receipt.settingsAfterReset = await page.evaluate(() => ({preferences: JSON.parse(localStorage.getItem('socratiq_learner_preferences_v1')), customAPI: JSON.parse(localStorage.getItem('userSettings'))?.customAPI}));
                  assert.deepEqual(receipt.settingsAfterReset, receipt.settingsBeforeReset, 'Global reset retains learning preferences and custom API settings');
                  await page.getByRole('button', {name: 'Ask SocratiQ', exact: true}).click();
                  await page.getByRole('button', {name: 'Study tools', exact: true}).click();
                  receipt.storageAfterReset = await page.evaluate(async () => {
                    const request = indexedDB.open('socratiqDB');
                    return new Promise((resolve, reject) => {
                      request.onerror = () => reject(request.error);
                      request.onsuccess = async () => {
                        const db = request.result;
                        const names = ['cards','quizHighScores','quizHistory','quizStats','quizAttempts','ongoingIncorrectQuestions','cumulativeQuizScores','pageCompletions'];
                        const rows = {};
                        for (const name of names) {
                          if (!db.objectStoreNames.contains(name)) continue;
                          rows[name] = await new Promise((resolve, reject) => {const read = db.transaction(name).objectStore(name).getAll(); read.onsuccess = () => resolve(read.result); read.onerror = () => reject(read.error);});
                        }
                        db.close(); resolve(rows);
                      };
                    });
                  });
                  assert.ok(Object.values(receipt.storageAfterReset).every(rows => rows.length === 0), 'Global reset clears card and quiz stores');
                }
                assert.ok(!receipt.requests.some(request => new URL(request.url).hostname.endsWith('workers.dev')), 'Production widget must use the configured public gateway');
                assert.deepEqual(receipt.errors, []);
                receipt.status = 'passed';
              } catch (error) {
                receipt.status = 'failed'; receipt.failure = error.stack;
                await writeFile(join(directory, 'failure-dom.html'), await page.locator('#widget-chat-container').evaluate(node => node.shadowRoot.innerHTML)).catch(() => {});
                await screenshot(page, directory, 'failure', receipt).catch(() => {});
                throw error;
              } finally {
                receipt.finished = new Date().toISOString();
                await writeFile(join(directory, 'receipt.json'), JSON.stringify(receipt, null, 2));
                manifest.cases.push({engine, volume, device, status: receipt.status, receipt: join(directory, 'receipt.json')});
                await writeFile(join(output, 'manifest.json'), JSON.stringify(manifest, null, 2));
                await context.close();
              }
            });
          }
        }
      } finally {await browser.close();}
    }
  } finally {
    await new Promise(resolve => server.close(resolve));
    manifest.finished = new Date().toISOString();
    await writeFile(join(output, 'manifest.json'), JSON.stringify(manifest, null, 2));
  }
});
