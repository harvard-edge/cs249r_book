// Opt-in: sends one public passage to the existing live AI service.
// Injects the local bundle into an isolated browser; does not modify the website.
import {chromium} from '@playwright/test';
import {mkdir, writeFile, readFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';

const output = process.argv[2];
if (!output) throw new Error('Usage: node tests/reader-live-smoke.mjs <receipt-directory> [public-chapter-url]');
const url = process.argv[3] || 'https://mlsysbook.ai/vol1/01_introduction/01_introduction.html';
if (new URL(url).hostname !== 'mlsysbook.ai') throw new Error('Live smoke checks use public MLSysBook passages only.');
await mkdir(output, {recursive: true});
const bundle = resolve('../books/shared/assets/scripts/socratiq/reader.js');
const browser = await chromium.launch();
const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
const receipts = [];
let responseReceipt;
page.on('response', response => {
  if (new URL(response.url()).origin + new URL(response.url()).pathname !== 'https://mlsysbook.ai/api/socratiq/ai/stream') return;
  if (response.request().method() !== 'POST') return;
  responseReceipt = (async () => {
    const receipt = {url: response.url(), status: response.status(), request: response.request().postDataJSON()};
    // Chromium can discard an SSE body when the reader closes it after [DONE].
    // Keep that limitation explicit; the visible answer is also saved below.
    try {receipt.response = await response.text();}
    catch (error) {receipt.streamCaptureError = error.message;}
    receipts.push(receipt);
  })();
});
try {
  await page.goto(url, {waitUntil: 'domcontentloaded'});
  await page.addScriptTag({path: bundle});
  const paragraphs = page.locator('main section p');
  const paragraphIndex = await paragraphs.evaluateAll(elements => elements.findIndex(element =>
    element.textContent.trim().length > 120 && /machine|system|model|agent|physical/i.test(element.textContent)
    && !element.closest('figure, figcaption, .callout-title-container')));
  if (paragraphIndex < 0) throw new Error('No substantive public passage found for the smoke check.');
  const passage = paragraphs.nth(paragraphIndex);
  await passage.scrollIntoViewIfNeeded();
  await passage.evaluate(element => {
    const range = document.createRange(); range.selectNodeContents(element);
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    document.dispatchEvent(new Event('selectionchange'));
  });
  await page.getByRole('button', {name: 'Explain this', exact: true}).click();
  await page.getByText('Explanation ready.', {exact: true}).waitFor({timeout: 60000});
  const answer = await page.locator('#socratiq-reader .reader-answer').last().innerText();
  if (!answer.trim()) throw new Error('No visible answer.');
  await page.screenshot({path: resolve(output, 'live-desktop.png')});
  await page.setViewportSize({width: 390, height: 844});
  await page.screenshot({path: resolve(output, 'live-mobile.png')});
  if (responseReceipt) await responseReceipt;
  await writeFile(resolve(output, 'response.json'), JSON.stringify({answer, requests: receipts}, null, 2));
  await writeFile(resolve(output, 'manifest.json'), JSON.stringify({
    kind: 'live inference smoke check, not a learning evaluation', timestamp: new Date().toISOString(), url,
    command: process.argv, sha: execFileSync('git', ['rev-parse', 'HEAD'], {encoding: 'utf8'}).trim(),
    dirty: Boolean(execFileSync('git', ['status', '--porcelain'], {encoding: 'utf8'}).trim()),
    bundleSha256: createHash('sha256').update(await readFile(bundle)).digest('hex'),
    responseSha256: createHash('sha256').update(await readFile(resolve(output, 'response.json'))).digest('hex'),
  }, null, 2));
  console.log(`Live explanation completed. Receipts: ${output}`);
} catch (error) {
  await page.screenshot({path: resolve(output, 'failure.png')}).catch(() => {});
  throw error;
} finally { await browser.close(); }
