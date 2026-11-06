import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile, mkdir, mkdtemp} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createServer} from 'vite';
import {chromium} from '@playwright/test';
import {fileURLToPath} from 'node:url';

// Constructed local quiz data. This checks export mechanics, with no model calls.
test('stored quiz history exports both PDF tables through the ESM plugin', {timeout: 60000}, async () => {
  const packageInfo = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));
  const server = await createServer({
    root: fileURLToPath(new URL('../src_shadow', import.meta.url)),
    configFile: false,
    cacheDir: await mkdtemp(join(tmpdir(), 'socratiq-pdf-test-')),
    server: {host: '127.0.0.1', port: 0, strictPort: true},
    optimizeDeps: {noDiscovery: true, include: Object.keys(packageInfo.dependencies)},
  });
  let browser;
  try {
    await server.listen();
    const port = server.httpServer.address().port;
    browser = await chromium.launch({headless: true});
    const page = await browser.newPage();
    await page.route('**/night-qa-blank', route => route.fulfill({
      contentType: 'text/html', body: '<!doctype html><title>Constructed PDF fixture</title>',
    }));
    await page.goto(`http://127.0.0.1:${port}/night-qa-blank`);
    // Load dependencies before creating the stored attempt, avoiding optimizer reloads.
    await page.evaluate(async () => {
      await import('/js/components/quiz/generateQuizPDF.js');
      const {initializeSocratiqDB} = await import('/js/libs/utils/indexDb.js');
      const db = await initializeSocratiqDB();
      const title = '1.1 Constructed PDF section (source)';
      const date = new Date().toISOString();
      const score = {correct: 1, total: 1};
      await db.put('quizHighScores', {quizTitle: title, score, percentageScore: '100.00', achievedAt: date});
      await db.put('quizHistory', {
        id: title, quizTitle: title,
        attempts: [{date, score,
          answers: [{question: 'Constructed question?', correctAnswer: 'Constructed answer',
            selectedAnswerText: 'Constructed answer', wasCorrect: true}],
          quizData: [{question: 'Constructed question?', answers: [
            {text: 'Constructed answer', correct: true}, {text: 'Constructed distractor', correct: false},
          ]}],
        }],
      });
      const {getQuizIdentity} = await import('/js/components/quiz/quiz-storage.js');
      for (const pageUrl of ['https://example.org/vol1/memory', 'https://example.org/vol2/memory']) {
        const quizId = getQuizIdentity(title, pageUrl);
        await db.put('quizHighScores', {quizTitle: quizId, title, quizId, pageUrl, score, percentageScore: '100.00', achievedAt: date});
        await db.put('quizHistory', {id: quizId, quizId, quizTitle: title, pageUrl,
          attempts: [{date, score, answers: [{question: 'Constructed scoped question?', correctAnswer: 'Constructed answer', selectedAnswerText: 'Constructed answer', wasCorrect: true}],
            quizData: [{question: 'Constructed scoped question?', answers: [{text: 'Constructed answer', correct: true}]}]}]});
      }
    });
    const downloadReady = page.waitForEvent('download');
    const result = await page.evaluate(async () => {
      const {generateQuizPDF} = await import('/js/components/quiz/generateQuizPDF.js');
      return generateQuizPDF();
    });
    const download = await downloadReady;
    const bytes = await readFile(await download.path());
    const content = bytes.toString('latin1');
    assert.ok(content.startsWith('%PDF-'));
    assert.match(content, /Total Attempts: 1/);
    assert.match(content, /Constructed question/);
    assert.match(content, /Constructed distractor/);
    // Verification metadata is base64; these literal PDF text runs are the
    // visible source labels, not strings found only in hidden metadata.
    assert.match(content, /\(Source: https:\/\/example\.org\/vol1\/memory\) Tj/);
    assert.match(content, /\(Source: https:\/\/example\.org\/vol2\/memory\) Tj/);
    assert.match(content, /\(Source: Unassigned source\) Tj/);
    assert.ok(result.hash);
    if (process.env.SOCRATIQ_PDF_TEST_OUTPUT) {
      await mkdir(process.env.SOCRATIQ_PDF_TEST_OUTPUT, {recursive: true});
      await download.saveAs(`${process.env.SOCRATIQ_PDF_TEST_OUTPUT}/constructed-quiz-report.pdf`);
    }
    const verification = await page.evaluate(async data => {
      const {verifyPDFReport} = await import('/js/components/quiz/verifyReport.js');
      const original = Uint8Array.from(data);
      const current = await verifyPDFReport(new File([original], 'constructed-quiz-report.pdf', {type: 'application/pdf'}));
      const pdfText = Array.from(original, byte => String.fromCharCode(byte)).join('');
      const encoded = pdfText.match(/\(base64:([A-Za-z0-9+/=]+)\)/)[1];
      const changed = btoa(atob(encoded).replace('"correct":1', '"correct":0'));
      const tamperedText = pdfText.replace(encoded, changed);
      const tampered = await verifyPDFReport(new File([Uint8Array.from(tamperedText, character => character.charCodeAt(0))], 'constructed-modified-metadata.pdf'));

      // Construct an older raw-JSON report using the same jsPDF literal encoder.
      const {jsPDF} = await import('/@id/jspdf');
      const {generateContentHash} = await import('/js/libs/utils/hashUtils.js');
      const legacy = new jsPDF();
      const timestamp = new Date().toISOString();
      const content = JSON.stringify({reportId: 'Constructed legacy report', highScores: [
        {quizTitle: 'ConstructedTitle', quizId: 'spaces (parentheses) \\ path'},
      ]});
      legacy.text('---VERIFY_CONTENT_START---', 10, 10);
      legacy.text(content, 10, 20);
      legacy.text('---VERIFY_CONTENT_END---', 10, 30);
      legacy.text('Verification Code:', 10, 40);
      legacy.text(generateContentHash(content, timestamp), 10, 50);
      legacy.text('Generated:', 10, 60);
      legacy.text(timestamp, 10, 70);
      const older = await verifyPDFReport(new File([legacy.output('arraybuffer')], 'constructed-legacy.pdf'));
      return {current, tampered, older};
    }, [...bytes]);
    assert.equal(verification.current.isValid, true, JSON.stringify(verification.current));
    assert.equal(verification.older.isValid, true, JSON.stringify(verification.older));
    assert.equal(verification.tampered.isValid, false, JSON.stringify(verification.tampered));
  } finally {
    await browser?.close();
    await server.close();
  }
});
