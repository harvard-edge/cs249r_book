import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile, mkdtemp} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import {chromium} from '@playwright/test';

test('review commits before refreshing card state for the next grade', {timeout: 60000}, async () => {
  const packageInfo = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));
  const server = await createServer({
    root: fileURLToPath(new URL('../src_shadow', import.meta.url)), configFile: false,
    cacheDir: await mkdtemp(join(tmpdir(), 'socratiq-review-test-')),
    server: {host: '127.0.0.1', port: 0, strictPort: true},
    optimizeDeps: {noDiscovery: true, include: Object.keys(packageInfo.dependencies)},
  });
  let browser;
  try {
    await server.listen();
    browser = await chromium.launch({headless: true});
    const page = await browser.newPage();
    await page.route('**/night-qa-blank', route => route.fulfill({contentType: 'text/html', body: '<!doctype html><title>Constructed review interleaving</title>'}));
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/night-qa-blank`);
    const result = await page.evaluate(async () => {
      const {SpacedRepetitionModal} = await import('/js/components/spaced_repetition/spaced-repetition-modal-handler.js');
      const {FlashCard} = await import('/js/components/spaced_repetition/spaced-repetition.js');
      const root = document.createElement('div');
      const card = new FlashCard('Constructed question', 'Constructed answer');
      let persisted = structuredClone(card);
      const trace = [];
      let commit;
      const modal = {
        shadowRoot: root, currentChapter: {chapter: 1}, chapterSets: new Map([[1, [card]]]),
        storageHandler: {
          inMemoryData: {currentDeck: [card]},
          saveToLocalStorage() {
            const snapshot = structuredClone(this.inMemoryData.currentDeck[0]);
            trace.push('save-start');
            return new Promise(resolve => {commit = () => {persisted = snapshot; trace.push('commit'); resolve(true);};});
          },
          getCurrentChapterStats: () => ({percentage: 0, learned: 0, total: 1}),
        },
        uiHandler: {updateProgress() {}},
        async immediateUpdate() {
          trace.push('reload');
          const restored = structuredClone(persisted);
          this.chapterSets.set(1, [restored]);
          this.storageHandler.inMemoryData.currentDeck = [restored];
        },
      };
      // Storage completion is controlled externally, as a pending IDB commit.
      const first = SpacedRepetitionModal.prototype.reviewCard.call(modal, 0, 3);
      await Promise.resolve();
      const beforeCommit = [...trace];
      commit();
      await first;
      const second = SpacedRepetitionModal.prototype.reviewCard.call(modal, 0, 3);
      await Promise.resolve();
      commit();
      await second;
      return {beforeCommit, trace, repetitions: persisted.repetitions, reviews: persisted.reviewHistory.length};
    });
    assert.deepEqual(result.beforeCommit, ['save-start']);
    assert.deepEqual(result.trace, ['save-start', 'commit', 'reload', 'save-start', 'commit', 'reload']);
    assert.equal(result.repetitions, 2);
    assert.equal(result.reviews, 2);
  } finally {
    await browser?.close();
    await server.close();
  }
});
