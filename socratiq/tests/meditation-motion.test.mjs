import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile, mkdir} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {testBrowser} from './legacy-test-browser.mjs';

// Constructed timer session: no provider requests or inferred health outcomes.
test('meditation controls remain still during breathing and respect reduced motion', {timeout: 30000}, async () => {
  const source = await readFile(new URL('../src_shadow/js/components/meditation/meditationTimer.js', import.meta.url), 'utf8');
  const directory = process.env.SOCRATIQ_TEST_RECEIPT_DIR || join(tmpdir(), 'socratiq-meditation-motion');
  await mkdir(directory, {recursive: true});
  const browser = await testBrowser.launch({headless: true});
  try {
    for (const reducedMotion of ['reduce', 'no-preference']) {
      const page = await browser.newPage({viewport: {width: 375, height: 667}, reducedMotion});
      await page.setContent('<!doctype html><body>Constructed timer accessibility check</body>');
      await page.evaluate(async source => {
        const module = await import(URL.createObjectURL(new Blob([source], {type: 'text/javascript'})));
        const host = document.createElement('div'); document.body.append(host);
        module.openMeditationTimer(host.attachShadow({mode: 'open'}));
      }, source);
      await page.locator('#med-svg').click({position: {x: 195, y: 105}});
      await page.locator('#med-play').click();
      await page.waitForTimeout(2000); // Sample away from the animation's stationary endpoint.
      const state = await page.locator('#med-card').evaluate(card => ({
        cardTransform: getComputedStyle(card).transform,
        dialAnimation: getComputedStyle(card.querySelector('#med-dial')).animationName,
      }));
      assert.equal(state.cardTransform, 'none');
      assert.equal(state.dialAnimation, reducedMotion === 'reduce' ? 'none' : 'med-breathe');
      await page.screenshot({path: join(directory, `timer-${reducedMotion}.png`)});
      await page.locator('#med-play').click({timeout: 1200});
      assert.equal(await page.locator('#med-status').textContent(), 'Paused');
      await page.locator('#med-close').click();
      await page.close();
    }
  } finally {await browser.close();}
});
