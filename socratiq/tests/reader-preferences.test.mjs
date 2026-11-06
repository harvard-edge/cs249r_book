import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const source = await readFile(new URL('../src_reader/preferences.js', import.meta.url), 'utf8');
const moduleForTest = () => import(`data:text/javascript;base64,${Buffer.from(source + `\n// ${Math.random()}`).toString('base64')}`);
const storage = initial => {
  const values = new Map(Object.entries(initial || {}));
  return {getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), values};
};

test('learner settings normalize bounded text and import existing study difficulty', async () => {
  const p = await moduleForTest();
  assert.deepEqual(p.normalizeLearnerPreferences({level: 'invalid', background: '  Python\u0000  ', instructions: 'x'.repeat(1000)}), {level: 'intermediate', background: 'Python', instructions: 'x'.repeat(800)});
  assert.equal(p.readLearnerPreferences(storage({userSettings: JSON.stringify({sliderValue: 0})})).level, 'beginner');
  assert.equal(p.readLearnerPreferences(storage({userSettings: '{damaged'})).level, 'intermediate');
});

test('saving synchronizes study level without losing unrelated settings; clear removes personal text', async () => {
  const p = await moduleForTest();
  const local = storage({userSettings: JSON.stringify({sliderValue: 2, customAPI: {enabled: false}, other: 'preserve'})});
  assert.equal(p.saveLearnerPreferences({level: 'beginner', background: 'Python', instructions: 'Use analogies'}, local).persisted, true);
  assert.deepEqual(p.readLearnerPreferences(local), {level: 'beginner', background: 'Python', instructions: 'Use analogies'});
  const legacy = JSON.parse(local.getItem('userSettings'));
  assert.equal(legacy.sliderValue, 0);
  assert.equal(legacy.other, 'preserve');
  assert.deepEqual(legacy.customAPI, {enabled: false});
  p.clearLearnerPreferences(local);
  assert.deepEqual(p.readLearnerPreferences(local), {level: 'intermediate', background: '', instructions: ''});
});

test('blocked storage keeps preferences usable for this page', async () => {
  const p = await moduleForTest();
  const blocked = {getItem() {throw Error('blocked');}, setItem() {throw Error('blocked');}};
  assert.equal(p.saveLearnerPreferences({level: 'advanced', background: 'Systems'}, blocked).persisted, false);
  assert.equal(p.readLearnerPreferences(blocked).background, 'Systems');
  p.clearLearnerPreferences(blocked);
  assert.equal(p.readLearnerPreferences(blocked).background, '');
});

test('preference saves scrub opted-out legacy API credentials without touching opted-in credentials', async () => {
  const p = await moduleForTest();
  const local = storage({userSettings: JSON.stringify({customAPI: {apiKey: 'constructed-not-a-secret', saveLocally: false}, unrelated: 'keep'})});
  p.saveLearnerPreferences({level: 'beginner'}, local);
  assert.equal(JSON.parse(local.getItem('userSettings')).customAPI.apiKey, '');
  assert.equal(JSON.parse(local.getItem('userSettings')).unrelated, 'keep');
  local.setItem('userSettings', JSON.stringify({customAPI: {apiKey: 'constructed-opted-in', saveLocally: true}}));
  p.saveLearnerPreferences({level: 'advanced'}, local);
  assert.equal(JSON.parse(local.getItem('userSettings')).customAPI.apiKey, 'constructed-opted-in');
});

test('custom API opt-out retains session credential but removes it from every settings save', async () => {
  const settingsSource = (await readFile(new URL('../src_shadow/js/components/settings/settings.js', import.meta.url), 'utf8'))
    .replace(/^import .*;\n/gm, '').split('export function setupModal')[0];
  const {SettingsManager} = await import(`data:text/javascript;base64,${Buffer.from('const showPopover = () => {};\n' + settingsSource).toString('base64')}`);
  const local = storage();
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', {configurable: true, value: local});
  try {
    const manager = Object.create(SettingsManager.prototype);
    manager.settings = {unrelated: 'preserved', customAPI: {}};
    manager.isInitializing = false;
    manager.dispatchAndUpdate_settings = () => {};
    manager.updateCustomAPI('openai', 'https://example.test/v1', 'constructed-model', 'constructed-session-key', false, true);
    assert.equal(manager.settings.customAPI.apiKey, 'constructed-session-key');
    assert.equal(JSON.parse(local.getItem('userSettings')).customAPI.apiKey, '');
    manager.updateCustomAPI('openai', 'https://example.test/v1', 'constructed-model', 'constructed-session-key', true, true);
    assert.equal(JSON.parse(local.getItem('userSettings')).customAPI.apiKey, 'constructed-session-key');
    manager.updateCustomAPI('openai', 'https://example.test/v1', 'constructed-model', 'constructed-session-key', false, true);
    assert.equal(manager.settings.customAPI.apiKey, 'constructed-session-key');
    manager.saveSettings();
    assert.equal(JSON.parse(local.getItem('userSettings')).customAPI.apiKey, '');
    assert.equal(JSON.parse(local.getItem('userSettings')).unrelated, 'preserved');
  } finally {
    if (previous) Object.defineProperty(globalThis, 'localStorage', previous); else delete globalThis.localStorage;
  }
});
