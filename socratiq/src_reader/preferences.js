// Local learner preferences, shared by the focused reader and lazy study workspace.
export const PREFERENCES_KEY = 'socratiq_learner_preferences_v1';
export const SELECTION_CLIP_KEY = 'socratiq_selection_clip_v1';
export const PREFERENCE_LIMITS = Object.freeze({background: 1200, instructions: 800});
export const SELECTION_CLIP_OPTIONS = Object.freeze([
  {id: 'explain', label: 'Explain this', shortLabel: 'Explain this', default: true},
  {id: 'ask', label: 'Ask…', shortLabel: 'Ask…', default: true},
  {id: 'check', label: 'Check my understanding', shortLabel: 'Check me', default: true},
  {id: 'taxonomy', label: 'Map to taxonomy', shortLabel: 'Taxonomy', default: true},
  {id: 'simplify', label: 'Simplify', shortLabel: 'Simplify', default: false},
  {id: 'example', label: 'Show an example', shortLabel: 'Example', default: false},
  {id: 'study', label: 'Open Study tools', shortLabel: 'Study tools', default: false},
]);
export const DEFAULT_CLIP_ACTIONS = Object.freeze(
  SELECTION_CLIP_OPTIONS.filter(option => option.default).map(option => option.id),
);
const VALID_CLIP_IDS = new Set(SELECTION_CLIP_OPTIONS.map(option => option.id));
function browserStorage() {try {return globalThis.localStorage;} catch {return null;}}
const LEVELS = ['beginner', 'intermediate', 'advanced'];
let temporary = null;
let temporaryClipActions = null;
const clean = (value, limit) => typeof value === 'string'
  ? value.replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, '').trim().slice(0, limit) : '';

export function normalizeClipActions(input) {
  if (!Array.isArray(input)) return [...DEFAULT_CLIP_ACTIONS];
  const valid = SELECTION_CLIP_OPTIONS
    .map(option => option.id)
    .filter(id => input.includes(id) && VALID_CLIP_IDS.has(id));
  return valid.length ? valid : ['explain'];
}

export function readClipActions(storage = browserStorage()) {
  if (temporaryClipActions) return [...temporaryClipActions];
  try {
    const saved = storage?.getItem(SELECTION_CLIP_KEY);
    if (saved && saved.length <= 2048) {
      const value = JSON.parse(saved);
      if (value?.version === 1 && Array.isArray(value.actions)) {
        return normalizeClipActions(value.actions);
      }
    }
  } catch { /* Unavailable or damaged storage must not prevent reading. */ }
  return [...DEFAULT_CLIP_ACTIONS];
}

export function saveClipActions(actions, storage = browserStorage()) {
  const normalized = normalizeClipActions(actions);
  let persisted = false;
  try {
    storage.setItem(SELECTION_CLIP_KEY, JSON.stringify({version: 1, actions: normalized}));
    persisted = true;
  } catch { /* Keep edits usable for this page when browser storage is blocked. */ }
  temporaryClipActions = persisted ? null : normalized;
  globalThis.dispatchEvent?.(new Event('socratiq:clip-actions-changed'));
  return {actions: normalized, persisted};
}

export function normalizeLearnerPreferences(input = {}) {
  const value = input && typeof input === 'object' && !Array.isArray(input) ? input : {};
  return {level: LEVELS.includes(value.level) ? value.level : 'intermediate',
    background: clean(value.background, PREFERENCE_LIMITS.background),
    instructions: clean(value.instructions, PREFERENCE_LIMITS.instructions)};
}

export function hasSavedLearnerPreferences(storage = browserStorage()) {
  try {
    const raw = storage?.getItem(PREFERENCES_KEY);
    if (!raw || raw.length > 8192) return false;
    const value = JSON.parse(raw);
    return value?.version === 1 && typeof value === 'object' && !Array.isArray(value);
  } catch {return false;}
}

export function readLearnerPreferences(storage = browserStorage()) {
  if (temporary) return {...temporary};
  try {
    const saved = storage?.getItem(PREFERENCES_KEY);
    if (saved && saved.length <= 8192) {
      const value = JSON.parse(saved);
      if (value?.version === 1 && typeof value === 'object' && !Array.isArray(value)) return normalizeLearnerPreferences(value);
    }
  } catch { /* Unavailable or damaged storage must not prevent reading. */ }
  let level = 'intermediate';
  try {
    const raw = storage?.getItem('userSettings');
    const legacy = raw && raw.length <= 65536 ? JSON.parse(raw) : null;
    if ([0, 1, 2, 3, '0', '1', '2', '3'].includes(legacy?.sliderValue)) {
      level = LEVELS[Math.min(Number(legacy.sliderValue), 2)];
    }
  } catch { /* Retain default without rewriting a legacy backup. */ }
  return normalizeLearnerPreferences({level});
}

export function saveLearnerPreferences(value, storage = browserStorage()) {
  const preferences = normalizeLearnerPreferences(value);
  let persisted = false;
  try {
    storage.setItem(PREFERENCES_KEY, JSON.stringify({version: 1, ...preferences}));
    persisted = true;
  } catch { /* Keep edits usable for this page when browser storage is blocked. */ }
  temporary = persisted ? null : preferences;
  // Update an existing legacy settings record without creating or discarding one.
  try {
    const raw = storage.getItem('userSettings');
    const legacy = raw && raw.length <= 65536 ? JSON.parse(raw) : null;
    if (legacy && typeof legacy === 'object' && !Array.isArray(legacy)) {
      if (!(preferences.level === 'advanced' && Number(legacy.sliderValue) === 3)) {
        legacy.sliderValue = LEVELS.indexOf(preferences.level);
      }
      if (legacy.customAPI && typeof legacy.customAPI === 'object' && legacy.customAPI.saveLocally !== true && 'apiKey' in legacy.customAPI) {
        legacy.customAPI = {...legacy.customAPI, apiKey: ''};
      }
      storage.setItem('userSettings', JSON.stringify(legacy));
    }
  } catch { /* Preserve damaged/blocked legacy settings. */ }
  globalThis.dispatchEvent?.(new Event('socratiq:preferences-changed'));
  return {preferences, persisted};
}

export function clearLearnerPreferences(storage = browserStorage()) {
  return saveLearnerPreferences({level: 'intermediate'}, storage);
}
