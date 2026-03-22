// LocalStorage-based progress tracking for the Heat Map

export interface AttemptRecord {
  questionId: string;
  competencyArea: string;
  track: string;
  level: string;
  selfScore: number; // 0-3: 0=skipped, 1=wrong, 2=partial, 3=correct
  timestamp: number;
}

export interface GauntletResult {
  id: string;
  track: string;
  level: string;
  questionCount: number;
  duration: number; // seconds
  attempts: AttemptRecord[];
  completedAt: number;
}

const STORAGE_KEY = 'staffml_progress';
const GAUNTLET_KEY = 'staffml_gauntlets';

function getStorage<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function setStorage<T>(key: string, value: T): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    console.warn('Failed to write to localStorage');
  }
}

export function getAttempts(): AttemptRecord[] {
  return getStorage<AttemptRecord[]>(STORAGE_KEY, []);
}

const MAX_ATTEMPTS = 500;
const MAX_GAUNTLETS = 50;

export function saveAttempt(attempt: AttemptRecord): void {
  const attempts = getAttempts();
  attempts.push(attempt);
  // Cap to prevent localStorage overflow (~5-10 MB limit)
  if (attempts.length > MAX_ATTEMPTS) {
    attempts.splice(0, attempts.length - MAX_ATTEMPTS);
  }
  setStorage(STORAGE_KEY, attempts);
}

export function getGauntletResults(): GauntletResult[] {
  return getStorage<GauntletResult[]>(GAUNTLET_KEY, []);
}

export function saveGauntletResult(result: GauntletResult): void {
  const results = getGauntletResults();
  results.push(result);
  if (results.length > MAX_GAUNTLETS) {
    results.splice(0, results.length - MAX_GAUNTLETS);
  }
  setStorage(GAUNTLET_KEY, results);
}

// Heat map data: competency × track → { attempted, correct, total }
export interface HeatCell {
  attempted: number;
  correct: number;
  total: number;
}

export function getHeatMapData(): Record<string, Record<string, HeatCell>> {
  const attempts = getAttempts();
  const map: Record<string, Record<string, HeatCell>> = {};

  attempts.forEach(a => {
    if (!map[a.competencyArea]) map[a.competencyArea] = {};
    if (!map[a.competencyArea][a.track]) {
      map[a.competencyArea][a.track] = { attempted: 0, correct: 0, total: 0 };
    }
    const cell = map[a.competencyArea][a.track];
    cell.attempted++;
    if (a.selfScore >= 2) cell.correct++;
  });

  return map;
}

export function getCompetencyScore(competencyArea: string): number {
  const attempts = getAttempts().filter(a => a.competencyArea === competencyArea);
  if (attempts.length === 0) return 0;
  const correct = attempts.filter(a => a.selfScore >= 2).length;
  return Math.round((correct / attempts.length) * 100);
}

export function clearProgress(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
    window.localStorage.removeItem(GAUNTLET_KEY);
  } catch {}
}
