import { getQuestions, Question } from "@/lib/corpus";

function hashString(s: string): number {
  let hash = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    hash = ((hash << 5) - hash) + c;
    hash |= 0;
  }
  return Math.abs(hash);
}

export function getDailyQuestions(): Question[] {
  const today = new Date().toISOString().slice(0, 10);
  const seed = hashString(today);
  const pool = getQuestions();
  if (pool.length === 0) return [];

  const primes = [1, 31, 97];
  const indices = primes.map(p => (seed * p) % pool.length);
  const unique: number[] = [];
  indices.forEach(i => { if (!unique.includes(i)) unique.push(i); });
  while (unique.length < 3 && unique.length < pool.length) {
    const next = (unique[unique.length - 1] + 7) % pool.length;
    if (!unique.includes(next)) unique.push(next);
  }
  return unique.slice(0, 3).map(i => pool[i]);
}

export function getTodayKey(): string {
  return new Date().toISOString().slice(0, 10);
}

export function isDailyCompleted(): boolean {
  try {
    const data = JSON.parse(window.localStorage.getItem('staffml_daily') || '{}');
    return data[getTodayKey()] === true;
  } catch { return false; }
}

export function markDailyCompleted(): void {
  try {
    const data = JSON.parse(window.localStorage.getItem('staffml_daily') || '{}');
    data[getTodayKey()] = true;
    const keys = Object.keys(data).sort().slice(-30);
    const trimmed: Record<string, boolean> = {};
    keys.forEach(k => trimmed[k] = data[k]);
    window.localStorage.setItem('staffml_daily', JSON.stringify(trimmed));
  } catch {}
}
