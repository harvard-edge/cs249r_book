// LocalStorage-based progress tracking for the Heat Map + Spaced Repetition

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

// SM-2 inspired spaced repetition card data
export interface SRCard {
  questionId: string;
  easeFactor: number; // starts at 2.5, min 1.3
  interval: number;   // days until next review
  repetitions: number;
  nextReview: number;  // timestamp
  lastScore: number;
}

const STORAGE_KEY = 'staffml_progress';
const GAUNTLET_KEY = 'staffml_gauntlets';
const SR_KEY = 'staffml_sr';

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

// ─── Attempts ────────────────────────────────────

export function getAttempts(): AttemptRecord[] {
  return getStorage<AttemptRecord[]>(STORAGE_KEY, []);
}

const MAX_ATTEMPTS = 500;
const MAX_GAUNTLETS = 50;

export function saveAttempt(attempt: AttemptRecord): void {
  const attempts = getAttempts();
  attempts.push(attempt);
  if (attempts.length > MAX_ATTEMPTS) {
    attempts.splice(0, attempts.length - MAX_ATTEMPTS);
  }
  setStorage(STORAGE_KEY, attempts);
}

// ─── Gauntlet Results ────────────────────────────

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

// ─── Spaced Repetition (SM-2) ────────────────────

function getSRCards(): Record<string, SRCard> {
  return getStorage<Record<string, SRCard>>(SR_KEY, {});
}

function saveSRCards(cards: Record<string, SRCard>): void {
  setStorage(SR_KEY, cards);
}

/**
 * Update SM-2 card after a review.
 * quality: 0=skip, 1=wrong, 2=partial, 3=nailed it
 * Maps to SM-2 quality: 0→0, 1→1, 2→3, 3→5
 */
export function updateSRCard(questionId: string, quality: number): void {
  const cards = getSRCards();
  const card = cards[questionId] || {
    questionId,
    easeFactor: 2.5,
    interval: 1,
    repetitions: 0,
    nextReview: 0,
    lastScore: 0,
  };

  // Map 0-3 self-score to SM-2 quality (0-5)
  const q = [0, 1, 3, 5][quality] ?? 0;

  if (q < 3) {
    // Failed: reset
    card.repetitions = 0;
    card.interval = 1;
  } else {
    if (card.repetitions === 0) {
      card.interval = 1;
    } else if (card.repetitions === 1) {
      card.interval = 3;
    } else {
      card.interval = Math.round(card.interval * card.easeFactor);
    }
    card.repetitions++;
  }

  // Update ease factor: EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
  card.easeFactor = Math.max(
    1.3,
    card.easeFactor + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
  );

  card.lastScore = quality;
  card.nextReview = Date.now() + card.interval * 24 * 60 * 60 * 1000;

  cards[questionId] = card;
  saveSRCards(cards);
}

/**
 * Get questions due for review, sorted by most overdue first.
 */
export function getDueQuestionIds(): string[] {
  const cards = getSRCards();
  const now = Date.now();
  return Object.values(cards)
    .filter(c => c.nextReview <= now)
    .sort((a, b) => a.nextReview - b.nextReview)
    .map(c => c.questionId);
}

/**
 * Get count of questions due for review.
 */
export function getDueCount(): number {
  const cards = getSRCards();
  const now = Date.now();
  return Object.values(cards).filter(c => c.nextReview <= now).length;
}

/**
 * Get SR stats for display.
 */
export function getSRStats(): {
  totalCards: number;
  dueNow: number;
  mastered: number; // interval > 7 days
  learning: number; // interval <= 7 days
} {
  const cards = getSRCards();
  const now = Date.now();
  const all = Object.values(cards);
  return {
    totalCards: all.length,
    dueNow: all.filter(c => c.nextReview <= now).length,
    mastered: all.filter(c => c.interval > 7).length,
    learning: all.filter(c => c.interval <= 7).length,
  };
}

// ─── Heat Map ────────────────────────────────────

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

// ─── Streaks ─────────────────────────────────────

const STREAK_KEY = 'staffml_streak';

interface StreakData {
  currentStreak: number;
  longestStreak: number;
  lastActiveDate: string; // YYYY-MM-DD
  activeDates: string[];  // last 90 days of activity
}

function getToday(): string {
  return new Date().toISOString().split('T')[0];
}

function getYesterday(): string {
  const d = new Date();
  d.setDate(d.getDate() - 1);
  return d.toISOString().split('T')[0];
}

export function getStreakData(): StreakData {
  return getStorage<StreakData>(STREAK_KEY, {
    currentStreak: 0,
    longestStreak: 0,
    lastActiveDate: '',
    activeDates: [],
  });
}

export function recordActivity(): StreakData {
  const data = getStreakData();
  const today = getToday();
  const yesterday = getYesterday();

  // Already recorded today
  if (data.lastActiveDate === today) return data;

  // Extend streak if active yesterday or starting fresh today
  if (data.lastActiveDate === yesterday) {
    data.currentStreak++;
  } else if (data.lastActiveDate !== today) {
    // Streak broken — reset to 1
    data.currentStreak = 1;
  }

  data.longestStreak = Math.max(data.longestStreak, data.currentStreak);
  data.lastActiveDate = today;

  // Track active dates (cap at 90 days for calendar display)
  if (!data.activeDates.includes(today)) {
    data.activeDates.push(today);
    if (data.activeDates.length > 90) {
      data.activeDates.shift();
    }
  }

  setStorage(STREAK_KEY, data);
  return data;
}

export function getStreakMilestone(streak: number): string | null {
  if (streak >= 100) return 'Centurion';
  if (streak >= 30) return 'Monthly Master';
  if (streak >= 14) return 'Two-Week Warrior';
  if (streak >= 7) return 'Weekly Streak';
  if (streak >= 3) return 'Getting Started';
  return null;
}

// ─── Clear All ───────────────────────────────────

export function clearProgress(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
    window.localStorage.removeItem(GAUNTLET_KEY);
    window.localStorage.removeItem(SR_KEY);
    window.localStorage.removeItem(STREAK_KEY);
  } catch {}
}
