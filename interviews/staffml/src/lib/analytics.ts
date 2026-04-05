// Lightweight analytics for StaffML
// All data is anonymous — no PII, no cookies, no persistent user IDs
// Events are stored locally AND batched to a remote endpoint for aggregate analysis

const ANALYTICS_KEY = 'staffml_analytics';
const SESSION_KEY = 'staffml_session_id';
const PENDING_KEY = 'staffml_analytics_pending';

// Remote endpoint — set via environment variable at build time
// Falls back to local-only if not configured
const ANALYTICS_ENDPOINT = process.env.NEXT_PUBLIC_ANALYTICS_URL || '';

// ─── Event Types ────────────────────────────────

export type AnalyticsEvent =
  // Session
  | { type: 'session_start'; isReturning: boolean; screenWidth: number }
  // Question lifecycle
  | { type: 'question_scored'; questionId: string; topic: string; zone: string; level: string; track: string; score: number }
  | { type: 'question_skipped'; topic: string; level: string }
  | { type: 'question_reported'; questionId: string; category?: string }
  | { type: 'question_thumbs'; questionId: string; topic: string; level: string; value: 'up' | 'down' }
  | { type: 'question_difficulty_feedback'; questionId: string; topic: string; level: string; perceived: 'too_easy' | 'about_right' | 'too_hard' }
  | { type: 'question_contributed'; topic: string; track: string }
  | { type: 'answer_response_time'; questionId: string; topic: string; level: string; seconds: number; napkinGrade?: string; hadUserAnswer: boolean }
  | { type: 'answer_revealed'; topic: string; zone: string; hadUserAnswer: boolean }
  // Gauntlet
  | { type: 'gauntlet_started'; track: string; level: string; questionCount: number }
  | { type: 'gauntlet_completed'; track: string; level: string; pct: number; questionCount: number }
  | { type: 'gauntlet_abandoned'; track: string; level: string; questionsAnswered: number }
  // Plans
  | { type: 'plan_started'; planId: string }
  | { type: 'plan_completed'; planId: string }
  | { type: 'daily_completed' }
  // Navigation
  | { type: 'page_view'; page: string }
  | { type: 'search_query'; query: string; topicResults: number; questionResults: number }
  // Star gate
  | { type: 'star_gate_shown' }
  | { type: 'star_gate_verified' }
  // Feedback
  | { type: 'improvement_suggested'; questionId: string }
  | { type: 'progress_exported' }
  | { type: 'progress_imported' };

interface StoredEvent {
  event: AnalyticsEvent;
  timestamp: number;
  sessionId: string;
}

// ─── Session Management ─────────────────────────
// Session ID is per-tab, non-persistent — no tracking across visits

function getSessionId(): string {
  try {
    let id = window.sessionStorage.getItem(SESSION_KEY);
    if (!id) {
      id = crypto.randomUUID();
      window.sessionStorage.setItem(SESSION_KEY, id);
    }
    return id;
  } catch {
    return 'unknown';
  }
}

// ─── Local Storage ──────────────────────────────

const MAX_LOCAL_EVENTS = 1000;

function getLocalEvents(): StoredEvent[] {
  try {
    const raw = window.localStorage.getItem(ANALYTICS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveLocalEvents(events: StoredEvent[]): void {
  try {
    const trimmed = events.slice(-MAX_LOCAL_EVENTS);
    window.localStorage.setItem(ANALYTICS_KEY, JSON.stringify(trimmed));
  } catch {}
}

// ─── Remote Batching ────────────────────────────
// Events are queued and sent in batches every 30 seconds
// If send fails, events stay in the pending queue for next attempt

let flushTimer: ReturnType<typeof setTimeout> | null = null;
const FLUSH_INTERVAL = 30_000; // 30 seconds
const MAX_BATCH_SIZE = 50;

function getPendingEvents(): StoredEvent[] {
  try {
    const raw = window.localStorage.getItem(PENDING_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function savePendingEvents(events: StoredEvent[]): void {
  try {
    // Cap pending queue to prevent unbounded growth if endpoint is down
    const trimmed = events.slice(-200);
    window.localStorage.setItem(PENDING_KEY, JSON.stringify(trimmed));
  } catch {}
}

async function flushToRemote(): Promise<void> {
  if (!ANALYTICS_ENDPOINT) return;

  const pending = getPendingEvents();
  if (pending.length === 0) return;

  const batch = pending.slice(0, MAX_BATCH_SIZE);

  try {
    const response = await fetch(ANALYTICS_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        events: batch.map(({ event, timestamp, sessionId }) => ({
          ...event,
          _ts: timestamp,
          _sid: sessionId,
        })),
      }),
      // Don't block the UI — use keepalive for background delivery
      keepalive: true,
    });

    if (response.ok) {
      // Remove sent events from pending queue
      const remaining = pending.slice(batch.length);
      savePendingEvents(remaining);
    }
    // If not ok, events stay in pending for retry
  } catch {
    // Network error — events stay in pending queue
  }
}

function scheduleFlush(): void {
  if (flushTimer) return;
  flushTimer = setTimeout(() => {
    flushTimer = null;
    flushToRemote();
  }, FLUSH_INTERVAL);
}

// ─── Public API ─────────────────────────────────

/** Track an analytics event. Stored locally and queued for remote delivery. */
export function track(event: AnalyticsEvent): void {
  const stored: StoredEvent = {
    event,
    timestamp: Date.now(),
    sessionId: getSessionId(),
  };

  // Always store locally (for dashboard)
  const events = getLocalEvents();
  events.push(stored);
  saveLocalEvents(events);

  // Queue for remote delivery if endpoint is configured
  if (ANALYTICS_ENDPOINT) {
    const pending = getPendingEvents();
    pending.push(stored);
    savePendingEvents(pending);
    scheduleFlush();
  }
}

/** Manually flush pending events to remote (e.g., on page unload) */
export function flushAnalytics(): void {
  flushToRemote();
}

/** Get all locally stored analytics events */
export function getAnalyticsEvents(): StoredEvent[] {
  return getLocalEvents();
}

/** Get count of events waiting to be sent */
export function getPendingCount(): number {
  return getPendingEvents().length;
}

/** Clear all local analytics data */
export function clearAnalytics(): void {
  try {
    window.localStorage.removeItem(ANALYTICS_KEY);
    window.localStorage.removeItem(PENDING_KEY);
  } catch {}
}

// ─── Aggregate Stats ────────────────────────────

export interface AnalyticsSummary {
  totalEvents: number;
  uniqueSessions: number;
  questionsScored: number;
  questionsReported: number;
  gauntletsCompleted: number;
  gauntletsAbandoned: number;
  dailyCompletions: number;
  improvementsSuggested: number;
  thumbsUp: number;
  thumbsDown: number;
  difficultyDistribution: Record<'too_easy' | 'about_right' | 'too_hard', number>;
  scoresByZone: Record<string, { total: number; count: number; avg: number }>;
  scoresByTopic: Record<string, { total: number; count: number; avg: number }>;
  scoresByLevel: Record<string, { total: number; count: number; avg: number }>;
  eventsByDay: Record<string, number>;
  topSkippedTopics: { topic: string; count: number }[];
}

export function computeSummary(events?: StoredEvent[]): AnalyticsSummary {
  const all = events || getLocalEvents();

  const sessions = new Set<string>();
  let questionsScored = 0;
  let questionsReported = 0;
  let gauntletsCompleted = 0;
  let gauntletsAbandoned = 0;
  let dailyCompletions = 0;
  let improvementsSuggested = 0;
  let thumbsUp = 0;
  let thumbsDown = 0;
  const difficultyDistribution: Record<'too_easy' | 'about_right' | 'too_hard', number> = {
    too_easy: 0, about_right: 0, too_hard: 0,
  };
  // Dedup feedback: only count latest per (questionId, sessionId)
  const latestThumbs = new Map<string, 'up' | 'down'>();
  const latestDifficulty = new Map<string, 'too_easy' | 'about_right' | 'too_hard'>();

  const scoresByZone: Record<string, { total: number; count: number; avg: number }> = {};
  const scoresByTopic: Record<string, { total: number; count: number; avg: number }> = {};
  const scoresByLevel: Record<string, { total: number; count: number; avg: number }> = {};
  const eventsByDay: Record<string, number> = {};
  const skipsByTopic: Record<string, number> = {};

  for (const { event, timestamp, sessionId } of all) {
    sessions.add(sessionId);

    const day = new Date(timestamp).toISOString().split('T')[0];
    eventsByDay[day] = (eventsByDay[day] || 0) + 1;

    switch (event.type) {
      case 'question_scored': {
        questionsScored++;
        if (!scoresByZone[event.zone]) scoresByZone[event.zone] = { total: 0, count: 0, avg: 0 };
        scoresByZone[event.zone].total += event.score;
        scoresByZone[event.zone].count++;
        if (!scoresByTopic[event.topic]) scoresByTopic[event.topic] = { total: 0, count: 0, avg: 0 };
        scoresByTopic[event.topic].total += event.score;
        scoresByTopic[event.topic].count++;
        if (!scoresByLevel[event.level]) scoresByLevel[event.level] = { total: 0, count: 0, avg: 0 };
        scoresByLevel[event.level].total += event.score;
        scoresByLevel[event.level].count++;
        break;
      }
      case 'question_skipped':
        skipsByTopic[event.topic] = (skipsByTopic[event.topic] || 0) + 1;
        break;
      case 'question_reported':
        questionsReported++;
        break;
      case 'gauntlet_completed':
        gauntletsCompleted++;
        break;
      case 'gauntlet_abandoned':
        gauntletsAbandoned++;
        break;
      case 'daily_completed':
        dailyCompletions++;
        break;
      case 'improvement_suggested':
        improvementsSuggested++;
        break;
      case 'question_thumbs':
        latestThumbs.set(`${event.questionId}:${sessionId}`, event.value);
        break;
      case 'question_difficulty_feedback':
        latestDifficulty.set(`${event.questionId}:${sessionId}`, event.perceived);
        break;
    }
  }

  // Aggregate deduplicated feedback (last-write-wins per question+session)
  Array.from(latestThumbs.values()).forEach(value => {
    if (value === 'up') thumbsUp++; else thumbsDown++;
  });
  Array.from(latestDifficulty.values()).forEach(perceived => {
    difficultyDistribution[perceived]++;
  });

  // Compute averages
  for (const v of Object.values(scoresByZone)) v.avg = v.count > 0 ? v.total / v.count : 0;
  for (const v of Object.values(scoresByTopic)) v.avg = v.count > 0 ? v.total / v.count : 0;
  for (const v of Object.values(scoresByLevel)) v.avg = v.count > 0 ? v.total / v.count : 0;

  // Top skipped topics
  const topSkippedTopics = Object.entries(skipsByTopic)
    .map(([topic, count]) => ({ topic, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);

  return {
    totalEvents: all.length,
    uniqueSessions: sessions.size,
    questionsScored,
    questionsReported,
    gauntletsCompleted,
    gauntletsAbandoned,
    dailyCompletions,
    improvementsSuggested,
    thumbsUp,
    thumbsDown,
    difficultyDistribution,
    scoresByZone,
    scoresByTopic,
    scoresByLevel,
    eventsByDay,
    topSkippedTopics,
  };
}
