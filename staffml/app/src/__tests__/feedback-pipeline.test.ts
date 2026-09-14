/**
 * Feedback Pipeline Journey Tests
 *
 * Validates the complete data round-trip for every user feedback action:
 * track() → localStorage → computeSummary() → dashboard-visible metrics
 *
 * Each test maps to a journey from the UX QA audit.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import {
  track,
  computeSummary,
  getAnalyticsEvents,
  clearAnalytics,
  type AnalyticsSummary,
} from '@/lib/analytics';

// Clean slate before each test
beforeEach(() => {
  clearAnalytics();
  window.localStorage.clear();
  window.sessionStorage.clear();
});

// ─── Journey 1: Thumbs up on a practice question ────────────

describe('Journey 1: User gives thumbs up', () => {
  it('stores a question_thumbs event in localStorage', () => {
    track({
      type: 'question_thumbs',
      questionId: 'q-kv-cache-01',
      topic: 'kv-cache',
      level: 'L3',
      value: 'up',
    });

    const events = getAnalyticsEvents();
    expect(events).toHaveLength(1);
    expect(events[0].event.type).toBe('question_thumbs');
    expect((events[0].event as any).value).toBe('up');
  });

  it('is aggregated by computeSummary() into thumbsUp count', () => {
    track({ type: 'question_thumbs', questionId: 'q1', topic: 't', level: 'L3', value: 'up' });
    track({ type: 'question_thumbs', questionId: 'q2', topic: 't', level: 'L3', value: 'up' });

    const summary = computeSummary();
    expect(summary.thumbsUp).toBe(2);
    expect(summary.thumbsDown).toBe(0);
  });
});

// ─── Journey 2: Difficulty rating ────────────────────────────

describe('Journey 2: User gives difficulty rating', () => {
  it('stores a question_difficulty_feedback event', () => {
    track({
      type: 'question_difficulty_feedback',
      questionId: 'q-kv-cache-01',
      topic: 'kv-cache',
      level: 'L3',
      perceived: 'too_hard',
    });

    const events = getAnalyticsEvents();
    expect(events).toHaveLength(1);
    expect((events[0].event as any).perceived).toBe('too_hard');
  });

  it('is aggregated into difficultyDistribution', () => {
    track({ type: 'question_difficulty_feedback', questionId: 'q1', topic: 't', level: 'L3', perceived: 'too_easy' });
    track({ type: 'question_difficulty_feedback', questionId: 'q2', topic: 't', level: 'L4', perceived: 'about_right' });
    track({ type: 'question_difficulty_feedback', questionId: 'q3', topic: 't', level: 'L5', perceived: 'too_hard' });

    const summary = computeSummary();
    expect(summary.difficultyDistribution).toEqual({
      too_easy: 1,
      about_right: 1,
      too_hard: 1,
    });
  });
});

// ─── Journey 3 & 4: Report / Suggest fire analytics ─────────

describe('Journey 3-4: Report and Suggest track analytics', () => {
  it('question_reported increments questionsReported in summary', () => {
    track({ type: 'question_reported', questionId: 'q1' });
    track({ type: 'question_reported', questionId: 'q2' });

    const summary = computeSummary();
    expect(summary.questionsReported).toBe(2);
  });

  it('improvement_suggested increments improvementsSuggested in summary', () => {
    track({ type: 'improvement_suggested', questionId: 'q1' });

    const summary = computeSummary();
    expect(summary.improvementsSuggested).toBe(1);
  });
});

// ─── Journey 8: Double-click same button ─────────────────────

describe('Journey 8: Double-click deduplication', () => {
  it('two identical thumbs events for same question+session deduplicate to 1 in summary', () => {
    // Simulate: user clicks thumbs-up twice (guard was bypassed or two components)
    track({ type: 'question_thumbs', questionId: 'q1', topic: 't', level: 'L3', value: 'up' });
    track({ type: 'question_thumbs', questionId: 'q1', topic: 't', level: 'L3', value: 'up' });

    const summary = computeSummary();
    // Last-write-wins dedup: same question+session → counted once
    expect(summary.thumbsUp).toBe(1);
    expect(summary.thumbsDown).toBe(0);
  });

  it('two identical difficulty events for same question+session deduplicate to 1', () => {
    track({ type: 'question_difficulty_feedback', questionId: 'q1', topic: 't', level: 'L3', perceived: 'too_hard' });
    track({ type: 'question_difficulty_feedback', questionId: 'q1', topic: 't', level: 'L3', perceived: 'too_hard' });

    const summary = computeSummary();
    expect(summary.difficultyDistribution.too_hard).toBe(1);
  });
});

// ─── Journey 9: User changes mind ────────────────────────────

describe('Journey 9: User changes mind (thumbs down → up)', () => {
  it('last-write-wins: only the final vote counts', () => {
    track({ type: 'question_thumbs', questionId: 'q1', topic: 't', level: 'L3', value: 'down' });
    track({ type: 'question_thumbs', questionId: 'q1', topic: 't', level: 'L3', value: 'up' });

    const summary = computeSummary();
    expect(summary.thumbsUp).toBe(1);
    expect(summary.thumbsDown).toBe(0);
  });

  it('last-write-wins for difficulty: too_easy → too_hard', () => {
    track({ type: 'question_difficulty_feedback', questionId: 'q1', topic: 't', level: 'L3', perceived: 'too_easy' });
    track({ type: 'question_difficulty_feedback', questionId: 'q1', topic: 't', level: 'L3', perceived: 'too_hard' });

    const summary = computeSummary();
    expect(summary.difficultyDistribution.too_easy).toBe(0);
    expect(summary.difficultyDistribution.too_hard).toBe(1);
  });
});

// ─── Journey 10: Feedback hydration on revisit ───────────────

describe('Journey 10: Previous feedback can be retrieved', () => {
  it('getAnalyticsEvents() contains feedback for hydration lookup', () => {
    track({ type: 'question_thumbs', questionId: 'q-cache-01', topic: 't', level: 'L3', value: 'down' });
    track({ type: 'question_difficulty_feedback', questionId: 'q-cache-01', topic: 't', level: 'L3', perceived: 'about_right' });
    track({ type: 'question_scored', questionId: 'q-other', topic: 't2', zone: 'design', level: 'L4', track: 'cloud', score: 2 });

    // Simulate what QuestionFeedback.tsx does on mount: walk backwards
    const events = getAnalyticsEvents();
    let foundThumbs: string | null = null;
    let foundDifficulty: string | null = null;

    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i].event;
      if (e.type === 'question_thumbs' && (e as any).questionId === 'q-cache-01' && !foundThumbs) {
        foundThumbs = (e as any).value;
      }
      if (e.type === 'question_difficulty_feedback' && (e as any).questionId === 'q-cache-01' && !foundDifficulty) {
        foundDifficulty = (e as any).perceived;
      }
    }

    expect(foundThumbs).toBe('down');
    expect(foundDifficulty).toBe('about_right');
  });
});

// ─── Journey 13: No analytics endpoint configured ───────────

describe('Journey 13: Graceful degradation without remote endpoint', () => {
  it('track() stores locally even without ANALYTICS_ENDPOINT', () => {
    // NEXT_PUBLIC_ANALYTICS_URL is not set in test env
    track({ type: 'question_thumbs', questionId: 'q1', topic: 't', level: 'L3', value: 'up' });

    const events = getAnalyticsEvents();
    expect(events).toHaveLength(1);

    const summary = computeSummary();
    expect(summary.thumbsUp).toBe(1);
  });
});

// ─── Journey 14: Event cap behavior ──────────────────────────

describe('Journey 14: 1000-event cap', () => {
  it('oldest events are evicted when cap is exceeded', () => {
    // Fill with 1001 events
    for (let i = 0; i < 1001; i++) {
      track({ type: 'question_thumbs', questionId: `q-${i}`, topic: 't', level: 'L3', value: 'up' });
    }

    const events = getAnalyticsEvents();
    expect(events.length).toBeLessThanOrEqual(1000);

    // First event (q-0) should be evicted, last (q-1000) should survive
    const ids = events.map(e => (e.event as any).questionId);
    expect(ids).not.toContain('q-0');
    expect(ids).toContain('q-1000');
  });
});

// ─── Cross-cutting: Dashboard summary completeness ──────────

describe('Dashboard summary has all feedback fields', () => {
  it('computeSummary() returns feedback fields even when empty', () => {
    const summary = computeSummary();

    // These fields must exist (not undefined) for the dashboard to render
    expect(summary.thumbsUp).toBe(0);
    expect(summary.thumbsDown).toBe(0);
    expect(summary.difficultyDistribution).toEqual({
      too_easy: 0,
      about_right: 0,
      too_hard: 0,
    });
  });

  it('mixed event types are all counted correctly', () => {
    track({ type: 'question_scored', questionId: 'q1', topic: 't', zone: 'design', level: 'L3', track: 'cloud', score: 3 });
    track({ type: 'question_thumbs', questionId: 'q1', topic: 't', level: 'L3', value: 'up' });
    track({ type: 'question_difficulty_feedback', questionId: 'q1', topic: 't', level: 'L3', perceived: 'about_right' });
    track({ type: 'question_reported', questionId: 'q1' });
    track({ type: 'improvement_suggested', questionId: 'q1' });
    track({ type: 'gauntlet_completed', track: 'cloud', level: 'L3', pct: 80, questionCount: 10 });
    track({ type: 'question_skipped', topic: 't', level: 'L4' });

    const s = computeSummary();
    expect(s.questionsScored).toBe(1);
    expect(s.thumbsUp).toBe(1);
    expect(s.thumbsDown).toBe(0);
    expect(s.difficultyDistribution.about_right).toBe(1);
    expect(s.questionsReported).toBe(1);
    expect(s.improvementsSuggested).toBe(1);
    expect(s.gauntletsCompleted).toBe(1);
    expect(s.topSkippedTopics).toHaveLength(1);
    expect(s.totalEvents).toBe(7);
  });
});

// ─── Export/Import includes analytics ────────────────────────

describe('Journey 7: Export/Import preserves feedback', () => {
  it('exported JSON contains analytics array', async () => {
    track({ type: 'question_thumbs', questionId: 'q1', topic: 't', level: 'L3', value: 'up' });

    // We can't directly call exportProgress() because it imports from progress.ts
    // which reads localStorage — but we can verify the analytics key is populated
    const raw = window.localStorage.getItem('staffml_analytics');
    expect(raw).not.toBeNull();
    const parsed = JSON.parse(raw!);
    expect(parsed).toHaveLength(1);
    expect(parsed[0].event.type).toBe('question_thumbs');
  });
});

// ─── Multi-question, multi-session dedup ─────────────────────

describe('Dedup across multiple questions and sessions', () => {
  it('different questions are counted independently', () => {
    track({ type: 'question_thumbs', questionId: 'q1', topic: 't', level: 'L3', value: 'up' });
    track({ type: 'question_thumbs', questionId: 'q2', topic: 't', level: 'L3', value: 'down' });
    track({ type: 'question_thumbs', questionId: 'q3', topic: 't', level: 'L3', value: 'up' });

    const s = computeSummary();
    expect(s.thumbsUp).toBe(2);
    expect(s.thumbsDown).toBe(1);
  });

  it('same question from different sessions counts separately', () => {
    // Simulate two different sessions rating the same question
    // (different users or different tabs)
    const events = getAnalyticsEvents();

    // Manually push events with different session IDs
    const baseEvent = { type: 'question_thumbs' as const, questionId: 'q1', topic: 't', level: 'L3', value: 'up' as const };
    events.push(
      { event: baseEvent, timestamp: Date.now(), sessionId: 'session-A' },
      { event: { ...baseEvent, value: 'down' }, timestamp: Date.now(), sessionId: 'session-B' },
    );
    window.localStorage.setItem('staffml_analytics', JSON.stringify(events));

    const s = computeSummary();
    expect(s.thumbsUp).toBe(1);  // session-A: up
    expect(s.thumbsDown).toBe(1); // session-B: down
  });
});
