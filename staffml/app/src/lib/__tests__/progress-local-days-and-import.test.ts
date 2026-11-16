import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

// A zone far from UTC so local-day vs UTC-day bugs are observable.
process.env.TZ = 'America/Los_Angeles';

import { localDateKey, previousLocalDateKey } from '../dates';
import { getDailyQuestions, getTodayKey } from '../daily';
import { getStreakData, recordActivity, importProgress, exportProgress, getAttempts } from '../progress';

beforeEach(() => window.localStorage.clear());
afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe('local calendar days (not UTC)', () => {
  // 2026-03-10 21:30 in Los Angeles == 2026-03-11 04:30 UTC (PDT, UTC-7).
  const lateEvening = new Date(2026, 2, 10, 21, 30);

  test('localDateKey uses the local day', () => {
    expect(localDateKey(lateEvening)).toBe('2026-03-10');
    expect(lateEvening.toISOString().slice(0, 10)).toBe('2026-03-11'); // the old (wrong) answer
    expect(previousLocalDateKey(lateEvening)).toBe('2026-03-09');
  });

  test('previousLocalDateKey crosses month, year and DST boundaries', () => {
    expect(previousLocalDateKey(new Date(2026, 2, 1, 12))).toBe('2026-02-28');
    expect(previousLocalDateKey(new Date(2026, 0, 1, 0, 5))).toBe('2025-12-31');
    expect(previousLocalDateKey(new Date(2026, 2, 8, 12))).toBe('2026-03-07'); // spring-forward day
  });

  test('daily key follows the local day', () => {
    vi.useFakeTimers();
    vi.setSystemTime(lateEvening);
    expect(getTodayKey()).toBe('2026-03-10');
    expect(getDailyQuestions()).toEqual(getDailyQuestions());
  });

  test('a streak counts consecutive local days, even across the UTC midnight', () => {
    vi.useFakeTimers();
    // Day 1 at 21:30 local (already "tomorrow" in UTC).
    vi.setSystemTime(new Date(2026, 2, 10, 21, 30));
    expect(recordActivity().streak.currentStreak).toBe(1);
    // Day 2 at 08:00 local. Under UTC keys both visits fall on 2026-03-11 and
    // the streak never advances; with local days it must become 2.
    vi.setSystemTime(new Date(2026, 2, 11, 8, 0));
    const res = recordActivity();
    expect(res.streak.currentStreak).toBe(2);
    expect(getStreakData().activeDates).toEqual(['2026-03-10', '2026-03-11']);
  });
});

describe('importProgress safety', () => {
  const attempt = { questionId: 'q1', competencyArea: 'memory', track: 'cloud', level: 'L3', selfScore: 3, timestamp: 1 };

  test('rejects an attempts array containing any malformed record', () => {
    for (const bad of [null, 42, 'x', {}, { questionId: 'q2' }, { questionId: 'q2', selfScore: '3' }]) {
      window.localStorage.clear();
      const json = JSON.stringify({ version: 1, attempts: [attempt, bad] });
      expect(importProgress(json)).toBe(false);
      expect(getAttempts()).toEqual([]); // nothing was written
    }
  });

  test('accepts a fully valid export (round trip)', () => {
    window.localStorage.setItem('staffml_progress', JSON.stringify([attempt]));
    const json = exportProgress();
    window.localStorage.clear();
    expect(importProgress(json)).toBe(true);
    expect(getAttempts()).toEqual([attempt]);
  });

  test('rolls back and returns false when a write fails part-way', () => {
    window.localStorage.setItem('staffml_progress', JSON.stringify([{ ...attempt, questionId: 'old' }]));
    window.localStorage.setItem('staffml_sr', JSON.stringify({ old: true }));

    const realSet = window.localStorage.setItem.bind(window.localStorage);
    let failing = false;
    vi.spyOn(window.localStorage, 'setItem').mockImplementation((k: string, v: string) => {
      // Simulate a quota error on the second key written by the import.
      if (k === 'staffml_sr' && !failing) { failing = true; throw new DOMException('quota', 'QuotaExceededError'); }
      realSet(k, v);
    });
    vi.spyOn(console, 'warn').mockImplementation(() => {});

    const json = JSON.stringify({ version: 1, attempts: [attempt], sr: { new: true } });
    expect(importProgress(json)).toBe(false);
    // The pre-import state is intact, not a half-imported mix.
    expect(getAttempts().map(a => a.questionId)).toEqual(['old']);
    expect(window.localStorage.getItem('staffml_sr')).toBe(JSON.stringify({ old: true }));
  });
});
