/**
 * Calendar-day keys in the user's LOCAL timezone ("YYYY-MM-DD").
 *
 * `Date#toISOString()` is always UTC, so using it for "today" makes the day
 * roll over at UTC midnight rather than the learner's midnight. For streaks,
 * the daily set, and "completed today" counts that is visibly wrong (e.g. a
 * learner in UTC-8 starts a new "day" at 4 pm). Use these helpers for any
 * user-facing day boundary.
 */
export function localDateKey(date: Date = new Date()): string {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

/** Local-day key for the day before `date` (calendar arithmetic, DST-safe). */
export function previousLocalDateKey(date: Date = new Date()): string {
  const d = new Date(date.getFullYear(), date.getMonth(), date.getDate() - 1);
  return localDateKey(d);
}
