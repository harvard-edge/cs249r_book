/**
 * Tests for useVisibilityPoll — the Page Visibility-aware setInterval
 * wrapper extracted from Nav's SR-due-count poll.
 *
 * Pins the four behaviors that make this hook useful versus a plain
 * setInterval:
 *   1. Fires immediately on mount (so the UI isn't blank for one interval).
 *   2. Fires repeatedly at the requested cadence while visible.
 *   3. Stops firing when the tab goes hidden.
 *   4. Catches up with an immediate call when the tab comes back, then
 *      resumes the cadence.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useVisibilityPoll } from '@/lib/hooks/useVisibilityPoll';

function setVisibility(state: 'visible' | 'hidden') {
  Object.defineProperty(document, 'visibilityState', {
    configurable: true,
    get: () => state,
  });
  document.dispatchEvent(new Event('visibilitychange'));
}

describe('useVisibilityPoll', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    setVisibility('visible');
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('fires immediately on mount when visible', () => {
    const cb = vi.fn();
    renderHook(() => useVisibilityPoll(cb, 1000));
    expect(cb).toHaveBeenCalledTimes(1);
  });

  it('fires repeatedly at the requested cadence while visible', () => {
    const cb = vi.fn();
    renderHook(() => useVisibilityPoll(cb, 1000));
    expect(cb).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(3000);
    expect(cb).toHaveBeenCalledTimes(4);   // mount + 3 ticks
  });

  it('stops firing when the tab is hidden', () => {
    const cb = vi.fn();
    renderHook(() => useVisibilityPoll(cb, 1000));
    expect(cb).toHaveBeenCalledTimes(1);

    setVisibility('hidden');
    vi.advanceTimersByTime(10_000);
    expect(cb).toHaveBeenCalledTimes(1);   // nothing fired while hidden
  });

  it('catches up immediately on resume and then resumes the cadence', () => {
    const cb = vi.fn();
    renderHook(() => useVisibilityPoll(cb, 1000));
    expect(cb).toHaveBeenCalledTimes(1);

    setVisibility('hidden');
    vi.advanceTimersByTime(5000);
    expect(cb).toHaveBeenCalledTimes(1);

    setVisibility('visible');
    expect(cb).toHaveBeenCalledTimes(2);   // immediate catch-up
    vi.advanceTimersByTime(2000);
    expect(cb).toHaveBeenCalledTimes(4);   // + 2 ticks
  });

  it('does not start the interval if mounted while already hidden', () => {
    setVisibility('hidden');
    const cb = vi.fn();
    renderHook(() => useVisibilityPoll(cb, 1000));
    vi.advanceTimersByTime(10_000);
    expect(cb).not.toHaveBeenCalled();

    setVisibility('visible');
    expect(cb).toHaveBeenCalledTimes(1);
  });

  it('isolates throws from the callback so mount setup completes and polling continues', () => {
    // Reviewer caught a subtle bug in an earlier iteration: if the
    // callback throws on the mount-time synchronous tick, the throw
    // propagates out of the effect body BEFORE React receives the
    // cleanup function. Any setInterval/listeners armed before the
    // throw leak forever. The hook now wraps every tick in try/catch
    // internally so the worst case is per-tick console errors, not a
    // leak. We also assert that the visibilitychange listener was
    // attached (it sits AFTER start() in the effect body, so a throw
    // there would skip it).
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
    let throwOnce = true;
    const cb = vi.fn(() => {
      if (throwOnce) { throwOnce = false; throw new Error('boom'); }
    });

    const { unmount } = renderHook(() => useVisibilityPoll(cb, 1000));
    expect(cb).toHaveBeenCalledTimes(1);
    expect(consoleError).toHaveBeenCalled();

    // Listener wired up despite the throw — toggling visibility works.
    setVisibility('hidden');
    vi.advanceTimersByTime(3000);
    expect(cb).toHaveBeenCalledTimes(1);   // paused
    setVisibility('visible');
    expect(cb).toHaveBeenCalledTimes(2);   // resume tick (no throw this time)

    vi.advanceTimersByTime(2000);
    expect(cb).toHaveBeenCalledTimes(4);   // interval kept ticking

    unmount();
    consoleError.mockRestore();
  });

  it('cleans up the interval and listener on unmount', () => {
    const cb = vi.fn();
    const { unmount } = renderHook(() => useVisibilityPoll(cb, 1000));
    expect(cb).toHaveBeenCalledTimes(1);
    unmount();
    vi.advanceTimersByTime(10_000);
    expect(cb).toHaveBeenCalledTimes(1);   // nothing fired post-unmount
  });

  it('always calls the latest callback closure, not the one captured on first render', () => {
    // Sanity-check the ref pattern: changing the callback between renders
    // must take effect without tearing down and re-arming the interval.
    let value = 0;
    const seen: number[] = [];
    const { rerender } = renderHook(
      ({ v }: { v: number }) => useVisibilityPoll(() => seen.push(v), 1000),
      { initialProps: { v: 0 } },
    );
    expect(seen).toEqual([0]);

    value = 1;
    rerender({ v: 1 });
    vi.advanceTimersByTime(1000);
    expect(seen).toEqual([0, 1]);
    void value;
  });
});
