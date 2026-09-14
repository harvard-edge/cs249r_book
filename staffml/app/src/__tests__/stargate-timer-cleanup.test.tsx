/**
 * StarGate timer-cleanup regression.
 *
 * Both `handleStar` (Star on GitHub) and `handleAlreadyStarred` (I already
 * starred) schedule `setTimeout(onVerified, 1500)` to give the user a
 * moment to see the "Thank you" confirmation before the parent dismisses
 * the gate. If the gate unmounts before 1500ms (parent route change,
 * programmatic close), the timer should be cancelled — otherwise it
 * fires `onVerified` on an unmounted parent later. Today the practice
 * page treats that as a no-op, but a future caller may not.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, fireEvent } from '@testing-library/react';

const markVerified = vi.fn();

vi.mock('@/lib/star-gate', () => ({
  fetchStarCount: () => new Promise(() => {}),    // never resolves
  getStarUrl: () => 'https://github.com/harvard-edge/cs249r_book',
  markVerified: (...args: unknown[]) => markVerified(...args),
}));

import StarGate from '@/components/StarGate';

describe('StarGate verify-delay timer cleanup', () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    markVerified.mockClear();
    // window.open in jsdom can warn / navigate; stub it out for the
    // "Star on GitHub" path.
    vi.spyOn(window, 'open').mockImplementation(() => null);
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('cancels the pending onVerified call if the gate unmounts after "Star on GitHub"', () => {
    const onVerified = vi.fn();
    const { getByRole, unmount } = render(<StarGate onVerified={onVerified} />);

    fireEvent.click(getByRole('button', { name: /Star on GitHub/i }));
    expect(markVerified).toHaveBeenCalledWith('starred');
    expect(onVerified).not.toHaveBeenCalled();    // delayed by 1.5s

    unmount();
    vi.advanceTimersByTime(5000);                 // way past the 1.5s delay
    expect(onVerified).not.toHaveBeenCalled();    // cleanup ran, timer cancelled
  });

  it('cancels the pending onVerified call if the gate unmounts after "I already starred"', () => {
    const onVerified = vi.fn();
    const { getByRole, unmount } = render(<StarGate onVerified={onVerified} />);

    fireEvent.click(getByRole('button', { name: /I already starred/i }));
    expect(markVerified).toHaveBeenCalledWith('honor');
    expect(onVerified).not.toHaveBeenCalled();

    unmount();
    vi.advanceTimersByTime(5000);
    expect(onVerified).not.toHaveBeenCalled();
  });

  it('still fires onVerified after 1.5s if the gate stays mounted', () => {
    const onVerified = vi.fn();
    const { getByRole } = render(<StarGate onVerified={onVerified} />);

    fireEvent.click(getByRole('button', { name: /I already starred/i }));
    expect(onVerified).not.toHaveBeenCalled();

    vi.advanceTimersByTime(1500);
    expect(onVerified).toHaveBeenCalledTimes(1);
  });
});
