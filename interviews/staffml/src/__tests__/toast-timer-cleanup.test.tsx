/**
 * Toast auto-dismiss timer cleanup regression.
 *
 * Each toast schedules a 4s setTimeout to auto-dismiss itself. The timer
 * handle used to be discarded, so it was never cleared: if the provider
 * unmounted within 4s the timer still fired setState on a dead tree, and a
 * manual dismiss left the timer running. Timers are now tracked in a ref,
 * cleared on manual dismiss, and all cleared on unmount.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { ToastProvider, useToast } from '@/components/Toast';

function Trigger() {
  const { show } = useToast();
  return (
    <button onClick={() => show({ type: 'info', title: 'hi' })}>fire toast</button>
  );
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('Toast auto-dismiss timer cleanup', () => {
  it('clears the pending auto-dismiss timer when the provider unmounts', () => {
    const clearSpy = vi.spyOn(globalThis, 'clearTimeout');
    const { unmount } = render(
      <ToastProvider>
        <Trigger />
      </ToastProvider>,
    );

    act(() => {
      fireEvent.click(screen.getByText('fire toast'));
    });

    const before = clearSpy.mock.calls.length;
    unmount();
    // Unmount must clear the still-pending 4s timer.
    expect(clearSpy.mock.calls.length).toBeGreaterThan(before);

    // Flushing timers after unmount must not throw (no setState on dead tree).
    expect(() => act(() => vi.runOnlyPendingTimers())).not.toThrow();
    clearSpy.mockRestore();
  });

  it('cancels the auto-dismiss timer on manual dismiss', () => {
    const clearSpy = vi.spyOn(globalThis, 'clearTimeout');
    render(
      <ToastProvider>
        <Trigger />
      </ToastProvider>,
    );

    act(() => {
      fireEvent.click(screen.getByText('fire toast'));
    });
    expect(screen.getByText('hi')).toBeInTheDocument();

    const before = clearSpy.mock.calls.length;
    act(() => {
      fireEvent.click(screen.getByRole('button', { name: /dismiss notification/i }));
    });
    // Manual dismiss must cancel the pending auto-dismiss timer.
    expect(clearSpy.mock.calls.length).toBeGreaterThan(before);

    // Advancing past the original 4s window must not throw — the timer was
    // cancelled, so its callback never runs (no double-removal / dead setState).
    expect(() => act(() => vi.advanceTimersByTime(5000))).not.toThrow();
    clearSpy.mockRestore();
  });
});
