/**
 * StarGate dialog accessibility guardrails.
 *
 * StarGate is a full-viewport blocking overlay, but it used to render as a
 * plain <div>: no role="dialog", no aria-modal, no aria-labelledby, no Escape
 * handler, and no focus management — so the dimmed page behind it stayed
 * keyboard-reachable and screen readers didn't announce it as a dialog.
 *
 * It now mirrors the KeyboardShortcutsOverlay / CommandPalette pattern:
 * dialog semantics, focus moved to the primary CTA on mount, focus restored
 * on unmount, Tab trapped within the surface, and Escape to dismiss.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

const markVerified = vi.fn();

vi.mock('@/lib/star-gate', () => ({
  // Never resolves — keeps the live star count in its loading state so the
  // test doesn't depend on network or timing.
  fetchStarCount: () => new Promise(() => {}),
  getStarUrl: () => 'https://github.com/harvard-edge/cs249r_book',
  markVerified: (...args: unknown[]) => markVerified(...args),
}));

import StarGate from '@/components/StarGate';

beforeEach(() => {
  markVerified.mockClear();
});

describe('StarGate dialog a11y', () => {
  it('exposes dialog semantics labelled by its heading', () => {
    render(<StarGate onVerified={() => {}} />);
    const dialog = screen.getByRole('dialog');
    expect(dialog).toHaveAttribute('aria-modal', 'true');
    expect(dialog).toHaveAttribute('aria-labelledby', 'stargate-title');
    expect(document.getElementById('stargate-title')).toHaveTextContent('Our only ask.');
  });

  it('moves focus to the primary CTA on mount', async () => {
    render(<StarGate onVerified={() => {}} />);
    const primary = screen.getByRole('button', { name: /Star on GitHub/i });
    await waitFor(() => expect(primary).toHaveFocus());
  });

  it('dismisses on Escape (counts as a dismiss) and restores focus on unmount', () => {
    const trigger = document.createElement('button');
    document.body.appendChild(trigger);
    trigger.focus();

    const onVerified = vi.fn();
    const { unmount } = render(<StarGate onVerified={onVerified} />);

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(markVerified).toHaveBeenCalledWith('dismissed');
    expect(onVerified).toHaveBeenCalledTimes(1);

    unmount();
    expect(trigger).toHaveFocus();
    trigger.remove();
  });
});
