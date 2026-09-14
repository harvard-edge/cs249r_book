/**
 * Toast accessibility guardrails.
 *
 * Toasts (badge unlocks, streak bumps, success messages) used to be invisible
 * to assistive tech: the container was a plain <div> with no live region, and
 * the dismiss button was an icon-only <button> with no accessible name. Screen
 * readers announced nothing on a toast, and "button" with no label on dismiss.
 *
 * We now expose the container as a polite live region and give the dismiss
 * button an aria-label — matching the existing VersionDriftToast pattern.
 */
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ToastProvider, useToast } from '@/components/Toast';

// Minimal consumer that fires a toast on click — exercises the public API.
function Trigger() {
  const { show } = useToast();
  return (
    <button onClick={() => show({ type: 'badge', title: 'Streak unlocked', description: '7-day streak' })}>
      fire toast
    </button>
  );
}

function renderWithToast() {
  return render(
    <ToastProvider>
      <Trigger />
    </ToastProvider>,
  );
}

describe('Toast a11y', () => {
  it('exposes the toast container as a labelled polite live region', () => {
    renderWithToast();
    const region = screen.getByRole('region', { name: /notifications/i });
    expect(region).toHaveAttribute('aria-live', 'polite');
    // aria-atomic=false so only the newly-added toast is announced, not the
    // whole stack re-read on every change.
    expect(region).toHaveAttribute('aria-atomic', 'false');
  });

  it('renders a fired toast with its title and description', () => {
    renderWithToast();
    fireEvent.click(screen.getByText('fire toast'));
    expect(screen.getByText('Streak unlocked')).toBeInTheDocument();
    expect(screen.getByText('7-day streak')).toBeInTheDocument();
  });

  it('gives the dismiss button an accessible name', () => {
    renderWithToast();
    fireEvent.click(screen.getByText('fire toast'));
    expect(
      screen.getByRole('button', { name: /dismiss notification/i }),
    ).toBeInTheDocument();
  });
});
