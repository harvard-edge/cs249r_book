/**
 * WaitlistModal a11y: focus management + focus trap.
 *
 * Before: the dialog had role="dialog" aria-modal="true" but Tab/Shift+Tab
 * escaped the modal into background controls behind the backdrop, and the
 * email input wasn't auto-focused on open — keyboard users landed in the
 * page body and had to hunt for the input.
 *
 * After: focus moves to the email input on open, Tab cycles within the
 * surface, and focus returns to the previously-focused element on unmount.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, fireEvent } from '@testing-library/react';
import { WaitlistModal } from '@/components/AskInterviewer';

describe('WaitlistModal a11y', () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('autofocuses the email input on open', () => {
    render(<WaitlistModal onClose={() => {}} endpoint="" />);
    vi.runAllTimers();   // setTimeout(0) for focus
    expect(document.activeElement?.tagName).toBe('INPUT');
    expect(document.activeElement?.getAttribute('type')).toBe('email');
  });

  it('restores focus to the previously-focused element on unmount', () => {
    // Set up an outside trigger that "had focus" before the modal opened.
    const trigger = document.createElement('button');
    trigger.textContent = 'Open waitlist';
    document.body.appendChild(trigger);
    trigger.focus();
    expect(document.activeElement).toBe(trigger);

    const { unmount } = render(<WaitlistModal onClose={() => {}} endpoint="" />);
    vi.runAllTimers();
    expect(document.activeElement).not.toBe(trigger);   // moved into modal

    unmount();
    expect(document.activeElement).toBe(trigger);       // restored
    trigger.remove();
  });

  it('cycles focus from the last focusable back to the first on Tab', () => {
    const { container } = render(<WaitlistModal onClose={() => {}} endpoint="" />);
    vi.runAllTimers();
    const surface = container.querySelector('[role="dialog"] > div') as HTMLElement;
    const focusables = surface.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), input:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    );
    const first = focusables[0];
    const last = focusables[focusables.length - 1];

    last.focus();
    expect(document.activeElement).toBe(last);
    fireEvent.keyDown(surface, { key: 'Tab' });
    expect(document.activeElement).toBe(first);
  });

  it('cycles focus from the first back to the last on Shift+Tab', () => {
    const { container } = render(<WaitlistModal onClose={() => {}} endpoint="" />);
    vi.runAllTimers();
    const surface = container.querySelector('[role="dialog"] > div') as HTMLElement;
    const focusables = surface.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), input:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    );
    const first = focusables[0];
    const last = focusables[focusables.length - 1];

    first.focus();
    expect(document.activeElement).toBe(first);
    fireEvent.keyDown(surface, { key: 'Tab', shiftKey: true });
    expect(document.activeElement).toBe(last);
  });

  it('does not steal focus when Tab is pressed in the middle of the trap', () => {
    // If focus is not on the boundary, Tab should fall through to the
    // browser's native handling — we only block at the edges.
    const { container } = render(<WaitlistModal onClose={() => {}} endpoint="" />);
    vi.runAllTimers();
    const surface = container.querySelector('[role="dialog"] > div') as HTMLElement;
    const focusables = surface.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), input:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    );
    // Focus a middle element.
    const middle = focusables[Math.floor(focusables.length / 2)];
    middle.focus();
    const event = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
    surface.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
  });
});
