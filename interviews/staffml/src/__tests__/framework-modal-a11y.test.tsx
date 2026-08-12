/**
 * Framework primitive-detail modal accessibility guardrails.
 *
 * The PrimitiveDetail overlay rendered as a plain <div>: no role="dialog",
 * aria-modal, or aria-labelledby, and no focus management — so screen readers
 * didn't announce it as a dialog and keyboard focus stayed in the dimmed page
 * behind it. (Escape was already handled at the page level.)
 *
 * It now carries dialog semantics labelled by the primitive name, moves focus
 * to the close button on mount, restores focus on unmount, and traps Tab —
 * mirroring KeyboardShortcutsOverlay / CommandPalette.
 */
import { describe, it, expect } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { PrimitiveDetail } from '@/app/framework/page';
import { primitives } from '@/data/designGrammar';

const sample = primitives[0];

describe('Framework PrimitiveDetail modal a11y', () => {
  it('exposes dialog semantics labelled by the primitive name', () => {
    render(<PrimitiveDetail primitive={sample} onClose={() => {}} onLinkClick={() => {}} />);
    const dialog = screen.getByRole('dialog');
    expect(dialog).toHaveAttribute('aria-modal', 'true');
    expect(dialog).toHaveAttribute('aria-labelledby', 'primitive-detail-title');
    expect(document.getElementById('primitive-detail-title')).toHaveTextContent(sample.name);
  });

  it('moves focus to the close button on mount and restores it on unmount', async () => {
    const trigger = document.createElement('button');
    document.body.appendChild(trigger);
    trigger.focus();

    const { unmount } = render(
      <PrimitiveDetail primitive={sample} onClose={() => {}} onLinkClick={() => {}} />,
    );
    const close = screen.getByRole('button', { name: 'Close' });
    await waitFor(() => expect(close).toHaveFocus());

    unmount();
    expect(trigger).toHaveFocus();
    trigger.remove();
  });
});
