/**
 * MetaTooltip a11y guardrails.
 *
 * Previously the trigger declared role="button" with no handler — screen
 * readers announced "button" but Enter/Space did nothing. We keep the
 * trigger focusable (tabIndex={0}) so keyboard users see the tooltip on
 * focus, but no longer claim it's a button.
 *
 * Also locks in the `withTooltip={false}` opt-out on LevelBadge for the
 * "rendered inside a <button>" case at vault/TopicDetail.tsx, which
 * otherwise produces nested interactive elements (invalid HTML +
 * double tab stop).
 */
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import MetaTooltip from '@/components/MetaTooltip';
import LevelBadge from '@/components/LevelBadge';

describe('MetaTooltip', () => {
  it('does not declare role="button" on the trigger', () => {
    render(
      <MetaTooltip title="L4" body="Analyze">
        <span>L4</span>
      </MetaTooltip>,
    );
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  it('keeps the trigger keyboard-focusable so the tooltip appears on focus', () => {
    const { container } = render(
      <MetaTooltip title="L4" body="Analyze">
        <span>L4</span>
      </MetaTooltip>,
    );
    const trigger = container.firstElementChild as HTMLElement;
    expect(trigger.tabIndex).toBe(0);
  });

  it('links the trigger to the tooltip via aria-describedby', () => {
    const { container } = render(
      <MetaTooltip title="L4" body="Analyze">
        <span>L4</span>
      </MetaTooltip>,
    );
    const trigger = container.firstElementChild as HTMLElement;
    const describedBy = trigger.getAttribute('aria-describedby');
    expect(describedBy).toBeTruthy();
    const tip = document.getElementById(describedBy!);
    expect(tip).not.toBeNull();
    expect(tip!.getAttribute('role')).toBe('tooltip');
  });
});

describe('LevelBadge', () => {
  it('renders a MetaTooltip by default (standalone use)', () => {
    const { container } = render(<LevelBadge level="L3" />);
    const root = container.firstElementChild as HTMLElement;
    expect(root.getAttribute('aria-describedby')).toBeTruthy();
  });

  it('omits the tooltip wrapper when withTooltip={false} (nested-in-button use)', () => {
    const { container } = render(<LevelBadge level="L3" withTooltip={false} />);
    const root = container.firstElementChild as HTMLElement;
    expect(root.getAttribute('aria-describedby')).toBeNull();
    expect(root.hasAttribute('tabindex')).toBe(false);
  });
});
