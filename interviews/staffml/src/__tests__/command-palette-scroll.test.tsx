/**
 * CommandPalette keyboard-scroll guardrail.
 *
 * The result list is scrollable (max-h-[60vh]). Arrowing down past the fold
 * used to move the highlighted row out of view because nothing scrolled the
 * active row back into the viewport. We now call scrollIntoView on the active
 * row whenever activeIdx changes, with `block: "nearest"` so an already-visible
 * row never causes a jump.
 *
 * jsdom doesn't implement Element.prototype.scrollIntoView, so we spy on it.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import CommandPalette from '@/components/CommandPalette';

// ─── Mock the data + routing deps so the palette renders standalone ──
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}));
vi.mock('@/lib/taxonomy', () => ({ searchTopics: () => [] }));
vi.mock('@/lib/corpus', () => ({ searchQuestions: () => [] }));
vi.mock('@/lib/corpus-provider', () => ({
  useVault: () => ({ apiBase: null }),
  vaultSearch: async () => [],
}));

function openPalette() {
  // The navbar dispatches this event to open the palette by click.
  act(() => {
    window.dispatchEvent(new Event('staffml:open-palette'));
  });
}

describe('CommandPalette active-row scrolling', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('scrolls the active row into view when navigating with ArrowDown', () => {
    const scrollSpy = vi
      .spyOn(Element.prototype, 'scrollIntoView')
      .mockImplementation(() => {});

    render(<CommandPalette />);
    openPalette();

    // With an empty query the static Pages list is shown — enough rows to navigate.
    const input = screen.getByLabelText(/command palette query/i);

    scrollSpy.mockClear();
    fireEvent.keyDown(input, { key: 'ArrowDown' });

    // The newly-active row (index 1) should have been scrolled into view.
    const calledRows = scrollSpy.mock.instances as Element[];
    expect(scrollSpy).toHaveBeenCalled();
    expect(calledRows.some(el => el.id === 'cmdk-row-1')).toBe(true);
    // And it must use block:"nearest" so visible rows never jump.
    expect(scrollSpy).toHaveBeenCalledWith({ block: 'nearest' });
  });

  it('scrolls back up to the active row on ArrowUp', () => {
    const scrollSpy = vi
      .spyOn(Element.prototype, 'scrollIntoView')
      .mockImplementation(() => {});

    render(<CommandPalette />);
    openPalette();
    const input = screen.getByLabelText(/command palette query/i);

    fireEvent.keyDown(input, { key: 'ArrowDown' });
    fireEvent.keyDown(input, { key: 'ArrowDown' });
    scrollSpy.mockClear();
    fireEvent.keyDown(input, { key: 'ArrowUp' });

    const calledRows = scrollSpy.mock.instances as Element[];
    expect(calledRows.some(el => el.id === 'cmdk-row-1')).toBe(true);
  });
});
