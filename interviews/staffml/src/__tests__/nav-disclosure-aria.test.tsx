/**
 * Nav disclosure ARIA guardrails.
 *
 * The "Tools" dropdown toggle and the mobile hamburger are disclosure buttons
 * that show/hide a menu, but they exposed no expanded/collapsed state to
 * assistive tech — the only cue was a visual chevron rotation. Screen-reader
 * users couldn't tell whether the menu was open. Both now carry
 * aria-expanded / aria-haspopup / aria-controls, and the controlled menu
 * gets a matching id.
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, within } from '@testing-library/react';

vi.mock('next/navigation', () => ({
  usePathname: () => '/',
}));

import Nav from '@/components/Nav';
import ThemeProvider from '@/components/ThemeProvider';

function renderNav() {
  return render(
    <ThemeProvider>
      <Nav />
    </ThemeProvider>,
  );
}

describe('Nav Tools dropdown ARIA', () => {
  it('reflects expanded state and wires aria-controls to the menu', () => {
    renderNav();
    const toggle = screen.getByRole('button', { name: 'Tools' });

    expect(toggle).toHaveAttribute('aria-haspopup', 'true');
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(toggle).toHaveAttribute('aria-controls', 'nav-tools-menu');

    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'true');

    const menu = document.getElementById('nav-tools-menu');
    expect(menu).not.toBeNull();
    // Sanity-check the controlled element actually holds the tool links.
    expect(within(menu as HTMLElement).getByText('Framework')).toBeInTheDocument();
  });
});

describe('Nav mobile hamburger ARIA', () => {
  it('reflects expanded state and wires aria-controls to the mobile menu', () => {
    renderNav();
    const toggle = screen.getByRole('button', { name: 'Toggle navigation menu' });

    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(toggle).toHaveAttribute('aria-controls', 'nav-mobile-menu');

    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    expect(document.getElementById('nav-mobile-menu')).not.toBeNull();
  });
});
