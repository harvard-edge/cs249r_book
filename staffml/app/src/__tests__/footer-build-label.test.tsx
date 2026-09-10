/**
 * Footer's build-date label must format in UTC.
 *
 * With Next's static export, the HTML is generated at build time and
 * hydrates on the client whenever the user visits. `toLocaleDateString`
 * without an explicit `timeZone` formats in the runtime's local TZ — so
 * the build server (UTC) and a client in a different TZ can format the
 * same ISO instant differently near day/year boundaries. That's a
 * hydration mismatch AND a wrong label.
 *
 * Mirrors the year-stability test for PaperCitationCard (PR #1843).
 */
import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/react';

// Mock the stats module BEFORE importing Footer so the constants are the
// ones we control. Each test below re-mocks per case via vi.doMock for a
// clean closure.
vi.mock('@/lib/stats', () => ({
  RELEASE_ID: '0.1.0',
  RELEASE_HASH: 'abcdef1234567890abcdef1234567890',
  BUILD_DATE: '2025-01-01T01:00:00Z',
}));

import Footer from '@/components/Footer';

describe('Footer build label', () => {
  it('formats the build date in UTC, not the runtime local timezone', () => {
    // 2025-01-01T01:00:00Z is Jan 1 2025 in UTC but Dec 31 2024 in PT
    // (UTC-8). The label must read "Jan 1, 2025" — the UTC value, matching
    // the build server — regardless of where the viewer sits.
    const { container } = render(<Footer />);
    expect(container.textContent).toContain('Jan 1, 2025');
    expect(container.textContent).not.toContain('Dec 31, 2024');
  });

  it('does not depend on the runtime clock', () => {
    // Force "now" to a different year than BUILD_DATE. If the formatting
    // ever regresses to use the runtime clock the assertion below
    // catches it.
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2099-07-04T00:00:00Z'));
    try {
      const { container } = render(<Footer />);
      expect(container.textContent).toContain('Jan 1, 2025');
      expect(container.textContent).not.toContain('2099');
    } finally {
      vi.useRealTimers();
    }
  });

});
