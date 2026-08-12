/**
 * PaperCitationCard year stability.
 *
 * Previously the BibTeX year came from `new Date().getFullYear()`, which
 * is the *render-time* year on the client. With Next's static export the
 * HTML is generated at build time and hydrates on the client whenever the
 * user visits — server (build) and client (hydrate) could disagree on the
 * year (e.g. around midnight UTC on Dec 31 / Jan 1, or just any time the
 * site is viewed in a year after the build year). That's a hydration
 * mismatch AND, more fundamentally, the wrong anchor for a citation.
 * The year now comes from the required `buildDate` prop (UTC).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render } from '@testing-library/react';
import PaperCitationCard from '@/components/PaperCitationCard';

const baseProps = {
  paperUrl: 'https://example.com/paper.pdf',
  releaseId: '0.1.0',
  releaseHash: 'abcdef1234567890abcdef1234567890',
};

function getCiteText(container: HTMLElement): string {
  return container.querySelector('pre')?.textContent ?? '';
}

describe('PaperCitationCard BibTeX year', () => {
  it('derives the year from buildDate, not from the current clock', () => {
    // Force "now" to a different year than buildDate. If the code ever
    // regresses to `new Date().getFullYear()` this test will catch it.
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2099-07-04T00:00:00Z'));
    try {
      const { container } = render(
        <PaperCitationCard {...baseProps} buildDate="2024-06-15T00:00:00Z" />,
      );
      const cite = getCiteText(container);
      expect(cite).toContain('@misc{staffml2024');
      expect(cite).toContain('year = {2024}');
      expect(cite).not.toContain('2099');
    } finally {
      vi.useRealTimers();
    }
  });

  it('updates the year when buildDate changes', () => {
    const { container, rerender } = render(
      <PaperCitationCard {...baseProps} buildDate="2024-06-15T00:00:00Z" />,
    );
    expect(getCiteText(container)).toContain('@misc{staffml2024');

    rerender(<PaperCitationCard {...baseProps} buildDate="2027-03-01T00:00:00Z" />);
    expect(getCiteText(container)).toContain('@misc{staffml2027');
    expect(getCiteText(container)).toContain('year = {2027}');
  });

  it('uses UTC so the year is stable regardless of viewer timezone', () => {
    // 2025-01-01T01:00:00Z is Jan 1 2025 in UTC but Dec 31 2024 in PT
    // (UTC-8). The citation year must be 2025 (the UTC year, matching
    // the build server) and not depend on where the viewer is sitting.
    const { container } = render(
      <PaperCitationCard {...baseProps} buildDate="2025-01-01T01:00:00Z" />,
    );
    expect(getCiteText(container)).toContain('@misc{staffml2025');
  });
});
