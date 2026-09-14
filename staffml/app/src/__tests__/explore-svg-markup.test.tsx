/**
 * Explore radial-chart markup guardrail.
 *
 * The center "zoom out" affordance used to be an HTML <button> wrapping an SVG
 * <circle>. A <button> is not valid SVG content — it doesn't focus reliably
 * across browsers and has no accessible name. The click handler now lives on
 * the <circle> directly; the keyboard-accessible navigation is the Breadcrumb
 * and the ExplorerPanel, and the chart is exposed to assistive tech as a
 * single labelled role="img".
 *
 * This test locks in that the SVG contains no nested HTML <button>.
 */
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import ExplorePage from '@/app/explore/page';

describe('Explore radial chart markup', () => {
  it('renders the chart as a labelled role="img" with no HTML <button> nested in the SVG', () => {
    const { container } = render(<ExplorePage />);
    const svg = container.querySelector('svg[role="img"]');
    expect(svg).not.toBeNull();
    expect(svg).toHaveAttribute('aria-label');
    // Invalid SVG content: there must be no HTML button inside the SVG.
    expect(svg!.querySelector('button')).toBeNull();
  });
});
