/**
 * ChainStrip a11y guardrails.
 *
 * The progress dots used to convey state purely by color/scale/opacity
 * and `title=` attributes — keyboard / screen-reader users couldn't tell
 * which step was current or what each step was about. We now render the
 * dots in an `<ol>` with `aria-current="step"` on the active dot and an
 * accessible name on each button.
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import ChainStrip from '@/components/ChainStrip';
import type { ChainInfo } from '@/lib/corpus';

const chain: ChainInfo = {
  chainId: 'kv-cache-tour',
  position: 1,
  total: 3,
  tier: 'primary',
  questions: [
    { id: 'q-a', title: 'What is a KV cache?',             level: 'L1', position: 0 },
    { id: 'q-b', title: 'Why does KV cache memory grow?',  level: 'L2', position: 1 },
    { id: 'q-c', title: 'Estimate KV cache for Llama 70B', level: 'L3', position: 2 },
  ],
};

describe('ChainStrip a11y', () => {
  it('renders the dots as a semantic ordered list', () => {
    render(<ChainStrip chain={chain} onNavigate={() => {}} />);
    const list = screen.getByRole('list', { name: /question chain progress/i });
    expect(within(list).getAllByRole('listitem')).toHaveLength(3);
  });

  it('marks only the current step with aria-current', () => {
    render(<ChainStrip chain={chain} onNavigate={() => {}} />);
    const list = screen.getByRole('list', { name: /question chain progress/i });
    const current = within(list).getAllByRole('button').filter(
      b => b.getAttribute('aria-current') === 'step',
    );
    expect(current).toHaveLength(1);
    expect(current[0].getAttribute('aria-label')).toMatch(/Part 2 of 3/);
  });

  it('gives each dot an accessible name with part, level, and title', () => {
    render(<ChainStrip chain={chain} onNavigate={() => {}} />);
    expect(
      screen.getByRole('button', { name: /Part 1 of 3.*L1.*What is a KV cache.*completed/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /Part 2 of 3.*L2.*Why does KV cache memory grow/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /Part 3 of 3.*L3.*Estimate KV cache for Llama 70B/i }),
    ).toBeInTheDocument();
  });

  it('only past steps are labelled as completed', () => {
    render(<ChainStrip chain={chain} onNavigate={() => {}} />);
    const list = screen.getByRole('list', { name: /question chain progress/i });
    const completed = within(list).getAllByRole('button').filter(
      b => /completed/i.test(b.getAttribute('aria-label') || ''),
    );
    expect(completed).toHaveLength(1);
    expect(completed[0].getAttribute('aria-label')).toMatch(/Part 1 of 3/);
  });

  it('every dot button is type="button" so it cannot accidentally submit a form', () => {
    render(<ChainStrip chain={chain} onNavigate={() => {}} />);
    const list = screen.getByRole('list', { name: /question chain progress/i });
    for (const button of within(list).getAllByRole('button')) {
      expect((button as HTMLButtonElement).type).toBe('button');
    }
  });

  it('clicking a dot still navigates to that question', () => {
    const onNavigate = vi.fn();
    render(<ChainStrip chain={chain} onNavigate={onNavigate} />);
    const part3 = screen.getByRole('button', { name: /Part 3 of 3/i });
    part3.click();
    expect(onNavigate).toHaveBeenCalledWith('q-c');
  });
});
