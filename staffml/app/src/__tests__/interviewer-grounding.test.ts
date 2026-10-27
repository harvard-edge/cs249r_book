/**
 * Interviewer grounding: what the Ask Interviewer panel sends the worker.
 *
 * 2026-10-27: the tutor redid napkin math from scratch and got it wrong, and
 * interview mode invented constraints that contradicted the question. The fix
 * sends the question's verified napkin math; interview mode sees only the
 * Assumptions section. These tests pin the pieces that fix depends on,
 * including the site/worker mirrors that must stay identical.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { interviewerContext, napkinAssumptions } from '@/components/AskInterviewer';

const NAPKIN = `**Assumptions & Constraints:**
- KV cache per token: 131,072 bytes.
- 128k context.

**Calculations:**
- Context memory: 128,000 x 131,072 bytes = 16.77 GB.

**Conclusion & Interpretation:**
- **Result: ~33GB required.**`;

describe('interviewerContext', () => {
  it('appends the explicit question to the scenario', () => {
    expect(interviewerContext({ scenario: 'Serve an 8B model.', question: 'Why is this impossible?' }))
      .toBe('Serve an 8B model.\n\nQuestion: Why is this impossible?');
  });

  it('falls back to the scenario when there is no question', () => {
    expect(interviewerContext({ scenario: 'Serve an 8B model.' })).toBe('Serve an 8B model.');
  });
});

describe('napkinAssumptions', () => {
  it('keeps the givens and drops the calculations and result', () => {
    const facts = napkinAssumptions(NAPKIN);
    expect(facts).toContain('131,072 bytes');
    expect(facts).not.toContain('Calculations');
    expect(facts).not.toContain('16.77');
    expect(facts).not.toContain('33GB');
  });

  it('shares nothing when the Calculations heading is missing', () => {
    expect(napkinAssumptions('Result: 33GB')).toBe('');
  });
});

describe('site and worker mirrors', () => {
  const site = readFileSync(resolve(__dirname, '../components/AskInterviewer.tsx'), 'utf8');
  const worker = readFileSync(resolve(__dirname, '../../worker/src/index.ts'), 'utf8');
  const pattern = (src: string) => src.match(/const DELIMITER_PATTERN = (\/.+\/gi);/)?.[1];

  it('strip the same delimiter tags', () => {
    expect(pattern(site)).toBeDefined();
    expect(pattern(site)).toBe(pattern(worker));
  });

  it('cut napkin math at the same heading', () => {
    const cut = 'napkin.search(/\\*\\*Calculations/i)';
    expect(site).toContain(cut);
    expect(worker).toContain(cut);
  });
});
