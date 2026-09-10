/**
 * Regression test — QuestionVisual resets its `failed` state when the
 * visual.path prop changes.
 *
 * Bug: after one question's image errored (`onError` set failed=true), the
 * component stayed in the error branch when the parent re-rendered with a
 * different question's visual, because failed=true short-circuited before
 * the new <img> could mount and clear the state.
 */
import { describe, it, expect } from 'vitest';
import { render, fireEvent, screen } from '@testing-library/react';
import QuestionVisual from '@/components/QuestionVisual';

const visualA = {
  kind: 'svg' as const,
  path: 'broken.svg',
  alt: 'A broken diagram of cache hierarchies',
  caption: 'Visual A',
};

const visualB = {
  kind: 'svg' as const,
  path: 'working.svg',
  alt: 'A working diagram of attention heads',
  caption: 'Visual B',
};

describe('QuestionVisual', () => {
  it('shows the error UI after the image errors', () => {
    render(<QuestionVisual track="cloud" visual={visualA} />);
    const img = screen.getByTestId('question-visual-img') as HTMLImageElement;
    fireEvent.error(img);
    expect(screen.getByText(/Diagram failed to load/i)).toBeInTheDocument();
  });

  it('clears the error UI when visual.path changes', () => {
    const { rerender } = render(<QuestionVisual track="cloud" visual={visualA} />);
    fireEvent.error(screen.getByTestId('question-visual-img'));
    expect(screen.getByText(/Diagram failed to load/i)).toBeInTheDocument();

    rerender(<QuestionVisual track="cloud" visual={visualB} />);
    expect(screen.queryByText(/Diagram failed to load/i)).not.toBeInTheDocument();
    expect(screen.getByTestId('question-visual-img')).toBeInTheDocument();
  });
});
