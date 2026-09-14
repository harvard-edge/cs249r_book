/**
 * Integration Tests — QuestionFeedback Component
 *
 * Renders the actual React component, simulates user clicks,
 * and verifies both the DOM state AND the analytics pipeline.
 * This catches bugs that unit tests miss — like a button that
 * renders but doesn't wire up its onClick correctly.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import QuestionFeedback from '@/components/QuestionFeedback';
import { getAnalyticsEvents, clearAnalytics } from '@/lib/analytics';

const mockQuestion = {
  id: 'q-test-001',
  title: 'Test Question',
  level: 'L3',
  track: 'cloud',
  topic: 'kv-cache',
  zone: 'design',
  competency_area: 'memory',
};

beforeEach(() => {
  clearAnalytics();
  window.localStorage.clear();
  window.sessionStorage.clear();
});

// ─── Rendering ───────────────────────────────────────────────

describe('QuestionFeedback renders correctly', () => {
  it('renders thumbs up and down buttons', () => {
    render(<QuestionFeedback question={mockQuestion} />);

    expect(screen.getByLabelText('This question was useful')).toBeInTheDocument();
    expect(screen.getByLabelText('This question was not useful')).toBeInTheDocument();
  });

  it('renders difficulty buttons', () => {
    render(<QuestionFeedback question={mockQuestion} />);

    expect(screen.getByLabelText('Difficulty: Easy')).toBeInTheDocument();
    expect(screen.getByLabelText('Difficulty: Right')).toBeInTheDocument();
    expect(screen.getByLabelText('Difficulty: Hard')).toBeInTheDocument();
  });

  it('renders report and suggest links', () => {
    render(<QuestionFeedback question={mockQuestion} />);

    expect(screen.getByText('Report issue')).toBeInTheDocument();
    expect(screen.getByText('Suggest improvement')).toBeInTheDocument();
  });
});

// ─── Thumbs Click → Analytics Pipeline ───────────────────────

describe('Thumbs up click fires analytics and updates DOM', () => {
  it('clicking thumbs up stores event and sets aria-pressed', () => {
    render(<QuestionFeedback question={mockQuestion} />);

    const btn = screen.getByLabelText('This question was useful');
    expect(btn).toHaveAttribute('aria-pressed', 'false');

    fireEvent.click(btn);

    // DOM: aria-pressed updated
    expect(btn).toHaveAttribute('aria-pressed', 'true');

    // Analytics: event stored
    const events = getAnalyticsEvents();
    expect(events).toHaveLength(1);
    expect(events[0].event.type).toBe('question_thumbs');
    expect((events[0].event as any).value).toBe('up');
    expect((events[0].event as any).questionId).toBe('q-test-001');
  });

  it('clicking thumbs down after thumbs up switches correctly', () => {
    render(<QuestionFeedback question={mockQuestion} />);

    const upBtn = screen.getByLabelText('This question was useful');
    const downBtn = screen.getByLabelText('This question was not useful');

    fireEvent.click(upBtn);
    fireEvent.click(downBtn);

    // DOM: down is pressed, up is not
    expect(upBtn).toHaveAttribute('aria-pressed', 'false');
    expect(downBtn).toHaveAttribute('aria-pressed', 'true');

    // Analytics: two events (up then down)
    const events = getAnalyticsEvents();
    expect(events).toHaveLength(2);
    expect((events[1].event as any).value).toBe('down');
  });
});

// ─── Dedup Guard: Same Button Twice ──────────────────────────

describe('Dedup guard prevents duplicate events', () => {
  it('clicking thumbs up twice fires only ONE event', () => {
    render(<QuestionFeedback question={mockQuestion} />);

    const btn = screen.getByLabelText('This question was useful');
    fireEvent.click(btn);
    fireEvent.click(btn); // second click — should be guarded

    const events = getAnalyticsEvents();
    expect(events).toHaveLength(1); // NOT 2
  });

  it('clicking same difficulty twice fires only ONE event', () => {
    render(<QuestionFeedback question={mockQuestion} />);

    const btn = screen.getByLabelText('Difficulty: Hard');
    fireEvent.click(btn);
    fireEvent.click(btn);

    const events = getAnalyticsEvents();
    expect(events).toHaveLength(1);
  });
});

// ─── Difficulty Click → Analytics Pipeline ───────────────────

describe('Difficulty click fires analytics', () => {
  it('clicking "Too Hard" stores correct event', () => {
    render(<QuestionFeedback question={mockQuestion} />);

    fireEvent.click(screen.getByLabelText('Difficulty: Hard'));

    const events = getAnalyticsEvents();
    expect(events).toHaveLength(1);
    expect(events[0].event.type).toBe('question_difficulty_feedback');
    expect((events[0].event as any).perceived).toBe('too_hard');
  });

  it('switching difficulty fires new event', () => {
    render(<QuestionFeedback question={mockQuestion} />);

    fireEvent.click(screen.getByLabelText('Difficulty: Easy'));
    fireEvent.click(screen.getByLabelText('Difficulty: Hard'));

    const events = getAnalyticsEvents();
    expect(events).toHaveLength(2);
    expect((events[0].event as any).perceived).toBe('too_easy');
    expect((events[1].event as any).perceived).toBe('too_hard');
  });
});

// ─── Report/Suggest Fire Analytics ───────────────────────────

describe('Report and Suggest links fire analytics on click', () => {
  it('clicking Report Issue fires question_reported event', () => {
    render(<QuestionFeedback question={mockQuestion} />);

    const link = screen.getByText('Report issue');
    fireEvent.click(link);

    const events = getAnalyticsEvents();
    expect(events).toHaveLength(1);
    expect(events[0].event.type).toBe('question_reported');
    expect((events[0].event as any).questionId).toBe('q-test-001');
  });

  it('clicking Suggest Improvement fires improvement_suggested event', () => {
    render(<QuestionFeedback question={mockQuestion} />);

    const link = screen.getByText('Suggest improvement');
    fireEvent.click(link);

    const events = getAnalyticsEvents();
    expect(events).toHaveLength(1);
    expect(events[0].event.type).toBe('improvement_suggested');
  });
});

// ─── Accessibility ───────────────────────────────────────────

describe('Accessibility: aria attributes', () => {
  it('thumbs buttons have aria-pressed that reflects state', () => {
    render(<QuestionFeedback question={mockQuestion} />);

    const upBtn = screen.getByLabelText('This question was useful');
    const downBtn = screen.getByLabelText('This question was not useful');

    // Initially: both unpressed
    expect(upBtn).toHaveAttribute('aria-pressed', 'false');
    expect(downBtn).toHaveAttribute('aria-pressed', 'false');

    // Click up
    fireEvent.click(upBtn);
    expect(upBtn).toHaveAttribute('aria-pressed', 'true');
    expect(downBtn).toHaveAttribute('aria-pressed', 'false');

    // Switch to down
    fireEvent.click(downBtn);
    expect(upBtn).toHaveAttribute('aria-pressed', 'false');
    expect(downBtn).toHaveAttribute('aria-pressed', 'true');
  });

  it('difficulty buttons have aria-pressed', () => {
    render(<QuestionFeedback question={mockQuestion} />);

    const easy = screen.getByLabelText('Difficulty: Easy');
    const hard = screen.getByLabelText('Difficulty: Hard');

    expect(easy).toHaveAttribute('aria-pressed', 'false');
    fireEvent.click(easy);
    expect(easy).toHaveAttribute('aria-pressed', 'true');
    expect(hard).toHaveAttribute('aria-pressed', 'false');
  });

  it('button groups have role="group"', () => {
    render(<QuestionFeedback question={mockQuestion} />);

    const groups = screen.getAllByRole('group');
    expect(groups.length).toBeGreaterThanOrEqual(2);
  });
});

// ─── Feedback Hydration on Revisit ───────────────────────────

describe('Previous feedback is hydrated on mount', () => {
  it('shows thumbs up if previously given', () => {
    // Pre-populate localStorage with a previous thumbs event
    const stored = [{
      event: { type: 'question_thumbs', questionId: 'q-test-001', topic: 'kv-cache', level: 'L3', value: 'up' },
      timestamp: Date.now(),
      sessionId: 'prev-session',
    }];
    window.localStorage.setItem('staffml_analytics', JSON.stringify(stored));

    render(<QuestionFeedback question={mockQuestion} />);

    const upBtn = screen.getByLabelText('This question was useful');
    expect(upBtn).toHaveAttribute('aria-pressed', 'true');
  });

  it('shows difficulty if previously given', () => {
    const stored = [{
      event: { type: 'question_difficulty_feedback', questionId: 'q-test-001', topic: 'kv-cache', level: 'L3', perceived: 'too_hard' },
      timestamp: Date.now(),
      sessionId: 'prev-session',
    }];
    window.localStorage.setItem('staffml_analytics', JSON.stringify(stored));

    render(<QuestionFeedback question={mockQuestion} />);

    const hardBtn = screen.getByLabelText('Difficulty: Hard');
    expect(hardBtn).toHaveAttribute('aria-pressed', 'true');
  });
});
