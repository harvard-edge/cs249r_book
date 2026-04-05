"use client";

import { useState, useEffect } from "react";
import { ThumbsUp, ThumbsDown, Flag, Lightbulb } from "lucide-react";
import clsx from "clsx";
import { track, getAnalyticsEvents } from "@/lib/analytics";
import { buildReportUrl, buildSuggestUrl } from "@/lib/issue-url";

interface QuestionFeedbackProps {
  question: {
    id: string;
    title: string;
    level: string;
    track: string;
    topic: string;
    zone: string;
    competency_area: string;
  };
}

export default function QuestionFeedback({ question }: QuestionFeedbackProps) {
  const [thumbs, setThumbs] = useState<'up' | 'down' | null>(null);
  const [difficulty, setDifficulty] = useState<'too_easy' | 'about_right' | 'too_hard' | null>(null);

  // Hydrate previous feedback from analytics events on mount
  useEffect(() => {
    const events = getAnalyticsEvents();
    // Walk backwards to find the most recent feedback for this question
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i].event;
      if (e.type === 'question_thumbs' && e.questionId === question.id) {
        setThumbs(e.value);
        break;
      }
    }
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i].event;
      if (e.type === 'question_difficulty_feedback' && e.questionId === question.id) {
        setDifficulty(e.perceived);
        break;
      }
    }
  }, [question.id]);

  const handleThumbs = (value: 'up' | 'down') => {
    if (thumbs === value) return; // guard: no duplicate events
    setThumbs(value);
    track({
      type: 'question_thumbs',
      questionId: question.id,
      topic: question.topic,
      level: question.level,
      value,
    });
  };

  const handleDifficulty = (perceived: 'too_easy' | 'about_right' | 'too_hard') => {
    if (difficulty === perceived) return; // guard: no duplicate events
    setDifficulty(perceived);
    track({
      type: 'question_difficulty_feedback',
      questionId: question.id,
      topic: question.topic,
      level: question.level,
      perceived,
    });
  };

  return (
    <div className="border-t border-border pt-4 space-y-3">
      {/* Thumbs + difficulty row */}
      <div className="flex items-center gap-4 flex-wrap">
        {/* Thumbs up/down */}
        <div className="flex items-center gap-1.5" role="group" aria-label="Question usefulness">
          <span className="text-[10px] font-mono text-textTertiary uppercase mr-1">Useful?</span>
          <button
            onClick={() => handleThumbs('up')}
            className={clsx(
              "p-1.5 rounded-md border transition-all",
              thumbs === 'up'
                ? "border-accentGreen/40 bg-accentGreen/10 text-accentGreen"
                : "border-transparent text-textMuted hover:text-accentGreen hover:bg-accentGreen/5"
            )}
            aria-label="This question was useful"
            aria-pressed={thumbs === 'up'}
          >
            <ThumbsUp className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => handleThumbs('down')}
            className={clsx(
              "p-1.5 rounded-md border transition-all",
              thumbs === 'down'
                ? "border-accentRed/40 bg-accentRed/10 text-accentRed"
                : "border-transparent text-textMuted hover:text-accentRed hover:bg-accentRed/5"
            )}
            aria-label="This question was not useful"
            aria-pressed={thumbs === 'down'}
          >
            <ThumbsDown className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Vertical divider */}
        <div className="w-px h-5 bg-border" />

        {/* Difficulty perception */}
        <div className="flex items-center gap-1.5" role="group" aria-label="Perceived difficulty">
          <span className="text-[10px] font-mono text-textTertiary uppercase mr-1">Difficulty?</span>
          {([
            { value: 'too_easy' as const, label: 'Easy', emoji: '😌' },
            { value: 'about_right' as const, label: 'Right', emoji: '👌' },
            { value: 'too_hard' as const, label: 'Hard', emoji: '🤯' },
          ]).map(({ value, label, emoji }) => (
            <button
              key={value}
              onClick={() => handleDifficulty(value)}
              className={clsx(
                "px-2 py-1 rounded-md border text-[10px] font-medium transition-all",
                difficulty === value
                  ? value === 'too_easy' ? "border-accentGreen/40 bg-accentGreen/10 text-accentGreen"
                  : value === 'about_right' ? "border-accentBlue/40 bg-accentBlue/10 text-accentBlue"
                  : "border-accentRed/40 bg-accentRed/10 text-accentRed"
                  : "border-transparent text-textMuted hover:border-borderHighlight hover:text-textSecondary"
              )}
              aria-label={`Difficulty: ${label}`}
              aria-pressed={difficulty === value}
            >
              {emoji} {label}
            </button>
          ))}
        </div>
      </div>

      {/* Report + suggest row */}
      <div className="flex items-center gap-4">
        <a
          href={buildReportUrl(question)}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 text-[11px] text-textSecondary hover:text-accentRed transition-colors"
          onClick={() => track({ type: 'question_reported', questionId: question.id })}
        >
          <Flag className="w-3.5 h-3.5" /> Report issue
        </a>
        <a
          href={buildSuggestUrl(question)}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 text-[11px] text-textSecondary hover:text-accentAmber transition-colors"
          onClick={() => track({ type: 'improvement_suggested', questionId: question.id })}
        >
          <Lightbulb className="w-3.5 h-3.5" /> Suggest improvement
        </a>
      </div>
    </div>
  );
}
