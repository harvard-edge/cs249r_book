"use client";

import { useState } from "react";
import clsx from "clsx";
import { CheckCircle2, XCircle, Circle } from "lucide-react";
import MarkdownText from "./MarkdownText";

/**
 * MCQ self-check widget for multiple-choice questions.
 *
 * Decision (A1): this is a SELF-CHECK, not a scored gate. It is deliberately
 * NOT wired into the spaced-repetition / self-rating pipeline — selecting an
 * option and revealing the answer changes nothing in `progress.ts`. The open
 * self-rating flow (Skip / Wrong / Partial / Nailed It) remains the single
 * source of truth for scoring, exactly as it is for open-response questions.
 *
 * Behavior:
 *   - Pre-reveal: options are selectable radio-style buttons. Distractors are
 *     kept; the learner commits to a pick before revealing.
 *   - Post-reveal (`revealed`): the correct option is marked green; the
 *     learner's pick, if wrong, is marked red. The existing explanation
 *     (common_mistake / realistic_solution) continues to render elsewhere.
 *
 * Rendered only when a question carries `details.options` + `correct_index`.
 */
export default function MCQOptions({
  options,
  correctIndex,
  revealed,
  selected,
  onSelect,
}: {
  options: string[];
  correctIndex: number;
  revealed: boolean;
  /** Controlled selected index, or null if nothing picked yet. */
  selected: number | null;
  onSelect: (index: number) => void;
}) {
  const letters = "ABCDEFGHIJ";
  return (
    <div className="mt-6 space-y-2" role="radiogroup" aria-label="Answer choices">
      {options.map((opt, idx) => {
        const isCorrect = idx === correctIndex;
        const isPicked = idx === selected;
        const showCorrect = revealed && isCorrect;
        const showWrongPick = revealed && isPicked && !isCorrect;

        return (
          <button
            key={idx}
            type="button"
            role="radio"
            aria-checked={isPicked}
            disabled={revealed}
            onClick={() => !revealed && onSelect(idx)}
            className={clsx(
              "w-full text-left flex items-start gap-3 p-3 rounded-lg border transition-all text-sm",
              revealed ? "cursor-default" : "cursor-pointer",
              showCorrect
                ? "border-accentGreen/50 bg-accentGreen/10"
                : showWrongPick
                ? "border-accentRed/50 bg-accentRed/10"
                : isPicked
                ? "border-accentBlue/50 bg-accentBlue/5"
                : "border-border hover:border-borderHighlight",
            )}
          >
            <span className="shrink-0 mt-0.5">
              {showCorrect ? (
                <CheckCircle2 className="w-4 h-4 text-accentGreen" />
              ) : showWrongPick ? (
                <XCircle className="w-4 h-4 text-accentRed" />
              ) : isPicked ? (
                <span className="w-4 h-4 rounded-full bg-accentBlue/20 border border-accentBlue flex items-center justify-center text-[9px] font-bold font-mono text-accentBlue">
                  {letters[idx]}
                </span>
              ) : (
                <Circle className="w-4 h-4 text-textTertiary/50" />
              )}
            </span>
            <span
              className={clsx(
                "leading-relaxed",
                showCorrect
                  ? "text-accentGreen font-medium"
                  : showWrongPick
                  ? "text-accentRed"
                  : "text-textSecondary",
              )}
            >
              <MarkdownText text={opt} />
            </span>
          </button>
        );
      })}
    </div>
  );
}
