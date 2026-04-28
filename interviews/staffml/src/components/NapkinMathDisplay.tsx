"use client";

import clsx from "clsx";
import MarkdownText from "./MarkdownText";

/**
 * Renders napkin math text with structured formatting.
 *
 * Handles common patterns in the corpus:
 * - Pipe-separated steps: "step1 | step2 | step3"
 * - Dash-prefixed bullets: "- step1\n- step2"
 * - Numbered steps: "1. **Title:** content"
 * - Inline markdown bold: **text**
 * - Arrow markers: => (final answers)
 * - Simple one-liners
 */
export default function NapkinMathDisplay({ text }: { text: string }) {
  const steps = parseSteps(text);

  if (steps.length === 1 && !steps[0].isResult) {
    // Simple one-liner — render inline
    return (
      <div className="font-mono text-[13px] text-textSecondary leading-relaxed">
        <MarkdownText text={steps[0].text} />
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {steps.map((step, i) => (
        <div
          key={i}
          className={clsx(
            "flex gap-3 items-start",
            step.isResult && "mt-1"
          )}
        >
          {/* Step indicator */}
          <div className="shrink-0 mt-0.5">
            {step.isResult ? (
              <span className="text-[11px] font-mono font-bold text-accentGreen bg-accentGreen/10 px-1.5 py-0.5 rounded">
                →
              </span>
            ) : (
              <span className="text-[11px] font-mono text-textMuted bg-surface px-1.5 py-0.5 rounded">
                {steps.filter((s, j) => j <= i && !s.isResult).length}
              </span>
            )}
          </div>

          {/* Step content */}
          <div
            className={clsx(
              "flex-1 font-mono text-[13px] leading-relaxed min-w-0",
              step.isResult
                ? "text-accentGreen font-semibold"
                : "text-textSecondary"
            )}
          >
            <MarkdownText text={step.text} />
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Parsing ──────────────────────────────────────────────────

interface Step {
  text: string;
  isResult: boolean;
}

function parseSteps(raw: string): Step[] {
  const trimmed = raw.trim();

  // Check if pipe-separated (common pattern: "step1 | step2 | => result")
  if (trimmed.includes(" | ")) {
    return trimmed.split(/\s*\|\s*/).map((part) => ({
      text: cleanStepText(part),
      isResult: part.trim().startsWith("=>") || part.trim().startsWith("- =>"),
    }));
  }

  // Check for numbered/bulleted lines
  const lines = trimmed.split("\n").filter((l) => l.trim());
  if (lines.length > 1) {
    return lines.map((line) => ({
      text: cleanStepText(line),
      isResult: line.trim().startsWith("=>"),
    }));
  }

  // Single line — check if it has => marker
  if (trimmed.startsWith("=>")) {
    return [{ text: cleanStepText(trimmed), isResult: true }];
  }

  // Long single-line text: split on sentence boundaries that contain calculations
  // Look for patterns like "result. Next sentence" or "value). Next"
  if (trimmed.length > 120) {
    const sentenceSteps = splitOnSentences(trimmed);
    if (sentenceSteps.length > 1) {
      return sentenceSteps;
    }
  }

  return [{ text: trimmed, isResult: false }];
}

/** Split a dense paragraph into steps at sentence boundaries */
function splitOnSentences(text: string): Step[] {
  // Split on ". " that follows a closing paren, number, unit, or word
  // but NOT on decimal points like "3.5" or abbreviations
  const parts = text.split(/(?<=[\d)%a-z])\.\s+(?=[A-Z])/g);
  if (parts.length <= 1) return [{ text, isResult: false }];

  return parts.map((part, i) => {
    const cleaned = part.trim();
    // Last sentence or sentences containing final "=" result tend to be conclusions
    const isResult = i === parts.length - 1 && /=\s*[\d,]+/.test(cleaned);
    return { text: cleaned, isResult };
  });
}

function cleanStepText(text: string): string {
  return text
    .replace(/^-\s*/, "") // strip leading dash
    .replace(/^\d+\.\s*/, "") // strip leading number
    .replace(/^=>\s*/, "") // strip => for result lines (we show our own indicator)
    .trim();
}
