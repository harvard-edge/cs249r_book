"use client";

import clsx from "clsx";

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
        <FormattedText text={steps[0].text} />
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
            <FormattedText text={step.text} />
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

  return [{ text: trimmed, isResult: false }];
}

function cleanStepText(text: string): string {
  return text
    .replace(/^-\s*/, "") // strip leading dash
    .replace(/^\d+\.\s*/, "") // strip leading number
    .replace(/^=>\s*/, "") // strip => for result lines (we show our own indicator)
    .trim();
}

// ─── Inline Formatting ───────────────────────────────────────

function FormattedText({ text }: { text: string }) {
  // Split on **bold** markers and inline code `backticks`
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`|~[^~]+~)/g);

  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith("**") && part.endsWith("**")) {
          return (
            <span key={i} className="font-bold text-textPrimary">
              {part.slice(2, -2)}
            </span>
          );
        }
        if (part.startsWith("`") && part.endsWith("`")) {
          return (
            <code
              key={i}
              className="text-accentBlue bg-accentBlue/10 px-1 py-0.5 rounded text-[12px]"
            >
              {part.slice(1, -1)}
            </code>
          );
        }
        if (part.startsWith("~") && part.endsWith("~")) {
          return (
            <span key={i} className="text-textTertiary line-through">
              {part.slice(1, -1)}
            </span>
          );
        }
        // Highlight numbers and units inline
        return <HighlightNumbers key={i} text={part} />;
      })}
    </>
  );
}

function HighlightNumbers({ text }: { text: string }) {
  // Highlight numbers with units (e.g., "5 ms", "3.35 TB/s", "989 TFLOPS")
  const parts = text.split(
    /([\d,]+(?:\.\d+)?(?:×\d+(?:\^\d+)?)?)\s*((?:TFLOPS|TOPS|GFLOPS|TB\/s|GB\/s|MB\/s|GB|MB|KB|ms|μs|ns|s|%|FLOPs\/byte|FLOPs|bytes?|FLOP)(?:\/s)?)/gi
  );

  if (parts.length === 1) return <>{text}</>;

  return (
    <>
      {parts.map((part, i) => {
        // Pattern: [before, number, unit, ...rest]
        // Groups come in triples: text, number, unit
        if (i % 3 === 1) {
          // This is the number part
          return (
            <span key={i} className="font-semibold text-textPrimary">
              {part}
            </span>
          );
        }
        if (i % 3 === 2) {
          // This is the unit part
          return (
            <span key={i} className="text-textTertiary">
              {" "}{part}
            </span>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}
